"""C: HyperKvasir.
  C1  polyp localisation: findings-screen 'polyp' heat-map vs the 1,000 expert polyp masks (pointing game).
  C2  video test of the UC grader: 15 'colitis' videos vs 31 clean-bowel (BBPS 2-3) videos, 1 frame per second.
Resumable: each part saves per-item outputs; run repeatedly until it prints DONE. Usage: python hyperkvasir_eval.py c1|c2|report"""
import os, sys, json, glob, time, subprocess, numpy as np
from PIL import Image
BASE = os.path.expanduser("~/mnt/MP_Cohn Disease")
HK = os.path.join(BASE, "temorary solution/hyper-kvasir")
OUT = os.path.join(BASE, "interim_build/results/bulk")
sys.path.insert(0, os.path.join(BASE, "interim_build/app"))
from backend.heads import SharedBackbone  # noqa: E402
from backend.core import softmax  # noqa: E402
T0 = time.time(); BUDGET = float(os.environ.get("BUDGET", 140))

def c1():
    out = os.path.join(OUT, "hyperkvasir_polyps"); os.makedirs(out, exist_ok=True)
    imgs = sorted(os.listdir(os.path.join(HK, "segmented-images/images")))
    done = set()
    for f in glob.glob(os.path.join(out, "part_*.json")): done.update(json.load(open(f)).keys())
    todo = [f for f in imgs if f not in done]
    print("C1 polyps: done", len(done), "todo", len(todo), flush=True)
    if not todo: print("C1 DONE"); return
    bb = SharedBackbone(); h = bb.heads["sb_findings_head"]; k = h.classes.index("polyp")
    res = {}
    for i in range(0, len(todo), 16):
        if time.time() - T0 > BUDGET: break
        names = todo[i:i + 16]
        ims = [Image.open(os.path.join(HK, "segmented-images/images", n)).convert("RGB") for n in names]
        pooled, fmap = bb.features(ims); P, C = h.predict(pooled, fmap)
        for j, n in enumerate(names):
            m = np.asarray(Image.open(os.path.join(HK, "segmented-images/masks", n)).convert("L")) > 127
            H, W = m.shape; cam = C[j, k]
            r, c = np.unravel_index(int(cam.argmax()), cam.shape)
            cy, cx = int((r + 0.5) * H / 7), int((c + 0.5) * W / 7)
            y0, y1, x0, x1 = max(0, cy - 15), min(H, cy + 16), max(0, cx - 15), min(W, cx + 16)  # 15-px tolerance
            centres = [m[int((rr + 0.5) * H / 7), int((cc + 0.5) * W / 7)] for rr in range(7) for cc in range(7)]
            res[n] = dict(hit=bool(m[y0:y1, x0:x1].any()), chance=float(np.mean(centres)), mask_frac=float(m.mean()),
                          p_polyp=float(P[j, k]), top=h.classes[int(P[j].argmax())])
    json.dump(res, open(os.path.join(out, f"part_{int(time.time())}.json"), "w"))
    print("C1 scored", len(res), "this call; remaining", len(todo) - len(res), flush=True)

def video_list():
    base = os.path.join(HK, "labeled-videos/lower-gi-tract")
    v = [("colitis", p) for p in sorted(glob.glob(os.path.join(base, "pathological-findings/colitis/*.avi")))]
    v += [("clean", p) for p in sorted(glob.glob(os.path.join(base, "quality-of-mucosal-view/BBPS-2-3/*.avi")))]
    return v

def frames_1fps(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", "fps=1,scale=224:224:flags=bicubic", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 224, 224, 3)

def c2():
    out = os.path.join(OUT, "hyperkvasir_videos"); vids = video_list()
    todo = [(g, p) for g, p in vids if not os.path.exists(os.path.join(out, os.path.basename(p) + ".npz"))]
    print("C2 videos: total", len(vids), "todo", len(todo), flush=True)
    if not todo: print("C2 DONE"); return
    bb = SharedBackbone(); h = bb.heads["uc_mayo_head"]; n = 0
    for g, p in todo:
        if time.time() - T0 > BUDGET: break
        fr = frames_1fps(p); ims = [Image.fromarray(f) for f in fr]
        pooled, fmap = bb.features(ims); P, _ = h.predict(pooled, fmap)
        np.savez(os.path.join(out, os.path.basename(p) + ".npz"), group=g, probs=P.astype(np.float32))
        n += 1; print(f"  {g:8s} {os.path.basename(p)[:8]} frames={len(fr)} active={P[:, 2:].sum(1).mean():.3f}", flush=True)
    print("C2 scored", n, "videos this call; remaining", len(todo) - n, flush=True)

def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool); r = np.argsort(np.argsort(s)) + 1.0
    for v in np.unique(s):  # tie-average
        t = s == v
        if t.sum() > 1: r[t] = r[t].mean()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))

def report():
    R = {}
    parts = {}
    for f in glob.glob(os.path.join(OUT, "hyperkvasir_polyps/part_*.json")): parts.update(json.load(open(f)))
    if parts:
        v = list(parts.values())
        R["polyp_localisation"] = dict(n=len(v), pointing_hit_rate=float(np.mean([x["hit"] for x in v])),
                                       chance=float(np.mean([x["chance"] for x in v])), mean_mask_area=float(np.mean([x["mask_frac"] for x in v])),
                                       mean_p_polyp=float(np.mean([x["p_polyp"] for x in v])), detected_at_0_5=float(np.mean([x["p_polyp"] >= 0.5 for x in v])),
                                       note="findings screen trained on SEE-AI capsule frames; HyperKvasir = colonoscopy (other camera and organ)")
    vs = []
    for f in sorted(glob.glob(os.path.join(OUT, "hyperkvasir_videos/*.npz"))):
        z = np.load(f); P = z["probs"]; act = P[:, 2:].sum(1); exp = P @ np.arange(4)
        vs.append(dict(video=os.path.basename(f)[:-8], group=str(z["group"]), frames=int(len(P)), mean_active=float(act.mean()),
                       p90_expected_mayo=float(np.percentile(exp, 90)), frac_active_frames=float((act >= 0.5).mean())))
    if vs:
        y = np.array([v["group"] == "colitis" for v in vs])
        R["uc_video_test"] = dict(n_colitis=int(y.sum()), n_clean=int((~y).sum()),
                                  auroc_mean_active=auroc([v["mean_active"] for v in vs], y),
                                  auroc_p90_expected_mayo=auroc([v["p90_expected_mayo"] for v in vs], y),
                                  auroc_frac_active_frames=auroc([v["frac_active_frames"] for v in vs], y),
                                  mean_active_colitis=float(np.mean([v["mean_active"] for v in vs if v["group"] == "colitis"])),
                                  mean_active_clean=float(np.mean([v["mean_active"] for v in vs if v["group"] == "clean"])),
                                  videos=vs, note="UC grader trained on LIMUC; HyperKvasir colitis videos (type not specified) vs BBPS 2-3 clean-bowel videos; 1 frame/s")
    json.dump(R, open(os.path.join(BASE, "interim_build/results/hyperkvasir_extra.json"), "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "videos"} for k, v in R.items()}, indent=1))

if __name__ == "__main__":
    {"c1": c1, "c2": c2, "report": report}[sys.argv[1]]()
