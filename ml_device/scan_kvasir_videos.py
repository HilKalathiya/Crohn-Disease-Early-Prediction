"""B2: scan whole Kvasir-Capsule videos (each = one complete capsule study, ~50,000 frames).
Stage 1: fold-1 CrohnIPI model scores every 10th frame.  Stage 2: 5-model ensemble re-scores the top 150 frames.
Resumable (segment files + per-video final file). Order alternates videos with / without expert inflammatory labels.
Usage: python scan_kvasir_videos.py [--seconds 140]"""
import os, sys, csv, json, time, glob, argparse, subprocess, numpy as np
from PIL import Image
import onnxruntime as ort
PREFIX = os.environ.get("MODEL_PREFIX", "crohnipi_effb0")  # crohnipi_effb0v2 = improved model
BASE = os.path.expanduser("~/mnt/MP_Cohn Disease")
KC = os.path.join(BASE, "temorary solution/Kvasir-Capsule")
MODELS = os.path.join(BASE, "interim_build/app/models")
OUT = os.path.join(BASE, "interim_build/results/bulk/kvasir_videos" + ("" if PREFIX == "crohnipi_effb0" else "_" + PREFIX))
THUMBS = os.path.join(OUT, "thumbs")
STRIDE, SEG, TOPN, FPS = 10, 15000, 150, 30
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1); STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
INFL = ("Ulcer", "Erosion", "Erythema")

def sess(k):
    so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.enable_cpu_mem_arena = False
    return ort.InferenceSession(os.path.join(MODELS, f"{PREFIX}_fold{k}.onnx"), so, providers=["CPUExecutionProvider"])

def prep(fr):  # fr: N x 224 x 224 x 3 uint8
    x = fr.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return np.ascontiguousarray((x - MEAN) / STD, np.float32)

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def video_order():
    meta = list(csv.DictReader(open(os.path.join(KC, "metadata.csv"), encoding="utf-8", errors="replace"), delimiter=";"))
    infl = {}
    for r in meta:
        infl.setdefault(r["video_id"], 0)
        if r["finding_class"] in INFL: infl[r["video_id"]] += 1
    pos = sorted([v for v in infl if infl[v] > 0], key=lambda v: -infl[v]); neg = sorted(v for v in infl if infl[v] == 0)
    order = []
    for i in range(max(len(pos), len(neg))):
        if i < len(pos): order.append(pos[i])
        if i < len(neg): order.append(neg[i])
    return order, infl

def nframes(path):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path],
                       capture_output=True, text=True).stdout.strip()
    return int(o)

def decode_segment(path, start, count):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start / FPS:.4f}", "-i", path, "-t", f"{count / FPS:.4f}",
           "-vf", f"select='not(mod(n\\,{STRIDE}))',scale=224:224:flags=bicubic", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 224, 224, 3)

def decode_frame(path, n, size=None):
    vf = ["-vf", f"scale={size}:{size}:flags=bicubic"] if size else []
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{n / FPS:.4f}", "-i", path, "-frames:v", "1", *vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    s = size or 336
    return np.frombuffer(raw, np.uint8).reshape(s, s, 3)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seconds", type=float, default=140); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); os.makedirs(THUMBS, exist_ok=True)
    order, infl = video_order()
    if a.limit: order = order[:a.limit]
    s1 = None; ens = None
    for v in order:
        final = os.path.join(OUT, f"{v}.npz")
        if os.path.exists(final): continue
        path = os.path.join(KC, "labelled_videos", f"{v}.mp4"); N = nframes(path)
        done_segs = sorted(x for x in glob.glob(os.path.join(OUT, f"{v}_seg*.npz")) if not x.endswith(".tmp.npz"))
        covered = max([int(np.load(s)["frame"].max()) + STRIDE for s in done_segs], default=0)
        for start in range(0, N, SEG):
            seg = os.path.join(OUT, f"{v}_seg{start:06d}.npz")
            if os.path.exists(seg) or start < covered: continue
            if time.time() - t0 > a.seconds: print(f"[pause] {v} at frame {start}/{N}", flush=True); return
            fr = decode_segment(path, start, min(SEG, N - start))
            if s1 is None: s1 = sess(1)
            outs = [s1.run(None, {"image": prep(fr[i:i + 16])}) for i in range(0, len(fr), 16)]
            P = np.concatenate([softmax(o[0]) for o in outs])
            FE = np.concatenate([o[1].mean((2, 3)) for o in outs]).astype(np.float16)  # pooled fold-1 features (for the quality filter)
            frames_n = np.arange(start, start + len(fr) * STRIDE, STRIDE)[:len(fr)]
            keep = np.argsort(-(1 - P[:, 0]))[:TOPN]  # cache candidate pixels so stage 2 needs no re-decoding
            np.savez(seg + ".tmp.npz", frame=frames_n, probs=P.astype(np.float32), feat=FE, cand_frame=frames_n[keep], cand_pix=fr[keep])
            os.replace(seg + ".tmp.npz", seg)  # atomic: a killed call never leaves a broken segment
            print(f"  {v} seg {start:6d}: {len(fr)} frames, {time.time() - t0:.0f}s", flush=True)
        segs = sorted(x for x in glob.glob(os.path.join(OUT, f"{v}_seg*.npz")) if not x.endswith(".tmp.npz"))
        if os.environ.get("STAGE1_ONLY") == "1":  # merge stage-1 segments into one file; stage 2 runs later
            if not any(sg.endswith("_seg_all.npz") for sg in segs):
                zs = [np.load(sg) for sg in segs]
                Fm = np.concatenate([z["frame"] for z in zs]); Pm = np.concatenate([z["probs"] for z in zs]); FEm = np.concatenate([z["feat"] for z in zs])
                Fm, u = np.unique(Fm, return_index=True); Pm, FEm = Pm[u], FEm[u]
                cf = np.concatenate([z["cand_frame"] for z in zs]); cp = np.concatenate([z["cand_pix"] for z in zs])
                cs = np.array([1 - Pm[np.searchsorted(Fm, n)][0] for n in cf]); keep = np.argsort(-cs)[:TOPN]
                np.savez(os.path.join(OUT, f"{v}_seg_all.npz"), frame=Fm, probs=Pm, feat=FEm, cand_frame=cf[keep], cand_pix=cp[keep])
                os.makedirs(os.path.join(OUT, "segments"), exist_ok=True)
                for sg in segs: os.replace(sg, os.path.join(OUT, "segments", os.path.basename(sg)))
                print(f"[stage 1 done] {v} sampled={len(Fm)} t={time.time() - t0:.0f}s", flush=True)
            continue
        if time.time() - t0 > a.seconds + 15: print(f"[pause] {v} before stage 2", flush=True); return
        F = np.concatenate([np.load(s)["frame"] for s in segs]); P = np.concatenate([np.load(s)["probs"] for s in segs])
        F, u = np.unique(F, return_index=True); P = P[u]
        FEAT = np.concatenate([np.load(s)["feat"] for s in segs])[u] if all("feat" in np.load(s).files for s in segs) else None
        lesion = 1 - P[:, 0]; top = np.sort(F[np.argsort(-lesion)[:TOPN]])
        if ens is None: ens = [sess(k) for k in range(1, 6)]
        cache = {}
        for sg in segs:
            z = np.load(sg)
            if "cand_frame" in z.files:
                cache.update({int(n): px for n, px in zip(z["cand_frame"], z["cand_pix"])})
        imgs = np.stack([cache[int(n)] if int(n) in cache else decode_frame(path, int(n), 224) for n in top])
        x = prep(imgs); PE = np.stack([np.concatenate([softmax(s.run(None, {"image": x[i:i + 16]})[0]) for i in range(0, len(x), 16)]) for s in ens])
        pe, sd = PE.mean(0), PE.std(0)
        best = top[np.argsort(-(1 - pe[:, 0]))[:12]]
        os.makedirs(os.path.join(THUMBS, v), exist_ok=True)
        for n in best:
            Image.fromarray(decode_frame(path, int(n))).save(os.path.join(THUMBS, v, f"frame_{int(n):06d}.jpg"), quality=85)
        np.savez(final, frame=F, probs=P.astype(np.float32), top_frames=top, top_probs_ens=pe.astype(np.float32), top_sd=sd.astype(np.float32),
                 n_frames=N, expert_inflammatory=infl[v], **({"feat": FEAT} if FEAT is not None else {}))
        os.makedirs(os.path.join(OUT, "segments"), exist_ok=True)
        for s in segs: os.replace(s, os.path.join(OUT, "segments", os.path.basename(s)))  # no deletes in the user's folder
        print(f"[video done] {v} frames={N} sampled={len(F)} expert_infl={infl[v]} max_lesion={lesion.max():.3f} flagged={(lesion >= 0.5).mean():.3f} t={time.time() - t0:.0f}s", flush=True)
    print("ALL VIDEOS DONE", flush=True)

if __name__ == "__main__":
    main()
