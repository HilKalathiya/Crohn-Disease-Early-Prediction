"""Extract every 50th frame (224x224 JPEG) of the 43 labelled Kvasir-Capsule videos + an index CSV.
The frames feed the GPU hard-negative experiment and a fixed, fast whole-video evaluation set (no ffmpeg needed later).
Resumable per video. Usage: python extract_video_frames.py [--seconds 140]"""
import os, csv, sys, time, argparse, subprocess, numpy as np
from PIL import Image
BASE = os.path.expanduser("~/mnt/MP_Cohn Disease")
KC = os.path.join(BASE, "temorary solution/Kvasir-Capsule")
OUT = os.path.join(BASE, "interim_build/data_kvasir_video_frames")
VID = os.path.join(BASE, "interim_build/results/bulk/kvasir_videos")
STRIDE, S = 50, 224
INFL = ("Ulcer", "Erosion", "Erythema")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seconds", type=float, default=140); a = ap.parse_args(); t0 = time.time()
    meta = list(csv.DictReader(open(os.path.join(KC, "metadata.csv"), encoding="utf-8", errors="replace"), delimiter=";"))
    lab = {}
    for r in meta: lab.setdefault(r["video_id"], []).append((int(r["frame_number"]), r["finding_class"]))
    vids = sorted(lab)
    for v in vids:
        done_flag = os.path.join(OUT, v, "_done.csv")
        if os.path.exists(done_flag): continue
        if time.time() - t0 > a.seconds: print("[pause] before", v, flush=True); return
        os.makedirs(os.path.join(OUT, v), exist_ok=True)
        path = os.path.join(KC, "labelled_videos", f"{v}.mp4")
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='not(mod(n\\,{STRIDE}))',scale={S}:{S}:flags=bicubic",
               "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE); k = 0; rows = []
        # v1 lesion scores from the stride-10 scan (stride-50 frames are a subset)
        z = np.load(os.path.join(VID, f"{v}.npz")); smap = dict(zip(z["frame"].tolist(), (1 - z["probs"][:, 0]).tolist()))
        infl = np.array([n for n, c in lab[v] if c in INFL]); anyl = lab[v]
        while True:
            buf = p.stdout.read(S * S * 3)
            if len(buf) < S * S * 3: break
            n = k * STRIDE; k += 1
            fn = f"frame_{n:06d}.jpg"
            Image.frombuffer("RGB", (S, S), buf).save(os.path.join(OUT, v, fn), quality=90)
            near = [c for m, c in anyl if abs(m - n) <= 25]
            rows.append(dict(video_id=v, frame=n, file=f"{v}/{fn}", video_has_inflammatory=int(len(infl) > 0),
                             expert_label_within_25=near[0] if near else "", v1_lesion_score=round(smap.get(n, float("nan")), 4)))
        p.wait()
        with open(done_flag, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"  {v}: {len(rows)} frames, t={time.time() - t0:.0f}s", flush=True)
    # merge all per-video indices
    allrows = []
    for v in vids:
        allrows += list(csv.DictReader(open(os.path.join(OUT, v, "_done.csv"))))
    with open(os.path.join(OUT, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print("ALL DONE", len(allrows), "frames", flush=True)

if __name__ == "__main__":
    main()
