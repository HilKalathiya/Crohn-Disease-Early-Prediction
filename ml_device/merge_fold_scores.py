"""Average per-fold bulk scores (FOLDS=k runs) into 5-model ensemble shards that analyze_kvasir_full.py reads."""
import os, glob, numpy as np
B = os.path.expanduser("~/mnt/MP_Cohn Disease/interim_build/results/bulk")
PREFIX = os.environ.get("MODEL_PREFIX", "crohnipi_effb0v2")
per = {}
for k in range(1, 6):
    d = {}
    for f in glob.glob(os.path.join(B, f"kvasir_frames_{PREFIX}_f{k}", "shard_*.npz")):
        with np.load(f) as z:
            for i, p in enumerate(z["path"]):
                d[str(p)] = (str(z["cls_dir"][i]), str(z["video_id"][i]), str(z["finding_class"][i]), z["probs"][i])
    per[k] = d; print("fold", k, len(d))
common = sorted(set.intersection(*[set(d) for d in per.values()]))
print("frames scored by all 5 folds:", len(common))
P = np.stack([np.stack([per[k][p][3] for p in common]) for k in range(1, 6)])
out = os.path.join(B, f"kvasir_frames_{PREFIX}"); os.makedirs(out, exist_ok=True)
meta = [per[1][p] for p in common]
np.savez_compressed(os.path.join(out, "shard_merged_000000_all.npz"), cls_dir=np.array([m[0] for m in meta]), video_id=np.array([m[1] for m in meta]),
                    finding_class=np.array([m[2] for m in meta]), path=np.array(common), probs=P.mean(0).astype(np.float32), probs_sd=P.std(0).astype(np.float32))
print("wrote", out)
