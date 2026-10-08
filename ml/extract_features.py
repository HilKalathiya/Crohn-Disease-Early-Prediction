"""Frozen-backbone embeddings (orig + h-flip TTA averaged) for transfer-learning baselines."""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from data import load_images, MEAN, STD, CACHE
from backbones import create
torch.set_num_threads(2)
d, X = load_images(224)
for key in sys.argv[1:]:
    out = os.path.join(CACHE, f"feat_{key}.npy")
    if os.path.exists(out): continue
    m = create(key, num_classes=0).eval().to(memory_format=torch.channels_last); t = time.time(); feats = []
    mean = np.array(m.pretrained_cfg['mean'], dtype=np.float32); std = np.array(m.pretrained_cfg['std'], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 64):
            x = torch.from_numpy(((X[i:i+64] / 255.0 - mean) / std).astype(np.float32)).permute(0, 3, 1, 2)
            f = (m(x) + m(torch.flip(x, dims=[3]))) / 2
            feats.append(f.numpy())
    np.save(out, np.concatenate(feats)); print(key, "done", time.time() - t, flush=True)
