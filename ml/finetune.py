"""Fine-tune EfficientNet-B0 on CrohnIPI, official 5-fold CV. Fixed recipe (no epoch selection on the test fold).

Usage: python ml/finetune.py <backbone_key> <fold|all> [epochs]
"""
import sys, os, time, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from data import load_images, CLASSES
from backbones import create
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
key, fold = sys.argv[1], sys.argv[2]; EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
BS, LR, WD, LS = 32, 6e-4, 1e-2, 0.1
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", key); os.makedirs(OUT, exist_ok=True)
d, X = load_images(224); y = d.y.values
tr = np.arange(len(y)) if fold == "all" else np.where(d.fold.values != int(fold))[0]
te = np.array([], int) if fold == "all" else np.where(d.fold.values == int(fold))[0]
model = create(key, num_classes=7).to(memory_format=torch.channels_last)
mean = torch.tensor(model.pretrained_cfg["mean"]).view(1, 3, 1, 1); std = torch.tensor(model.pretrained_cfg["std"]).view(1, 3, 1, 1)
cnt = np.bincount(y[tr], minlength=7); w = torch.tensor((cnt.max() / cnt) ** 0.5, dtype=torch.float32)  # inverse-sqrt class weights
crit = nn.CrossEntropyLoss(weight=w / w.mean(), label_smoothing=LS)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
steps = EPOCHS * math.ceil(len(tr) / BS); warm = math.ceil(len(tr) / BS)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

def to_tensor(idx):
    x = torch.from_numpy(np.ascontiguousarray(X[np.sort(idx)])).permute(0, 3, 1, 2).float() / 255.0
    return x, np.sort(idx)

def augment(x):
    n = x.shape[0]
    # random rotation (capsule field of view is circular -> rotation invariant), scale and flips via one affine warp
    ang = torch.rand(n) * 2 * math.pi; sc = 1 / (0.85 + 0.25 * torch.rand(n)); flip = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos * flip, -sin, (torch.rand(n) - 0.5) * 0.1], 1), torch.stack([sin * flip, cos, (torch.rand(n) - 0.5) * 0.1], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="zeros", align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.3; c = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.3; s = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.2
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * c + m * b; g = x.mean(1, keepdim=True); x = (x - g) * s + g
    return x.clamp(0, 1)

def predict(idx, tta=True):
    model.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            x, _ = to_tensor(idx[i:i + 64]); x = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
            views = [x, x.flip(3), x.flip(2), x.rot90(1, (2, 3))] if tta else [x]
            P.append(torch.stack([F.softmax(model(v), 1) for v in views]).mean(0).numpy())
    return np.concatenate(P)

log = []
for ep in range(EPOCHS):
    model.train(); perm = np.random.permutation(tr); t0 = time.time(); tl = 0
    for i in range(0, len(perm), BS):
        x, idx = to_tensor(perm[i:i + BS]); x = ((augment(x) - mean) / std).contiguous(memory_format=torch.channels_last)
        loss = crit(model(x), torch.from_numpy(y[idx])); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
    log.append(dict(epoch=ep + 1, train_loss=tl / len(tr), sec=time.time() - t0)); print(key, fold, log[-1], flush=True)
torch.save(model.state_dict(), os.path.join(OUT, f"fold{fold}.pt"))
if len(te):
    P = predict(te); np.save(os.path.join(OUT, f"oof_fold{fold}.npy"), P); np.save(os.path.join(OUT, f"idx_fold{fold}.npy"), te)
    acc = (P.argmax(1) == y[te]).mean(); bacc = np.mean([(P.argmax(1)[y[te] == c] == c).mean() for c in range(7)])
    print(key, fold, "TEST acc", round(acc, 4), "bal_acc", round(bacc, 4), flush=True)
json.dump(log, open(os.path.join(OUT, f"log_fold{fold}.json"), "w"))
