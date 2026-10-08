"""Train the CrohnIPI lesion model on an NVIDIA GPU (Windows or Linux), official 5-fold CV.

Run from the `interim_build/gpu` folder:
    python train_gpu.py --backbone effb0 --epochs 30
    python train_gpu.py --backbone effb2 --epochs 30 --batch 16 --accum 2

What it does, per fold:
  * trains on the 4 other official CrohnIPI folds with a FIXED recipe (no epoch picking on the test fold)
  * keeps an exponential moving average (EMA) of the weights and evaluates the EMA model (decided in advance)
  * predicts the held-out fold with 8-view test-time augmentation (4 rotations x horizontal flip;
    valid because the capsule view is a circle with no up or down)
  * saves  runs_gpu/<backbone>/fold<k>.pt , oof_fold<k>.npy , idx_fold<k>.npy , log_fold<k>.json

Then run  evaluate_oof.py  (metrics + comparison with the current model) and  export_to_app.py  (ONNX for the app).
"""
import argparse, copy, json, math, os, sys, time, warnings
import numpy as np
# harmless with mixed precision: the gradient scaler may skip the first step(s) while it calibrates
warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`")
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import timm
except ImportError:
    sys.exit("timm is missing:  pip install timm")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.normpath(os.path.join(HERE, "..", "..", "temorary solution", "DataCrohnIPI"))
CLASSES = ["N", "E", "O", "AU", "U3-10", "U>10", "S"]
# Memory layout. Channels-last is OFF by default: on the RTX 3050 laptop (cuDNN 9.19) bench_gpu.py measured it 7x slower
# (B0: 35 vs 261 images/s). Same model and same numbers either way; only speed differs. --channels-last turns it back on.
MEMFMT = torch.contiguous_format
# EfficientNet family only: pooled features go straight into one linear layer, so the app's
# class-activation maps (W . feature map) stay mathematically exact.
ARCH = {
    "effb0": "efficientnet_b0.ra_in1k",
    "effb2": "efficientnet_b2.ra_in1k",
    "effb3": "efficientnet_b3.ra2_in1k",
    "effv2s": "tf_efficientnetv2_s.in21k_ft_in1k",
}


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="effb0", choices=sorted(ARCH))
    p.add_argument("--folds", default="1,2,3,4,5", help="comma list of official folds to run, or 'all' = train on every CrohnIPI frame")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16, help="images per GPU step (16 fits a 4 GB RTX 3050 for B0/B2)")
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (effective batch = batch x accum)")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.998, help="EMA decay; 0 disables EMA")
    p.add_argument("--img", type=int, default=224, help="input size (the app expects 224 unless you change crohnipi_meta.json)")
    p.add_argument("--data", default=DEFAULT_DATA, help="folder with CrohnIPI_description.csv and imgs/")
    p.add_argument("--out", default=None, help="output folder (default runs_gpu/<backbone>)")
    p.add_argument("--weights", default=None, help="optional local ImageNet .pth instead of downloading")
    p.add_argument("--no-pretrained", action="store_true", help="random init (only for smoke tests)")
    p.add_argument("--cpu", action="store_true", help="force CPU (smoke tests only)")
    p.add_argument("--limit", type=int, default=0, help="use only N images (smoke tests only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--channels-last", action="store_true", help="channels-last memory layout (slower on the RTX 3050; see bench_gpu.py)")
    # hard-negative mode (fix for whole-video ranking): add frames of lesion-free Kvasir-Capsule videos as "normal"
    p.add_argument("--kvasir-index", default=None, help="data_kvasir_video_frames/index.csv; enables hard negatives")
    p.add_argument("--kvasir-group", default=None, choices=["A", "B"], help="use negatives only from this half of the videos (test on the other half)")
    p.add_argument("--neg-per-video", type=int, default=400, help="negative frames per lesion-free training video (60%% hardest by the v1 model, 40%% random)")
    return p.parse_args()


def load_data(data_dir, size, cache_dir, limit=0):
    csv = os.path.join(data_dir, "CrohnIPI_description.csv")
    if not os.path.exists(csv):
        sys.exit(f"Cannot find {csv}. Pass --data <folder containing CrohnIPI_description.csv and imgs/>")
    d = pd.read_csv(csv)
    d.columns = ["frame", "label", "fold"]
    d["y"] = d.label.map(CLASSES.index)
    d["row"] = np.arange(len(d))  # index into the official CSV order (used by evaluate_oof.py)
    if limit:
        d = d[d.groupby("fold").cumcount() < max(1, limit // 5)].reset_index(drop=True)
    tag = f"_lim{limit}" if limit else ""
    cache = os.path.join(cache_dir, f"crohnipi_{size}{tag}.npy")
    if os.path.exists(cache):
        X = np.load(cache)
    else:
        print(f"Caching {len(d)} images at {size}x{size} -> {cache}", flush=True)
        X = np.stack([np.asarray(Image.open(os.path.join(data_dir, "imgs", f)).convert("RGB").resize((size, size), Image.BICUBIC))
                      for f in d.frame])
        np.save(cache, X)
    return d, X


def load_kvasir_negatives(index_csv, group, per_video, seed=0):
    """Frames from lesion-free videos of one half (A/B) -> uint8 array, all labelled 0 (normal)."""
    import csv
    sys.path.insert(0, HERE)
    from kvasir_groups import groups
    g = groups(index_csv); root = os.path.dirname(index_csv); rng = np.random.default_rng(seed)
    rows = [r for r in csv.DictReader(open(index_csv)) if g[r["video_id"]] == group and r["video_has_inflammatory"] == "0"]
    by_v = {}
    for r in rows:
        by_v.setdefault(r["video_id"], []).append(r)
    files = []
    for v, rs in sorted(by_v.items()):
        sc = np.array([float(r["v1_lesion_score"]) if r["v1_lesion_score"] not in ("", "nan") else 0.0 for r in rs])
        n_hard = min(len(rs), int(per_video * 0.6)); order = np.argsort(-sc)
        pick = list(order[:n_hard]); rest = order[n_hard:]
        pick += list(rng.choice(rest, size=min(len(rest), per_video - n_hard), replace=False)) if len(rest) else []
        files += [rs[i]["file"] for i in pick]
    X = np.stack([np.asarray(Image.open(os.path.join(root, f)).convert("RGB")) for f in files])
    print(f"Hard negatives: {len(files)} frames from {len(by_v)} lesion-free videos of half {group}", flush=True)
    return X, files, sorted(by_v)


def make_model(key, weights=None, pretrained=True):
    m = timm.create_model(ARCH[key], pretrained=(pretrained and weights is None), num_classes=7)
    if weights:
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith(("classifier.", "fc.", "head."))}
        missing, _ = m.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith(("classifier", "fc", "head"))]
        if bad:
            sys.exit(f"--weights does not match {ARCH[key]}: missing {bad[:5]}")
    return m


class EMA:
    """Exponential moving average of all weights (BatchNorm statistics included)."""

    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        self.steps = 0
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for e, p in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)


def augment(x):
    """x: float tensor in [0,1], B x 3 x H x W, on the training device."""
    n, dev = x.shape[0], x.device
    ang = torch.rand(n, device=dev) * 2 * math.pi
    sc = 1 / (0.85 + 0.25 * torch.rand(n, device=dev))
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    tx, ty = (torch.rand(n, device=dev) - 0.5) * 0.1, (torch.rand(n, device=dev) - 0.5) * 0.1
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="zeros", align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3
    s = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.2
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * c + m * b
    g = x.mean(1, keepdim=True)
    x = (x - g) * s + g
    return x.clamp(0, 1)


def tta_views(x):
    out = []
    for k in range(4):
        r = torch.rot90(x, k, (2, 3))
        out += [r, r.flip(3)]
    return out


def batch_tensor(X, idx, dev):
    x = torch.from_numpy(np.ascontiguousarray(X[idx])).to(dev, non_blocking=True)
    return x.permute(0, 3, 1, 2).float().div_(255.0)


@torch.no_grad()
def predict(model, X, idx, mean, std, dev, amp):
    model.eval()
    P = []
    for i in range(0, len(idx), 32):
        x = (batch_tensor(X, idx[i:i + 32], dev) - mean) / std
        x = x.contiguous(memory_format=MEMFMT)
        probs = []
        for v in tta_views(x):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                logits = model(v)
            probs.append(F.softmax(logits.float(), 1))
        P.append(torch.stack(probs).mean(0).cpu().numpy())
    return np.concatenate(P)


def run_fold(fold, args, d, X, dev, out_dir, extra=None):
    """fold = 1..5: hold out that official fold.  fold = 0: train on all CrohnIPI (no CrohnIPI test).
    extra = uint8 array of additional normal frames (hard negatives), training only."""
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    amp = dev.type == "cuda"
    y = d.y.values
    tr = np.where(d.fold.values != fold)[0]
    te = np.where(d.fold.values == fold)[0]
    if extra is not None and len(extra):
        X = np.concatenate([np.asarray(X), extra]); y = np.concatenate([y, np.zeros(len(extra), y.dtype)])
        tr = np.concatenate([tr, np.arange(len(d), len(d) + len(extra))])

    model = make_model(args.backbone, args.weights, pretrained=not args.no_pretrained).to(dev)
    model = model.to(memory_format=MEMFMT)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)

    cnt = np.bincount(y[tr], minlength=7).astype(np.float64)
    w = (cnt.max() / np.maximum(cnt, 1)) ** 0.5
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32, device=dev),
                               label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(tr) / (args.batch * args.accum))
    total = args.epochs * steps_per_epoch
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, args.ema) if args.ema > 0 else None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tag = f"fold{fold}" if fold else "all"
    ckpt = os.path.join(out_dir, f"{tag}_resume.pt")  # written after every epoch, so a crash loses at most one epoch
    log, start = [], 0
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        if (st.get("backbone"), st.get("epochs"), st.get("n_train"), st.get("img")) == (args.backbone, args.epochs, len(tr), args.img):
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"])
            if ema and st.get("ema") is not None:
                ema.model.load_state_dict(st["ema"]); ema.steps = st["ema_steps"]
            log, start = st["log"], st["epoch"]
            np.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"].cpu())
            if dev.type == "cuda" and st.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([r.cpu() for r in st["cuda_rng"]])
            print(f"[{args.backbone} {tag}] resuming after epoch {start} of {args.epochs}", flush=True)
        else:
            print(f"[{args.backbone} {tag}] {ckpt} was made with other settings; starting from epoch 1", flush=True)
    for ep in range(start, args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        t0, tl, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, i in enumerate(range(0, len(perm), args.batch)):
            idx = np.sort(perm[i:i + args.batch])
            x = (augment(batch_tensor(X, idx, dev)) - mean) / std
            x = x.contiguous(memory_format=MEMFMT)
            yb = torch.from_numpy(y[idx]).to(dev)
            for attempt in range(3):
                try:
                    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                        loss = crit(model(x), yb) / args.accum
                    scaler.scale(loss).backward()
                    break
                except Exception as e:  # torch.OutOfMemoryError or "CUDA error: out of memory"
                    if "out of memory" not in str(e).lower() or attempt == 2:
                        print(f"\nTraining stopped in epoch {ep + 1}. Run the same command again: it resumes after epoch {ep}.", flush=True)
                        raise
                    print("  GPU out of memory (is another program using the GPU?) - freeing memory and retrying this batch", flush=True)
                    loss = None
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    time.sleep(10)
            last = i + args.batch >= len(perm)
            if (bi + 1) % args.accum == 0 or last:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema:
                    ema.update(model)
            tl += loss.item() * args.accum * len(idx)
            seen += len(idx)
            if ep == 0 and bi % 100 == 99:  # progress inside the first epoch, so a long epoch does not look frozen
                el = time.time() - t0
                print(f"  epoch 1: {seen}/{len(perm)} images, {el:.0f} s (about {el * len(perm) / seen:.0f} s per epoch)", flush=True)
        rec = dict(epoch=ep + 1, train_loss=round(tl / seen, 4), sec=round(time.time() - t0, 1),
                   lr=round(sched.get_last_lr()[0], 7))
        if dev.type == "cuda":
            rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        log.append(rec)
        print(f"[{args.backbone} fold {fold}] {rec}", flush=True)
        st = dict(epoch=ep + 1, epochs=args.epochs, backbone=args.backbone, n_train=len(tr), img=args.img, log=log,
                  model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                  ema=ema.model.state_dict() if ema else None, ema_steps=ema.steps if ema else 0,
                  np_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None)
        torch.save(st, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
        del st

    final = ema.model if ema else model
    torch.save(final.state_dict(), os.path.join(out_dir, f"{tag}.pt"))
    try:
        os.remove(ckpt)
    except OSError:
        pass
    if not len(te):
        json.dump(dict(args={k: v for k, v in vars(args).items()}, epochs=log), open(os.path.join(out_dir, f"log_{tag}.json"), "w"), indent=1)
        print(f"[{args.backbone} {tag}] saved (no CrohnIPI test fold)", flush=True)
        return dict(fold=tag)
    P = predict(final, X, te, mean, std, dev, amp)
    rows = d.row.values[te]
    np.save(os.path.join(out_dir, f"oof_fold{fold}.npy"), P)
    np.save(os.path.join(out_dir, f"idx_fold{fold}.npy"), rows)
    yt = y[te]
    acc7 = float((P.argmax(1) == yt).mean())
    bacc = float(np.mean([(P.argmax(1)[yt == c] == c).mean() for c in range(7) if (yt == c).any()]))
    binacc = float(((1 - P[:, 0] >= 0.5) == (yt != 0)).mean())
    summary = dict(fold=fold, n_test=int(len(te)), acc7=round(acc7, 4), bal_acc7=round(bacc, 4), bin_acc_at_0_5=round(binacc, 4))
    print(f"[{args.backbone} fold {fold}] TEST {summary}", flush=True)
    json.dump(dict(args={k: v for k, v in vars(args).items()}, epochs=log, test=summary),
              open(os.path.join(out_dir, f"log_fold{fold}.json"), "w"), indent=1)
    del model, final, ema
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main():
    global MEMFMT
    args = get_args()
    MEMFMT = torch.channels_last if args.channels_last else torch.contiguous_format
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    else:
        print("WARNING: no CUDA GPU found, running on CPU (slow). Check:  python -c \"import torch; print(torch.cuda.is_available())\"",
              flush=True)
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    out_dir = args.out or os.path.join(HERE, "runs_gpu", args.backbone + (f"_lim{args.limit}" if args.limit else ""))
    os.makedirs(out_dir, exist_ok=True)
    d, X = load_data(args.data, args.img, out_dir, args.limit)
    print(f"Data: {len(d)} labelled frames from {args.data}", flush=True)
    folds = [0] if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    extra = None
    if args.kvasir_index:
        if not args.kvasir_group:
            sys.exit("--kvasir-index needs --kvasir-group A or B")
        extra, files, vids = load_kvasir_negatives(args.kvasir_index, args.kvasir_group, args.neg_per_video, args.seed)
        json.dump(dict(group=args.kvasir_group, videos=vids, files=files), open(os.path.join(out_dir, "hard_negatives.json"), "w"))
    results = []
    t0 = time.time()
    for k in folds:
        results.append(run_fold(k, args, d, X, dev, out_dir, extra))
    print(f"\nDone {len(folds)} fold(s) in {(time.time() - t0) / 60:.1f} min. Outputs in {out_dir}")
    for r in results:
        print("  ", r)
    print("\nNext:  python evaluate_oof.py --runs", os.path.relpath(out_dir, HERE))


if __name__ == "__main__":
    main()
