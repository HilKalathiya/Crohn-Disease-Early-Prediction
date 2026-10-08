"""One-minute GPU speed test: why is training slow?  Run while nothing else uses the GPU:

    python bench_gpu.py

Times training steps of the EfficientNet models with the same data path, augmentation and EMA as train_gpu.py,
with and without channels-last memory layout and mixed precision, and prints images per second for each.
"""
import time, sys, os
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_gpu import make_model, augment, batch_tensor, EMA  # noqa: E402


def bench(key, img, batch, channels_last, amp, ema_on, steps=15, dev=torch.device("cuda")):
    torch.manual_seed(0)
    m = make_model(key, pretrained=False).to(dev)
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    ema = EMA(m, 0.998) if ema_on else None
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = nn.CrossEntropyLoss()
    X = np.random.randint(0, 255, (batch * 4, img, img, 3), dtype=np.uint8)
    y = torch.randint(0, 7, (batch,), device=dev)
    m.train()
    t = None
    for i in range(steps + 5):
        if i == 5:
            torch.cuda.synchronize(); t = time.time()
        idx = np.sort(np.random.choice(len(X), batch, replace=False))
        x = augment(batch_tensor(X, idx, dev))
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            loss = crit(m(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if ema:
            ema.update(m)
        loss.item()
    torch.cuda.synchronize()
    return steps * batch / (time.time() - t)


def main():
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible to PyTorch.")
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuDNN {torch.backends.cudnn.version()}")
    rows = [("effb0", 224, 32), ("effb2", 320, 8)]
    print(f"{'model':6s} {'size':>4s} {'batch':>5s} {'channels_last':>13s} {'mixed_prec':>10s} {'EMA':>4s} {'images/s':>9s}")
    for key, img, b in rows:
        for cl in (True, False):
            for amp in (True, False):
                r = bench(key, img, b, cl, amp, ema_on=True)
                print(f"{key:6s} {img:4d} {b:5d} {str(cl):>13s} {str(amp):>10s} {'yes':>4s} {r:9.1f}", flush=True)
        r = bench(key, img, b, True, True, ema_on=False)
        print(f"{key:6s} {img:4d} {b:5d} {'True':>13s} {'True':>10s} {'no':>4s} {r:9.1f}", flush=True)
        torch.cuda.empty_cache()
    print("\nSend this table to Claude. Also run  nvidia-smi  during training: GPU-Util should be near 100%.")


if __name__ == "__main__":
    main()
