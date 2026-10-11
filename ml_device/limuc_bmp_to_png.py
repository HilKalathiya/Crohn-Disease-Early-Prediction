"""Losslessly convert the LIMUC colonoscopy images from .bmp to .png: pixel-identical, about 38% of the size
(3.4 GB -> ~1.3 GB). Every PNG is decoded again and compared with the BMP pixels before the BMP is moved away.

    python ml_device/limuc_bmp_to_png.py                 # from interim_build; resumable
    BUDGET=150 python ml_device/limuc_bmp_to_png.py      # stop starting new files after 150 s (run again to continue)

Originals are MOVED (not deleted) to  <MP_Cohn Disease>/_TO_DELETE/temorary solution/Labeled Images for .../*.bmp .
The project code does not care about the extension: ml_device/pack.py and pack2.py list every file in the LIMUC
folders and open it with Pillow.
"""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))                       # MP_Cohn Disease
LIM = os.path.join(ROOT, "temorary solution", "Labeled Images for Ulcerative Colitis (LIMUC)")
TRASH = os.path.join(ROOT, "_TO_DELETE")
BUDGET = float(os.environ.get("BUDGET", "0")) or float("inf")


def convert(bmp):
    png = bmp[:-4] + ".png"
    with Image.open(bmp) as im:
        im.load(); mode, size, px = im.mode, im.size, im.tobytes()
        tmp = png + ".part"
        im.save(tmp, "PNG", compress_level=6)
    with Image.open(tmp) as chk:
        chk.load()
        ok = chk.mode == mode and chk.size == size and chk.tobytes() == px
    if not ok:                                              # keep the BMP; park the bad PNG with the trash
        bad = os.path.join(TRASH, "failed_png", os.path.relpath(tmp, ROOT))
        os.makedirs(os.path.dirname(bad), exist_ok=True); os.replace(tmp, bad)
        return bmp, 0, 0, False
    b_bmp = os.path.getsize(bmp)
    os.replace(tmp, png)
    dst = os.path.join(TRASH, os.path.relpath(bmp, ROOT))
    os.makedirs(os.path.dirname(dst), exist_ok=True); os.replace(bmp, dst)
    return bmp, b_bmp, os.path.getsize(png), True


def main():
    if not os.path.isdir(LIM):
        sys.exit(f"Not found: {LIM}")
    todo = sorted(os.path.join(r, f) for r, _, fs in os.walk(LIM) for f in fs if f.lower().endswith(".bmp"))
    print(f"{len(todo)} BMP files left", flush=True)
    t0, n, b_in, b_out, bad = time.time(), 0, 0, 0, 0
    with ProcessPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2)))) as ex:
        pending = set(); it = iter(todo)
        while True:
            while len(pending) < 32 and time.time() - t0 < BUDGET:
                nxt = next(it, None)
                if nxt is None: break
                pending.add(ex.submit(convert, nxt))
            if not pending: break
            done = next(as_completed(pending)); pending.discard(done)
            _, a, b, ok = done.result(); n += 1; b_in += a; b_out += b; bad += (not ok)
    left = sum(1 for r, _, fs in os.walk(LIM) for f in fs if f.lower().endswith(".bmp"))
    print(f"converted {n - bad} files ({b_in / 1e9:.2f} GB -> {b_out / 1e9:.2f} GB) in {time.time() - t0:.0f} s; "
          f"{bad} failed checks; {left} BMP files left", flush=True)


if __name__ == "__main__":
    main()
