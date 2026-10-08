"""Shared image + model utilities (onnxruntime only - no PyTorch needed to run the app)."""
import io, os, json, base64, numpy as np, onnxruntime as ort
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)

def _providers():
    """GPU if onnxruntime-gpu is installed and CROHNSIGHT_DEVICE is not 'cpu'; otherwise CPU (the default package)."""
    want = os.environ.get("CROHNSIGHT_DEVICE", "auto").lower()
    avail = ort.get_available_providers()
    if want != "cpu" and "CUDAExecutionProvider" in avail:
        if hasattr(ort, "preload_dlls"):  # finds CUDA/cuDNN DLLs shipped with PyTorch or the nvidia-* pip wheels
            try:
                ort.preload_dlls()
            except Exception:
                pass
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

PROVIDERS = _providers()

def session(name):
    so = ort.SessionOptions(); so.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    so.enable_cpu_mem_arena = False  # keep memory flat across large studies
    return ort.InferenceSession(os.path.join(MODELS, name), so, providers=PROVIDERS)

def read_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data)); im.load()
    return im.convert("RGB")

def to_input(images, size=224):
    x = np.stack([np.asarray(im.resize((size, size), Image.BICUBIC), np.float32) / 255.0 for im in images])
    return ((x - IMNET_MEAN) / IMNET_STD).transpose(0, 3, 1, 2).astype(np.float32).copy()

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True); e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)

def normalise_cam(cam):
    """cam: (..., 7, 7) -> scaled to [0,1] per map, ReLU first (only positive evidence)."""
    cam = np.maximum(cam, 0); mx = cam.reshape(*cam.shape[:-2], -1).max(-1)[..., None, None]
    return np.where(mx > 0, cam / np.maximum(mx, 1e-8), 0)

def thumb_b64(im: Image.Image, size=112, q=80):
    b = io.BytesIO(); im.resize((size, size), Image.BICUBIC).save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

def load_json(name, default=None):
    p = os.path.join(MODELS, name)
    return json.load(open(p)) if os.path.exists(p) else default
