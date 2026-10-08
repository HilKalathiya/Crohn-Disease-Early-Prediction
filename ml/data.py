"""CrohnIPI loading + in-memory cache (full-frame resize, no crop: lesions can sit at the FOV edge)."""
import os, numpy as np, pandas as pd
from PIL import Image
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "DataCrohnIPI")
CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")
CLASSES = ["N", "E", "O", "AU", "U3-10", "U>10", "S"]
CLASS_NAMES = {"N": "Normal mucosa", "E": "Erythema", "O": "Oedema", "AU": "Aphthoid ulceration",
               "U3-10": "Ulcer 3-10 mm", "U>10": "Ulcer >10 mm", "S": "Stenosis"}
# 4-level severity tier (CECDAI/Lewis-inspired: ulcer size + stenosis weigh most)
SEVERITY = {"N": 0, "E": 1, "O": 1, "AU": 2, "U3-10": 2, "U>10": 3, "S": 3}
TIER_NAMES = ["None", "Mild (inflammatory)", "Moderate (ulcerative)", "Severe (large ulcer / stenosis)"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32); STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def table():
    d = pd.read_csv(os.path.join(ROOT, "CrohnIPI_description.csv"))
    d.columns = ["frame", "label", "fold"]
    d["y"] = d.label.map(CLASSES.index)
    d["tier"] = d.label.map(SEVERITY)
    return d

def load_images(size=224):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"imgs_{size}.npy")
    d = table()
    if not os.path.exists(p):
        arr = np.stack([np.asarray(Image.open(os.path.join(ROOT, "imgs", f)).convert("RGB").resize((size, size), Image.BICUBIC)) for f in d.frame])
        np.save(p, arr)
    return d, np.load(p, mmap_mode="r")
