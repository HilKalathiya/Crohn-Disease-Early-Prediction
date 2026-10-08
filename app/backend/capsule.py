"""Module 1 - Crohn's capsule lesion analyser (CrohnIPI fold ensemble; model set in crohnipi_meta.json) + general small-bowel screen."""
import numpy as np
from .core import session, to_input, softmax, normalise_cam, load_json, thumb_b64, MODELS
import os

CLASSES = ["N", "E", "O", "AU", "U3-10", "U>10", "S"]
NAMES = {"N": "Normal mucosa", "E": "Erythema", "O": "Oedema (villous swelling)", "AU": "Aphthoid ulceration",
         "U3-10": "Ulcer 3-10 mm", "U>10": "Ulcer > 10 mm", "S": "Stenosis (narrowing)"}
TIER_OF = np.array([0, 1, 1, 2, 2, 3, 3])
TIERS = [
    dict(name="No lesion", color="#2e9e6b", text="Mucosa looks normal in this frame."),
    dict(name="Mild", color="#d4a017", text="Inflammatory change only (erythema / oedema). Low weight in capsule activity scores."),
    dict(name="Moderate", color="#e07b22", text="Ulcerative lesion (aphthoid or 3-10 mm ulcer). Aphthoid ulcers are regarded as the earliest visible Crohn's lesion."),
    dict(name="Severe", color="#c8372d", text="Large ulcer (>10 mm) or stenosis - the findings that weigh most in Lewis / CECDAI scores."),
]

class CapsuleModel:
    def __init__(self):
        self.meta = load_json("crohnipi_meta.json", {})
        # switch models by editing crohnipi_meta.json: "model_prefixes" (several seeds, averaged) or "model_prefix" (one)
        prefixes = self.meta.get("model_prefixes") or [self.meta.get("model_prefix", "crohnipi_effb0")]
        self.prefix = "+".join(prefixes)
        self.size = int(self.meta.get("input_size", 224))
        self.folds = []
        for p in prefixes:
            for k in range(1, 6):
                f = f"{p}_fold{k}.onnx"
                if os.path.exists(os.path.join(MODELS, f)):
                    head = np.load(os.path.join(MODELS, f"{p}_fold{k}_head.npz"))
                    self.folds.append((session(f), head["W"], head["b"]))
        self.threshold = float(self.meta.get("lesion_threshold", 0.5))
        # screening mode: a lower cut-off fixed on CrohnIPI training data to catch 99% of lesions (see crohnipi_meta.json)
        self.screen_threshold = float(self.meta.get("screening_threshold", self.threshold))

    @property
    def ready(self): return len(self.folds) > 0

    def _run(self, x):
        probs, cams = [], []
        for s, W, b in self.folds:
            logits, fmap = s.run(None, {"image": x})
            probs.append(softmax(logits)); cams.append(np.einsum("ck,bkhw->bchw", W, fmap))
        return np.stack(probs), np.mean(cams, 0)  # (models,B,7), (B,7,h,w)

    def analyse(self, images, with_cam=True):
        out = []
        for i in range(0, len(images), 16):
            x = to_input(images[i:i + 16], self.size); P, C = self._run(x)
            pm, ps = P.mean(0), P.std(0); C = normalise_cam(C)
            for j in range(len(x)):
                p = pm[j]; tier_p = np.array([p[TIER_OF == t].sum() for t in range(4)]); t = int(tier_p.argmax())
                lesion = float(1 - p[0]); top = int(p.argmax())
                r = dict(probs={c: float(p[k]) for k, c in enumerate(CLASSES)}, probs_sd={c: float(ps[j][k]) for k, c in enumerate(CLASSES)},
                         top_class=CLASSES[top], top_name=NAMES[CLASSES[top]], lesion_score=lesion, lesion=lesion >= self.threshold,
                         lesion_screening=lesion >= self.screen_threshold,
                         tier=t, tier_name=TIERS[t]["name"], tier_color=TIERS[t]["color"], tier_text=TIERS[t]["text"],
                         tier_probs=tier_p.tolist(), agreement=float((P[:, j].argmax(1) == top).mean()), n_models=len(self.folds),
                         agree_count=int((P[:, j].argmax(1) == top).sum()))
                if with_cam:
                    lesion_cam = (C[j, 1:] * p[1:, None, None]).sum(0); lesion_cam = lesion_cam / max(lesion_cam.max(), 1e-8)
                    r["cam"] = {c: np.round(C[j, k], 3).tolist() for k, c in enumerate(CLASSES)}; r["cam"]["lesion"] = np.round(lesion_cam, 3).tolist()
                out.append(r)
        return out

def study_summary(results, threshold, screen_threshold=None):
    n = len(results); tiers = [r["tier"] for r in results]; scores = np.array([r["lesion_score"] for r in results])
    flagged = int((scores >= threshold).sum()); order = np.argsort(-scores)
    counts = {c: int(sum(r["top_class"] == c for r in results)) for c in CLASSES}
    # simple frame-level burden index (0-100): mean expected severity tier over frames, scaled
    exp_tier = np.array([np.dot(r["tier_probs"], [0, 1, 2, 3]) for r in results])
    return dict(frames=n, flagged=flagged, flagged_pct=100 * flagged / max(n, 1), worst_tier=int(max(tiers) if tiers else 0),
                worst_tier_name=TIERS[int(max(tiers) if tiers else 0)]["name"], tier_counts=[int(tiers.count(t)) for t in range(4)],
                class_counts=counts, burden_index=float(100 * exp_tier.mean() / 3) if n else 0.0,
                review_order=order.tolist(), review_needed=flagged, reading_saved_pct=100 * (1 - flagged / max(n, 1)),
                flagged_screening=int((scores >= (screen_threshold if screen_threshold is not None else threshold)).sum()))
