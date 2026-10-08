"""Modules 2 & 3 - linear heads on a shared ImageNet EfficientNet-B0 backbone:
   * UC Mayo grader (trained on LIMUC, externally tested on HyperKvasir)
   * small-bowel multi-finding screen (trained on Kvasir-Capsule)."""
import os, numpy as np
from .core import session, to_input, softmax, normalise_cam, load_json, MODELS

class LinearHead:
    """Standardised logistic regression: logit = coef . ((f - mu) / sd) + b.  Spatial evidence map = sum_k coef_ck/sd_k * fmap_k."""
    def __init__(self, name):
        z = np.load(os.path.join(MODELS, f"{name}.npz"), allow_pickle=True)
        self.mu, self.sd, self.coef, self.b = z["mu"], z["sd"], z["coef"], z["b"]; self.classes = [str(c) for c in z["classes"]]
        self.meta = load_json(f"{name}_meta.json", {}); self.multilabel = bool(self.meta.get("multilabel", False))
        self.T = float(self.meta.get("temperature", 1.0))  # temperature scaling fitted on out-of-fold LIMUC predictions
    def predict(self, pooled, fmap):
        logits = ((pooled - self.mu) / self.sd) @ self.coef.T + self.b
        p = 1 / (1 + np.exp(-logits)) if self.multilabel else softmax(logits / self.T)
        cam = np.einsum("ck,bkhw->bchw", self.coef / self.sd, fmap); return p, normalise_cam(cam)

class SharedBackbone:
    def __init__(self):
        self.ok = os.path.exists(os.path.join(MODELS, "backbone_effb0_ra.onnx"))
        self.s = session("backbone_effb0_ra.onnx") if self.ok else None
        self.heads = {}
        for n in ["uc_mayo_head", "sb_findings_head"]:
            if os.path.exists(os.path.join(MODELS, f"{n}.npz")): self.heads[n] = LinearHead(n)
    def features(self, images, batch=16):
        """Run in small batches: one big batch makes onnxruntime allocate GBs of activations."""
        P, F = [], []
        for i in range(0, len(images), batch):
            pooled, fmap = self.s.run(None, {"image": to_input(images[i:i + batch])}); P.append(pooled); F.append(fmap)
        return np.concatenate(P), np.concatenate(F)

MAYO_TEXT = {0: "Mayo 0 - normal / inactive mucosa", 1: "Mayo 1 - mild: erythema, decreased vascular pattern",
             2: "Mayo 2 - moderate: marked erythema, absent vascular pattern, erosions",
             3: "Mayo 3 - severe: spontaneous bleeding, ulceration"}

def uc_grade(bb, images):
    h = bb.heads["uc_mayo_head"]; pooled, fmap = bb.features(images); P, C = h.predict(pooled, fmap); out = []
    for j in range(len(images)):
        p = P[j]; m = int(p.argmax()); expected = float(np.dot(p, np.arange(4)))
        out.append(dict(probs=p.tolist(), mayo=m, mayo_text=MAYO_TEXT[m], expected_mayo=expected,
                        active_prob=float(p[2:].sum()), remission=bool(p[:2].sum() >= 0.5),
                        cam={str(k): np.round(C[j, k], 3).tolist() for k in range(4)}))
    return out

def sb_findings(bb, images, pooled=None, fmap=None):
    h = bb.heads["sb_findings_head"]
    if pooled is None: pooled, fmap = bb.features(images)
    P, C = h.predict(pooled, fmap); out = []
    for j in range(len(pooled)):
        p = P[j]; order = np.argsort(-p)
        out.append(dict(probs={c: float(p[k]) for k, c in enumerate(h.classes)}, top=h.classes[int(order[0])],
                        abnormal_prob=float(p.max()) if h.multilabel else float(1 - p[h.classes.index("normal")])))
    return out
