"""Module 3 - HMP2 flare-risk forecaster. Random forest stored as plain numpy arrays (no scikit-learn needed at runtime);
per-prediction feature contributions via Saabas path attribution (prediction = bias + sum of contributions)."""
import os, numpy as np
from .core import MODELS, load_json

class FlareModel:
    def __init__(self):
        p = os.path.join(MODELS, "flare_rf.npz"); self.ok = os.path.exists(p)
        if not self.ok: return
        z = np.load(p); self.L, self.R, self.F, self.TH, self.V = z["left"], z["right"], z["feature"], z["threshold"], z["value"]
        self.med, self.ind = z["medians"], z["indicator_features"]
        m = load_json("flare_meta.json"); self.features, self.nice = m["features"], m["nice"]; self.platt = m.get("platt", {"a": 1.0, "b": 0.0})
        self.timelines = load_json("flare_timelines.json", {}); self.metrics = load_json("flare_metrics.json", {})

    def _prep(self, x):
        miss = np.isnan(x); xi = np.where(miss, self.med, x); return np.concatenate([xi, miss[self.ind].astype(float)])

    def predict(self, feats: dict):
        n = len(self.features); x = np.array([float(feats[f]) if feats.get(f) is not None else np.nan for f in self.features])
        xi = self._prep(x); contrib = np.zeros(len(xi)); bias = 0.0; pred = 0.0; T = self.L.shape[0]
        for t in range(T):
            nd = 0; bias += self.V[t, 0]
            while self.L[t, nd] != -1:
                f = self.F[t, nd]; nx = self.L[t, nd] if xi[f] <= self.TH[t, nd] else self.R[t, nd]
                contrib[f] += self.V[t, nx] - self.V[t, nd]; nd = nx
            pred += self.V[t, nd]
        pred /= T; bias /= T; contrib /= T; c = contrib[:n].copy()
        lp = np.log(np.clip(pred, 1e-4, 1 - 1e-4) / (1 - np.clip(pred, 1e-4, 1 - 1e-4)))
        risk = 1 / (1 + np.exp(-(self.platt["a"] * lp + self.platt["b"])))  # Platt-recalibrated (fit on out-of-fold predictions)
        for j, f in enumerate(self.ind): c[f] += contrib[n + j]   # fold 'was missing' indicators into their feature
        items = sorted([dict(feature=self.nice[f], key=f, value=None if np.isnan(x[i]) else float(x[i]), contribution=float(c[i]))
                        for i, f in enumerate(self.features)], key=lambda d: -abs(d["contribution"]))
        base_rate = self.metrics.get("summary", {}).get("flare_rate", 0.1)
        rel = risk / base_rate; level = "High" if rel >= 2.5 else "Elevated" if rel >= 1.5 else "Low"
        return dict(risk=float(risk), raw_score=float(pred), relative=float(rel), level=level, base_rate=base_rate, bias=float(bias),
                    contributions=items[:10], imputed=[self.nice[f] for i, f in enumerate(self.features) if np.isnan(x[i])])
