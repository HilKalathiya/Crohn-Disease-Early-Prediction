"""Transfer-learning baseline: frozen ImageNet backbone + logistic-regression head, official CrohnIPI 5 folds."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from data import table, CACHE
from metrics import evaluate, short
d = table(); y = d.y.values; folds = d.fold.values; res = {}
for key in sys.argv[1:]:
    F = np.load(os.path.join(CACHE, f"feat_{key}.npy")); P = np.zeros((len(y), 7))
    for k in range(1, 6):
        tr, te = folds != k, folds == k
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
        clf.fit(F[tr], y[tr]); P[te] = clf.predict_proba(F[te])
    m = evaluate(y, P); res[key] = m; print(key, short(m), flush=True)
    np.save(os.path.join(CACHE, f"oof_probe_{key}.npy"), P)
os.makedirs("artifacts/crohnipi", exist_ok=True)
old = json.load(open("artifacts/crohnipi/probe_metrics.json")) if os.path.exists("artifacts/crohnipi/probe_metrics.json") else {}
old.update(res); json.dump(old, open("artifacts/crohnipi/probe_metrics.json", "w"), indent=1)
