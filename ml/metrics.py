"""Shared evaluation for CrohnIPI: 7-class, binary (lesion vs normal), 4-level severity tier."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                             cohen_kappa_score, recall_score, precision_score)
from data import CLASSES, SEVERITY

TIER_OF = np.array([SEVERITY[c] for c in CLASSES])

def tier_probs(P):
    return np.stack([P[:, TIER_OF == t].sum(1) for t in range(4)], 1)

def evaluate(y, P):
    pred = P.argmax(1); yb = (y != 0).astype(int); pb = 1 - P[:, 0]
    T, yt = tier_probs(P), TIER_OF[y]; pt = T.argmax(1)
    out = {
        "multiclass": dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                           macro_f1=f1_score(y, pred, average="macro"),
                           per_class_recall=dict(zip(CLASSES, recall_score(y, pred, average=None, labels=range(7)).tolist())),
                           per_class_precision=dict(zip(CLASSES, precision_score(y, pred, average=None, labels=range(7), zero_division=0).tolist())),
                           confusion=confusion_matrix(y, pred, labels=range(7)).tolist()),
        "binary": dict(auroc=roc_auc_score(yb, pb), accuracy=accuracy_score(yb, pb >= 0.5),
                       sensitivity=recall_score(yb, pb >= 0.5), specificity=recall_score(1 - yb, pb < 0.5)),
        "severity": dict(accuracy=accuracy_score(yt, pt), macro_f1=f1_score(yt, pt, average="macro"),
                         qwk=cohen_kappa_score(yt, pt, weights="quadratic"), confusion=confusion_matrix(yt, pt, labels=range(4)).tolist()),
    }
    return out

def short(m):
    return (f"7cls acc {m['multiclass']['accuracy']:.3f} bal {m['multiclass']['balanced_accuracy']:.3f} mF1 {m['multiclass']['macro_f1']:.3f} | "
            f"bin AUROC {m['binary']['auroc']:.3f} acc {m['binary']['accuracy']:.3f} sens {m['binary']['sensitivity']:.3f} spec {m['binary']['specificity']:.3f} | "
            f"tier acc {m['severity']['accuracy']:.3f} QWK {m['severity']['qwk']:.3f}")
