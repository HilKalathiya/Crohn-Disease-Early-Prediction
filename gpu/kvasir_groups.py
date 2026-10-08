"""Deterministic split of the 43 labelled Kvasir-Capsule videos into two halves (A/B) for the hard-negative experiment.
Videos WITH expert inflammatory labels and videos WITHOUT are each alternated A, B, A, ... so both halves get both kinds."""
import csv
from collections import defaultdict


def groups(index_csv):
    infl = defaultdict(int); has = {}
    for r in csv.DictReader(open(index_csv)):
        v = r["video_id"]; has[v] = int(r["video_has_inflammatory"])
        if r["expert_label_within_25"] in ("Ulcer", "Erosion", "Erythema"):
            infl[v] += 1
    pos = sorted([v for v in has if has[v]], key=lambda v: (-infl[v], v))
    neg = sorted(v for v in has if not has[v])
    g = {}
    for lst in (pos, neg):
        for i, v in enumerate(lst):
            g[v] = "A" if i % 2 == 0 else "B"
    return g
