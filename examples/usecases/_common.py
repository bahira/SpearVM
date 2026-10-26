"""Utilitaires partages des cas d'usage (donnees reelles embarquees dans scikit-learn)."""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "docs", "bench", "usecases")


def save(name, payload):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name + ".json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print("resultats ->", os.path.relpath(path, ROOT))


def timed(fn, reps=1):
    best = float("inf")
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best
