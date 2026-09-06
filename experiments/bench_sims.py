"""Etage 10 : impact des nouveaux noyaux sur les cas d'usage reels du Lab.

On instancie chaque simulation du serveur (celles que le front Three.js pilote)
et on chronometre `step()` — c'est le temps qui decide du nombre d'images par
seconde tenable, pas les GFLOPS d'un GEMM carre.

Deux processus : SPUR_MM_LEGACY=1 (ancien noyau) puis v2, la variable etant lue
une seule fois au premier appel.

Sortie : results/sims_e2e_t{threads}.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

# (id, parametres, nombre de pas chronometres) — reglages par defaut de l'UI
CASES = [
    ("flowfield", {"grid": 24, "hidden": 64}, 20),
    ("flowfield", {"grid": 32, "hidden": 128}, 10),
    ("implicit", {"grid": 48, "hidden": 64}, 10),
    ("implicit", {"grid": 56, "hidden": 128}, 6),
    ("trainer", {"hidden": 96, "batch": 2048, "steps": 4, "eval_grid": 72}, 20),
    ("trainer", {"hidden": 128, "batch": 4096, "steps": 4, "eval_grid": 96}, 10),
    ("wavefield", {"size": 160, "substeps": 2}, 20),
]


def run_child() -> dict:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "web" / "server"))
    from spearvm_sim.sims import create  # noqa: PLC0415

    out = {}
    for sim_id, params, steps in CASES:
        sim = create(sim_id, dict(params))
        sim.step(1 / 60)  # warmup (alloue les buffers, chauffe les caches)
        best = float("inf")
        for _ in range(steps):
            t0 = time.perf_counter()
            sim.step(1 / 60)
            best = min(best, time.perf_counter() - t0)
        label = f"{sim_id} {'/'.join(f'{k}={v}' for k, v in params.items())}"
        out[label] = round(best * 1e3, 3)
    return out


def main() -> None:
    if os.environ.get("SPUR_SIM_CHILD") == "1":
        print(json.dumps(run_child()))
        return

    runs = {}
    for label, legacy in (("legacy", "1"), ("v2", "0")):
        env = dict(os.environ, SPUR_SIM_CHILD="1", SPUR_MM_LEGACY=legacy)
        proc = subprocess.run([sys.executable, __file__], env=env,
                              capture_output=True, text=True, check=True)
        runs[label] = json.loads(proc.stdout.strip().splitlines()[-1])

    threads = os.environ.get("OMP_NUM_THREADS", "?")
    print(f"threads={threads}\n{'cas':>52} {'legacy':>9} {'v2':>9} {'gain':>6} {'fps v2':>8}")
    report = {}
    for label in runs["legacy"]:
        old, new = runs["legacy"][label], runs["v2"][label]
        report[label] = {"legacy_ms": old, "v2_ms": new, "gain": round(old / new, 3),
                         "fps_v2": round(1000.0 / new, 1)}
        print(f"{label:>52} {old:9.2f} {new:9.2f} {old / new:6.2f} {1000.0 / new:8.1f}")

    out = RESULTS / f"sims_e2e_t{threads}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
