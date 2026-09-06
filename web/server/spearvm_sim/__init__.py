"""Package serveur du SpearVM Simulation Lab.

Reglage des pools de threads **avant** l'import de numpy (donc d'OpenBLAS) :
le serveur alterne en permanence des appels BLAS (numpy) et des regions OpenMP
(noyaux SpearVM). Par defaut OpenBLAS laisse ses threads *spinner* ~20 ms apres
chaque appel ; sur une machine 2 vCPU cela affame la region OpenMP suivante et
un matmul 256^3 passe de 0.4 ms a 18 ms (x45). `OPENBLAS_THREAD_TIMEOUT=1`
gare les threads immediatement et supprime l'interference.

Toute valeur deja presente dans l'environnement est respectee.
"""

from __future__ import annotations

import os

__version__ = "1.0.0"

_THREAD_DEFAULTS = {
    # OpenBLAS : parking rapide des threads (evite le vol de coeur decrit plus haut)
    "OPENBLAS_THREAD_TIMEOUT": "1",
    # GOMP : spin actif court, meilleur pour nos noyaux appeles en rafale
    "OMP_WAIT_POLICY": "ACTIVE",
}

for _key, _value in _THREAD_DEFAULTS.items():
    os.environ.setdefault(_key, _value)

__all__ = ["__version__"]
