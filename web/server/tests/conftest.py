import os
import sys
from pathlib import Path

import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

# les benchs de test doivent rester courts et deterministes
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


@pytest.fixture(scope="session")
def kernels():
    from spearvm_sim.kernels import get_kernels

    return get_kernels()
