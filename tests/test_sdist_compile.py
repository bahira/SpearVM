"""Teste le fallback de compilation : package sans binaire, sources C seules."""
import os
import shutil
import subprocess
import sys
import tempfile

tmp = tempfile.mkdtemp(prefix="spur_sdist_")
pkg = os.path.join(tmp, "spur_math")
os.makedirs(pkg)
shutil.copy("spur_math/__init__.py", pkg)
os.makedirs(os.path.join(pkg, "src"))
shutil.copy("src/spur_kernels.c", os.path.join(pkg, "src"))

env = dict(os.environ)
env["PYTHONPATH"] = tmp
code = (
    "import spur_math, numpy as np; "
    "g = spur_math.gelu(np.array([1.0])); "
    "ref = 0.997729 * (1.0 * np.clip(0.306923 + 0.501, 0, 1.002)) - 0.004004; "
    "assert abs(float(g[0]) - float(ref)) < 1e-12, (g, ref); "
    "print('fallback compile OK, gelu(1)=%.4f' % float(g[0]))"
)
r = subprocess.run([sys.executable, "-c", code], env=env,
                   capture_output=True, text=True)
shutil.rmtree(tmp, ignore_errors=True)


def test_fallback_compiles_and_matches():
    assert "OK" in r.stdout, r.stdout + r.stderr[-500:]
