"""Compile les noyaux natifs dans spur_math/ puis construit le package.

Usage:
    python build_package.py          # build wheel uniquement
    python build_package.py --install  # build + install dans l'environnement courant
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, "spur_math")

CC = os.environ.get("CC", "gcc")
FLAGS = ["-O3", "-mavx2", "-mfma", "-fopenmp", "-shared"]
SRC = os.path.join(ROOT, "src", "spur_kernels.c")


def compile_native():
    if sys.platform == "win32":
        out = os.path.join(PKG, "spur_kernels.dll")
    else:
        out = os.path.join(PKG, "libspur_kernels.so")
    cmd = [CC] + FLAGS + ["-o", out, SRC, "-lm"]
    print("[build]", " ".join(cmd))
    subprocess.check_call(cmd)
    print("[ok]", out)
    return out


def compile_jit():
    """JIT map/loop kernel : windows x64 uniquement (ABI Win64 codee en dur)."""
    if sys.platform != "win32":
        return
    src = os.path.join(ROOT, "src", "spur.c")
    out = os.path.join(PKG, "spur_jit.dll")
    cmd = [CC, "-O2", "-mavx2", "-mfma", "-I", os.path.join(ROOT, "include"),
           "-shared", "-o", out, src, "-lm"]
    print("[build]", " ".join(cmd))
    subprocess.check_call(cmd)
    print("[ok]", out)


def main():
    compile_native()
    compile_jit()
    subprocess.check_call([sys.executable, "-m", "pip", "wheel",
                           "--no-deps", "--no-build-isolation",
                           "-w", "dist", ROOT], cwd=ROOT)
    print("\nWheel: dist/")
    if "--install" in sys.argv:
        import glob
        whl = sorted(glob.glob(os.path.join(ROOT, "dist", "*.whl")))[-1]
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--force-reinstall", "--no-deps", whl])
        print("Installe:", os.path.basename(whl))


if __name__ == "__main__":
    main()
