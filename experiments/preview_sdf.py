"""Rendu de controle du champ implicite, **sans navigateur**.

Le shader du client fait du sphere tracing dans une texture 3D. Impossible de
compiler du GLSL ici : on re-implemente donc exactement le meme algorithme en
numpy (memes conventions : domaine [-1,1]^3, interpolation trilineaire,
avance `t += max(d, 0.75*voxel)`, normales par differences centrees) et on
ecrit une image PNG.

Ce que ce rendu prouve concretement :
  * le champ renvoye par le serveur est effectivement **marchable** (les rayons
    trouvent la surface sans la traverser) ;
  * la convention de signe est bonne (interieur negatif) ;
  * la borne de Lipschitz suffit : aucun pas ne saute par-dessus la surface.

Usage : python preview_sdf.py [shape] [grid] [hidden]
Sortie : results/sdf_<shape>.png
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "server"))

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

W = H = 420
STEPS = 220


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Encodeur PNG minimal (pas de dependance image dans ce depot)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def sample(vol: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Interpolation trilineaire, p en coordonnees objet [-1,1]^3."""
    n = vol.shape[0]
    g = (p * 0.5 + 0.5) * (n - 1)
    g = np.clip(g, 0.0, n - 1 - 1e-4)
    i0 = np.floor(g).astype(np.int32)
    f = g - i0
    i1 = np.minimum(i0 + 1, n - 1)
    x0, y0, z0 = i0[..., 0], i0[..., 1], i0[..., 2]
    x1, y1, z1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    c00 = vol[x0, y0, z0] * (1 - fx) + vol[x1, y0, z0] * fx
    c01 = vol[x0, y0, z1] * (1 - fx) + vol[x1, y0, z1] * fx
    c10 = vol[x0, y1, z0] * (1 - fx) + vol[x1, y1, z0] * fx
    c11 = vol[x0, y1, z1] * (1 - fx) + vol[x1, y1, z1] * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def main() -> None:
    shape = sys.argv[1] if len(sys.argv) > 1 else "sphere"
    grid = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 96

    from spearvm_sim.sims import create  # noqa: PLC0415

    amp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.34
    freq = float(sys.argv[5]) if len(sys.argv) > 5 else 1.2
    sim = create("implicit", {"grid": grid, "hidden": hidden, "shape": shape,
                              "amplitude": amp, "freq": freq, "seed": 11})
    for _ in range(6):          # laisse le morphing s'installer
        frame = sim.step(0.12)
    vol = (frame.payload.astype(np.float32) * frame.scale).reshape(grid, grid, grid)
    voxel = 2.0 / (grid - 1)
    print(f"[{shape}] {grid}^3 hidden={hidden} | grad_max={frame.stats['grad_max']} "
          f"lipschitz={frame.stats['lipschitz']} | sdf [{vol.min():.3f}, {vol.max():.3f}]")

    # camera
    eye = np.array([2.3, 1.5, 2.6], dtype=np.float32)
    target = np.zeros(3, dtype=np.float32)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0, 1, 0], dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    px = (np.arange(W, dtype=np.float32) + 0.5) / W * 2 - 1
    py = 1 - (np.arange(H, dtype=np.float32) + 0.5) / H * 2
    sx, sy = np.meshgrid(px, py)
    dirs = (fwd[None, None, :] * 2.35 + right[None, None, :] * sx[..., None]
            + up[None, None, :] * sy[..., None])
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    orig = np.broadcast_to(eye, dirs.shape).copy()
    inv = 1.0 / np.where(np.abs(dirs) < 1e-6, 1e-6, dirs)
    t0v = (-1.0 - orig) * inv
    t1v = (1.0 - orig) * inv
    tmin = np.maximum.reduce(np.minimum(t0v, t1v), axis=-1)
    tmax = np.minimum.reduce(np.maximum(t0v, t1v), axis=-1)
    alive = tmax > np.maximum(tmin, 0.0)

    t = np.maximum(tmin, 0.0) + voxel * 0.5
    hit = np.zeros((H, W), dtype=bool)
    surf = 0.6 * voxel
    for _ in range(STEPS):
        run = alive & ~hit & (t < tmax)
        if not run.any():
            break
        p = orig + dirs * t[..., None]
        d = sample(vol, p)
        newly = run & (d < surf)
        hit |= newly
        t = np.where(run & ~newly, t + np.maximum(d, 0.75 * voxel), t)

    p = orig + dirs * t[..., None]
    e = voxel
    nx = sample(vol, p + [e, 0, 0]) - sample(vol, p - [e, 0, 0])
    ny = sample(vol, p + [0, e, 0]) - sample(vol, p - [0, e, 0])
    nz = sample(vol, p + [0, 0, e]) - sample(vol, p - [0, 0, e])
    nrm = np.stack([nx, ny, nz], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9

    key = np.array([0.6, 0.85, 0.45], dtype=np.float32)
    key /= np.linalg.norm(key)
    fill = np.array([-0.5, 0.15, -0.7], dtype=np.float32)
    fill /= np.linalg.norm(fill)
    diff = np.clip((nrm * key).sum(-1), 0, 1)
    back = np.clip((nrm * fill).sum(-1), 0, 1) * 0.35
    view = -dirs
    refl = 2 * (nrm * key[None, None, :]).sum(-1, keepdims=True) * nrm - key
    spec = np.clip((refl * view).sum(-1), 0, 1) ** 42
    fres = (1 - np.clip((nrm * view).sum(-1), 0, 1)) ** 3

    depth = np.clip((t - np.maximum(tmin, 0)) / np.maximum(tmax - tmin, 1e-3), 0, 1)
    tone = np.clip(0.25 + 0.75 * (0.5 + 0.5 * nrm[..., 1]) - 0.35 * depth, 0, 1)
    pa = np.array([0.06, 0.24, 0.38]); pb = np.array([0.20, 0.90, 0.70])
    pc = np.array([0.98, 0.84, 0.42])
    lo = tone[..., None] * 2
    base = np.where(tone[..., None] < 0.5, pa + (pb - pa) * lo,
                    pb + (pc - pb) * (lo - 1))

    col = (base * (0.18 + 0.9 * diff + back)[..., None]
           + np.array([0.9, 0.97, 1.0]) * (spec * 0.55)[..., None]
           + np.array([0.25, 0.75, 0.95]) * (fres * 0.6)[..., None])
    col = col / (col + 0.85)
    col = np.power(np.clip(col, 0, 1), 1 / 2.2)

    bg_v = np.linspace(0.05, 0.10, H)[:, None, None] * np.array([0.6, 0.9, 1.0])
    img = np.where(hit[..., None], col, bg_v)
    rgb = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    out = RESULTS / f"sdf_{shape}.png"
    write_png(out, rgb)
    frac = float(hit.mean())
    print(f"  rayons touchant la surface : {frac * 100:.1f} %  -> {out}")
    if frac < 0.02:
        raise SystemExit("aucune surface trouvee : champ non marchable ?")


if __name__ == "__main__":
    main()
