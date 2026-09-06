"""Champ implicite : proprietes que le rendu GPU exige du serveur.

Le client fait du *sphere tracing* : a chaque pas il avance de la distance
renvoyee par le champ. Ce rendu n'est correct que si le serveur garantit
trois choses, testees ici plutot que supposees :

  1. convention de signe (interieur negatif, exterieur positif) ;
  2. champ **1-lipschitzien** (|grad d| <= 1) apres correction ;
  3. marchabilite reelle : on rejoue l'algorithme du shader en numpy et on
     verifie que les rayons atteignent la surface sans la traverser.
"""

from __future__ import annotations

import numpy as np
import pytest

from spearvm_sim.sims import create


def _volume(frame, n: int) -> np.ndarray:
    return (frame.payload.astype(np.float32) * frame.scale).reshape(n, n, n)


@pytest.fixture(scope="module")
def sim_frame():
    sim = create("implicit", {"grid": 32, "hidden": 32, "shape": "sphere", "seed": 5})
    frame = sim.step(0.1)
    return sim, frame


def test_frame_est_un_volume_int16(sim_frame):
    _, frame = sim_frame
    assert frame.kind == "sdf_grid"
    assert frame.shape == (32, 32, 32)
    assert frame.payload.dtype == np.int16
    assert frame.payload.size == 32 ** 3


def test_signe_interieur_negatif(sim_frame):
    _, frame = sim_frame
    vol = _volume(frame, 32)
    assert vol[16, 16, 16] < 0.0          # centre : dans la matiere
    for corner in ((0, 0, 0), (0, 0, -1), (-1, -1, -1), (-1, 0, -1)):
        assert vol[corner] > 0.0          # coins du domaine : dehors


@pytest.mark.parametrize("shape", ["sphere", "tore", "boite", "gyroide"])
def test_champ_1_lipschitzien(shape):
    """Condition de validite du sphere tracing, verifiee sur la grille."""
    sim = create("implicit", {"grid": 32, "hidden": 32, "shape": shape, "seed": 7})
    frame = sim.step(0.1)
    vol = _volume(frame, 32)
    voxel = 2.0 / 31
    gx, gy, gz = np.gradient(vol, voxel)
    grad = np.sqrt(gx * gx + gy * gy + gz * gz).max()
    # marge de quantification int16 : ~scale/voxel par difference centree
    assert grad <= 1.0 + 5e-3, f"{shape}: |grad| max = {grad:.4f}"
    assert frame.stats["lipschitz"] <= 1.0 + 1e-6


def test_le_volume_est_marchable(sim_frame):
    """Rejoue le sphere tracing du shader : les rayons doivent toucher."""
    _, frame = sim_frame
    n = 32
    vol = _volume(frame, n)
    voxel = 2.0 / (n - 1)

    def sample(p: np.ndarray) -> float:
        g = np.clip((p * 0.5 + 0.5) * (n - 1), 0, n - 1 - 1e-4)
        i0 = np.floor(g).astype(int)
        f = g - i0
        i1 = np.minimum(i0 + 1, n - 1)
        acc = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1])
                         * (f[2] if dz else 1 - f[2]))
                    idx = (i1[0] if dx else i0[0], i1[1] if dy else i0[1],
                           i1[2] if dz else i0[2])
                    acc += w * float(vol[idx])
        return acc

    hits = 0
    rays = 0
    rng = np.random.default_rng(0)
    for _ in range(24):
        # rayon partant d'un point du bord vers le centre (+ jitter)
        origin = np.array([0.99, 0.0, 0.0])
        origin[1:] = rng.uniform(-0.35, 0.35, size=2)
        direction = -origin / np.linalg.norm(origin)
        rays += 1
        t = 0.0
        for _ in range(400):
            p = origin + direction * t
            if np.abs(p).max() > 1.0:
                break
            d = sample(p)
            assert d > -0.35, "la marche a traverse la surface (champ non lipschitzien)"
            if d < 0.6 * voxel:
                hits += 1
                break
            t += max(d, 0.75 * voxel)
    assert hits >= rays * 0.8, f"seulement {hits}/{rays} rayons ont touche la surface"


def test_deterministe_et_sensible_a_la_graine():
    a = create("implicit", {"grid": 32, "hidden": 32, "seed": 3}).step(0.1)
    b = create("implicit", {"grid": 32, "hidden": 32, "seed": 3}).step(0.1)
    c = create("implicit", {"grid": 32, "hidden": 32, "seed": 4}).step(0.1)
    assert np.array_equal(a.payload, b.payload)
    assert not np.array_equal(a.payload, c.payload)


def test_stats_publiees_pour_l_ui(sim_frame):
    _, frame = sim_frame
    for key in ("grid", "hidden", "points", "k_in", "gflops", "mflop_per_tick",
                "grad_max", "lipschitz", "surface_frac", "voxel", "shape"):
        assert key in frame.stats, key
    assert frame.stats["points"] == 32 ** 3
    assert frame.stats["mflop_per_tick"] > 0


def test_changement_de_primitive_sans_realloc():
    """`shape` n'est pas structurel : le champ doit changer sans reconstruire."""
    sim = create("implicit", {"grid": 32, "hidden": 32, "shape": "sphere"})
    first = _volume(sim.step(0.1), 32)
    sim.reconfigure({"shape": "tore"})
    second = _volume(sim.step(0.1), 32)
    assert not np.allclose(first, second)
    assert second.shape == (32, 32, 32)


# --- Attention Lab -----------------------------------------------------------
@pytest.mark.parametrize("fmt", ["f32", "bf16", "i8"])
def test_attention_lab_formats(fmt):
    """Les trois formats doivent produire la meme structure d'attention."""
    sim = create("attention", {"format": fmt, "context": 256, "heads": 4,
                               "kv_heads": 2, "d_head": 32, "d_ff": 512})
    frame = sim.step(0.1)
    assert frame.kind == "attention_map"
    assert frame.shape == (4, 256)
    w = (frame.payload.astype(np.float32) * frame.scale).reshape(4, 256)
    assert np.all(w >= 0.0)
    for h in range(4):
        assert w[h].sum() == pytest.approx(1.0, abs=2e-3), "chaque tete somme a 1"
    assert frame.stats["format"] == fmt
    assert frame.stats["tokens_par_s"] > 0
    assert frame.stats["poids_Mo"] > 0


def test_attention_lab_quantification_reduit_l_empreinte():
    def poids(fmt):
        return create("attention", {"format": fmt, "context": 256, "heads": 4,
                                    "kv_heads": 2, "d_head": 32,
                                    "d_ff": 512}).step(0.1).stats["poids_Mo"]
    f32, bf16, i8 = poids("f32"), poids("bf16"), poids("i8")
    assert bf16 == pytest.approx(f32 / 2, rel=0.02)
    assert i8 == pytest.approx(f32 / 4, rel=0.05)


def test_attention_lab_temperature_concentre_l_attention():
    """Le piquant des requetes doit reellement changer l'entropie mesuree."""
    def entropie(t):
        sim = create("attention", {"format": "f32", "context": 512, "heads": 4,
                                   "kv_heads": 2, "d_head": 32, "d_ff": 512,
                                   "temperature": t})
        for _ in range(4):
            f = sim.step(0.1)
        return f.stats["entropie"]
    plate, piquee = entropie(0.5), entropie(6.0)
    assert plate > 0.95, "a faible temperature l'attention doit rester etalee"
    assert piquee < 0.7, "a forte temperature elle doit se concentrer"
    assert piquee < plate
