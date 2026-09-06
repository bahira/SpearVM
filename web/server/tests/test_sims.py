"""Comportement des trois simulations temps reel."""

from __future__ import annotations

import numpy as np
import pytest

from spearvm_sim.protocol import decode_frame
from spearvm_sim.sims import REGISTRY, create, describe_all


def test_catalogue_expose_les_metadonnees():
    described = describe_all()
    assert {d["id"] for d in described} == set(REGISTRY)
    for sim in described:
        assert sim["title"] and sim["description"]
        assert sim["kernels"]
        assert all({"key", "label", "kind", "default"} <= set(p) for p in sim["params"])


@pytest.mark.parametrize("sim_id", sorted(REGISTRY))
def test_chaque_simulation_produit_un_frame_valide(sim_id):
    sim = create(sim_id)
    for _ in range(3):
        frame = sim.step(1 / 30)
        blob = sim.encode(frame)
        header, arr = decode_frame(blob)
        assert header["magic"] == "spearvm.sim.v1"
        assert header["compute_ms"] >= 0.0
        assert arr is not None
        assert np.isfinite(np.asarray(arr, dtype=np.float64)).all()
        assert int(np.prod(header["shape"])) == arr.size
        assert header["stats"]["backend"]
    assert sim.tick == 3


@pytest.mark.parametrize("sim_id", sorted(REGISTRY))
def test_parametres_valides_et_clampes(sim_id):
    sim = create(sim_id)
    sim.reconfigure({"inconnu": 42})
    for spec in sim.params_spec:
        if spec.kind == "range":
            sim.reconfigure({spec.key: 1e9})
            assert sim.params[spec.key] <= spec.maximum
            sim.reconfigure({spec.key: -1e9})
            assert sim.params[spec.key] >= spec.minimum
        elif spec.kind == "select":
            before = sim.params[spec.key]
            sim.reconfigure({spec.key: "valeur-illegale"})
            assert sim.params[spec.key] == before
    sim.step(1 / 30)  # doit rester operationnel apres reconfiguration


def test_flowfield_champ_incompressible_et_borne():
    sim = create("flowfield", {"grid": 16, "hidden": 32})
    frame = sim.step(1 / 12)
    n = 16
    vel = (frame.payload.astype(np.float32) * frame.scale).reshape(n, n, n, 3)

    # le rotationnel discret est a divergence nulle (mesure avant saturation)
    assert frame.stats["div_rel"] < 1e-4
    # la saturation tanh borne le champ : indispensable pour la texture 8 bits
    assert float(np.abs(vel).max()) <= 1.0 + 1e-6
    assert frame.shape == (n, n, n, 3)


def test_flowfield_reseed_change_le_champ():
    sim = create("flowfield", {"grid": 16, "hidden": 32})
    before = sim.step(0.0).payload.copy()
    sim.command({"type": "reseed", "seed": 42})
    after = sim.step(0.0).payload
    assert not np.array_equal(before, after)


def test_wavefield_impulsion_puis_dissipation():
    sim = create("wavefield", {"size": 96, "driver": False, "damping": 0.02})
    sim.command({"type": "clear"})
    sim.step(1 / 30)
    sim.command({"type": "pulse", "x": 0.5, "y": 0.5, "amp": 1.0})
    peak_after_pulse = sim.step(1 / 30).stats["peak"]
    assert peak_after_pulse > 0.4

    for _ in range(200):
        frame = sim.step(1 / 30)
    assert frame.stats["peak"] < peak_after_pulse       # l'energie se dissipe
    assert np.isfinite(sim.u).all()                      # schema stable


def test_wavefield_ignore_les_commandes_invalides():
    sim = create("wavefield", {"size": 96})
    sim.command({"type": "pulse", "x": "nan", "y": None})
    sim.command({"type": "inconnu"})
    assert np.isfinite(sim.step(1 / 30).payload).all()


def test_trainer_converge_reellement():
    sim = create("trainer", {"hidden": 64, "eval_grid": 48, "batch": 1024, "steps": 6})
    first = sim.step(1 / 20).stats["loss"]
    for _ in range(60):
        stats = sim.step(1 / 20).stats
    assert stats["loss"] < first / 5          # la backprop SpearVM apprend
    assert stats["rmse"] < 0.25
    assert stats["steps"] == 61 * 6
    assert len(stats["history"]) > 10


def test_trainer_envoie_la_cible_au_demarrage():
    sim = create("trainer", {"eval_grid": 48, "target": "crater"})
    frames = sim.initial_frames()
    assert len(frames) == 1
    assert frames[0].kind == "target_grid"
    assert frames[0].payload.size == 48 * 48


def test_trainer_reset_repart_de_zero():
    sim = create("trainer", {"hidden": 32, "eval_grid": 48, "steps": 4})
    for _ in range(15):
        sim.step(1 / 20)
    trained = sim.step(1 / 20).stats["loss"]
    sim.command({"type": "reset", "seed": 11})
    assert sim.step(1 / 20).stats["loss"] > trained
