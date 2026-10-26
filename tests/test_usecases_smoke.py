"""Test de fumee des cas d'usage : chaque script tourne de bout en bout (epochs reduites,
sans ecrire les JSON commites). Garantit que les exemples ne cassent pas avec les APIs.
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USE = os.path.join(ROOT, "examples", "usecases")


def _load(name):
    import sys
    if USE not in sys.path:
        sys.path.insert(0, USE)
    spec = importlib.util.spec_from_file_location(name, os.path.join(USE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def no_save(monkeypatch):
    # Les scripts font `import _common` : on patche ce module-la (celui de sys.modules),
    # sinon les JSON commites seraient reecrits par le test.
    import sys
    if USE not in sys.path:
        sys.path.insert(0, USE)
    import _common
    saved = {}
    monkeypatch.setattr(_common, "save", lambda name, payload: saved.__setitem__(name, payload))
    return saved


def test_breast_cancer_smoke(no_save, monkeypatch, capsys):
    m = _load("breast_cancer_screening")
    monkeypatch.setattr(m, "EPOCHS", 2)
    m.main()
    r = no_save["breast_cancer_screening"]["results"]
    assert set(r) == {"spearvm_mlp", "pytorch_mlp", "logistic_regression"}
    for name in r:
        assert 0.0 <= r[name]["accuracy"]["mean"] <= 1.0


def test_wine_smoke(no_save, monkeypatch, capsys):
    m = _load("wine_quality_classify")
    monkeypatch.setattr(m, "EPOCHS", 2)
    m.main()
    r = no_save["wine_quality_classify"]["results"]
    assert np.isfinite(r["spearvm_mlp"]["log_loss"]["mean"])


def test_digits_train_smoke(no_save, monkeypatch, capsys):
    m = _load("digits_mlp_train")
    monkeypatch.setattr(m, "EPOCHS", 2)
    m.main()
    assert "digits_mlp_train" in no_save


def test_batch_serving_smoke(no_save, capsys):
    m = _load("digits_batch_serving")
    m.main()
    p = no_save["digits_batch_serving"]
    assert p["max_abs_logit_diff_vs_spearvm"]["spearvm_packed"] < 1e-4
