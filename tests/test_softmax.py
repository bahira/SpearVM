"""exp et softmax AVX2 : contrat de precision et cas limites.

Le softmax est la brique que l'attention exige ; sa correction se joue sur les
bords (lignes vides, longueurs variables, valeurs extremes), pas sur le cas
nominal. On teste donc surtout ceux-la.
"""
import numpy as np
import pytest

import spur_math as sm


def _ref_softmax(x, n=None):
    x = np.asarray(x, dtype=np.float64)
    if n is not None:
        x = x[:n]
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()


@pytest.mark.parametrize("dtype,ulp", [(np.float32, 4.0), (np.float64, 8.0)])
def test_exp_precision_ulp(dtype, ulp):
    lo = -87.0 if dtype == np.float32 else -700.0
    x = np.linspace(lo, 10.0, 200001).astype(dtype)
    got = sm.exp(x).astype(np.float64)
    ref = np.exp(x.astype(np.float64))
    rel = np.abs(got - ref) / ref
    eps = float(np.finfo(dtype).eps)
    assert rel.max() <= ulp * eps, f"{rel.max()/eps:.2f} ulp > {ulp}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exp_cas_limites(dtype):
    # tres negatif -> 0 sans NaN ; les grands positifs restent finis ou inf
    x = np.array([-1e4, -800.0, -200.0, -87.0, 0.0, 1.0], dtype=dtype)
    y = sm.exp(x)
    assert not np.isnan(y).any()
    assert (y[:2] >= 0).all()
    assert y[4] == pytest.approx(1.0, rel=1e-6)
    assert y[5] == pytest.approx(np.e, rel=1e-6)


@pytest.mark.parametrize("dtype,tol", [(np.float32, 5e-6), (np.float64, 1e-14)])
def test_softmax_lignes(dtype, tol):
    rng = np.random.default_rng(0)
    for rows, cols in [(1, 1), (3, 7), (37, 129), (8, 1024)]:
        X = (rng.standard_normal((rows, cols)) * 5).astype(dtype)
        Y = sm.softmax(X)
        assert np.abs(Y.sum(axis=1) - 1.0).max() <= tol
        for i in range(rows):
            assert np.abs(Y[i].astype(np.float64) - _ref_softmax(X[i])).max() <= tol


@pytest.mark.parametrize("dtype,tol", [(np.float32, 5e-6), (np.float64, 1e-14)])
def test_softmax_longueurs_variables(dtype, tol):
    """Forme causale : la ligne i n'a que lengths[i] entrees valides."""
    rng = np.random.default_rng(1)
    rows, cols = 64, 96
    X = (rng.standard_normal((rows, cols)) * 4).astype(dtype)
    lengths = np.minimum(np.arange(1, rows + 1), cols).astype(np.int32)
    Y = sm.softmax(X, lengths=lengths)
    for i in range(rows):
        n = int(lengths[i])
        assert np.all(Y[i, n:] == 0.0), f"ligne {i} : queue non nulle"
        assert Y[i, :n].sum() == pytest.approx(1.0, abs=tol)
        assert np.abs(Y[i, :n].astype(np.float64) - _ref_softmax(X[i], n)).max() <= tol


def test_softmax_ligne_vide_et_saturation():
    X = np.array([[1.0, 2.0, 3.0], [1e30, -1e30, 0.0], [-1e30, -1e30, -1e30]])
    Y = sm.softmax(X, lengths=np.array([0, 3, 3], dtype=np.int32))
    assert np.all(Y[0] == 0.0)                       # longueur nulle -> zeros
    assert not np.isnan(Y).any()                     # pas de NaN sur les extremes
    assert Y[1, 0] == pytest.approx(1.0)             # un seul gagnant
    assert Y[2].sum() == pytest.approx(1.0)          # tous egaux -> uniforme
    assert Y[2, 0] == pytest.approx(1 / 3, abs=1e-6)


def test_softmax_invariance_par_translation():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((16, 64))
    a = sm.softmax(X)
    b = sm.softmax(X + 137.0)
    assert np.abs(a - b).max() <= 1e-12


def test_out_et_validation():
    X = np.zeros((4, 8), dtype=np.float32)
    buf = np.empty((4, 8), dtype=np.float32)
    assert sm.softmax(X, out=buf) is buf
    with pytest.raises(ValueError):
        sm.softmax(X, out=np.empty((4, 9), dtype=np.float32))
    with pytest.raises(ValueError):
        sm.softmax(X, lengths=np.zeros(3, dtype=np.int32))
    with pytest.raises(ValueError):
        sm.softmax(np.zeros(8, dtype=np.float32))     # 1D refuse
    y = np.empty(16, dtype=np.float64)
    assert sm.exp(np.zeros(16)) is not y
    assert sm.exp(np.zeros(16), out=y) is y


# --- tuile d'attention -------------------------------------------------------
def _ref_attention(q, k, v, scale, lengths=None):
    qd, kd, vd = (np.asarray(a, dtype=np.float64) for a in (q, k, v))
    s = (qd @ kd.T) * scale
    if lengths is not None:
        s = np.where(np.arange(k.shape[0])[None, :] >= np.asarray(lengths)[:, None],
                     -np.inf, s)
    s -= s.max(axis=1, keepdims=True)
    w = np.exp(s)
    return (w / w.sum(axis=1, keepdims=True)) @ vd


@pytest.mark.parametrize("tq,tk,d", [(1, 1, 8), (4, 16, 8), (7, 33, 24), (64, 512, 64)])
def test_attention_tile_vs_reference(tq, tk, d):
    rng = np.random.default_rng(4)
    q = (rng.standard_normal((tq, d)) / np.sqrt(d)).astype(np.float32)
    k = (rng.standard_normal((tk, d)) / np.sqrt(d)).astype(np.float32)
    v = rng.standard_normal((tk, d)).astype(np.float32)
    got = sm.attention_tile(q, k, v)
    ref = _ref_attention(q, k, v, 1.0 / np.sqrt(d))
    assert np.isfinite(got).all()
    assert np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9) <= 5e-6


def test_attention_tile_causale():
    """lengths encode le masque causal : la requete i ne voit que i+1 cles."""
    rng = np.random.default_rng(5)
    tq = tk = 48
    d = 32
    q = (rng.standard_normal((tq, d)) / np.sqrt(d)).astype(np.float32)
    k = (rng.standard_normal((tk, d)) / np.sqrt(d)).astype(np.float32)
    v = rng.standard_normal((tk, d)).astype(np.float32)
    lengths = np.arange(1, tq + 1, dtype=np.int32)
    got = sm.attention_tile(q, k, v, lengths=lengths)
    ref = _ref_attention(q, k, v, 1.0 / np.sqrt(d), lengths)
    assert np.abs(got - ref).max() / np.abs(ref).max() <= 5e-6
    # la premiere requete ne voit que la premiere cle : sortie = v[0]
    assert np.abs(got[0] - v[0]).max() <= 1e-5


def test_attention_tile_validation():
    q = np.zeros((4, 8), dtype=np.float32)
    k = np.zeros((16, 8), dtype=np.float32)
    v = np.zeros((16, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        sm.attention_tile(q, k, np.zeros((15, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        sm.attention_tile(q, k, v, lengths=np.zeros(3, dtype=np.int32))
    with pytest.raises(ValueError):
        sm.attention_tile(q, k, v, out=np.zeros((4, 9), dtype=np.float32))
