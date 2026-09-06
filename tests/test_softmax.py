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


# --- attention multi-tetes et cache KV ---------------------------------------
def _ref_mha(q, k, v, lengths=None):
    tq, h_q, d = q.shape
    tk, h_kv, _ = k.shape
    rep = h_q // h_kv
    kr = np.repeat(k, rep, axis=1).astype(np.float64)
    vr = np.repeat(v, rep, axis=1).astype(np.float64)
    out = np.zeros((tq, h_q, d))
    for h in range(h_q):
        s = (q[:, h].astype(np.float64) @ kr[:, h].T) / np.sqrt(d)
        if lengths is not None:
            s = np.where(np.arange(tk)[None, :] >= np.asarray(lengths)[:, None], -np.inf, s)
        s -= s.max(axis=1, keepdims=True)
        w = np.exp(s)
        out[:, h] = (w / w.sum(axis=1, keepdims=True)) @ vr[:, h]
    return out


@pytest.mark.parametrize("tq,tk,d,h_q,h_kv", [
    (1, 1, 8, 1, 1), (4, 64, 32, 8, 2), (7, 33, 16, 3, 3), (16, 512, 64, 12, 4),
])
def test_attention_mha(tq, tk, d, h_q, h_kv):
    rng = np.random.default_rng(6)
    q = (rng.standard_normal((tq, h_q, d)) / np.sqrt(d)).astype(np.float32)
    k = (rng.standard_normal((tk, h_kv, d)) / np.sqrt(d)).astype(np.float32)
    v = rng.standard_normal((tk, h_kv, d)).astype(np.float32)
    lengths = np.minimum(np.arange(tk - tq + 1, tk + 1), tk).astype(np.int32)
    lengths = np.maximum(lengths, 1).astype(np.int32)
    for lens in (None, lengths):
        got = sm.attention_mha(q, k, v, lengths=lens)
        ref = _ref_mha(q, k, v, lens)
        assert np.isfinite(got).all()
        assert np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9) <= 5e-6


def test_attention_mha_coherente_avec_tile():
    """Le chemin multi-tetes doit redonner le chemin tete par tete."""
    rng = np.random.default_rng(7)
    tq, tk, d, h_q, h_kv = 8, 96, 32, 6, 3
    q = (rng.standard_normal((tq, h_q, d)) / np.sqrt(d)).astype(np.float32)
    k = (rng.standard_normal((tk, h_kv, d)) / np.sqrt(d)).astype(np.float32)
    v = rng.standard_normal((tk, h_kv, d)).astype(np.float32)
    lengths = np.arange(tk - tq + 1, tk + 1, dtype=np.int32)
    mha = sm.attention_mha(q, k, v, lengths=lengths)
    rep = h_q // h_kv
    for h in range(h_q):
        tile = sm.attention_tile(np.ascontiguousarray(q[:, h]),
                                 np.ascontiguousarray(k[:, h // rep]),
                                 np.ascontiguousarray(v[:, h // rep]),
                                 lengths=lengths)
        assert np.abs(mha[:, h] - tile).max() <= 2e-6


def test_kv_cache_bit_a_bit():
    """Le cache packe doit donner exactement le meme resultat que l'appel complet."""
    rng = np.random.default_rng(8)
    tk, d, h_q, h_kv = 256, 32, 8, 2
    k = (rng.standard_normal((tk, h_kv, d)) / np.sqrt(d)).astype(np.float32)
    v = rng.standard_normal((tk, h_kv, d)).astype(np.float32)
    cache = sm.KVCache(k, v)
    assert cache.nbytes == 2 * tk * d * h_kv * 4
    for tq in (1, 3, 16):
        q = (rng.standard_normal((tq, h_q, d)) / np.sqrt(d)).astype(np.float32)
        lengths = np.full(tq, tk, dtype=np.int32)
        a = cache.attend(q, lengths=lengths)
        b = sm.attention_mha(q, k, v, lengths=lengths)
        assert np.array_equal(a, b), "le chemin packe doit etre bit-a-bit identique"


def test_mha_validation():
    q = np.zeros((4, 6, 8), dtype=np.float32)
    k = np.zeros((16, 4, 8), dtype=np.float32)
    v = np.zeros((16, 4, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        sm.attention_mha(q, k, v)                      # 6 n'est pas multiple de 4
    with pytest.raises(ValueError):
        sm.attention_mha(np.zeros((4, 8), dtype=np.float32), k, v)   # 2D refuse
    with pytest.raises(ValueError):
        sm.KVCache(k, np.zeros((15, 4, 8), dtype=np.float32))


# --- cache KV en bf16 --------------------------------------------------------
def test_kv_cache_bf16_empreinte_et_precision():
    rng = np.random.default_rng(9)
    tk, d, h_q, h_kv = 512, 32, 8, 2
    k = (rng.standard_normal((tk, h_kv, d)) / np.sqrt(d)).astype(np.float32)
    v = rng.standard_normal((tk, h_kv, d)).astype(np.float32)
    c32 = sm.KVCache(k, v)
    cbf = sm.KVCache(k, v, dtype="bf16")
    assert cbf.nbytes * 2 == c32.nbytes, "bf16 doit diviser l'empreinte par deux"

    q = (rng.standard_normal((4, h_q, d)) / np.sqrt(d)).astype(np.float32)
    a, b = c32.attend(q), cbf.attend(q)
    rel = np.abs(a - b).max() / np.abs(a).max()
    # 8 bits de mantisse : ~4e-3 par element, moyenne sur tk termes -> ~1e-3
    assert rel <= 6e-3, f"ecart bf16 {rel:.2e} trop grand"
    assert rel > 1e-5, "un ecart nul signifierait que bf16 n'est pas applique"


def test_kv_cache_bf16_causal_et_reference():
    rng = np.random.default_rng(10)
    tk, d, h_q, h_kv = 128, 32, 4, 2
    k = (rng.standard_normal((tk, h_kv, d)) / np.sqrt(d)).astype(np.float32)
    v = rng.standard_normal((tk, h_kv, d)).astype(np.float32)
    q = (rng.standard_normal((8, h_q, d)) / np.sqrt(d)).astype(np.float32)
    lengths = np.arange(tk - 7, tk + 1, dtype=np.int32)
    got = sm.KVCache(k, v, dtype="bf16").attend(q, lengths=lengths)
    ref = _ref_mha(q, k, v, lengths)
    assert np.isfinite(got).all()
    assert np.abs(got - ref).max() / np.abs(ref).max() <= 6e-3


def test_kv_cache_dtype_invalide():
    k = np.zeros((8, 2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        sm.KVCache(k, k, dtype="int8")


# --- poids quantifies --------------------------------------------------------
@pytest.mark.parametrize("dtype,tol,ratio", [("bf16", 4e-3, 2), ("i8", 2e-2, 4)])
def test_quantized_weight(dtype, tol, ratio):
    rng = np.random.default_rng(11)
    n, k = 256, 192
    w = (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float32)
    q = sm.QuantizedWeight(w, dtype=dtype)
    assert q.nbytes * ratio <= w.nbytes * 1.05, "empreinte non reduite comme attendu"
    for m in (1, 2, 5, 33):
        a = (rng.standard_normal((m, k)) / np.sqrt(k)).astype(np.float32)
        got = q.matmul(a)
        ref = a.astype(np.float64) @ w.astype(np.float64).T
        rel = np.abs(got - ref).max() / np.abs(ref).max()
        assert rel <= tol, f"{dtype} m={m} : erreur {rel:.2e}"
        assert np.isfinite(got).all()


def test_quantized_weight_dequantize():
    """La dequantification doit reproduire ce que le noyau utilise vraiment."""
    rng = np.random.default_rng(12)
    w = (rng.standard_normal((64, 48)) / 7).astype(np.float32)
    for dtype, tol in (("bf16", 1e-6), ("i8", 5e-5)):
        q = sm.QuantizedWeight(w, dtype=dtype)
        a = (rng.standard_normal((3, 48)) / 7).astype(np.float32)
        direct = q.matmul(a)
        via = a.astype(np.float64) @ q.dequantize().astype(np.float64).T
        assert np.abs(direct - via).max() / max(np.abs(via).max(), 1e-9) <= tol


def test_quantized_weight_echelle_par_ligne():
    """Une ligne de tres faible amplitude ne doit pas etre ecrasee par les autres."""
    w = np.zeros((2, 32), dtype=np.float32)
    w[0] = 1.0
    w[1] = 1e-4
    q = sm.QuantizedWeight(w, dtype="i8")
    a = np.ones((1, 32), dtype=np.float32)
    got = q.matmul(a)[0]
    assert got[0] == pytest.approx(32.0, rel=1e-2)
    assert got[1] == pytest.approx(32e-4, rel=1e-2), "echelle par ligne non appliquee"


def test_quantized_weight_validation():
    w = np.zeros((8, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        sm.QuantizedWeight(w, dtype="f16")
    with pytest.raises(ValueError):
        sm.QuantizedWeight(np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError):
        sm.QuantizedWeight(w).matmul(np.zeros((2, 5), dtype=np.float32))
