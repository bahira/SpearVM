import numpy as np
import spur_math as sm
from spur_math.graph import Graph


def _ffn(X, W1, b1, W2):
    g = Graph()
    x = g.input("x")
    h = g.gelu(g.matmul_nt(x, W1, bias=b1))
    y = g.matmul_nt(h, W2)
    return g, y


def test_fusion_rewrites_matmul_gelu():
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((64, 32)).astype(np.float32)
    W2 = rng.standard_normal((16, 64)).astype(np.float32)
    b1 = rng.standard_normal(64).astype(np.float32)
    g, y = _ffn(None, W1, b1, W2)
    st = g.stats()
    assert st["fused_matmul_gelu"] == 1
    assert st["after_fusion"] == st["nodes"] - 1  # gelu absorbe dans le matmul


def test_graph_fused_equals_unfused_and_numpy():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((37, 48)).astype(np.float32)
    W1 = rng.standard_normal((70, 48)).astype(np.float32)
    b1 = rng.standard_normal(70).astype(np.float32)
    W2 = rng.standard_normal((11, 70)).astype(np.float32)
    g, y = _ffn(X, W1, b1, W2)
    fused = g.run({"x": X}, fuse=True)[y]
    plain = g.run({"x": X}, fuse=False)[y]
    assert np.abs(fused - plain).max() < 1e-3 * np.abs(plain).max()
    T = X.astype(np.float64) @ W1.astype(np.float64).T + b1
    H = sm.gelu(T.astype(np.float32))
    ref = H.astype(np.float64) @ W2.astype(np.float64).T
    assert np.abs(fused - ref).max() < 1e-2 * max(1.0, np.abs(ref).max())


def test_no_fusion_when_matmul_shared():
    g = Graph()
    x = g.input("x")
    W = np.eye(4, dtype=np.float32)
    m = g.matmul_nt(x, W)          # consomme par gelu ET par la sortie
    g.gelu(m)
    out = g.matmul_nt(m, W)
    assert g.stats()["fused_matmul_gelu"] == 0
    X = np.ones((2, 4), dtype=np.float32)
    r = g.run({"x": X})
    assert r[out].shape == (2, 4)
