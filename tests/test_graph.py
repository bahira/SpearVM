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
    H = sm.gelu_erf(T.astype(np.float32))
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


def _fd(f, x, eps=1e-6):
    num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        num[i] = (f(xp) - f(xm)) / (2 * eps)
    return num


def test_graph_backward_gradcheck_f64():
    """Gradient du graphe FFN (matmul -> bias -> gelu -> matmul) vs differences finies."""
    rng = np.random.default_rng(3)
    X = rng.normal(0, 0.5, (5, 4))
    W1 = rng.normal(0, 0.5, (6, 4))
    b1 = rng.normal(0, 0.1, 6)
    W2 = rng.normal(0, 0.5, (3, 6))
    G = rng.normal(0, 1, (5, 3))
    g = Graph()
    x = g.input("x")
    w1 = g.input("w1")
    bb = g.input("b1")
    w2 = g.input("w2")
    h = g.gelu(g.matmul_nt(x, w1, bias=bb))
    y = g.matmul_nt(h, w2)
    feeds = {"x": X, "w1": W1, "b1": b1, "w2": W2}
    gr = g.backward(y, feeds, G, wrt=[x, w1, bb, w2])

    def loss_for(name):
        def f(v):
            fd = dict(feeds); fd[name] = v
            return float(np.sum(g.forward(fd)[y] * G))
        return f

    for nid, name, base in [(x, "x", X), (w1, "w1", W1), (bb, "b1", b1), (w2, "w2", W2)]:
        num = _fd(loss_for(name), base.copy())
        assert np.abs(num - gr[nid]).max() < 1e-5, name


def test_graph_sgd_training_reduces_loss():
    """Entrainement reel : SGD sur un FFN via Graph.backward, la perte doit chuter."""
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (64, 8)).astype(np.float32)
    Wt = rng.normal(0, 1, (4, 8)).astype(np.float32)
    Y = np.tanh(X @ Wt.T)  # cible
    W1 = (rng.normal(0, 0.3, (16, 8))).astype(np.float32)
    W2 = (rng.normal(0, 0.3, (4, 16))).astype(np.float32)
    losses = []
    for step in range(120):
        g = Graph()
        x = g.input("x"); w1 = g.input("w1"); w2 = g.input("w2")
        y = g.matmul_nt(g.gelu(g.matmul_nt(x, w1)), w2)
        out = g.forward({"x": X, "w1": W1, "w2": W2})[y]
        diff = out - Y
        losses.append(float((diff ** 2).mean()))
        dout = (2 * diff / diff.size).astype(np.float32)
        gr = g.backward(y, {"x": X, "w1": W1, "w2": W2}, dout, wrt=[w1, w2])
        W1 = W1 - 0.5 * gr[w1].astype(np.float32)
        W2 = W2 - 0.5 * gr[w2].astype(np.float32)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
