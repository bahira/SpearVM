"""MLP d'entrainement SpearVM : gradcheck (f64) et accord avec PyTorch (f32)."""
import numpy as np
import pytest

from spur_math.nn import SpearMLP, Adam, train_step, PackedMLP


def test_gradcheck_f64_all_layers():
    net = SpearMLP([5, 7, 4, 3], seed=1, dtype=np.float64)
    rng = np.random.default_rng(2)
    X = rng.standard_normal((6, 5))
    y = rng.integers(0, 3, 6)
    _, grads = net.loss_and_grads(X, y)
    eps = 1e-6
    for li, (W, b) in enumerate(net.params):
        for arr, g in ((W, grads[li][0]), (b, grads[li][1])):
            flat = arr.reshape(-1)
            gflat = g.reshape(-1)
            for idx in rng.choice(flat.size, size=min(12, flat.size), replace=False):
                old = flat[idx]
                flat[idx] = old + eps
                lp = net.loss_and_grads(X, y)[0]
                flat[idx] = old - eps
                lm = net.loss_and_grads(X, y)[0]
                flat[idx] = old
                num = (lp - lm) / (2 * eps)
                assert abs(num - gflat[idx]) < 1e-6, (li, idx, num, gflat[idx])


def test_matches_pytorch_gradients_f32():
    torch = pytest.importorskip("torch")
    net = SpearMLP([12, 32, 16, 5], seed=3, dtype=np.float32)
    rng = np.random.default_rng(4)
    X = rng.standard_normal((20, 12)).astype(np.float32)
    y = rng.integers(0, 5, 20)
    loss, grads = net.loss_and_grads(X, y)

    layers = []
    for W, b in net.params:
        lin = torch.nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(W)); lin.bias.copy_(torch.from_numpy(b))
        layers.append(lin)
    h = torch.from_numpy(X)
    for i, lin in enumerate(layers):
        h = lin(h)
        if i < len(layers) - 1:
            h = torch.nn.functional.gelu(h)  # exacte
    tl = torch.nn.functional.cross_entropy(h, torch.from_numpy(y).long())
    tl.backward()
    assert abs(loss - float(tl.detach())) < 1e-5
    for li, lin in enumerate(layers):
        assert np.abs(grads[li][0] - lin.weight.grad.numpy()).max() < 1e-4
        assert np.abs(grads[li][1] - lin.bias.grad.numpy()).max() < 1e-4


def test_training_learns_and_packed_inference_agrees():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    y = (X[:, 0] + 2 * X[:, 1] > 0).astype(int) + (X[:, 2] > 0.5).astype(int)
    net = SpearMLP([8, 32, 3], seed=0)
    opt = Adam(net, lr=1e-2)
    l0 = train_step(net, opt, X, y)
    for _ in range(200):
        l = train_step(net, opt, X, y)
    assert l < 0.5 * l0
    assert net.accuracy(X, y) > 0.9
    P = PackedMLP(net)
    assert np.abs(P.predict_logits(X) - net.predict_logits(X)).max() < 1e-4


def test_packed_mlp_fused_path_matches_training_forward_large_batch():
    # Regression : PackedMLP utilise le GEMM + GELU fusionne ; doit rester identique
    # au forward d'entrainement a grand lot (ancien chemin numpy supprime).
    import numpy as np
    from spur_math.nn import SpearMLP, PackedMLP
    net = SpearMLP([64, 128, 10], seed=3)
    X = np.random.default_rng(3).standard_normal((1024, 64)).astype(np.float32)
    ref = net.predict_logits(X)
    out = PackedMLP(net).predict_logits(X)
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 1e-4
