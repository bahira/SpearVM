"""Cas d'usage 3 : service d'inference par lots (reconnaissance de chiffres, donnees reelles).

On entraine le MLP digits (SpearVM), puis on le sert avec 3 moteurs :
  - SpearVM PackedMLP (poids pre-packes, GELU exacte fusionnee)
  - PyTorch CPU (torch.nn, meme poids)
  - numpy pur (matmul BLAS + GELU exacte via scipy.special.erf)
Mesure : latence p50/p90 et debit (echantillons/s) selon la taille de lot ; sorties verifiees identiques.
"""
import numpy as np
import torch
from scipy.special import erf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import _common as C
from spur_math.nn import SpearMLP, Adam, train_step, PackedMLP


def numpy_forward(params, X):
    H = X
    for i, (W, b) in enumerate(params):
        T = H @ W.T + b
        H = 0.5 * T * (1 + erf(T / np.sqrt(2))) if i < len(params) - 1 else T
    return H


def bench(fn, X, reps):
    fn(X)
    ts = []
    import time
    for _ in range(reps):
        t = time.perf_counter(); fn(X); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2], ts[int(0.9 * (len(ts) - 1))]


def main():
    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    net = SpearMLP([64, 128, 10], seed=0)
    opt = Adam(net, lr=3e-3)
    rng = np.random.default_rng(1)
    for _ in range(40):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 64):
            idx = perm[s:s + 64]
            train_step(net, opt, Xtr[idx], ytr[idx])
    acc = net.accuracy(Xte, yte)

    packed = PackedMLP(net)
    tl = [torch.nn.Linear(W.shape[1], W.shape[0]) for W, _ in net.params]
    with torch.no_grad():
        for lin, (W, b) in zip(tl, net.params):
            lin.weight.copy_(torch.from_numpy(W)); lin.bias.copy_(torch.from_numpy(b))

    def torch_fwd(Xn):
        h = torch.from_numpy(Xn)
        with torch.no_grad():
            for i, lin in enumerate(tl):
                h = lin(h)
                if i < len(tl) - 1:
                    h = torch.nn.functional.gelu(h)
        return h.numpy()

    engines = {
        "spearvm_packed": packed.predict_logits,
        "pytorch_cpu": torch_fwd,
        "numpy_blas": lambda Xn: numpy_forward(net.params, Xn),
    }
    # verification : memes predictions
    ref = net.predict_logits(Xte)
    checks = {k: float(np.abs(f(Xte) - ref).max()) for k, f in engines.items()}
    print("ecart max logits vs SpearVM :", {k: f"{v:.1e}" for k, v in checks.items()})
    print(f"precision test : {acc:.4f}")

    results = {}
    for B in (1, 32, 256, 1024):
        Xb = np.tile(Xte, (int(np.ceil(B / len(Xte))), 1))[:B].copy()
        reps = 200 if B <= 32 else 60
        results[B] = {}
        for name, f in engines.items():
            p50, p90 = bench(f, Xb, reps)
            results[B][name] = {"p50_ms": round(p50 * 1e3, 4), "p90_ms": round(p90 * 1e3, 4),
                                "samples_per_s": round(B / p50)}
        row = "  ".join(f"{k}: p50 {v['p50_ms']:8.4f} ms ({v['samples_per_s']:>9,} /s)"
                        for k, v in results[B].items())
        print(f"lot {B:5d} | {row}")
    C.save("digits_batch_serving", {
        "dataset": "sklearn load_digits, 25% test",
        "model": "MLP [64,128,10], GELU exacte, entraine 40 epochs (SpearVM)",
        "test_accuracy": round(acc, 4),
        "max_abs_logit_diff_vs_spearvm": checks,
        "latency_and_throughput_by_batch": results,
        "machine": "VM 2 coeurs, AVX-512",
    })


if __name__ == "__main__":
    main()
