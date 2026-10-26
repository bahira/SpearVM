"""Cas d'usage 4 : classification de vins par origine (3 cultivars), donnees reelles
(sklearn load_wine, 178 echantillons, 13 mesures chimiques). Validation croisee 5 plis stratifiee.

Comparaison : MLP SpearVM, MLP PyTorch (meme architecture), regression logistique (baseline).
Metriques : precision, F1 macro, log-loss (qualite des probabilites).
"""
import numpy as np
import torch
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import _common as C
from spur_math.nn import SpearMLP, Adam, train_step

EPOCHS, BATCH, LR, HIDDEN = 150, 16, 3e-3, 32


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def train_spear(Xtr, ytr, sizes, seed):
    net = SpearMLP(sizes, seed=seed)
    opt = Adam(net, lr=LR)
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), BATCH):
            idx = perm[s:s + BATCH]
            train_step(net, opt, Xtr[idx], ytr[idx])
    return lambda X: softmax(net.predict_logits(X))


def train_torch(Xtr, ytr, sizes, seed):
    torch.manual_seed(seed)
    tl = [torch.nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def fwd(x):
        h = x
        for i, lin in enumerate(tl):
            h = lin(h)
            if i < len(tl) - 1:
                h = torch.nn.functional.gelu(h)
        return h

    opt = torch.optim.Adam([p for lin in tl for p in lin.parameters()], lr=LR)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), BATCH):
            idx = torch.from_numpy(perm[s:s + BATCH])
            opt.zero_grad()
            torch.nn.functional.cross_entropy(fwd(Xt[idx]), yt[idx]).backward()
            opt.step()
    return lambda X: torch.softmax(fwd(torch.from_numpy(X)), 1).detach().numpy()


def metrics(P, y):
    pred = P.argmax(1)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, np.clip(P, 1e-7, 1), labels=[0, 1, 2]),
    }


def main():
    d = load_wine()
    X = d.data.astype(np.float32)
    y = d.target.astype(np.int64)
    sizes = [X.shape[1], HIDDEN, 3]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    names = ["spearvm_mlp", "pytorch_mlp", "logistic_regression"]
    rows = {n: [] for n in names}
    times = {n: 0.0 for n in names}
    for k, (tr, te) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]).astype(np.float32)
        Xte = sc.transform(X[te]).astype(np.float32)
        ytr, yte = y[tr], y[te]
        fn, dt = C.timed(lambda: train_spear(Xtr, ytr, sizes, seed=k))
        times["spearvm_mlp"] += dt; rows["spearvm_mlp"].append(metrics(fn(Xte), yte))
        fn, dt = C.timed(lambda: train_torch(Xtr, ytr, sizes, seed=k))
        times["pytorch_mlp"] += dt; rows["pytorch_mlp"].append(metrics(fn(Xte), yte))
        lr, dt = C.timed(lambda: LogisticRegression(max_iter=5000).fit(Xtr, ytr))
        times["logistic_regression"] += dt
        rows["logistic_regression"].append(metrics(lr.predict_proba(Xte), yte))

    summary = {}
    for n in names:
        summary[n] = {m: {"mean": round(float(np.mean([r[m] for r in rows[n]])), 4),
                          "std": round(float(np.std([r[m] for r in rows[n]])), 4)}
                      for m in ("accuracy", "f1_macro", "log_loss")}
        summary[n]["train_seconds_total_5folds"] = round(times[n], 3)
        print(f"{n:20s} acc {summary[n]['accuracy']['mean']:.4f}  F1 {summary[n]['f1_macro']['mean']:.4f}  "
              f"logloss {summary[n]['log_loss']['mean']:.4f}  | {times[n]:.2f} s")
    C.save("wine_quality_classify", {
        "dataset": "sklearn load_wine (178 vins, 13 features, 3 cultivars), CV 5 plis stratifiee",
        "model": f"MLP {sizes}, GELU exacte, Adam lr={LR}, batch {BATCH}, {EPOCHS} epochs; standardisation fittee par pli",
        "results": summary,
    })


if __name__ == "__main__":
    main()
