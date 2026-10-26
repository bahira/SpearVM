"""Cas d'usage 2 : aide au diagnostic (tumeur maligne / benigne), donnees cliniques reelles
(sklearn breast_cancer, 569 patientes, 30 mesures cellulaires). Validation croisee 5 plis stratifiee.

Comparaison : MLP SpearVM, MLP PyTorch (meme architecture), regression logistique (baseline classique).
Metriques : precision, AUC ROC, rappel sur les tumeurs malignes (cas a ne pas rater).
"""
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import _common as C
from spur_math.nn import SpearMLP, Adam, train_step

EPOCHS, BATCH, LR, HIDDEN = 150, 32, 3e-3, 32


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
    return lambda X: torch.softmax(fwd(torch.from_numpy(X)), 1).detach().numpy()[:, 1]


def train_spear(Xtr, ytr, sizes, seed):
    net = SpearMLP(sizes, seed=seed)
    opt = Adam(net, lr=LR)
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), BATCH):
            idx = perm[s:s + BATCH]
            train_step(net, opt, Xtr[idx], ytr[idx])

    def proba(X):
        Z = net.predict_logits(X)
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        return P[:, 1]
    return proba


def metrics(p, y):
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "auc": roc_auc_score(y, p),
        "recall_malignant": recall_score(y, pred, pos_label=0),  # classe 0 = malin
    }


def main():
    d = load_breast_cancer()
    X = d.data.astype(np.float32)
    y = (1 - d.target).astype(np.int64)  # 1 = maligne (classe positive)
    sizes = [X.shape[1], HIDDEN, 2]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    rows = {"spearvm_mlp": [], "pytorch_mlp": [], "logistic_regression": []}
    times = {"spearvm_mlp": 0.0, "pytorch_mlp": 0.0, "logistic_regression": 0.0}
    for k, (tr, te) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]).astype(np.float32)
        Xte = sc.transform(X[te]).astype(np.float32)
        ytr, yte = y[tr], y[te]

        (fn), dt = C.timed(lambda: train_spear(Xtr, ytr, sizes, seed=k))
        times["spearvm_mlp"] += dt
        rows["spearvm_mlp"].append(metrics(fn(Xte), yte))

        (fn), dt = C.timed(lambda: train_torch(Xtr, ytr, sizes, seed=k))
        times["pytorch_mlp"] += dt
        rows["pytorch_mlp"].append(metrics(fn(Xte), yte))

        lr, dt = C.timed(lambda: LogisticRegression(max_iter=5000).fit(Xtr, ytr))
        times["logistic_regression"] += dt
        rows["logistic_regression"].append(metrics(lr.predict_proba(Xte)[:, 1], yte))

    summary = {}
    for name, rs in rows.items():
        summary[name] = {m: {"mean": round(float(np.mean([r[m] for r in rs])), 4),
                             "std": round(float(np.std([r[m] for r in rs])), 4)}
                         for m in ("accuracy", "auc", "recall_malignant")}
        summary[name]["train_seconds_total_5folds"] = round(times[name], 3)
        print(f"{name:20s} acc {summary[name]['accuracy']['mean']:.4f}  "
              f"AUC {summary[name]['auc']['mean']:.4f}  "
              f"rappel malin {summary[name]['recall_malignant']['mean']:.4f}  "
              f"| {times[name]:.2f} s")
    C.save("breast_cancer_screening", {
        "dataset": "sklearn load_breast_cancer (569 patientes, 30 features), CV 5 plis stratifiee",
        "model": f"MLP {sizes}, GELU exacte, Adam lr={LR}, batch {BATCH}, {EPOCHS} epochs; standardisation fittee par pli",
        "results": summary,
    })


if __name__ == "__main__":
    main()
