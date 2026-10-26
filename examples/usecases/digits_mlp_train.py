"""Cas d'usage 1 : entrainer un MLP sur des chiffres manuscrits reels (sklearn digits, 1797 images 8x8).

Meme initialisation, memes batches, meme Adam : SpearVM (spur_math.nn) contre PyTorch CPU.
Mesure : precision test, temps d'entrainement, accord des poids finaux.
"""
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import _common as C
import spur_math as sm
from spur_math.nn import SpearMLP, Adam, train_step

EPOCHS, BATCH, LR, HIDDEN = 40, 64, 3e-3, 128


def main():
    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    sizes = [X.shape[1], HIDDEN, 10]

    # --- SpearVM ---
    net = SpearMLP(sizes, seed=0)
    init = [(W.copy(), b.copy()) for W, b in net.params]
    opt = Adam(net, lr=LR)
    rng = np.random.default_rng(1)

    def spear_epoch():
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), BATCH):
            idx = perm[s:s + BATCH]
            train_step(net, opt, Xtr[idx], ytr[idx])

    _, t_spear = C.timed(lambda: [spear_epoch() for _ in range(EPOCHS)])
    acc_spear = net.accuracy(Xte, yte)

    # --- PyTorch : meme init, memes batches (meme seed de permutation) ---
    torch.manual_seed(0)
    tl = [torch.nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    with torch.no_grad():
        for lin, (W, b) in zip(tl, init):
            lin.weight.copy_(torch.from_numpy(W)); lin.bias.copy_(torch.from_numpy(b))

    def fwd(x):
        h = x
        for i, lin in enumerate(tl):
            h = lin(h)
            if i < len(tl) - 1:
                h = torch.nn.functional.gelu(h)
        return h

    topt = torch.optim.Adam([p for lin in tl for p in lin.parameters()], lr=LR)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    rng2 = np.random.default_rng(1)

    def torch_epoch():
        perm = rng2.permutation(len(Xtr))
        for s in range(0, len(Xtr), BATCH):
            idx = torch.from_numpy(perm[s:s + BATCH])
            topt.zero_grad()
            loss = torch.nn.functional.cross_entropy(fwd(Xt[idx]), yt[idx])
            loss.backward()
            topt.step()

    _, t_torch = C.timed(lambda: [torch_epoch() for _ in range(EPOCHS)])
    with torch.no_grad():
        acc_torch = float((fwd(torch.from_numpy(Xte)).argmax(1).numpy() == yte).mean())
        Wt = [lin.weight.numpy() for lin in tl]
    dW = max(np.abs(W - Wref).max() for W, Wref in zip([p[0] for p in net.params], Wt))

    res = {
        "dataset": "sklearn load_digits (1797 x 8x8, 10 classes), split 75/25 stratifie",
        "model": f"MLP {sizes}, GELU exacte, Adam lr={LR}, batch {BATCH}, {EPOCHS} epochs",
        "spearvm": {"test_accuracy": round(acc_spear, 4), "train_seconds": round(t_spear, 3)},
        "pytorch_cpu": {"test_accuracy": round(acc_torch, 4), "train_seconds": round(t_torch, 3)},
        "max_weight_diff_after_training": float(dW),
        "note": "les poids divergent par accumulation d'arrondis fp32 ; la metrique utile est la precision",
    }
    print(f"SpearVM : acc test {acc_spear:.4f} | {t_spear:.2f} s")
    print(f"PyTorch : acc test {acc_torch:.4f} | {t_torch:.2f} s")
    print(f"ecart max des poids apres entrainement : {dW:.2e}")
    C.save("digits_mlp_train", res)


if __name__ == "__main__":
    main()
