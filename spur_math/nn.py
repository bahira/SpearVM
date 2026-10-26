"""MLP d'entrainement sur SpearVM : forward, backward exact et Adam.

Couches cachees : GELU exacte (gelu_erf, champion superspear) ; couche de sortie lineaire.
Perte : entropie croisee softmax. Tout en float32 (ou float64 pour gradcheck).

    net = SpearMLP([64, 256, 10], seed=0)
    loss = net.train_step(Xb, yb, opt)   # Adam
    logits = net.predict_logits(X)
"""
import numpy as np

from . import gelu_backward, gelu_erf, matmul_nt, PackedNT


class SpearMLP:
    def __init__(self, sizes, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.dtype = np.dtype(dtype)
        self.params = []
        for fin, fout in zip(sizes[:-1], sizes[1:]):
            W = (rng.standard_normal((fout, fin)) * np.sqrt(2.0 / fin)).astype(self.dtype)
            b = np.zeros(fout, dtype=self.dtype)
            self.params.append([W, b])

    # ---- forward ----
    def _forward(self, X):
        H = np.ascontiguousarray(X, dtype=self.dtype)
        acts, pres = [H], []
        L = len(self.params)
        for i, (W, b) in enumerate(self.params):
            T = matmul_nt(H, W) + b
            pres.append(T)
            if i < L - 1:
                H = gelu_erf(T)
                acts.append(H)
            else:
                H = T
        return H, acts, pres

    def predict_logits(self, X):
        return self._forward(X)[0]

    # ---- perte + gradients ----
    def loss_and_grads(self, X, y):
        Z, acts, pres = self._forward(X)
        B = Z.shape[0]
        Zs = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Zs)
        P = E / E.sum(axis=1, keepdims=True)
        loss = float(-np.mean(np.log(P[np.arange(B), y] + 1e-30)))
        dZ = P.copy()
        dZ[np.arange(B), y] -= 1.0
        dZ /= B
        grads = [None] * len(self.params)
        for i in range(len(self.params) - 1, -1, -1):
            W, _ = self.params[i]
            Hin = acts[i]
            dW = matmul_nt(np.ascontiguousarray(dZ.T), np.ascontiguousarray(Hin.T))
            db = dZ.sum(axis=0)
            grads[i] = [dW, db]
            if i > 0:
                dH = matmul_nt(dZ, np.ascontiguousarray(W.T))
                dZ = gelu_backward(dH, pres[i - 1])
        return loss, grads

    def accuracy(self, X, y):
        return float((self.predict_logits(X).argmax(axis=1) == y).mean())


class Adam:
    def __init__(self, net, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.net, self.lr, self.b1, self.b2, self.eps = net, lr, b1, b2, eps
        self.wd = weight_decay
        self.t = 0
        self.m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in net.params]
        self.v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in net.params]

    def step(self, grads):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for i, ((W, b), (gW, gb)) in enumerate(zip(self.net.params, grads)):
            if self.wd:
                gW = gW + self.wd * W
            for j, (p, g) in enumerate(((W, gW), (b, gb))):
                m, v = self.m[i][j], self.v[i][j]
                m *= self.b1; m += (1 - self.b1) * g
                v *= self.b2; v += (1 - self.b2) * g * g
                p -= (self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)).astype(p.dtype)


def train_step(net, opt, X, y):
    loss, grads = net.loss_and_grads(X, y)
    opt.step(grads)
    return loss


class PackedMLP:
    """Inference : poids pre-packes (PackedNT), GELU exacte. Pour le serving par lots."""

    def __init__(self, net):
        self.packs = [(PackedNT(W), b.astype(np.float32)) for W, b in net.params]

    def predict_logits(self, X):
        H = np.ascontiguousarray(X, dtype=np.float32)
        L = len(self.packs)
        for i, (P, b) in enumerate(self.packs):
            if i < L - 1:
                H = P.matmul_gelu(H, b)   # GEMM + biais + GELU exacte fusionnes
            else:
                H = P.matmul(H) + b
        return H
