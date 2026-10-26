"""Benchmark end-to-end d'un encodeur Transformer (architecture BERT/BGE-small) sur CPU.

Comparaison, meme modele, memes entrees, meme machine :
  1. PyTorch CPU (reference, GELU exact)
  2. ONNX Runtime CPU (graphe exporte depuis le meme modele PyTorch)
  3. Hybride SpearVM : attention/LayerNorm en PyTorch, FFN (matmul+GELU+matmul)
     via spur_math (matmul_nt_gelu fusionne + matmul_nt)

Limites assumees (a lire avant de citer les chiffres) :
  - poids ALEATOIRES (seed fixe) : Hugging Face est inaccessible depuis le sandbox.
    Le cout de calcul (FLOPs, formes) est identique a un vrai checkpoint ;
    la qualite des sorties n'est pas evaluee, seulement l'ecart entre chemins.
  - embeddings / tokenizer exclus : entree = tenseur (B,T,d) identique pour tous.
  - GELU de SpearVM = approximation polynomiale (erreur max ~0.08) : on mesure
    l'ecart de sortie resultant (max abs, cosinus) par rapport a la reference.

Usage : python examples/bench_encoder_e2e.py [--layers 12] [--reps 20]
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import spur_math as sm  # noqa: E402


class Layer(nn.Module):
    def __init__(self, d, nh, ff):
        super().__init__()
        self.nh = nh
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff); self.f2 = nn.Linear(ff, d)

    def attn(self, h):
        B, T, D = h.shape
        dh = D // self.nh
        q = self.q(h).view(B, T, self.nh, dh).transpose(1, 2)
        k = self.k(h).view(B, T, self.nh, dh).transpose(1, 2)
        v = self.v(h).view(B, T, self.nh, dh).transpose(1, 2)
        s = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
        return self.o((s @ v).transpose(1, 2).reshape(B, T, D))

    def forward(self, x):  # pre-LN, GELU exact (reference)
        x = x + self.attn(self.ln1(x))
        return x + self.f2(F.gelu(self.f1(self.ln2(x))))


class Encoder(nn.Module):
    def __init__(self, layers, d, nh, ff):
        super().__init__()
        self.layers = nn.ModuleList([Layer(d, nh, ff) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.ln(x).mean(dim=1)


def spear_forward(model, packed, x):
    """Hybride : attention/LN en torch, FFN en SpearVM (float32, poids pre-packes)."""
    with torch.no_grad():
        for l, (P1, b1, P2, b2) in zip(model.layers, packed):
            x = x + l.attn(l.ln1(x))
            h = l.ln2(x).numpy()
            B, T, D = h.shape
            H = P1.matmul_gelu(h.reshape(-1, D), b1)              # gelu(h W1^T + b1)
            Y = P2.matmul(H) + b2                                 # H W2^T + b2
            x = x + torch.from_numpy(Y.reshape(B, T, D))
        return model.ln(x).mean(dim=1)


def timeit(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return {"median_ms": round(statistics.median(ts), 2),
            "p90_ms": round(ts[int(0.9 * (len(ts) - 1))], 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    ff = 4 * a.d
    model = Encoder(a.layers, a.d, a.heads, ff).eval()

    # poids numpy float32 pour SpearVM (copies contiguës, une seule fois)
    # poids pre-packes une fois (le cout de packing est hors de la boucle d'inference)
    W = []
    for l in model.layers:
        W.append((sm.PackedNT(l.f1.weight.detach().numpy()),
                  np.ascontiguousarray(l.f1.bias.detach().numpy(), dtype=np.float32),
                  sm.PackedNT(l.f2.weight.detach().numpy()),
                  np.ascontiguousarray(l.f2.bias.detach().numpy(), dtype=np.float32)))

    # export ONNX (chemin ORT)
    import onnxruntime as ort
    onnx_path = os.path.join("/tmp", f"enc_{a.layers}L_{a.d}.onnx")
    results = {"config": vars(a) | {"ff": ff, "onnxruntime": ort.__version__,
                                    "torch": torch.__version__,
                                    "cpu_flags_avx512": os.popen("grep -c avx512f /proc/cpuinfo").read().strip()}}
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = a.threads
    sess_opts.inter_op_num_threads = 1
    for B in a.batch:
        x = torch.randn(B, a.seq, a.d, generator=torch.Generator().manual_seed(B))
        with torch.no_grad():
            ref = model(x).numpy()
        torch.onnx.export(model, (x,), onnx_path, input_names=["x"],
                          output_names=["y"], opset_version=17, dynamo=False)
        sess = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])
        xn = x.numpy()
        ort_out = sess.run(None, {"x": xn})[0]

        def run_torch():
            with torch.no_grad():
                model(x)

        def run_ort():
            sess.run(None, {"x": xn})

        def run_spear():
            spear_forward(model, W, x)

        sp_out = spear_forward(model, W, x).numpy()
        r = {}
        r["torch_cpu"] = timeit(run_torch, a.reps)
        r["onnxruntime_cpu"] = timeit(run_ort, a.reps)
        r["spearvm_ffn_hybrid"] = timeit(run_spear, a.reps)
        cos = float(np.sum(sp_out * ref) / (np.linalg.norm(sp_out) * np.linalg.norm(ref)))
        r["accuracy_vs_torch"] = {
            "max_abs_diff": float(np.abs(sp_out - ref).max()),
            "cosine": round(cos, 8),
            "ort_max_abs_diff": float(np.abs(ort_out - ref).max()),
        }
        r["speedup_vs_torch"] = round(r["torch_cpu"]["median_ms"] / r["spearvm_ffn_hybrid"]["median_ms"], 3)
        r["speedup_vs_ort"] = round(r["onnxruntime_cpu"]["median_ms"] / r["spearvm_ffn_hybrid"]["median_ms"], 3)
        results[f"batch_{B}"] = r
        print(f"\n=== batch {B} (seq {a.seq}, {a.layers} couches, d={a.d}) ===")
        print(f"  PyTorch CPU        : {r['torch_cpu']['median_ms']:9.2f} ms (p90 {r['torch_cpu']['p90_ms']})")
        print(f"  ONNX Runtime CPU   : {r['onnxruntime_cpu']['median_ms']:9.2f} ms (p90 {r['onnxruntime_cpu']['p90_ms']})")
        print(f"  SpearVM FFN hybride: {r['spearvm_ffn_hybrid']['median_ms']:9.2f} ms (p90 {r['spearvm_ffn_hybrid']['p90_ms']})")
        print(f"  precision SpearVM vs torch : max|diff|={r['accuracy_vs_torch']['max_abs_diff']:.3e}  cos={cos:.6f}")
        print(f"  speedup vs torch = x{r['speedup_vs_torch']}   vs ORT = x{r['speedup_vs_ort']}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)
        print("\njson ->", a.json)


if __name__ == "__main__":
    main()
