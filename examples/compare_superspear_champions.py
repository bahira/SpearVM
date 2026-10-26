"""Comparatif des champions superspear (github.com/bahira/superspear, ledger/*.json)
vs les noyaux SpearVM, sur une grille commune [-8,8]. Requiert scipy (reference erf)."""
import sys, numpy as np
from math import erf
from scipy.special import erf as serf  # noqa
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spur_math as sm

x = np.linspace(-8, 8, 160001)
x64 = x.astype(np.float64)

def ex_gelu(x): return 0.5 * x * (1 + serf(x / np.sqrt(2)))
def ex_sig(x): return 1 / (1 + np.exp(-x))
def ex_silu(x): return x * ex_sig(x)
def ex_tanh(x): return np.tanh(x)

def pade_tanh(x):  # SpearVM CHAMPIONS : Pade[3/4], clamp +-4
    y = np.clip(x, -4, 4)
    a, b, c, d = 0.994894946, 0.076611228, 0.402171314, 0.005670342
    return (a * y + b * y ** 3) / (1 + c * y ** 2 + d * y ** 4)

def sv_sigmoid(x): return 0.5 + 0.5 * pade_tanh(x / 2)   # forme SpearVM (SigmoidALU)
def sv_silu(x): return x * sv_sigmoid(x)

# superspear : formes du registre ledger/*.json
def ss_gelu_fast(x): return 1.010719 * np.maximum(x, 0) - 0.057684
def ss_sig_fast(x): return 0.619014 * (x / (1.369594 + np.abs(x))) + 0.5
def ss_silu_fast(x): return 1.016356 * np.maximum(x, 0) - 0.15849
def ss_tanh_sat(x): return 1.007047 * np.maximum(-0.808092, np.minimum(x, 1)) - 0.067587

def stats(name, approx, ref, dom=None):
    e = approx - ref
    print(f"{name:42s} Linf={np.abs(e).max():.3e}  MSE={np.mean(e**2):.3e}")

print("== GELU (domaine [-8,8]) ==")
G = ex_gelu(x64)
stats("SpearVM gelu v1 (kernel actuel)", sm.gelu(x64.astype(np.float64)) if False else
      0.997729 * (x64 * np.clip(0.306923 * x64 + 0.501, 0, 1.002)) - 0.004004, G)
q = sm.gelu_quintic(x64); stats("SpearVM gelu_quintic", q, G)
e = sm.gelu_erf(x64); stats("SpearVM gelu_erf", e, G)
stats("superspear gelu fast (relu)", ss_gelu_fast(x64), G)

print("== SIGMOID ==")
S = ex_sig(x64)
stats("SpearVM sigmoid (tanh Pade[3/4])", sv_sigmoid(x64), S)
stats("superspear sigmoid fast (rationnel)", ss_sig_fast(x64), S)

print("== SiLU ==")
L = ex_silu(x64)
stats("SpearVM silu (x*sig Pade)", sv_silu(x64), L)
stats("superspear silu fast (relu)", ss_silu_fast(x64), L)

print("== TANH ==")
T = np.tanh(x64)
stats("SpearVM tanh Pade[3/4] (clamp 4)", pade_tanh(x64), T)
stats("superspear tanh_sat fast (clip)", ss_tanh_sat(x64), T)
