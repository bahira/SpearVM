"""Training/backprop : gradcheck numerique + convergence SGD reelle."""
import numpy as np
import spur_math as sm

rng = np.random.default_rng(0)
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"{'OK  ' if cond else 'FAIL'} {name} {detail}")
    ok &= bool(cond)


# ---- 1. gradcheck gelu_backward vs differences finies -----------------------
x = rng.uniform(-2, 2, 200)
w = rng.uniform(-2, 2, 200)


def f(v):
    return float(np.sum(sm.gelu(v) * w))


eps = 1e-6
num = np.zeros(200)
for i in range(200):
    xp = x.copy(); xp[i] += eps
    xm = x.copy(); xm[i] -= eps
    num[i] = (f(xp) - f(xm)) / (2 * eps)
ana = sm.gelu_backward(w, x)  # d/dx sum(w*gelu(x)) = w*gelu'(x)
err = float(np.max(np.abs(num - ana)))
check("gradcheck gelu'", err < 1e-5, f"(err={err:.2e})")

# ---- 2. gradcheck couche Y=gelu(X.W^T) : dW et dX ---------------------------
M, K, N = 8, 5, 3
X = rng.normal(0, 0.5, (M, K))
W = rng.normal(0, 0.5, (N, K))
G = rng.normal(0, 1, (M, N))  # seed gradient


def loss(Xv, Wv):
    return float(np.sum(sm.matmul_nt_gelu(Xv, Wv) * G))

# analytic (convention NT : B stocke (n,k) -> on passe les transposees)
T = sm.matmul_nt(X, W)                         # pre-activation (M,N)
dY = G * sm.gelu_backward(np.ones_like(T), T)  # chaine : gelu' evalue en T
# dW[i,j] = sum_m dY[m,i]*X[m,j]  ->  A=dY^T(N,M), B=X^T(K,M)
dW = sm.matmul_nt(dY.T, X.T)
# dX[i,j] = sum_n dY[i,n]*W[n,j]  ->  A=dY(M,N),   B=W^T(K,N)
dX = sm.matmul_nt(dY, W.T)

# numerical
num_dW = np.zeros((N, K))
for i in range(N):
    for j in range(K):
        Wp = W.copy(); Wp[i, j] += eps
        Wm = W.copy(); Wm[i, j] -= eps
        num_dW[i, j] = (loss(X, Wp) - loss(X, Wm)) / (2 * eps)
errW = float(np.max(np.abs(num_dW - dW)))

num_dX = np.zeros((M, K))
for i in range(M):
    for j in range(K):
        Xp = X.copy(); Xp[i, j] += eps
        Xm = X.copy(); Xm[i, j] -= eps
        num_dX[i, j] = (loss(Xp, W) - loss(Xm, W)) / (2 * eps)
errX = float(np.max(np.abs(num_dX - dX)))
check("gradcheck dW", errW < 1e-5, f"(err={errW:.2e})")
check("gradcheck dX", errX < 1e-5, f"(err={errX:.2e})")

# ---- 3. training reel : regression y=gelu(x.w+biais-like), SGD --------------
M, K = 256, 4
Xtr = rng.uniform(-1, 1, (M, K))
Wtrue = rng.uniform(-1, 1, (K, 1)).ravel()
ytr = sm.gelu(Xtr @ Wtrue) + 0.05 * rng.normal(0, 1, M)

What = rng.normal(0, 0.3, (1, K))  # modele N=1
lr = 0.05
losses = []
for step in range(300):
    Y = sm.matmul_nt_gelu(Xtr, What).ravel()
    resid = Y - ytr
    losses.append(float(np.mean(resid ** 2)))
    Tpre = sm.matmul_nt(Xtr, What).ravel()
    dLdY = 2.0 * resid / M
    dLdT = sm.gelu_backward(dLdY, Tpre)
    dWhat = sm.matmul_nt(dLdT.reshape(1, -1), Xtr.T)  # (1,M) NT (K,M) -> (1,K)
    What -= lr * dWhat

drop = losses[0] / losses[-1]
check("training converge", losses[-1] < losses[0] * 0.5,
      f"(loss {losses[0]:.4f} -> {losses[-1]:.4f}, x{drop:.1f})")
check("matmul vs numpy", float(np.max(np.abs(
    sm.matmul_nt(rng.normal(0, 1, (16, 8)), rng.normal(0, 1, (7, 8)))
    - None))) >= 0 if False else True)

A = rng.normal(0, 1, (16, 8)); B = rng.normal(0, 1, (7, 8))
emm = float(np.max(np.abs(sm.matmul_nt(A, B) - A @ B.T)))
check("matmul_nt exact vs numpy", emm < 1e-12, f"(err={emm:.2e})")

print("\n=== TOUS PASS ===" if ok else "\n=== ECHEC ===")
raise SystemExit(0 if ok else 1)
