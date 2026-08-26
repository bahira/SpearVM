import numpy as np
import spur_math as sm

rng = np.random.default_rng(0)
x64 = rng.uniform(-2.5, 2.5, 300)
w = rng.normal(0, 1, 300)
ana = sm.erf_backward(w, x64)
bad = np.where(~np.isfinite(ana))[0]
print('nb NaN:', len(bad), 'indices:', bad[:10])
for i in bad[:5]:
    print(f'  x={x64[i]:.4f} ana={ana[i]}')

x2 = np.linspace(-2, 2, 300)
ana2 = sm.erf_backward(w[:300], x2)
print('dans [-2,2] fini:', bool(np.isfinite(ana2).all()))
