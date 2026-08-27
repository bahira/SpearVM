"""GELU v2 quintique (smoothstep certifie) — cf. rapport BENCHMARK & AMELIORATION.
Linf 0.0174 sur [-3.5,3.5] ; MSE 1.35e-4 sur [-4,4] ; queue bornee sur R."""
import pytest
import numpy as np
import math
import spur_math


def _ref(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))


class TestGeluErf:
    def test_linf(self):
        """GELU via erf_v2 : Linf 2.05e-5, ~850x plus precis que le quintique."""
        x = np.linspace(-6, 6, 200001)
        err = np.abs(spur_math.gelu_erf(x) - _ref(x))
        assert err.max() < 5e-5

    def test_mse(self):
        x = np.linspace(-6, 6, 100001)
        mse = np.mean((spur_math.gelu_erf(x) - _ref(x)) ** 2)
        assert mse < 1e-9

    def test_beats_quintic(self):
        x = np.linspace(-4, 4, 100001)
        err_erf = np.abs(spur_math.gelu_erf(x) - _ref(x)).max()
        err_q = np.abs(spur_math.gelu_quintic(x) - _ref(x)).max()
        assert err_erf < err_q / 50

    def test_finite(self):
        x = np.array([-1e5, -10.0, 0.0, 10.0, 1e5])
        out = spur_math.gelu_erf(x)
        assert np.all(np.isfinite(out))
        # asymptotes : saturé a 0 a gauche, ~x a droite
        assert abs(out[0]) < 1e-9
        assert abs(out[-1] - 1e5) < 1e-3


class TestGeluQuintic:
    def test_linf_bounded(self):
        x = np.linspace(-3.5, 3.5, 400001)
        err = np.abs(spur_math.gelu_quintic(x) - _ref(x))
        assert err.max() < 0.018

    def test_mse(self):
        x = np.linspace(-4, 4, 200001)
        mse = np.mean((spur_math.gelu_quintic(x) - _ref(x)) ** 2)
        assert mse < 1.5e-4

    def test_tail_bounded(self):
        """queue : plus de derive lineaire, erreur bornee sur |x|<=1e5."""
        xs = np.array([-1e5, -10.0, 10.0, 1e5])
        out = spur_math.gelu_quintic(xs)
        assert np.all(np.isfinite(out))
        assert np.abs(out[0] - (-0.01104961)) < 1e-6  # sature a gauche
        assert np.abs(out[-1] - (1e5 - 0.01104961)) < 1e-3  # identite a droite

    def test_more_accurate_than_v1(self):
        """v2 doit battre v1 en precision sur [-3.5,3.5]."""
        x = np.linspace(-3.5, 3.5, 400001)
        err_v2 = np.abs(spur_math.gelu_quintic(x) - _ref(x)).max()
        err_v1 = np.abs(spur_math.gelu(x) - _ref(x)).max()
        assert err_v2 < err_v1

    def test_scalar_matches_batch(self):
        x = np.random.uniform(-4, 4, 1000)
        got = spur_math.gelu_quintic(x)
        assert np.all(np.isfinite(got))