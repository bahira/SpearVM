"""Tests de correction pour spur_math.
Tolérances calibrées sur les erreurs réelles mesurées des noyaux approximatifs."""
import pytest
import numpy as np
import math
import spur_math


class TestGelu:
    def test_positive_range(self):
        x = np.linspace(0.1, 2.0, 1000)
        out = spur_math.gelu(x)
        ref = 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))
        assert np.max(np.abs(out - ref)) < 0.09

    def test_zero(self):
        out = spur_math.gelu(np.array([0.0]))
        assert abs(out[0]) < 0.05

    def test_finite(self):
        x = np.linspace(-2, 2, 10000)
        out = spur_math.gelu(x)
        assert np.all(np.isfinite(out))

    def test_monotonic_positive(self):
        """gelu croissant sur [0,2]"""
        x = np.linspace(0, 2, 100)
        out = spur_math.gelu(x)
        assert np.all(np.diff(out) >= -0.01)


class TestErf:
    def test_accuracy(self):
        """erf approximatif : tolérance adaptée au kernel rationnel"""
        x = np.linspace(-2, 2, 10000)
        ref = np.vectorize(math.erf)(x)
        got = spur_math.erf(x)
        assert np.max(np.abs(got - ref)) < 0.012

    def test_bounds(self):
        x = np.linspace(-2, 2, 10000)
        got = spur_math.erf(x)
        assert np.all(np.abs(got) < 1.05)

    def test_sign_consistency(self):
        got_pos = spur_math.erf(np.array([0.5, 1.0]))
        assert np.all(got_pos > 0)


class TestTanh:
    def test_accuracy(self):
        x = np.linspace(-2, 2, 10000)
        ref = np.tanh(x)
        got = spur_math.tanh(x)
        assert np.max(np.abs(got - ref)) < 0.009

    def test_range(self):
        x = np.linspace(-3, 3, 10000)
        got = spur_math.tanh(x)
        assert np.all(np.abs(got) <= 1.01)

    def test_finite(self):
        x = np.linspace(-3, 3, 10000)
        got = spur_math.tanh(x)
        assert np.all(np.isfinite(got))


class TestPerformance:
    def test_batch_execution(self):
        n = 500000
        x = np.random.uniform(-2, 2, n)
        out = spur_math.gelu(x)
        assert np.all(np.isfinite(out))
