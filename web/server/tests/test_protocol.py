"""Protocole binaire : alignement, quantification, aller-retour."""

from __future__ import annotations

import struct

import numpy as np

from spearvm_sim.protocol import decode_frame, encode_frame, quantize_i16


def test_roundtrip_float32():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    blob = encode_frame("test", data.reshape(-1), tick=3, sim_time=1.5,
                        compute_ms=2.25, shape=data.shape, stats={"a": 1})
    header, arr = decode_frame(blob)
    assert header["kind"] == "test"
    assert header["tick"] == 3
    assert header["dtype"] == "f32"
    assert header["stats"]["a"] == 1
    assert arr is not None and arr.shape == (2, 3, 4)
    assert np.array_equal(arr, data)


def test_charge_utile_alignee_sur_4_octets():
    """Le client cree des vues typees : le decalage doit rester aligne."""
    for kind in ("a", "ab", "abc", "abcd"):
        blob = encode_frame(kind, np.zeros(4, dtype=np.float32), tick=0,
                            sim_time=0.0, compute_ms=0.0, shape=(4,))
        (hlen,) = struct.unpack_from("<I", blob, 0)
        start = 4 + hlen + ((-hlen) % 4)
        assert start % 4 == 0
        assert (len(blob) - start) == 16


def test_quantification_i16_borne_l_erreur():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(5000).astype(np.float32) * 3.0
    q, scale = quantize_i16(data)
    restored = q.astype(np.float32) * scale
    assert np.max(np.abs(restored - data)) <= scale  # <= 1 LSB
    assert q.dtype == np.int16


def test_quantification_signal_nul():
    q, scale = quantize_i16(np.zeros(16, dtype=np.float32))
    assert scale == 1.0
    assert not q.any()


def test_roundtrip_int16_via_frame():
    data = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    q, scale = quantize_i16(data, amplitude=1.0)
    blob = encode_frame("h", q, tick=1, sim_time=0.1, compute_ms=0.5,
                        shape=(100,), scale=scale)
    header, arr = decode_frame(blob)
    assert header["dtype"] == "i16"
    assert arr is not None
    assert np.max(np.abs(arr.astype(np.float32) * header["scale"] - data)) <= scale


def test_frame_sans_charge_utile():
    header, arr = decode_frame(
        encode_frame("meta", None, tick=0, sim_time=0.0, compute_ms=0.0)
    )
    assert header["dtype"] == "none"
    assert arr is None
