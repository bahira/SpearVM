"""Protocole binaire des frames de simulation (serveur -> navigateur).

Un frame = en-tete JSON longueur-prefixee + charge utile binaire alignee 4o :

    u32  headerLen      (little endian, longueur du JSON *sans* padding)
    u8[] headerJson     (utf-8, puis padding 0 jusqu'au multiple de 4)
    u8[] payload        (Float32Array ou Int16Array selon header["dtype"])

L'en-tete porte toujours : `kind`, `tick`, `time`, `compute_ms`, `dtype`,
`shape`, `scale` (facteur de dequantification) et `stats` (libre).
Cette forme evite un schema binaire rigide tout en gardant le cout CPU/octet
d'un flux typed-array pur cote client.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Iterable

import numpy as np

MAGIC = "spearvm.sim.v1"


def quantize_i16(data: np.ndarray, amplitude: float | None = None) -> tuple[np.ndarray, float]:
    """Quantifie en int16 symetrique. Retourne (donnees, scale) avec
    valeur_reelle = donnees * scale. Divise par deux la bande passante."""
    arr = np.asarray(data, dtype=np.float32)
    if amplitude is None:
        amplitude = float(np.max(np.abs(arr))) if arr.size else 0.0
    if not np.isfinite(amplitude) or amplitude <= 1e-12:
        return np.zeros(arr.shape, dtype=np.int16), 1.0
    scale = amplitude / 32767.0
    out = np.clip(np.rint(arr / scale), -32767, 32767).astype(np.int16)
    return out, scale


def encode_frame(
    kind: str,
    payload: np.ndarray | None,
    *,
    tick: int,
    sim_time: float,
    compute_ms: float,
    shape: Iterable[int] = (),
    scale: float = 1.0,
    stats: dict[str, Any] | None = None,
) -> bytes:
    """Serialise un frame. `payload` doit etre contigu float32 ou int16."""
    if payload is None:
        buf = b""
        dtype = "none"
    else:
        arr = np.ascontiguousarray(payload)
        if arr.dtype == np.float32:
            dtype = "f32"
        elif arr.dtype == np.int16:
            dtype = "i16"
        else:  # normalisation defensive
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            dtype = "f32"
        buf = arr.tobytes()

    header = {
        "magic": MAGIC,
        "kind": kind,
        "tick": tick,
        "time": round(float(sim_time), 4),
        "compute_ms": round(float(compute_ms), 3),
        "dtype": dtype,
        "shape": list(shape),
        "scale": float(scale),
        "stats": stats or {},
    }
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(raw)) % 4
    return struct.pack("<I", len(raw)) + raw + b"\x00" * pad + buf


def decode_frame(blob: bytes) -> tuple[dict[str, Any], np.ndarray | None]:
    """Decodeur miroir — utilise par les tests (et utile aux clients Python)."""
    (hlen,) = struct.unpack_from("<I", blob, 0)
    header = json.loads(blob[4 : 4 + hlen].decode("utf-8"))
    start = 4 + hlen + ((-hlen) % 4)
    body = blob[start:]
    if header["dtype"] == "none":
        return header, None
    dt = np.float32 if header["dtype"] == "f32" else np.int16
    arr = np.frombuffer(body, dtype=dt)
    shape = tuple(header.get("shape") or ())
    if shape and int(np.prod(shape)) == arr.size:
        arr = arr.reshape(shape)
    return header, arr
