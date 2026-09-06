"""Attention Lab : le mecanisme d'attention, en direct, avec ses vrais couts.

Ce cas d'usage n'anime pas une jolie surface : il execute reellement un bloc de
decodeur SpearVM (attention multi-tetes sur cache KV + FFN) token apres token,
et pousse au navigateur la **carte d'attention** effectivement calculee, avec
les couts mesures du tick.

Ce que la scene rend visible :
  * la structure des poids d'attention par tete et par position ;
  * le prix du format des poids (f32 / bf16 / int8) sur le debit reel ;
  * le fait que l'attention finit par dominer quand le contexte s'allonge.

Charge utile : (H_q x contexte) poids quantifies en int16 — 8 x 512 = 8 Ko par
frame, soit deux ordres de grandeur sous les autres simulations.
"""

from __future__ import annotations

import time

import numpy as np

from ..protocol import quantize_i16
from .base import Frame, ParamSpec, Simulation

FORMATS = ("f32", "bf16", "i8")


def _rmsnorm(x, w, eps=1e-6):
    n = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
    return ((x / n) * w).astype(np.float32)


class AttentionLabSim(Simulation):
    sim_id = "attention"
    title = "Attention Lab"
    subtitle = "decodage reel : carte d'attention et cout du format"
    description = (
        "Un bloc de decodeur SpearVM tourne token apres token sur un cache KV "
        "packe. La carte d'attention reellement calculee est renvoyee telle "
        "quelle, avec le debit mesure du format de poids choisi."
    )
    kernels_used = ("attention_mha", "KVCache", "QuantizedWeight", "softmax",
                    "matmul_nt_gelu", "matmul_nt")
    default_rate = 12.0
    structural_params = ("context", "heads", "kv_heads", "d_head", "d_ff", "format", "seed")

    params_spec = (
        ParamSpec("context", "Contexte (tokens)", "select", 512,
                  options=[256, 512, 1024, 2048],
                  help="Taille du cache KV contre lequel chaque token decode"),
        ParamSpec("heads", "Tetes de requetes", "select", 8, options=[4, 8, 12, 16]),
        ParamSpec("kv_heads", "Tetes de cles (GQA)", "select", 2, options=[1, 2, 4, 8]),
        ParamSpec("d_head", "Dimension par tete", "select", 64, options=[32, 64, 96]),
        ParamSpec("d_ff", "Largeur du FFN", "select", 2048,
                  options=[1024, 2048, 3072, 4096]),
        ParamSpec("format", "Format des poids", "select", "i8", options=list(FORMATS),
                  help="f32 = reference ; bf16 = /2 ; int8 = /4 (echelle par ligne)"),
        ParamSpec("focus", "Tete affichee en relief", "range", 0, minimum=0,
                  maximum=15, step=1, help="Les autres restent visibles en fond"),
        ParamSpec("temperature", "Piquant des requetes", "range", 3.0, minimum=0.2,
                  maximum=8.0, step=0.1,
                  help="Amplitude des requetes : concentre ou etale l'attention"),
        ParamSpec("seed", "Graine", "range", 5, minimum=1, maximum=64, step=1),
    )

    # ------------------------------------------------------------------
    def setup(self) -> None:
        import spur_math as sm  # noqa: PLC0415

        self._sm = sm
        p = self.params
        self.tk = int(p["context"])
        self.h_q = int(p["heads"])
        self.h_kv = min(int(p["kv_heads"]), self.h_q)
        while self.h_q % self.h_kv:
            self.h_kv -= 1
        self.dh = int(p["d_head"])
        self.dm = self.h_q * self.dh
        self.d_ff = int(p["d_ff"])
        rng = np.random.default_rng(int(p["seed"]))

        def w(*shape):
            return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

        self.wq = w(self.dm, self.dm)
        self.wo = w(self.dm, self.dm)
        self.w1 = w(self.d_ff, self.dm)
        self.w2 = w(self.dm, self.d_ff)
        self.n1 = np.ones(self.dm, dtype=np.float32)
        self.n2 = np.ones(self.dm, dtype=np.float32)

        # cache KV : structure en "sujets" pour que la carte ait du sens
        base = rng.standard_normal((8, self.h_kv, self.dh)).astype(np.float32)
        base /= np.linalg.norm(base, axis=-1, keepdims=True)
        idx = rng.integers(0, 8, size=self.tk)
        self.base = base
        self.topics = idx
        k = base[idx] + 0.35 * rng.standard_normal((self.tk, self.h_kv, self.dh))
        v = rng.standard_normal((self.tk, self.h_kv, self.dh))
        # NB : `self.k` est deja la facade de noyaux (classe de base) — ne pas
        # l'ecraser. Et on ne divise PAS K par sqrt(dh) : `attend` applique
        # deja l'echelle, le faire deux fois aplatit les logits (entropie 1).
        self.kmat = np.ascontiguousarray(k, dtype=np.float32)
        self.vmat = np.ascontiguousarray(v, dtype=np.float32)
        self.cache = sm.KVCache(self.kmat, self.vmat)

        fmt = str(p["format"])
        self.fmt = fmt if fmt in FORMATS else "i8"
        if self.fmt == "f32":
            self.qw = None
            self.w_bytes = sum(a.nbytes for a in (self.wq, self.wo, self.w1, self.w2))
        else:
            self.qw = {n: sm.QuantizedWeight(getattr(self, n), dtype=self.fmt)
                       for n in ("wq", "wo", "w1", "w2")}
            self.w_bytes = sum(q.nbytes for q in self.qw.values())

        self.x = (rng.standard_normal((1, self.dm)) / np.sqrt(self.dm)).astype(np.float32)
        self.tokens = 0
        self._rate_ema = 0.0
        self._flops = (2.0 * (2 * self.dm * self.dm + 2 * self.dm * self.d_ff)
                       + 4.0 * self.tk * self.dh * self.h_q)

    # ------------------------------------------------------------------
    def _mm(self, name, a):
        if self.qw is None:
            return self._sm.matmul_nt(a, getattr(self, name))
        return self.qw[name].matmul(a)

    def command(self, message: dict) -> None:
        if message.get("type") == "reseed":
            self.params["seed"] = int(message.get("seed",
                                                  (int(self.params["seed"]) % 64) + 1))
            self.setup()

    def step(self, dt: float) -> Frame:
        sm = self._sm
        self.tick += 1
        self.sim_time += dt
        temp = float(self.params["temperature"])

        with self.timer():
            # Le debit annonce doit mesurer le DECODAGE, pas la visualisation :
            # la carte d'attention est calculee ensuite, hors de ce chronometre.
            t_block = time.perf_counter()
            h = _rmsnorm(self.x, self.n1)
            q = self._mm("wq", h).reshape(1, self.h_q, self.dh)
            # Une requete aleatoire donnerait une attention uniforme (entropie 1) :
            # aucun interet a regarder. On l'oriente vers un "sujet" du cache qui
            # tourne lentement — c'est ce que fait un vrai modele, et cela rend la
            # carte lisible sans changer un seul chemin de calcul.
            topic = (self.tick // 6) % self.base.shape[0]
            rep_kv = self.h_q // self.h_kv
            aim = np.repeat(self.base[topic][None], rep_kv, axis=1) if rep_kv > 1 \
                else self.base[topic][None]
            q = np.ascontiguousarray(
                (0.25 * q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
                 + aim.reshape(1, self.h_q, self.dh))
                * np.float32(temp * np.sqrt(self.dh)), dtype=np.float32)
            a = self.cache.attend(np.ascontiguousarray(q)).reshape(1, -1)
            h = self.x + self._mm("wo", a)
            n = _rmsnorm(h, self.n2)
            z = self._mm("w1", n)
            u = np.clip(0.306923 * z + 0.501, 0.0, 1.002).astype(np.float32)
            g = (0.997729 * (z * u) - 0.004004).astype(np.float32)
            self.x = h + self._mm("w2", g)
            self.x = np.ascontiguousarray(
                self.x / (np.abs(self.x).max() + 1e-6), dtype=np.float32)

            block_ms = (time.perf_counter() - t_block) * 1e3

            # carte d'attention : les memes noyaux, mais on garde les poids
            # (cout de visualisation, exclu du debit annonce)
            rep = self.h_q // self.h_kv
            scores = np.empty((self.h_q, self.tk), dtype=np.float32)
            for hd in range(self.h_q):
                sm.matmul_nt(np.ascontiguousarray(q[:, hd]),
                             np.ascontiguousarray(self.kmat[:, hd // rep]),
                             out=scores[hd:hd + 1])
            weights = sm.softmax(scores * np.float32(1.0 / np.sqrt(self.dh)))

        self.tokens += 1
        rate = 1.0 / max(block_ms / 1e3, 1e-9)
        self._rate_ema = rate if self._rate_ema == 0 else 0.85 * self._rate_ema + 0.15 * rate
        gflops = (self._flops / 1e9) / max(block_ms / 1e3, 1e-9)
        top = float(weights.max())
        # entropie normalisee : 0 = une seule position, 1 = attention uniforme
        w64 = weights.astype(np.float64) + 1e-12
        ent = float((-(w64 * np.log(w64)).sum(axis=1) / np.log(self.tk)).mean())

        payload, scale = quantize_i16(weights, amplitude=max(top, 1e-6))
        return Frame(
            kind="attention_map",
            payload=payload.reshape(-1),
            shape=(self.h_q, self.tk),
            scale=scale,
            stats={
                "context": self.tk, "heads": self.h_q, "kv_heads": self.h_kv,
                "d_head": self.dh, "d_model": self.dm, "d_ff": self.d_ff,
                "format": self.fmt,
                "poids_Mo": round(self.w_bytes / 1e6, 2),
                "kv_Mo": round(self.cache.nbytes / 1e6, 2),
                "tokens": self.tokens,
                "tokens_par_s": round(self._rate_ema, 1),
                "decode_ms": round(block_ms, 3),
                "carte_ms": round(max(self._compute_ms - block_ms, 0.0), 3),
                "gflops": round(gflops, 1),
                "mflop_par_token": round(self._flops / 1e6, 1),
                "attention_max": round(top, 4),
                "entropie": round(ent, 4),
                "focus": int(self.params["focus"]) % self.h_q,
            },
        )

    def metadata(self) -> dict:
        return {"bloc": "RMSNorm -> attention GQA -> RMSNorm -> FFN GELU",
                "formats": list(FORMATS)}
