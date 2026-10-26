"""Petit runtime de graphe de calcul au-dessus des noyaux SpearVM (CPU, sans GPU).

Usage :
    g = Graph()
    x  = g.input("x")
    h  = g.gelu(g.matmul_nt(x, W1, bias=b1))   # reecrit en noyau fusionne
    y  = g.matmul_nt(h, W2)
    out = g.run({"x": X}, fuse=True)[y]

La passe de fusion remplace  matmul_nt -> gelu  par  matmul_nt_gelu  quand le
resultat du matmul n'a pas d'autre consommateur (sinon il faudrait le garder).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import gelu, gelu_backward, matmul_backward, matmul_nt, matmul_nt_gelu


@dataclass
class _Node:
    op: str
    inputs: List[int]
    attrs: Dict[str, Any] = field(default_factory=dict)


def _acc(grads, idx, g):
    grads[idx] = g if idx not in grads else grads[idx] + g


class Graph:
    def __init__(self):
        self.nodes: List[_Node] = []

    # ---- construction ----
    def _add(self, op, inputs, **attrs):
        self.nodes.append(_Node(op, list(inputs), attrs))
        return len(self.nodes) - 1

    def input(self, name):
        return self._add("input", [], name=name)

    def const(self, value):
        return self._add("const", [], value=np.asarray(value))

    def _arg(self, v):
        return v if isinstance(v, int) else self.const(v)

    def matmul_nt(self, a, b, bias=None):
        ins = [self._arg(a), self._arg(b)]
        if bias is not None:
            ins.append(self._arg(bias))
        return self._add("matmul_nt", ins, has_bias=bias is not None)

    def gelu(self, a):
        return self._add("gelu", [self._arg(a)])

    # ---- passes ----
    def _consumers(self):
        cons = {i: 0 for i in range(len(self.nodes))}
        for n in self.nodes:
            for i in n.inputs:
                cons[i] += 1
        return cons

    def fused_plan(self):
        """Retourne la liste de nœuds (index -> (op, inputs)) apres fusion."""
        cons = self._consumers()
        plan = {}
        skip = set()
        for idx, n in enumerate(self.nodes):
            if n.op == "gelu":
                src = n.inputs[0]
                s = self.nodes[src]
                if s.op == "matmul_nt" and cons[src] == 1 and src not in skip:
                    plan[idx] = ("matmul_nt_gelu", s.inputs, s.attrs)
                    skip.add(src)
                    continue
        for idx, n in enumerate(self.nodes):
            if idx in skip or idx in plan:
                continue
            plan[idx] = (n.op, n.inputs, n.attrs)
        return plan, skip

    # ---- execution ----
    def run(self, feeds: Dict[str, Any], fuse: bool = True):
        plan, skip = self.fused_plan() if fuse else (
            {i: (n.op, n.inputs, n.attrs) for i, n in enumerate(self.nodes)},
            set())
        vals: Dict[int, np.ndarray] = {}
        for idx in range(len(self.nodes)):
            if idx in skip:
                continue
            op, ins, attrs = plan[idx]
            a = [vals[i] for i in ins]
            if op == "input":
                vals[idx] = np.asarray(feeds[attrs["name"]])
            elif op == "const":
                vals[idx] = attrs["value"]
            elif op == "matmul_nt":
                vals[idx] = matmul_nt(a[0], a[1])
                if attrs.get("has_bias"):
                    vals[idx] = vals[idx] + a[2]
            elif op == "matmul_nt_gelu":
                bias = a[2] if len(a) > 2 else None
                vals[idx] = matmul_nt_gelu(a[0], a[1], bias)
            elif op == "gelu":
                vals[idx] = gelu(a[0])
            else:
                raise ValueError(f"op inconnue : {op}")
        return {i: vals[i] for i in vals}

    # ---- autodiff (reverse-mode, sur le graphe non fusionne) ----
    def forward(self, feeds):
        """Evalue tout le graphe (sans fusion) et renvoie toutes les valeurs."""
        vals: Dict[int, np.ndarray] = {}
        for idx, n in enumerate(self.nodes):
            a = [vals[i] for i in n.inputs]
            if n.op == "input":
                vals[idx] = np.asarray(feeds[n.attrs["name"]])
            elif n.op == "const":
                vals[idx] = n.attrs["value"]
            elif n.op == "matmul_nt":
                v = matmul_nt(a[0], a[1])
                if n.attrs.get("has_bias"):
                    v = v + a[2]
                vals[idx] = v
            elif n.op == "gelu":
                vals[idx] = gelu(a[0])
            else:
                raise ValueError(f"op inconnue : {n.op}")
        return vals

    def backward(self, out, feeds, dout, wrt):
        """Gradients de `out` (seed `dout`) par rapport aux noeuds `wrt`.

        Retourne {node_id: grad}. Gradients accumules si un noeud est reutilise.
        """
        vals = self.forward(feeds)
        grads: Dict[int, np.ndarray] = {out: np.asarray(dout, dtype=vals[out].dtype)}
        for idx in range(out, -1, -1):
            if idx not in grads:
                continue
            n = self.nodes[idx]
            g = grads[idx]
            if n.op in ("input", "const"):
                continue
            if n.op == "gelu":
                x = vals[n.inputs[0]]
                _acc(grads, n.inputs[0], gelu_backward(g, x))
            elif n.op == "matmul_nt":
                A, B = vals[n.inputs[0]], vals[n.inputs[1]]
                dA, dB = matmul_backward(g, A, B)
                _acc(grads, n.inputs[0], dA)
                _acc(grads, n.inputs[1], dB)
                if n.attrs.get("has_bias"):
                    _acc(grads, n.inputs[2], g.sum(axis=0))
        return {w: grads[w] for w in wrt if w in grads}

    def stats(self):
        plan, _ = self.fused_plan()
        ops = [v[0] for v in plan.values()]
        return {"nodes": len(self.nodes), "after_fusion": len(ops),
                "fused_matmul_gelu": ops.count("matmul_nt_gelu")}
