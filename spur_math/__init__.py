import ctypes
import glob
import os
import platform
import shutil
import subprocess

import numpy as np

# les noyaux sont des intrinsics x86 AVX2/FMA : refus clair ailleurs
if platform.machine().lower() not in ("x86_64", "amd64"):
    raise ImportError(
        f"spur_math : architecture '{platform.machine()}' non supportee "
        "(noyaux AVX2 x86-64 uniquement ; Mac ARM/Apple Silicon non couvert)."
    )

_PKG = os.path.dirname(os.path.abspath(__file__))
_SRC_C = os.path.join(_PKG, "src", "spur_kernels.c")


def _try_compile():
    """Pas de binaire embarque (sdist Linux/macOS) : compilation unique en cache."""
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc or not os.path.exists(_SRC_C):
        return None
    cache = os.environ.get("SPUR_CACHE_DIR",
                           os.path.join(_PKG, "_native"))
    try:
        os.makedirs(cache, exist_ok=True)
    except OSError:
        import tempfile
        cache = os.path.join(tempfile.gettempdir(), "spur_math_native")
        os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, "libspur_kernels.so")
    if not os.path.exists(out):
        cmd = [cc, "-O3", "-mavx2", "-mfma", "-fopenmp", "-shared",
               "-o", out, _SRC_C]
        try:
            subprocess.check_call(cmd)
        except (subprocess.CalledProcessError, OSError):
            return None
    return out


_dll_path = os.path.join(_PKG, "spur_kernels.dll")
if not os.path.exists(_dll_path):
    cands = sorted(glob.glob(os.path.join(_PKG, "libspur_kernels*.so"))) or \
            sorted(glob.glob(os.path.join(_PKG, "_native",
                                          "libspur_kernels*.so")))
    if cands:
        _dll_path = cands[0]
    else:
        _compiled = _try_compile()
        if _compiled:
            _dll_path = _compiled
        else:
            raise ImportError(
                "spur_math : binaire introuvable et compilation impossible.\n"
                "- Windows : reinstallez depuis le wheel win_amd64\n"
                "- Linux/macOS : installez gcc/clang puis reessayez ; ou:\n"
                "  gcc -O3 -mavx2 -mfma -fopenmp -shared "
                "-o spur_math/libspur_kernels.so src/spur_kernels.c"
            )

_pkg_dir = os.path.dirname(os.path.abspath(_dll_path))
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_pkg_dir)  # runtimes MinGW a cote du .dll

_dll = ctypes.CDLL(_dll_path)

# refus propre (au lieu de SIGILL) sur CPU sans AVX2+FMA
_dll.spur_cpu_ok.restype = ctypes.c_int
if not _dll.spur_cpu_ok():
    raise ImportError(
        "spur_math : ce paquet requiert un CPU avec AVX2 et FMA "
        "(Intel Haswell 2013+, AMD Zen 2017+)."
    )


def _bind(name):
    f = getattr(_dll, name)
    f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                  ctypes.c_longlong]
    f.restype = None
    return f


_batch_gelu = _bind("spur_batch_gelu")
_batch_erf = _bind("spur_batch_erf")
_batch_tanh = _bind("spur_batch_tanh")
_batch_gelu_quintic = _bind("spur_batch_gelu_quintic")
_batch_gelu_erf = _bind("spur_batch_gelu_erf")


def _bind_mm(name, with_bias=False):
    f = getattr(_dll, name)
    if with_bias:
        f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                      ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong]
    else:
        f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                      ctypes.POINTER(ctypes.c_double),
                      ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong]
    f.restype = None
    return f


def _bind_bw(name):
    f = getattr(_dll, name)
    f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                  ctypes.POINTER(ctypes.c_double), ctypes.c_longlong]
    f.restype = None
    return f


_matmul_nt = _bind_mm("spur_matmul_nt")
_matmul_nt_gelu = _bind_mm("spur_matmul_nt_gelu", with_bias=True)
_gelu_backward = _bind_bw("spur_batch_gelu_backward")

_PF = ctypes.POINTER(ctypes.c_float)
_batch_gelu_f32 = getattr(_dll, "spur_batch_gelu_f32")
_batch_gelu_f32.argtypes = [_PF, _PF, ctypes.c_longlong]
_batch_gelu_f32.restype = None

_matmul_nt_f32 = getattr(_dll, "spur_matmul_nt_f32")
_matmul_nt_f32.argtypes = [_PF, _PF, _PF] + [ctypes.c_longlong] * 3
_matmul_nt_f32.restype = None

_matmul_nt_gelu_f32 = getattr(_dll, "spur_matmul_nt_gelu_f32")
_matmul_nt_gelu_f32.argtypes = [_PF, _PF, ctypes.c_void_p, _PF] + [
    ctypes.c_longlong] * 3
_matmul_nt_gelu_f32.restype = None

_gelu_backward_f32 = getattr(_dll, "spur_batch_gelu_backward_f32")
_gelu_backward_f32.argtypes = [_PF, _PF, _PF, ctypes.c_longlong]
_gelu_backward_f32.restype = None

for _n in ("spur_batch_erf_backward", "spur_batch_tanh_backward",
           "spur_batch_sigmoid_backward"):
    _f = getattr(_dll, _n)
    _f.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3 + [ctypes.c_longlong]
    _f.restype = None
_erf_backward = _dll.spur_batch_erf_backward
_tanh_backward = _dll.spur_batch_tanh_backward
_sigmoid_backward = _dll.spur_batch_sigmoid_backward


def _dest(m, n, dt, out):
    """Tampon de sortie : fourni par l'appelant, sinon alloue *sans* mise a zero.

    Les noyaux ecrivent chaque case de C (verifie par tests/test_matmul_v2.py :
    C pre-rempli de NaN, aucun NaN ne survit), donc `np.empty` suffit — cela
    evite un memset de m*n elements a chaque appel. Le parametre `out=` permet
    en plus de reutiliser un tampon dans une boucle d'inference : sur une
    couche (64000, 64) l'allocation seule coutait plus cher que le calcul.
    """
    if out is None:
        return np.empty((m, n), dtype=dt)
    if out.shape != (m, n):
        raise ValueError(f"out attendu de forme {(m, n)}, recu {out.shape}")
    if out.dtype != dt:
        raise ValueError(f"out attendu en {np.dtype(dt).name}, recu {out.dtype.name}")
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("out doit etre C-contigu")
    return out


# --- exp et softmax (noyaux AVX2 minimax, cf. docs/TRANSCEND.md) ------------
_PD_ = ctypes.POINTER(ctypes.c_double)
_PF_ = ctypes.POINTER(ctypes.c_float)
_exp_f64 = _dll.spur_batch_exp
_exp_f64.argtypes = [_PD_, _PD_, ctypes.c_longlong]
_exp_f64.restype = None
_exp_f32 = _dll.spur_batch_exp_f32
_exp_f32.argtypes = [_PF_, _PF_, ctypes.c_longlong]
_exp_f32.restype = None
_softmax_f64 = _dll.spur_softmax_rows
_softmax_f64.argtypes = [_PD_, _PD_, ctypes.c_longlong, ctypes.c_longlong,
                         ctypes.POINTER(ctypes.c_int)]
_softmax_f64.restype = None
_softmax_f32 = _dll.spur_softmax_rows_f32
_softmax_f32.argtypes = [_PF_, _PF_, ctypes.c_longlong, ctypes.c_longlong,
                         ctypes.POINTER(ctypes.c_int)]
_softmax_f32.restype = None


def exp(x, out=None):
    """exp(x) vectorise AVX2. Precision mesuree : 1.69 ulp (f32), 2.12 ulp (f64)."""
    dt = np.float32 if np.asarray(x).dtype == np.float32 else np.float64
    x = np.ascontiguousarray(x, dtype=dt)
    y = np.empty_like(x) if out is None else out
    if y.shape != x.shape or y.dtype != dt or not y.flags["C_CONTIGUOUS"]:
        raise ValueError("out doit etre C-contigu, de meme forme et meme dtype")
    ptr = _PF_ if dt == np.float32 else _PD_
    fn = _exp_f32 if dt == np.float32 else _exp_f64
    fn(x.ctypes.data_as(ptr), y.ctypes.data_as(ptr), x.size)
    return y


def softmax(x, lengths=None, out=None):
    """Softmax par ligne sur un tableau 2D.

    `lengths` : (rows,) entiers, nombre d'entrees valides par ligne (le reste
    est mis a zero). C'est la forme causale utile a l'attention : aucun -inf a
    materialiser. Mesure : x11.8 vs numpy vectorise sur un masque triangulaire.
    """
    dt = np.float32 if np.asarray(x).dtype == np.float32 else np.float64
    x = np.ascontiguousarray(x, dtype=dt)
    if x.ndim != 2:
        raise ValueError(f"softmax attend un tableau 2D, recu {x.ndim}D")
    rows, cols = x.shape
    y = np.empty_like(x) if out is None else out
    if y.shape != x.shape or y.dtype != dt or not y.flags["C_CONTIGUOUS"]:
        raise ValueError("out doit etre C-contigu, de meme forme et meme dtype")
    lp = None
    if lengths is not None:
        lengths = np.ascontiguousarray(lengths, dtype=np.int32)
        if lengths.shape != (rows,):
            raise ValueError(f"lengths attendu ({rows},), recu {lengths.shape}")
        lp = lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    ptr = _PF_ if dt == np.float32 else _PD_
    fn = _softmax_f32 if dt == np.float32 else _softmax_f64
    fn(x.ctypes.data_as(ptr), y.ctypes.data_as(ptr), rows, cols, lp)
    return y


_attention_tile = _dll.spur_attention_tile_f32
_attention_tile.argtypes = [_PF_, _PF_, _PF_, _PF_, ctypes.c_longlong,
                            ctypes.c_longlong, ctypes.c_longlong, ctypes.c_float,
                            ctypes.POINTER(ctypes.c_int), _PF_]
_attention_tile.restype = None


def attention_tile(q, k, v, scale=None, lengths=None, out=None, scratch=None):
    """O = softmax(scale . Q.K^T, masque causal) . V  — float32.

    q:(tq,d) k:(tk,d) v:(tk,d). `lengths` (tq,) = nombre de cles visibles par
    requete (None = toutes). `scale` par defaut 1/sqrt(d).

    Les deux produits sont en convention NT (celle des noyaux GEMM v2) et
    l'echelle est absorbee par le softmax : aucune passe supplementaire sur la
    matrice de scores, aucun masque -inf a materialiser.
    """
    q = np.ascontiguousarray(q, dtype=np.float32)
    k = np.ascontiguousarray(k, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    tq, d = q.shape
    tk = k.shape[0]
    if k.shape[1] != d or v.shape != (tk, d):
        raise ValueError(f"formes incompatibles : q{q.shape} k{k.shape} v{v.shape}")
    vt = np.ascontiguousarray(v.T)            # V transpose -> second GEMM en NT
    o = np.empty((tq, d), dtype=np.float32) if out is None else out
    if o.shape != (tq, d) or o.dtype != np.float32 or not o.flags["C_CONTIGUOUS"]:
        raise ValueError("out doit etre (tq,d) float32 C-contigu")
    buf = np.empty((tq, tk), dtype=np.float32) if scratch is None else scratch
    if buf.shape != (tq, tk) or buf.dtype != np.float32:
        raise ValueError("scratch doit etre (tq,tk) float32")
    lp = None
    if lengths is not None:
        lengths = np.ascontiguousarray(lengths, dtype=np.int32)
        if lengths.shape != (tq,):
            raise ValueError(f"lengths attendu ({tq},), recu {lengths.shape}")
        lp = lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    _attention_tile(q.ctypes.data_as(_PF_), k.ctypes.data_as(_PF_),
                    vt.ctypes.data_as(_PF_), o.ctypes.data_as(_PF_),
                    tq, tk, d, float(1.0 / np.sqrt(d)) if scale is None else float(scale),
                    lp, buf.ctypes.data_as(_PF_))
    return o


_attention_mha = _dll.spur_attention_mha_f32
_attention_mha.argtypes = [_PF_, _PF_, _PF_, _PF_, ctypes.c_longlong,
                           ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
                           ctypes.c_int, ctypes.c_float,
                           ctypes.POINTER(ctypes.c_int)]
_attention_mha.restype = None


_kv_pack_size = _dll.spur_kv_pack_size_f32
_kv_pack_size.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
_kv_pack_size.restype = ctypes.c_longlong
_kv_pack = _dll.spur_kv_pack_f32
_kv_pack.argtypes = [_PF_, _PF_, ctypes.c_longlong, ctypes.c_longlong,
                     ctypes.c_int, _PF_]
_kv_pack.restype = None
_attention_mha_packed = _dll.spur_attention_mha_packed_f32
_attention_mha_packed.argtypes = [_PF_, _PF_, _PF_, ctypes.c_longlong,
                                  ctypes.c_longlong, ctypes.c_longlong,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                  ctypes.POINTER(ctypes.c_int)]
_attention_mha_packed.restype = None


_U16 = ctypes.POINTER(ctypes.c_uint16)
_kv_pack_size_bf16 = _dll.spur_kv_pack_size_bf16
_kv_pack_size_bf16.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
_kv_pack_size_bf16.restype = ctypes.c_longlong
_kv_pack_bf16 = _dll.spur_kv_pack_bf16
_kv_pack_bf16.argtypes = [_PF_, _PF_, ctypes.c_longlong, ctypes.c_longlong,
                          ctypes.c_int, _U16]
_kv_pack_bf16.restype = None
_attention_mha_bf16 = _dll.spur_attention_mha_packed_bf16
_attention_mha_bf16.argtypes = [_PF_, _U16, _PF_, ctypes.c_longlong,
                                ctypes.c_longlong, ctypes.c_longlong,
                                ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                ctypes.POINTER(ctypes.c_int)]
_attention_mha_bf16.restype = None


_I8 = ctypes.POINTER(ctypes.c_int8)
_quant_i8 = _dll.spur_quant_i8_rows
_quant_i8.argtypes = [_PF_, _I8, _PF_, ctypes.c_longlong, ctypes.c_longlong]
_quant_i8.restype = None
_quant_bf16 = _dll.spur_quant_bf16_rows
_quant_bf16.argtypes = [_PF_, _U16, ctypes.c_longlong, ctypes.c_longlong]
_quant_bf16.restype = None
_gemm_i8 = _dll.spur_gemm_nt_i8b_f32
_gemm_i8.argtypes = [_PF_, _I8, _PF_, _PF_, ctypes.c_longlong, ctypes.c_longlong,
                     ctypes.c_longlong]
_gemm_i8.restype = None
_gemm_bf16w = _dll.spur_gemm_nt_bf16w_f32
_gemm_bf16w.argtypes = [_PF_, _U16, _PF_, ctypes.c_longlong, ctypes.c_longlong,
                        ctypes.c_longlong]
_gemm_bf16w.restype = None


class QuantizedWeight:
    """Matrice de poids (n, k) stockee en bf16 ou int8, produit en NT.

    A m petit (decodage), un GEMM ne fait que 2.m flops par poids lu : le temps
    est decide par le nombre d'octets par poids, pas par le calcul. Reduire le
    format est donc le seul levier reel.

        w8 = sm.QuantizedWeight(w, dtype="i8")   # 4 octets -> 1
        y = w8.matmul(x)                         # y = x . w^T

    int8 : quantification symetrique **par ligne de sortie** (une echelle par
    neurone, ce qui suit sa dynamique propre). bf16 : troncature exacte du
    float32, sans echelle.
    """

    __slots__ = ("n", "k", "dtype", "_q", "_scales")

    def __init__(self, w, dtype="i8"):
        w = np.ascontiguousarray(w, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(f"poids attendus 2D (n, k), recus {w.shape}")
        if dtype not in ("bf16", "i8"):
            raise ValueError(f"dtype attendu 'bf16' ou 'i8', recu {dtype!r}")
        self.n, self.k = w.shape
        self.dtype = dtype
        if dtype == "i8":
            self._q = np.empty((self.n, self.k), dtype=np.int8)
            self._scales = np.empty(self.n, dtype=np.float32)
            _quant_i8(w.ctypes.data_as(_PF_), self._q.ctypes.data_as(_I8),
                      self._scales.ctypes.data_as(_PF_), self.n, self.k)
        else:
            self._q = np.empty((self.n, self.k), dtype=np.uint16)
            self._scales = None
            _quant_bf16(w.ctypes.data_as(_PF_), self._q.ctypes.data_as(_U16),
                        self.n, self.k)

    @property
    def nbytes(self):
        return self._q.nbytes + (self._scales.nbytes if self._scales is not None else 0)

    def dequantize(self):
        """Reconstruit les poids float32 (pour mesurer l'erreur de quantification)."""
        if self.dtype == "i8":
            return self._q.astype(np.float32) * self._scales[:, None]
        return (self._q.astype(np.uint32) << 16).view(np.float32)

    def matmul(self, a, out=None):
        """y = a . w^T. a:(m,k) float32 -> (m,n)."""
        a = np.ascontiguousarray(a, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != self.k:
            raise ValueError(f"a attendu (m, {self.k}), recu {a.shape}")
        m = a.shape[0]
        y = np.empty((m, self.n), dtype=np.float32) if out is None else out
        if y.shape != (m, self.n) or y.dtype != np.float32 or not y.flags["C_CONTIGUOUS"]:
            raise ValueError("out doit etre (m,n) float32 C-contigu")
        if self.dtype == "i8":
            _gemm_i8(a.ctypes.data_as(_PF_), self._q.ctypes.data_as(_I8),
                     self._scales.ctypes.data_as(_PF_), y.ctypes.data_as(_PF_),
                     m, self.k, self.n)
        else:
            _gemm_bf16w(a.ctypes.data_as(_PF_), self._q.ctypes.data_as(_U16),
                        y.ctypes.data_as(_PF_), m, self.k, self.n)
        return y


_kv_index_size = _dll.spur_kv_index_size_f32
_kv_index_size.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int]
_kv_index_size.restype = ctypes.c_longlong
_kv_index_build = _dll.spur_kv_index_build_f32
_kv_index_build.argtypes = [_PF_, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int, _PF_]
_kv_index_build.restype = None
_attention_sparse = _dll.spur_attention_sparse_f32
_attention_sparse.argtypes = [_PF_, _PF_, _PF_, _PF_, ctypes.c_longlong,
                              ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int,
                              ctypes.c_int, ctypes.c_float, ctypes.c_longlong,
                              ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int]
_attention_sparse.restype = ctypes.c_longlong


class KVCache:
    """Cache K/V packe une fois, reutilisable par des dizaines de tuiles.

    Pour tq petit (decodage, micro-blocs QSA), le packing est en O(tk.d.H_kv)
    alors que le calcul n'est qu'en O(tq.tk.d.H_q) : il domine l'appel. Le
    packer une fois transforme le regime.

        cache = sm.KVCache(k, v)          # k,v : (tk, H_kv, d) float32
        out = cache.attend(q, lengths=L)  # q : (tq, H_q, d)
    """

    __slots__ = ("tk", "d", "h_kv", "dtype", "_buf", "_index", "_bs", "_f")

    def __init__(self, k, v, dtype="f32"):
        """`dtype` : "f32" ou "bf16" (empreinte divisee par deux)."""
        k = np.ascontiguousarray(k, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        if k.ndim != 3 or v.shape != k.shape:
            raise ValueError(f"k et v doivent etre (tk, H_kv, d) identiques : "
                             f"{k.shape} vs {v.shape}")
        if dtype not in ("f32", "bf16"):
            raise ValueError(f"dtype attendu 'f32' ou 'bf16', recu {dtype!r}")
        self.tk, self.h_kv, self.d = k.shape
        self.dtype = dtype
        self._index = None
        self._bs = 0
        self._f = 0
        if dtype == "bf16":
            n = int(_kv_pack_size_bf16(self.tk, self.d, self.h_kv))
            self._buf = np.empty(n, dtype=np.uint16)
            _kv_pack_bf16(k.ctypes.data_as(_PF_), v.ctypes.data_as(_PF_),
                          self.tk, self.d, self.h_kv, self._buf.ctypes.data_as(_U16))
        else:
            n = int(_kv_pack_size(self.tk, self.d, self.h_kv))
            self._buf = np.empty(n, dtype=np.float32)
            _kv_pack(k.ctypes.data_as(_PF_), v.ctypes.data_as(_PF_),
                     self.tk, self.d, self.h_kv, self._buf.ctypes.data_as(_PF_))

    @property
    def nbytes(self):
        return self._buf.nbytes

    def build_index(self, block=8, factor=8):
        """Construit l'arbre de resumes qui rend le routage sous-quadratique.

        `block` : tokens par bloc (granularite de selection) ;
        `factor` : arite de l'arbre. Cout : ~1/block du cache en memoire.
        """
        if self.dtype != "f32":
            raise ValueError("l'index n'est disponible que pour un cache f32")
        n = int(_kv_index_size(self.tk, self.d, self.h_kv, block, factor))
        self._index = np.empty(n, dtype=np.float32)
        self._bs, self._f = int(block), int(factor)
        _kv_index_build(self._buf.ctypes.data_as(_PF_), self.tk, self.d,
                        self.h_kv, self._bs, self._f,
                        self._index.ctypes.data_as(_PF_))
        return self

    def attend_sparse(self, q, pos, budget=512, beam=0, scale=None, out=None):
        """Attention creuse : seuls les blocs les mieux notes sont lus.

        `pos` : position absolue de la premiere requete de la tuile. Les blocs
        entierement anterieurs a la tuile sont candidats a la selection ; la
        queue locale (du debut du bloc courant a la derniere requete) est
        toujours incluse, et la causalite est portee par les longueurs.

        Renvoie (sortie, tokens_lus).
        """
        if getattr(self, "_index", None) is None:
            raise ValueError("appeler build_index() avant attend_sparse()")
        q = np.ascontiguousarray(q, dtype=np.float32)
        if q.ndim != 3 or q.shape[2] != self.d:
            raise ValueError(f"q attendu (tq, H_q, {self.d}), recu {q.shape}")
        tq, h_q, d = q.shape
        if h_q % self.h_kv:
            raise ValueError(f"H_q={h_q} n'est pas un multiple de H_kv={self.h_kv}")
        o = np.empty((tq, h_q, d), dtype=np.float32) if out is None else out
        if o.shape != (tq, h_q, d) or o.dtype != np.float32 or not o.flags["C_CONTIGUOUS"]:
            raise ValueError("out doit etre (tq,H_q,d) float32 C-contigu")
        if beam <= 0:
            beam = max(-(-(budget // self._bs) // self._f), 8)
        got = _attention_sparse(q.ctypes.data_as(_PF_), self._buf.ctypes.data_as(_PF_),
                                self._index.ctypes.data_as(_PF_), o.ctypes.data_as(_PF_),
                                tq, self.tk, d, h_q, self.h_kv,
                                float(1.0 / np.sqrt(d)) if scale is None else float(scale),
                                int(pos), int(budget), self._bs, self._f, int(beam))
        if got < 0:
            raise ValueError("parametres d'attention creuse invalides")
        return o, int(got)

    def attend(self, q, scale=None, lengths=None, out=None):
        q = np.ascontiguousarray(q, dtype=np.float32)
        if q.ndim != 3 or q.shape[2] != self.d:
            raise ValueError(f"q attendu (tq, H_q, {self.d}), recu {q.shape}")
        tq, h_q, d = q.shape
        if h_q % self.h_kv:
            raise ValueError(f"H_q={h_q} n'est pas un multiple de H_kv={self.h_kv}")
        o = np.empty((tq, h_q, d), dtype=np.float32) if out is None else out
        if o.shape != (tq, h_q, d) or o.dtype != np.float32 or not o.flags["C_CONTIGUOUS"]:
            raise ValueError("out doit etre (tq,H_q,d) float32 C-contigu")
        lp = None
        if lengths is not None:
            lengths = np.ascontiguousarray(lengths, dtype=np.int32)
            if lengths.shape != (tq,):
                raise ValueError(f"lengths attendu ({tq},), recu {lengths.shape}")
            lp = lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        sc = float(1.0 / np.sqrt(d)) if scale is None else float(scale)
        if self.dtype == "bf16":
            _attention_mha_bf16(q.ctypes.data_as(_PF_), self._buf.ctypes.data_as(_U16),
                                o.ctypes.data_as(_PF_), tq, self.tk, d, h_q,
                                self.h_kv, sc, lp)
        else:
            _attention_mha_packed(q.ctypes.data_as(_PF_), self._buf.ctypes.data_as(_PF_),
                                  o.ctypes.data_as(_PF_), tq, self.tk, d, h_q,
                                  self.h_kv, sc, lp)
        return o


def attention_mha(q, k, v, scale=None, lengths=None, out=None):
    """Attention multi-tetes (GQA) en une seule descente C — float32.

    q:(tq, H_q, d)   k,v:(tk, H_kv, d)   -> (tq, H_q, d)
    H_q doit etre un multiple de H_kv (les tetes de requetes partagent alors une
    tete de cles, comme en GQA). `lengths` (tq,) porte le masque causal.

    Les tetes K/V ne sont packees qu'une fois par groupe, le parallelisme
    OpenMP porte sur les tetes de requetes, et rien ne repasse par Python entre
    les tetes : c'est ce qui sort les petites tuiles du regime ou le cout
    d'appel domine.
    """
    q = np.ascontiguousarray(q, dtype=np.float32)
    k = np.ascontiguousarray(k, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v doivent etre 3D (t, tetes, d)")
    tq, h_q, d = q.shape
    tk, h_kv, dk = k.shape
    if dk != d or v.shape != k.shape:
        raise ValueError(f"formes incompatibles : q{q.shape} k{k.shape} v{v.shape}")
    if h_q % h_kv:
        raise ValueError(f"H_q={h_q} n'est pas un multiple de H_kv={h_kv}")
    o = np.empty((tq, h_q, d), dtype=np.float32) if out is None else out
    if o.shape != (tq, h_q, d) or o.dtype != np.float32 or not o.flags["C_CONTIGUOUS"]:
        raise ValueError("out doit etre (tq,H_q,d) float32 C-contigu")
    lp = None
    if lengths is not None:
        lengths = np.ascontiguousarray(lengths, dtype=np.int32)
        if lengths.shape != (tq,):
            raise ValueError(f"lengths attendu ({tq},), recu {lengths.shape}")
        lp = lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    _attention_mha(q.ctypes.data_as(_PF_), k.ctypes.data_as(_PF_),
                   v.ctypes.data_as(_PF_), o.ctypes.data_as(_PF_),
                   tq, tk, d, h_q, h_kv,
                   float(1.0 / np.sqrt(d)) if scale is None else float(scale), lp)
    return o


def matmul_nt(a, b, out=None):
    """C = A . B^T. a:(m,k), b:(n,k) -> (m,n). float32 ou float64.

    `out` : tampon (m,n) reutilisable, meme dtype et C-contigu (optionnel).
    """
    dt = np.float32 if np.asarray(a).dtype == np.float32 else np.float64
    a = np.ascontiguousarray(a, dtype=dt)
    b = np.ascontiguousarray(b, dtype=dt)
    m, k = a.shape
    n = b.shape[0]
    c = _dest(m, n, dt, out)
    if dt == np.float32:
        _matmul_nt_f32(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       m, k, n)
    else:
        _matmul_nt(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   m, k, n)
    return c


def matmul_nt_gelu(a, b, bias=None, out=None):
    """C = gelu(A . B^T + bias). float32 ou float64. bias:(n,) ou None.

    `out` : tampon (m,n) reutilisable, meme dtype et C-contigu (optionnel).
    """
    dt = np.float32 if np.asarray(a).dtype == np.float32 else np.float64
    a = np.ascontiguousarray(a, dtype=dt)
    b = np.ascontiguousarray(b, dtype=dt)
    m, k = a.shape
    n = b.shape[0]
    c = _dest(m, n, dt, out)
    bp = None
    if bias is not None:
        bias = np.ascontiguousarray(bias, dtype=dt)
        assert bias.shape == (n,), f"bias attendu ({n},), recu {bias.shape}"
        bp = bias.ctypes.data_as(ctypes.c_void_p)
    if dt == np.float32:
        _matmul_nt_gelu_f32(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            bp,
                            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            m, k, n)
    else:
        _matmul_nt_gelu(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        bp,
                        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        m, k, n)
    return c


def gelu_backward(dY, x):
    """dX = dY * gelu'(x). float32 ou float64."""
    dt = np.float32 if np.asarray(x).dtype == np.float32 else np.float64
    dYc = np.ascontiguousarray(dY, dtype=dt)
    xc = np.ascontiguousarray(x, dtype=dt)
    out = np.zeros_like(xc)
    if dt == np.float32:
        _gelu_backward_f32(dYc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           xc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           xc.size)
    else:
        _gelu_backward(dYc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       xc.size)
    return out


def matmul_backward(dY, A, B):
    """Gradients de C=A.B^T : retourne (dA, dB).

    NT : out[i,j]=sum_k X[i,k]*Y[j,k] =>
      dA[m,p] = sum_n dY[m,n]*B[n,p]  -> matmul_nt(dY,   B.T)
      dB[n,p] = sum_m dY[m,n]*A[m,p]  -> matmul_nt(dY.T, A.T)
    """
    dY = np.asarray(dY)
    dt = np.float32 if dY.dtype == np.float32 else np.float64
    dA = matmul_nt(dY, np.ascontiguousarray(np.asarray(B, dtype=dt).T))
    dB = matmul_nt(np.ascontiguousarray(dY.T),
                   np.ascontiguousarray(np.asarray(A, dtype=dt).T))
    return dA, dB


def erf_backward(dY, x):
    """dX = dY * erf_approx'(x)."""
    dYc = np.ascontiguousarray(dY, dtype=np.float64)
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    _erf_backward(dYc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  xc.size)
    return out


def tanh_backward(dY, x):
    """dX = dY * tanh_approx'(x)."""
    dYc = np.ascontiguousarray(dY, dtype=np.float64)
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    _tanh_backward(dYc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   xc.size)
    return out


def sigmoid_backward(dY, x):
    """dX = dY * sigmoid_approx'(x)."""
    dYc = np.ascontiguousarray(dY, dtype=np.float64)
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    _sigmoid_backward(dYc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      xc.size)
    return out


def gelu(x):
    """GELU approximatif AVX2. Erreur max 0.079 sur [-2,2] (f32: +1e-6 bruit)."""
    if np.asarray(x).dtype == np.float32:
        xc = np.ascontiguousarray(x, dtype=np.float32)
        out = np.zeros_like(xc)
        _batch_gelu_f32(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        xc.size)
        return out
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    _batch_gelu(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), xc.size)
    return out


def gelu_quintic(x):
    """GELU v2 quintique smoothstep certifie. Linf 0.0174 sur [-3.5,3.5],
    MSE 1.35e-4 sur [-4,4], queue bornee sur R. f64 uniquement."""
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    _batch_gelu_quintic(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        xc.size)
    return out


def gelu_erf(x):
    """GELU haute precision via erf_v2 certifie : Linf 2.05e-5, MSE 8.3e-11.
    ~850x plus precis que gelu_quintic. f64 uniquement."""
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    _batch_gelu_erf(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    xc.size)
    return out


def erf(x):
    """erf approximatif AVX2. Erreur max 0.011 sur [-2, 2]."""
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    fn = _batch_erf
    fn(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), xc.size)
    return out


def tanh(x):
    """tanh approximatif AVX2. Erreur max 0.008 sur [-3, 3]."""
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    fn = _batch_tanh
    fn(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), xc.size)
    return out
