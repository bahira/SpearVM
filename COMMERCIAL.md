# Commercial distribution

SpearVM is now organized as a dual-license project:

- **Community edition:** `LICENSE` (MIT), including commercial use under the
  MIT terms.
- **Commercial edition:** `LICENSE-COMMERCIAL.md`, for customers requiring
  proprietary redistribution rights, support, indemnity, or an SLA.

The commercial document in this repository is a reference terms document, not
an executed contract. Do not market a commercial entitlement until the terms
have been reviewed by counsel and signed by the copyright holder and customer.

## Production use cases

The repository includes working, testable use cases:

1. **CPU inference and feature transforms:** `matmul_nt_gelu`, `exp`,
   `softmax`, and reusable output buffers.
2. **Transformer-style attention:** `attention_tile`, `attention_mha`,
   `KVCache`, and `QuantizedWeight`.
3. **Training workloads:** GELU and matmul backward passes, plus the live
   trainer in `web/server/spearvm_sim/sims/trainer.py`.
4. **Real-time simulation service:** FastAPI REST, binary WebSocket frames,
   rate limiting, health endpoints, and a Three.js client.

## Production checklist

Before deploying commercially:

```bash
# Native and Python checks
python -m pip install -r web/server/requirements-dev.txt
python -m pytest tests web/server/tests -q

# Frontend checks
cd web/client
npm ci
npm run build
```

The runtime requires an x86-64 CPU with AVX2/FMA for native kernels. The web
server has a NumPy fallback, and its actual backend is exposed by
`GET /api/health`.

Set these variables in production rather than relying on development defaults:

```text
SPEARVM_LOG_LEVEL=warning
SPEARVM_CORS=https://your-frontend.example
SPEARVM_MAX_CLIENTS=8
SPEARVM_MAX_RATE=60
SPEARVM_FORCE_FALLBACK=0
```

For a commercial deployment, place the service behind TLS termination and an
identity-aware reverse proxy. The project does not itself implement user
accounts, billing, license-key enforcement, or tenant isolation.
