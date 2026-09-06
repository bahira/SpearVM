"""Serveur FastAPI : REST (capacites, bench) + WebSocket (flux de simulation).

Une connexion WebSocket = une instance de simulation dediee, cadencee par le
serveur, qui pousse des frames binaires. Le client peut a tout moment envoyer
des parametres, une cadence ou une commande (impulsion, reseed...).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import bench as bench_mod
from .auth import AuthService, MemoryRateLimiter, Principal, RedisRateLimiter, build_auth_service
from .config import settings
from .kernels import get_kernels
from .metrics import ACTIVE_CLIENTS, LATENCY, REQUESTS, WS_MESSAGES, render as render_metrics
from .sims import REGISTRY, create, describe_all

log = logging.getLogger("spearvm.app")

START_TIME = time.time()
_clients = {"count": 0}
_tenant_clients: dict[str, int] = {}
_bench_cache: dict[str, Any] = {"report": None, "at": 0.0}
BENCH_TTL_S = 30.0
_auth: AuthService = build_auth_service(
    settings.auth_required, settings.api_keys, settings.database_url
)
try:
    _rate_limiter = RedisRateLimiter(settings.redis_url) if settings.redis_url else MemoryRateLimiter()
except Exception:
    _rate_limiter = MemoryRateLimiter()


def _api_key_from_request(request: Request) -> str | None:
    value = request.headers.get("x-api-key")
    if value:
        return value
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    return token if scheme.lower() == "bearer" and token else None


def _check_rate(principal: Principal) -> None:
    if not _rate_limiter.allow(principal.tenant_id, settings.rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


def require_http_auth(request: Request) -> Principal:
    try:
        principal = _auth.require(_api_key_from_request(request))
        _check_rate(principal)
        return principal
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail="authentication required") from exc


def require_admin(principal: Principal = Depends(require_http_auth)) -> Principal:
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return principal


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    k = get_kernels()
    log.info("SpearVM simulation server — backend=%s native=%s", k.name, k.native)
    log.info("simulations: %s", ", ".join(REGISTRY))
    if settings.serve_static:
        log.info("statique servi depuis %s", settings.static_dir)
    else:
        log.info("pas de build client (%s) — mode API seule", settings.static_dir)
    yield
    log.info("arret du serveur")


app = FastAPI(
    title="SpearVM Simulation Lab",
    version="1.0.0",
    description="Simulations Three.js temps reel adossees aux noyaux SIMD SpearVM.",
    lifespan=lifespan,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    LATENCY.labels(path).observe(time.perf_counter() - started)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cache-Control", "no-store" if request.url.path.startswith("/api/") else "public, max-age=3600")
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness endpoint: the process is alive even in NumPy fallback mode."""
    k = get_kernels()
    return {
        "status": "ok",
        "ready": True,
        "uptime_s": round(time.time() - START_TIME, 1),
        "clients": _clients["count"],
        "max_clients": settings.max_clients,
        "kernels": k.capabilities(),
        "simulations": list(REGISTRY),
        "static": settings.serve_static,
    }


@app.get("/api/ready")
def ready() -> JSONResponse:
    """Readiness endpoint suitable for load balancers and container probes."""
    try:
        backend = get_kernels()
        return JSONResponse({"status": "ready", "backend": backend.capabilities()})
    except Exception as exc:  # noqa: BLE001 - probes must return a useful 503
        log.exception("readiness check failed")
        return JSONResponse({"status": "not_ready", "error": str(exc)}, status_code=503)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(render_metrics(), media_type="text/plain; version=0.0.4")


@app.get("/api/simulations")
def simulations(_: Principal = Depends(require_http_auth)) -> dict[str, Any]:
    k = get_kernels()
    return {"backend": k.capabilities(), "simulations": describe_all()}


@app.get("/api/bench")
async def get_bench(
    quick: bool = Query(default=False, description="Version courte (~200 ms)"),
    refresh: bool = Query(default=False, description="Ignore le cache"),
    _: Principal = Depends(require_http_auth),
) -> JSONResponse:
    now = time.time()
    cached = _bench_cache["report"]
    if cached and not refresh and (now - _bench_cache["at"]) < BENCH_TTL_S and cached.get("quick") == quick:
        return JSONResponse({**cached, "cached": True})
    report = await asyncio.to_thread(bench_mod.full_report, quick)
    _bench_cache["report"] = report
    _bench_cache["at"] = now
    return JSONResponse({**report, "cached": False})


@app.get("/api/gradcheck")
async def gradcheck(_: Principal = Depends(require_http_auth)) -> dict[str, Any]:
    return await asyncio.to_thread(bench_mod.gradcheck_report)


@app.get("/api/admin/keys")
def list_api_keys(_: Principal = Depends(require_admin)) -> list[dict[str, Any]]:
    return [{"key_id": key.key_id, "tenant_id": key.tenant_id, "role": key.role}
            for key in _auth.list_keys()]


@app.post("/api/admin/keys")
def create_api_key(payload: dict[str, Any] = Body(...), _: Principal = Depends(require_admin)) -> dict[str, str]:
    tenant_id = str(payload.get("tenant_id", "")).strip()
    role = str(payload.get("role", "viewer"))
    if not tenant_id or role not in {"admin", "viewer"}:
        raise HTTPException(status_code=400, detail="tenant_id and role=admin|viewer are required")
    secret = secrets.token_urlsafe(32)
    key = _auth.create_key(tenant_id, secret, role)
    return {"key_id": key.key_id, "tenant_id": key.tenant_id, "role": key.role, "api_key": secret}


@app.delete("/api/admin/keys/{key_id}", status_code=204)
def revoke_api_key(key_id: str, _: Principal = Depends(require_admin)) -> None:
    _auth.revoke_key(key_id)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
def _parse_params(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _clamp_rate(value: Any, fallback: float) -> float:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(settings.min_rate, min(settings.max_rate, rate))


@app.websocket("/ws/sim/{sim_id}")
async def sim_socket(websocket: WebSocket, sim_id: str) -> None:
    if sim_id not in REGISTRY:
        await websocket.close(code=4404, reason=f"simulation inconnue: {sim_id}")
        return

    api_key = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
    principal = _auth.authenticate(api_key)
    if principal is None:
        await websocket.close(code=4401, reason="authentication required")
        return
    tenant_id = principal.tenant_id
    if not _rate_limiter.allow(tenant_id, settings.rate_limit_per_minute):
        await websocket.close(code=4429, reason="rate limit exceeded")
        return
    if _clients["count"] >= settings.max_clients:
        await websocket.close(code=4429, reason="trop de clients simultanes")
        return
    if _tenant_clients.get(tenant_id, 0) >= settings.max_clients_per_tenant:
        await websocket.close(code=4429, reason="limite client du tenant atteinte")
        return

    await websocket.accept()
    _clients["count"] += 1
    ACTIVE_CLIENTS.set(_clients["count"])
    _tenant_clients[tenant_id] = _tenant_clients.get(tenant_id, 0) + 1
    params = _parse_params(websocket.query_params.get("params"))
    sim = create(sim_id, params)
    rate = _clamp_rate(websocket.query_params.get("rate"), sim.default_rate)
    log.info("ws open sim=%s rate=%.1f clients=%d", sim_id, rate, _clients["count"])

    stop = asyncio.Event()
    lock = asyncio.Lock()  # protege l'etat de sim entre commandes et pas de calcul

    async def reader() -> None:
        """Boucle de reception : parametres, cadence, commandes."""
        nonlocal rate
        try:
            while not stop.is_set():
                raw = await websocket.receive_text()
                WS_MESSAGES.labels(tenant_id).inc()
                if not _rate_limiter.allow(tenant_id, settings.rate_limit_per_minute):
                    await websocket.close(code=4429, reason="rate limit exceeded")
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                async with lock:
                    if mtype == "params":
                        sim.reconfigure(msg.get("params") or {})
                        await _send_json(websocket, {
                            "type": "params",
                            "params": sim.params,
                            "meta": sim.metadata(),
                        })
                        for frame in sim.initial_frames():
                            await websocket.send_bytes(sim.encode(frame))
                    elif mtype == "rate":
                        rate = _clamp_rate(msg.get("value"), rate)
                    elif mtype == "command":
                        sim.command(msg)
                    elif mtype == "ping":
                        await _send_json(websocket, {"type": "pong", "t": msg.get("t")})
        except Exception as exc:  # noqa: BLE001 - deconnexion client = fin normale
            if not isinstance(exc, (WebSocketDisconnect, RuntimeError)):
                log.debug("lecture ws terminee: %s", exc)
        finally:
            stop.set()

    async def writer() -> None:
        last = time.perf_counter()
        deadline = last
        try:
            while not stop.is_set():
                period = 1.0 / rate
                now = time.perf_counter()
                if deadline < now - 0.5:  # on a pris trop de retard : on resynchronise
                    deadline = now
                sleep_for = max(0.0, deadline - now)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=sleep_for)
                    break
                except asyncio.TimeoutError:
                    pass
                deadline += period

                dt = min(max(time.perf_counter() - last, 1e-4), 0.25)
                last = time.perf_counter()
                async with lock:
                    frame = await asyncio.to_thread(sim.step, dt)
                    blob = sim.encode(frame)
                await websocket.send_bytes(blob)
        except Exception as exc:  # noqa: BLE001 - idem cote emission
            if not isinstance(exc, (WebSocketDisconnect, RuntimeError, ConnectionError)):
                log.debug("emission ws terminee: %s", exc)
        finally:
            stop.set()

    try:
        await _send_json(websocket, {
            "type": "hello",
            "sim": sim.describe(),
            "params": sim.params,
            "meta": sim.metadata(),
            "rate": rate,
            "backend": get_kernels().capabilities(),
            "protocol": "spearvm.sim.v1",
        })
        for frame in sim.initial_frames():
            await websocket.send_bytes(sim.encode(frame))
        await asyncio.gather(reader(), writer())
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("erreur websocket sim=%s", sim_id)
    finally:
        stop.set()
        _clients["count"] = max(0, _clients["count"] - 1)
        ACTIVE_CLIENTS.set(_clients["count"])
        _tenant_clients[tenant_id] = max(0, _tenant_clients.get(tenant_id, 1) - 1)
        if _tenant_clients[tenant_id] == 0:
            del _tenant_clients[tenant_id]
        # le client peut avoir disparu : la fermeture est best-effort
        with contextlib.suppress(Exception):
            await websocket.close()
        log.info("ws close sim=%s clients=%d", sim_id, _clients["count"])


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload, separators=(",", ":")))


# ---------------------------------------------------------------------------
# Statique (build client) — monte en dernier pour ne pas masquer /api et /ws
# ---------------------------------------------------------------------------
_PLACEHOLDER = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>SpearVM Simulation Lab</title>
<style>body{background:#0b0f14;color:#dbe6f0;font:15px/1.6 ui-monospace,monospace;
padding:3rem;max-width:44rem;margin:auto}code{color:#7ee0c0}a{color:#7ecdf0}</style></head>
<body><h1>SpearVM Simulation Lab — API</h1>
<p>Le build client n'est pas present. Lancez le front en dev :</p>
<pre><code>cd web/client &amp;&amp; npm install &amp;&amp; npm run dev</code></pre>
<p>ou construisez-le puis relancez ce serveur :</p>
<pre><code>cd web/client &amp;&amp; npm run build</code></pre>
<p>Endpoints : <a href="/api/health">/api/health</a> ·
<a href="/api/simulations">/api/simulations</a> ·
<a href="/api/bench?quick=true">/api/bench?quick=true</a> ·
<a href="/docs">/docs</a></p></body></html>"""


if settings.serve_static:
    app.mount("/assets", StaticFiles(directory=str(settings.static_dir / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):  # noqa: ANN201
        candidate = (settings.static_dir / full_path).resolve()
        try:
            inside = candidate.is_relative_to(settings.static_dir.resolve())
        except AttributeError:  # py<3.9
            inside = str(candidate).startswith(str(settings.static_dir.resolve()))
        if full_path and inside and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(settings.static_dir / "index.html")

else:

    @app.get("/", include_in_schema=False)
    def placeholder() -> HTMLResponse:
        return HTMLResponse(_PLACEHOLDER)

    @app.get("/{full_path:path}", include_in_schema=False)
    def placeholder_catch(full_path: str) -> HTMLResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="endpoint inconnu")
        return HTMLResponse(_PLACEHOLDER)
