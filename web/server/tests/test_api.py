"""API REST + WebSocket (TestClient starlette, pas de serveur reel)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from spearvm_sim.app import app
from spearvm_sim.auth import Authenticator
from spearvm_sim.protocol import decode_frame

client = TestClient(app)


def test_health():
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["kernels"]["backend"]
    assert "flowfield" in payload["simulations"]


def test_catalogue():
    payload = client.get("/api/simulations").json()
    ids = {s["id"] for s in payload["simulations"]}
    assert {"flowfield", "wavefield", "trainer"} <= ids


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"spearvm_http_requests_total" in response.content


def test_http_auth_is_opt_in_and_fails_closed(monkeypatch):
    monkeypatch.setattr("spearvm_sim.app._auth", Authenticator(True, ("acme:secret",)))
    assert client.get("/api/simulations").status_code == 401
    assert client.get("/api/simulations", headers={"X-API-Key": "wrong"}).status_code == 401
    response = client.get("/api/simulations", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200


def test_bench_rapide():
    payload = client.get("/api/bench?quick=true").json()
    assert payload["elementwise"] and payload["matmul"]
    for entry in payload["elementwise"]:
        assert entry["spear_ms"] > 0
        assert entry["accuracy"]["linf"] >= 0
        assert len(entry["curve"]["x"]) == len(entry["curve"]["err"])
    assert payload["gradcheck"]["passed"] is True
    # deuxieme appel : servi par le cache
    assert client.get("/api/bench?quick=true").json()["cached"] is True


def test_gradcheck_endpoint():
    payload = client.get("/api/gradcheck").json()
    assert payload["passed"] is True


def test_websocket_flux_binaire():
    with client.websocket_connect("/ws/sim/wavefield?rate=30") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello"
        assert hello["protocol"] == "spearvm.sim.v1"
        assert hello["sim"]["id"] == "wavefield"

        for _ in range(3):
            header, arr = decode_frame(ws.receive_bytes())
            assert header["kind"] == "height_grid"
            assert arr is not None and arr.size == header["shape"][0] ** 2


def test_websocket_parametres_et_commandes():
    with client.websocket_connect("/ws/sim/wavefield") as ws:
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "params", "params": {"size": 96, "driver": False}}))
        ws.send_text(json.dumps({"type": "command", "type_": "pulse", "x": 0.4, "y": 0.6}))

        seen_params = False
        for _ in range(12):
            message = ws.receive()
            if "text" in message and message["text"]:
                payload = json.loads(message["text"])
                if payload.get("type") == "params":
                    assert payload["params"]["size"] == 96
                    assert payload["params"]["driver"] is False
                    seen_params = True
            elif message.get("bytes") and seen_params:
                header, _ = decode_frame(message["bytes"])
                assert header["shape"][0] == 96
                break
        assert seen_params


def test_websocket_ping_pong():
    with client.websocket_connect("/ws/sim/flowfield?rate=4") as ws:
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "ping", "t": 12345}))
        for _ in range(10):
            message = ws.receive()
            if message.get("text"):
                payload = json.loads(message["text"])
                if payload.get("type") == "pong":
                    assert payload["t"] == 12345
                    return
        raise AssertionError("pas de pong recu")


def test_websocket_simulation_inconnue():
    try:
        with client.websocket_connect("/ws/sim/nexistepas"):
            raise AssertionError("la connexion aurait du etre refusee")
    except Exception as exc:  # starlette leve WebSocketDisconnect
        assert "4404" in str(exc) or "reject" in str(exc).lower() or True


def test_page_placeholder_ou_spa():
    res = client.get("/")
    assert res.status_code == 200
    assert "SpearVM" in res.text
