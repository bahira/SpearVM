"""Prometheus metrics kept behind a tiny module boundary."""

from prometheus_client import Counter, Gauge, Histogram, generate_latest

REQUESTS = Counter("spearvm_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("spearvm_http_request_duration_seconds", "HTTP latency", ["path"])
ACTIVE_CLIENTS = Gauge("spearvm_websocket_clients", "Active WebSocket clients")
WS_MESSAGES = Counter("spearvm_websocket_messages_total", "WebSocket messages", ["tenant"])


def render() -> bytes:
    return generate_latest()
