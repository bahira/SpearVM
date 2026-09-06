"""Authentication, API-key lifecycle, and tenant rate limiting.

The HTTP/WebSocket layer depends on this module instead of knowing whether the
staging/production deployment uses memory, PostgreSQL, or Redis. Local demos
remain zero-dependency; production selects adapters with environment variables.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    key_id: str = "local"
    role: str = "admin"


@dataclass(frozen=True)
class ApiKey:
    key_id: str
    tenant_id: str
    digest: str
    revoked_at: float | None = None
    role: str = "viewer"


def hash_api_key(secret: str, salt: str | None = None) -> str:
    """Hash a secret without storing the bearer token.

    PBKDF2-HMAC-SHA256 is available in the Python standard library and keeps
    the service easy to deploy. A random salt makes identical keys distinct.
    """
    salt = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt.encode(), 210_000)
    return f"pbkdf2_sha256$210000${salt}${derived.hex()}"


def verify_api_key(secret: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt.encode(), int(rounds))
        return hmac.compare_digest(actual.hex(), expected)
    except (TypeError, ValueError):
        return False


class KeyStore(Protocol):
    def list_active(self) -> list[ApiKey]: ...
    def create(self, tenant_id: str, secret: str) -> ApiKey: ...
    def revoke(self, key_id: str) -> None: ...


class MemoryKeyStore:
    def __init__(self, entries: tuple[str, ...] = ()) -> None:
        self._keys: dict[str, ApiKey] = {}
        for entry in entries:
            tenant, separator, secret = entry.partition(":")
            if separator and tenant.strip() and secret:
                self.create(tenant.strip(), secret, "admin")

    def list_active(self) -> list[ApiKey]:
        return [key for key in self._keys.values() if key.revoked_at is None]

    def create(self, tenant_id: str, secret: str, role: str = "viewer") -> ApiKey:
        key = ApiKey(str(uuid.uuid4()), tenant_id, hash_api_key(secret), role=role)
        self._keys[key.key_id] = key
        return key

    def revoke(self, key_id: str) -> None:
        key = self._keys.get(key_id)
        if key:
            self._keys[key_id] = ApiKey(key.key_id, key.tenant_id, key.digest, time.time())


class PostgresKeyStore:
    """Small PostgreSQL adapter; schema creation is idempotent."""

    def __init__(self, url: str) -> None:
        import psycopg

        self._psycopg = psycopg
        self.url = url
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spearvm_api_keys (
                    key_id UUID PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    revoked_at TIMESTAMPTZ NULL,
                    role TEXT NOT NULL DEFAULT 'viewer'
                )
            """)
            conn.execute("ALTER TABLE spearvm_api_keys ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'viewer'")

    def _connect(self):
        return self._psycopg.connect(self.url)

    def list_active(self) -> list[ApiKey]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key_id, tenant_id, digest, EXTRACT(EPOCH FROM revoked_at), role "
                "FROM spearvm_api_keys WHERE revoked_at IS NULL"
            ).fetchall()
        return [ApiKey(str(row[0]), row[1], row[2], row[3], row[4]) for row in rows]

    def create(self, tenant_id: str, secret: str, role: str = "viewer") -> ApiKey:
        key = ApiKey(str(uuid.uuid4()), tenant_id, hash_api_key(secret), role=role)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO spearvm_api_keys (key_id, tenant_id, digest, role) VALUES (%s, %s, %s, %s)",
                (key.key_id, key.tenant_id, key.digest, key.role),
            )
        return key

    def revoke(self, key_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE spearvm_api_keys SET revoked_at = now() WHERE key_id = %s", (key_id,))


class AuthService:
    def __init__(self, required: bool, store: KeyStore) -> None:
        self.required = required
        self.store = store

    def authenticate(self, secret: str | None) -> Principal | None:
        if not self.required and not secret:
            return Principal("public")
        if not secret:
            return None
        for key in self.store.list_active():
            if verify_api_key(secret, key.digest):
                return Principal(key.tenant_id, key.key_id, key.role)
        return None

    def require(self, secret: str | None) -> Principal:
        principal = self.authenticate(secret)
        if principal is None:
            raise PermissionError("authentication required")
        return principal

    def list_keys(self) -> list[ApiKey]:
        return self.store.list_active()

    def create_key(self, tenant_id: str, secret: str, role: str = "viewer") -> ApiKey:
        return self.store.create(tenant_id, secret, role)

    def revoke_key(self, key_id: str) -> None:
        self.store.revoke(key_id)


class Authenticator:
    """Backward-compatible facade for tests and local integrations."""

    def __init__(self, required: bool, entries: tuple[str, ...]) -> None:
        self._service = AuthService(required, MemoryKeyStore(entries))

    def authenticate(self, secret: str | None) -> Principal | None:
        return self._service.authenticate(secret)

    def require(self, secret: str | None) -> Principal:
        return self._service.require(secret)


def build_auth_service(required: bool, api_keys: tuple[str, ...], database_url: str | None) -> AuthService:
    if database_url:
        try:
            return AuthService(required, PostgresKeyStore(database_url))
        except Exception:
            if required:
                raise
    return AuthService(required, MemoryKeyStore(api_keys))


class RateLimiter:
    def allow(self, tenant_id: str, limit: int, window_s: int = 60) -> bool:
        raise NotImplementedError


class MemoryRateLimiter(RateLimiter):
    def __init__(self) -> None:
        self._buckets: dict[tuple[str, int], int] = {}

    def allow(self, tenant_id: str, limit: int, window_s: int = 60) -> bool:
        bucket = int(time.time() // window_s)
        key = (tenant_id, bucket)
        self._buckets[key] = self._buckets.get(key, 0) + 1
        return self._buckets[key] <= limit


class RedisRateLimiter(RateLimiter):
    def __init__(self, url: str) -> None:
        import redis

        self.client = redis.Redis.from_url(url, decode_responses=True)

    def allow(self, tenant_id: str, limit: int, window_s: int = 60) -> bool:
        bucket = int(time.time() // window_s)
        key = f"spearvm:rate:{tenant_id}:{bucket}"
        count = self.client.incr(key)
        if count == 1:
            self.client.expire(key, window_s + 1)
        return count <= limit
