"""Small, dependency-free API-key authentication boundary.

Authentication is disabled by default for local demos. Production deployments
must set SPEARVM_AUTH_REQUIRED=1 and SPEARVM_API_KEYS to comma-separated
``tenant:key`` pairs. Keys are compared in constant time and never returned.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    tenant_id: str


class Authenticator:
    def __init__(self, required: bool, entries: tuple[str, ...]) -> None:
        self.required = required
        self._keys = self._parse(entries)

    @staticmethod
    def _parse(entries: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
        parsed: list[tuple[str, str]] = []
        for entry in entries:
            tenant, separator, key = entry.partition(":")
            if separator and tenant and key:
                parsed.append((tenant.strip(), key.strip()))
        return tuple(parsed)

    def authenticate(self, key: str | None) -> Principal | None:
        if not self.required and not key:
            return Principal("public")
        if not key:
            return None
        for tenant, expected in self._keys:
            if hmac.compare_digest(key, expected):
                return Principal(tenant)
        return None

    def require(self, key: str | None) -> Principal:
        principal = self.authenticate(key)
        if principal is None:
            raise PermissionError("authentication required")
        return principal
