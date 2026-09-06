"""Authentication primitives: fail closed and tenant identity are stable."""

from spearvm_sim.auth import (
    Authenticator,
    MemoryKeyStore,
    MemoryRateLimiter,
    hash_api_key,
    verify_api_key,
)


def test_auth_disabled_keeps_local_demo_usable():
    principal = Authenticator(False, ()).require(None)
    assert principal.tenant_id == "public"


def test_auth_required_fails_closed_without_a_key():
    auth = Authenticator(True, ("acme:secret",))
    assert auth.authenticate(None) is None
    assert auth.authenticate("wrong") is None
    assert auth.authenticate("secret").tenant_id == "acme"


def test_auth_returns_only_tenant_identity():
    auth = Authenticator(True, ("acme:secret", "beta:other"))
    principal = auth.require("secret")
    assert principal.tenant_id == "acme"
    assert not hasattr(principal, "key")


def test_key_hash_and_rotation_revoke():
    digest = hash_api_key("secret")
    assert "secret" not in digest
    assert verify_api_key("secret", digest)
    assert not verify_api_key("wrong", digest)

    store = MemoryKeyStore()
    key = store.create("acme", "secret")
    assert len(store.list_active()) == 1
    store.revoke(key.key_id)
    assert store.list_active() == []


def test_memory_rate_limit_is_per_tenant():
    limiter = MemoryRateLimiter()
    assert limiter.allow("a", 1)
    assert not limiter.allow("a", 1)
    assert limiter.allow("b", 1)
