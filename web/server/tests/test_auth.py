"""Authentication primitives: fail closed and tenant identity are stable."""

from spearvm_sim.auth import Authenticator


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
