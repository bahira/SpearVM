#!/usr/bin/env python3
"""Create or revoke a PostgreSQL SpearVM API key.

Examples:
  DATABASE_URL=postgresql://... python scripts/provision_key.py create acme
  DATABASE_URL=postgresql://... python scripts/provision_key.py revoke KEY_ID
"""

from __future__ import annotations

import argparse
import os
import secrets

from spearvm_sim.auth import PostgresKeyStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage SpearVM tenant API keys")
    parser.add_argument("action", choices=("create", "revoke"))
    parser.add_argument("value", help="tenant id for create, key id for revoke")
    parser.add_argument("--role", choices=("admin", "viewer"), default="viewer")
    args = parser.parse_args()
    url = os.environ.get("DATABASE_URL")
    if not url:
        parser.error("DATABASE_URL is required")
    store = PostgresKeyStore(url)
    if args.action == "create":
        secret = secrets.token_urlsafe(32)
        key = store.create(args.value, secret, args.role)
        print(f"key_id={key.key_id}")
        print(f"api_key={secret}")
        print("Store the api_key now; only its hash is persisted.")
    else:
        store.revoke(args.value)
        print(f"revoked={args.value}")


if __name__ == "__main__":
    main()
