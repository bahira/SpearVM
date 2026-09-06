#!/usr/bin/env python3
"""Small staging smoke/load test; no third-party dependency required."""

from __future__ import annotations

import argparse
import concurrent.futures
import time
import urllib.request


def hit(url: str) -> int:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/api/ready")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        statuses = list(pool.map(hit, [args.url] * args.requests))
    elapsed = time.perf_counter() - started
    successful = sum(status == 200 for status in statuses)
    print(f"requests={args.requests} success={successful} elapsed_s={elapsed:.3f}")
    if successful != args.requests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
