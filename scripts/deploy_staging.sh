#!/usr/bin/env bash
set -euo pipefail

COMPOSE=(docker compose -f docker-compose.staging.yml)
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be injected by your secret manager}"

command -v docker >/dev/null || {
  echo "docker is required; install Docker Engine/Desktop and retry" >&2
  exit 127
}
docker compose version >/dev/null

"${COMPOSE[@]}" up --build -d

for attempt in $(seq 1 30); do
  if curl --fail --silent http://127.0.0.1:${SPEARVM_PORT:-8000}/api/ready >/dev/null; then
    echo "SpearVM staging is ready"
    python scripts/load_test.py --url http://127.0.0.1:${SPEARVM_PORT:-8000}/api/ready
    if [[ -n "${TENANT_ID:-}" ]]; then
      echo "Provisioning tenant ${TENANT_ID}; capture the one-time api_key output in your secret manager."
      "${COMPOSE[@]}" exec -T api python /app/scripts/provision_key.py create "$TENANT_ID" --role "${TENANT_ROLE:-viewer}"
    fi
    exit 0
  fi
  sleep 2
done

"${COMPOSE[@]}" ps
"${COMPOSE[@]}" logs --tail=100 api
exit 1
