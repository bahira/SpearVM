#!/usr/bin/env bash
set -euo pipefail

: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
backup_dir="${BACKUP_DIR:-./backups}"
mkdir -p "$backup_dir"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
file="$backup_dir/spearvm_${timestamp}.sql.gz"

docker compose -f docker-compose.staging.yml exec -T postgres \
  pg_dump -U spearvm -d spearvm | gzip > "$file"

echo "created $file"
