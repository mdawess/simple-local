#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-config.yml}"
ACTION="${2:-up}"
CONTAINER=simple-local-mysql
VOLUME=simple-local-mysql-data
IMAGE=mysql:8

[ -f .env ] && { set -a; . ./.env; set +a; }

SETTINGS="$(uv run --quiet python -c '
import sys
from simple_local.config import load
m = load(sys.argv[1]).logging.mysql
if m is None:
    raise SystemExit(9)
print("\t".join([m.host, str(m.port), m.user, m.password, m.database]))
' "$CONFIG" 2>/tmp/simple-local-mysql-cfg.err)" || {
  if [ $? = 9 ]; then
    [ "$ACTION" = up ] || echo "mysql: $CONFIG has no logging.mysql — pass CONFIG=<the config you serve>" >&2
    exit 0
  fi
  echo "mysql: could not read logging.mysql from $CONFIG — skipping" >&2
  tail -2 /tmp/simple-local-mysql-cfg.err >&2
  exit 0
}

IFS=$'\t' read -r DB_HOST DB_PORT DB_USER DB_PASS DB_NAME <<<"$SETTINGS"

case "$DB_HOST" in
  localhost | 127.0.0.1 | ::1) ;;
  *) exit 0 ;;
esac

run_sql() { docker exec -i "$CONTAINER" mysql -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" "$@"; }
ready() { docker exec "$CONTAINER" mysqladmin ping -u "$DB_USER" -p"$DB_PASS" --silent >/dev/null 2>&1; }

case "$ACTION" in
  down)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "mysql: removed $CONTAINER" || echo "mysql: not running"
    exit 0
    ;;
  shell)
    exec docker exec -it "$CONTAINER" mysql -u "$DB_USER" -p"$DB_PASS" "$DB_NAME"
    ;;
  tail)
    run_sql -e "SELECT id, created_at, endpoint, model, status, duration_ms,
      prompt_tokens, completion_tokens, LEFT(COALESCE(error, ''), 60) AS error
      FROM request_log ORDER BY id DESC LIMIT 20"
    exit 0
    ;;
esac

ready && exit 0

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker start "$CONTAINER" >/dev/null
else
  echo "mysql: creating $CONTAINER on :$DB_PORT (first boot initializes the data dir, ~30s)"
  docker run -d --name "$CONTAINER" --restart unless-stopped \
    -e "MYSQL_ROOT_PASSWORD=$DB_PASS" -e "MYSQL_DATABASE=$DB_NAME" \
    -p "$DB_PORT:3306" -v "$VOLUME:/var/lib/mysql" "$IMAGE" >/dev/null
fi

printf 'mysql: waiting'
for _ in $(seq 1 120); do
  ready && { echo " ready"; exit 0; }
  printf '.'
  sleep 0.5
done
echo
echo "mysql: not ready yet — serving anyway, request logging will retry" >&2
