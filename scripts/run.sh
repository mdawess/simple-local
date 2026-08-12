#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

available() {
  for base in implementations examples; do
    [ -d "$ROOT/$base" ] && (cd "$ROOT/$base" && ls -d */ 2>/dev/null | tr -d /)
  done | sort -u | tr '\n' ' '
}

EXAMPLE="${1:-}"
if [ -z "$EXAMPLE" ]; then
  echo "usage: scripts/run.sh <name> [args...]" >&2
  echo "available: $(available)" >&2
  exit 1
fi
shift

DIR=""
for base in implementations examples; do
  if [ -d "$ROOT/$base/$EXAMPLE" ]; then DIR="$ROOT/$base/$EXAMPLE"; break; fi
done
[ -n "$DIR" ] || { echo "no such example: $EXAMPLE (available: $(available))" >&2; exit 1; }

if [ -f "$DIR/run.py" ]; then
  ENTRY="run.py"
elif [ -f "$DIR/$EXAMPLE.py" ]; then
  ENTRY="$EXAMPLE.py"
else
  echo "no entrypoint in $DIR (expected run.py or $EXAMPLE.py)" >&2
  exit 1
fi

WITH=()
if [ -f "$DIR/deps.txt" ]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    for pkg in $line; do WITH+=(--with "$pkg"); done
  done < "$DIR/deps.txt"
fi

[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }

PORT=8081
HOST=localhost
if [ -f "$DIR/config.yml" ]; then
  PORT="$(awk '/^ *port:/ {print $2; exit}' "$DIR/config.yml" 2>/dev/null || echo 8081)"
  HOST="$(awk '/^ *host:/ {print $2; exit}' "$DIR/config.yml" 2>/dev/null || echo localhost)"
fi
if ! curl -s -o /dev/null "http://$HOST:$PORT/health" 2>/dev/null; then
  echo "note: server not reachable on $HOST:$PORT — start it with" >&2
  echo "      make serve CONFIG=${DIR#"$ROOT"/}/config.yml" >&2
fi

cd "$DIR"
exec uv run "${WITH[@]}" python "$ENTRY" "$@"
