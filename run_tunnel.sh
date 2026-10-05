#!/usr/bin/env bash
# Tunnel supervisor: keeps cloudflared alive and keeps its (ephemeral) URL
# registered at the public discovery endpoint, so every client device picks
# it up automatically — no manual URL pasting, ever.
set -u
LOG=/tmp/antigravity/tunnel.log
URL_FILE=/tmp/antigravity/current-tunnel-url
API=https://aruga-api.pages.dev/bridge-url
TOKEN=$(cat "$HOME/Projects/aruga-api/.api_token" 2>/dev/null || true)
mkdir -p /tmp/antigravity
: > "$LOG"

"$HOME/.local/bin/cloudflared" tunnel --url http://127.0.0.1:8080 --logfile "$LOG" &
CF=$!

push_url() {
  [ -n "${TOKEN:-}" ] || return 1
  curl -sS -m 8 -X POST "$API" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $TOKEN" \
    --data "{\"url\":\"$1\"}" >/dev/null 2>&1
}

# Watcher: registers the URL once available; re-registers if it ever rotates
# while the process is alive. Retries failed pushes every cycle.
(
  while kill -0 "$CF" 2>/dev/null; do
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" 2>/dev/null | tail -1)
    if [ -n "$URL" ] && [ "$(cat "$URL_FILE" 2>/dev/null)" != "$URL" ]; then
      if push_url "$URL"; then
        echo "$URL" > "$URL_FILE"
        echo "[tunnel] registered: $URL"
      fi
    fi
    sleep 10
  done
) &
WATCH=$!

wait "$CF"
EXIT=$?
kill "$WATCH" 2>/dev/null
echo "[tunnel] cloudflared exited ($EXIT) — systemd will restart it"
exit "$EXIT"
