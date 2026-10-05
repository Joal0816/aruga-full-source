#!/usr/bin/env bash
# ARUGA service launcher — systemd-managed (self-healing + auto URL registration).
# Safe to run any time: starts only what's down, then prints full health.
set -u

for unit in aruga-bridge aruga-tunnel; do
  if ! systemctl --user is-active --quiet "$unit.service" 2>/dev/null; then
    echo "[start] $unit was down — starting"
    systemctl --user start "$unit.service" 2>/dev/null \
      || echo "[error] could not start $unit (check: systemctl --user status $unit)"
  fi
done
sleep 2

BRIDGE=$(curl -s --max-time 3 http://127.0.0.1:8080/status || echo "DOWN")
LOCAL_URL=$(cat /tmp/antigravity/current-tunnel-url 2>/dev/null || true)
DISCOVERED=$(curl -s --max-time 5 https://aruga-api.pages.dev/bridge-url \
  | python3 -c "import sys,json;print(json.load(sys.stdin).get('url') or '')" 2>/dev/null || true)
SITE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 https://aruga.joalvergs.tech/)
API=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 https://aruga-api.pages.dev/)

echo ""
echo "bridge     : $BRIDGE"
echo "bridge unit: $(systemctl --user is-active aruga-bridge.service 2>/dev/null || echo '?')"
echo "tunnel unit: $(systemctl --user is-active aruga-tunnel.service 2>/dev/null || echo '?')"
echo "site       : https://aruga.joalvergs.tech/ ($SITE)"
echo "api        : https://aruga-api.pages.dev/ ($API)"
echo "discovery  : ${DISCOVERED:-not registered yet}"
echo ""
if [ -n "$DISCOVERED" ]; then
  echo "✔ Devices auto-discover this URL — no pasting needed."
  echo "  Manual override (if ever required): $DISCOVERED"
else
  echo "! Discovery has no URL yet — check: journalctl --user -u aruga-tunnel -n 20"
fi
