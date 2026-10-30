#!/usr/bin/env bash
# Put the Ivan site on the internet through a Cloudflare tunnel: https://<hostname>/ with Cloudflare's certificate,
# no port opened on the router, every path forwarded (the app's one password guards them all). Run ON THE SERVER, with sudo:
#
#   sudo deploy/setup_cloudflare_tunnel.sh [hostname]        (default nba.prattflix.com)
#
# Steps (each skipped when already done, so it can be rerun):
#   1. install cloudflared from Cloudflare's apt repository
#   2. cloudflared tunnel login: prints a URL; open it in any browser, log in to Cloudflare, pick prattflix.com
#   3. create the tunnel "ivan-nba" and point <hostname> at it (a CNAME in your Cloudflare DNS)
#   4. write /etc/cloudflared/config.yml (deploy/cloudflared/config.yml.template) and install the cloudflared service
#   5. refuse to finish unless the site has a password (IVAN_PASSWORD_HASH in the app's .env) and IVAN_PUBLIC=1 is set
#   6. check https://<hostname>/login answers
set -euo pipefail
HOST="${1:-nba.prattflix.com}"
NAME="ivan-nba"
ROOT="${IVAN_REMOTE_ROOT:-/opt/ivan/app}"
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }

ENVF="$ROOT/.env"
grep -qE '^IVAN(_DESK)?_PASSWORD_HASH=.+' "$ENVF" 2>/dev/null || {
  echo "!! no IVAN_PASSWORD_HASH in $ENVF: make one with  venv/bin/python scripts/password.py  and add it first"; exit 1; }
grep -qE '^IVAN(_DESK)?_PUBLIC=1' "$ENVF" || { echo "IVAN_PUBLIC=1" >> "$ENVF"; echo "== added IVAN_PUBLIC=1 to $ENVF"; }

if ! command -v cloudflared >/dev/null; then
  echo "== installing cloudflared"
  mkdir -p --mode=0755 /usr/share/keyrings
  curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg -o /usr/share/keyrings/cloudflare-main.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared $(. /etc/os-release; echo "$VERSION_CODENAME") main" \
    > /etc/apt/sources.list.d/cloudflared.list
  apt-get update -qq && apt-get install -y -qq cloudflared
fi

CF="/root/.cloudflared"
if [[ ! -f "$CF/cert.pem" ]]; then
  echo "== log in to Cloudflare: open the URL below in any browser and pick prattflix.com"
  cloudflared tunnel login
fi

if ! cloudflared tunnel list -o json 2>/dev/null | grep -q "\"name\":\"$NAME\""; then
  echo "== creating tunnel $NAME"
  cloudflared tunnel create "$NAME"
fi
TID=$(cloudflared tunnel list -o json | python3 -c "import json,sys; print(next(t['id'] for t in json.load(sys.stdin) if t['name']=='$NAME'))")
mkdir -p /etc/cloudflared
cp "$CF/$TID.json" /etc/cloudflared/ 2>/dev/null || true
echo "== pointing $HOST at the tunnel"
cloudflared tunnel route dns --overwrite-dns "$NAME" "$HOST" || true

# the app: the address the ivan-web unit listens on
APP=$(systemctl show ivan-web -p ExecStart --value | grep -o -- '--host [^ ]* --port [^ ]*' | awk '{print $2 ":" $4}')
[[ -n "$APP" ]] || { echo "!! could not read the ivan-web unit's host and port"; exit 1; }
sed -e "s/__TUNNEL_ID__/$TID/g" -e "s/__HOSTNAME__/$HOST/g" -e "s/__APP__/$APP/g" "$ROOT/deploy/cloudflared/config.yml.template" > /etc/cloudflared/config.yml
cloudflared tunnel ingress validate --config /etc/cloudflared/config.yml

if ! systemctl list-unit-files | grep -q '^cloudflared.service'; then
  cloudflared --config /etc/cloudflared/config.yml service install
fi
systemctl restart cloudflared
# the app reads .env at start: restart it so IVAN_PUBLIC applies
OLD=$(systemctl show ivan-web -p MainPID --value); [[ "$OLD" != "0" ]] && kill "$OLD"; sleep 4

echo "== checking"
for i in $(seq 1 15); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "https://$HOST/login" || true)
  [[ "$code" == "200" ]] && break; sleep 2
done
echo "https://$HOST/login -> $code (want 200)"
[[ "$code" == "200" ]] && echo "done: give out https://$HOST/ (the NBA desk is https://$HOST/nba/)" || { echo "!! not right yet: journalctl -u cloudflared -n 30"; exit 1; }
