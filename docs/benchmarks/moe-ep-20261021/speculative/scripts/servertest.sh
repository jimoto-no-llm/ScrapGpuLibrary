#!/bin/bash
# usage: servertest.sh OUTPREFIX server-args...   (call through go.sh/bundle.sh so the GPU lock is held)
# starts llama-server on port 8090 with the given args, waits for /health, runs client.py, stops the server by PID (never pkill).
HERE="$(dirname "$0")"; . "$HERE/env.sh"
OUTP=$1; shift
"$LB/llama-server" --port 8090 --host 127.0.0.1 -np 1 --no-webui "$@" > "$OUTP.server.log" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8090/health > /dev/null 2>&1 && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -5 "$OUTP.server.log"; exit 3; }
  sleep 1
done
python3 -I "$HERE/client.py" 8090 "$OUTP.json" 192
