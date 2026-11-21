#!/usr/bin/env bash
# Offline smoke test: lint, unit/integration tests, CLI flow on the SYNTHETIC fixture,
# API + Streamlit health checks. Uses a throwaway data dir; never touches ./data.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="${VENV:-.venv}/bin"
DATA="$(mktemp -d)"
export QUANTLAB_DATA_DIR="$DATA" LIVE_TRADING_ENABLED=false
trap 'kill ${API_PID:-} ${ST_PID:-} 2>/dev/null || true; rm -rf "$DATA"' EXIT

echo "== ruff";   "$BIN/ruff" check .
echo "== pytest"; "$BIN/pytest" -q -p no:warnings

echo "== CLI (synthetic fixture)"
SNAP=$("$BIN/quantlab" data ingest --symbols SPY QQQ IWM GLD TLT --start 2015-01-01 --end 2025-12-31 --fixture synthetic \
       | "$BIN/python" -c "import json,sys;print(json.load(sys.stdin)['snapshot_id'])")
"$BIN/quantlab" data check --snapshot "$SNAP" > /dev/null
RUN=$("$BIN/quantlab" backtest run --strategy ma-crossover --symbol SPY --snapshot "$SNAP" \
      | "$BIN/python" -c "import json,sys;print(json.load(sys.stdin)['run_id'])")
"$BIN/quantlab" report export --run "$RUN" --format md > /dev/null
"$BIN/quantlab" experiment run --id dq-001 --snapshot "$SNAP" --seeds 1 2 --strategies sma-crossover > /dev/null 2>&1
"$BIN/quantlab" ai explain --run "$RUN" | tail -1

echo "== API + dashboard health"
"$BIN/uvicorn" quantlab.api.app:app --port 8765 > /dev/null 2>&1 & API_PID=$!
"$BIN/streamlit" run apps/dashboard/streamlit_app.py --server.headless true --server.port 8766 > /dev/null 2>&1 & ST_PID=$!
for i in $(seq 1 60); do
  if curl -sf localhost:8765/health > /dev/null && curl -sf localhost:8766/_stcore/health > /dev/null; then break; fi
  sleep 1
done
curl -sf localhost:8765/health; echo
curl -sf localhost:8766/_stcore/health; echo
echo "SMOKE TEST PASSED"
