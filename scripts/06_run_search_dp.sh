#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../code" && pwd)"
PYTHON="${PYTHON:-python}"

: "${OUT_ROOT:?Set OUT_ROOT}"
: "${ROOTS_FILE:?Set ROOTS_FILE}"

SEARCH_GRAPH_DIR="${SEARCH_GRAPH_DIR:-$OUT_ROOT/search_graph}"
SEARCH_OUT_DIR="${SEARCH_OUT_DIR:-$OUT_ROOT/search_results}"

EXTRA_ARGS=()
if [[ -n "${INITIAL_COMPONENTS_DIR:-}" ]]; then
  EXTRA_ARGS+=(--initial-components-dir "$INITIAL_COMPONENTS_DIR")
fi
if [[ "${REUSE_EXISTING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--reuse-existing)
fi

"$PYTHON" "$CODE_DIR/run_search_dp.py" \
  --graph-dir "$SEARCH_GRAPH_DIR" \
  --roots-file "$ROOTS_FILE" \
  --outbase "$SEARCH_OUT_DIR" \
  --top-k "${TOP_K:-50}" \
  --max-nodes "${MAX_NODES:-20}" \
  --dp-mode "${DP_MODE:-at-most}" \
  --edge-cost-weight "${EDGE_COST_WEIGHT:-1.0}" \
  --singleton-root-target-size "${SINGLETON_ROOT_TARGET_SIZE:-50}" \
  "${EXTRA_ARGS[@]}"
