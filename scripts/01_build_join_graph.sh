#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../code" && pwd)"
PYTHON="${PYTHON:-python}"

: "${COLLECTION:?Set COLLECTION}"
: "${OUT_ROOT:?Set OUT_ROOT}"

JOIN_GRAPH_DIR="${JOIN_GRAPH_DIR:-$OUT_ROOT/join_graph}"
JOIN_TAU="${JOIN_TAU:-0.80}"
LIMIT_PER_COL="${LIMIT_PER_COL:-500}"

QDRANT_ARGS=()
if [[ -n "${QDRANT_URL:-}" ]]; then
  QDRANT_ARGS+=(--qdrant-url "$QDRANT_URL")
elif [[ -n "${QDRANT_PATH:-}" ]]; then
  QDRANT_ARGS+=(--qdrant-path "$QDRANT_PATH")
else
  echo "Set QDRANT_URL or QDRANT_PATH" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${QUERY_HNSW_EF:-}" ]]; then
  EXTRA_ARGS+=(--query-hnsw-ef "$QUERY_HNSW_EF")
fi

"$PYTHON" "$CODE_DIR/build_graph.py" \
  "${QDRANT_ARGS[@]}" \
  --collection "$COLLECTION" \
  --graph-mode join \
  --tau-sim "$JOIN_TAU" \
  --limit-per-col "$LIMIT_PER_COL" \
  --output-dir "$JOIN_GRAPH_DIR" \
  --join-topk-cols "${JOIN_TOPK_COLS:-1}" \
  "${EXTRA_ARGS[@]}"
