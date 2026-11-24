#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../code" && pwd)"
PYTHON="${PYTHON:-python}"

: "${COLLECTION:?Set COLLECTION}"
: "${OUT_ROOT:?Set OUT_ROOT}"

UNION_GRAPH_DIR="${UNION_GRAPH_DIR:-$OUT_ROOT/union_graph}"
UNION_TAU="${UNION_TAU:-0.85}"
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
  --graph-mode union \
  --tau-sim "$UNION_TAU" \
  --limit-per-col "$LIMIT_PER_COL" \
  --output-dir "$UNION_GRAPH_DIR" \
  "${EXTRA_ARGS[@]}"
