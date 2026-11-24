#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../code" && pwd)"
PYTHON="${PYTHON:-python}"

: "${OUT_ROOT:?Set OUT_ROOT}"
: "${DATASET_CACHE:?Set DATASET_CACHE}"

JOIN_GRAPH_DIR="${JOIN_GRAPH_DIR:-$OUT_ROOT/join_graph}"
UNION_GRAPH_DIR="${UNION_GRAPH_DIR:-$OUT_ROOT/union_graph}"
MERGED_GRAPH_DIR="${MERGED_GRAPH_DIR:-$OUT_ROOT/merged_graph}"

"$PYTHON" "$CODE_DIR/merge_join_union_graph.py" \
  --join-dir "$JOIN_GRAPH_DIR" \
  --union-dir "$UNION_GRAPH_DIR" \
  --dataset-cache "$DATASET_CACHE" \
  --outdir "$MERGED_GRAPH_DIR" \
  --node-top-k "${NODE_TOP_K:-5}" \
  --progress-every "${MERGE_PROGRESS_EVERY:-1000}"
