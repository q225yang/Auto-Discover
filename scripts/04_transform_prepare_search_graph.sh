#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../code" && pwd)"
PYTHON="${PYTHON:-python}"

: "${OUT_ROOT:?Set OUT_ROOT}"

MERGED_GRAPH_DIR="${MERGED_GRAPH_DIR:-$OUT_ROOT/merged_graph}"
SEARCH_GRAPH_DIR="${SEARCH_GRAPH_DIR:-$OUT_ROOT/search_graph}"

"$PYTHON" "$CODE_DIR/prepare_search_graph.py" \
  --graph-dir "$MERGED_GRAPH_DIR" \
  --outdir "$SEARCH_GRAPH_DIR"
