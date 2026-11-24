#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../code" && pwd)"
PYTHON="${PYTHON:-python}"

: "${OUT_ROOT:?Set OUT_ROOT}"

SEARCH_GRAPH_DIR="${SEARCH_GRAPH_DIR:-$OUT_ROOT/search_graph}"
INITIAL_COMPONENTS_DIR="${INITIAL_COMPONENTS_DIR:-$OUT_ROOT/initial_components}"

"$PYTHON" "$CODE_DIR/build_initial_components.py" \
  --graph-dir "$SEARCH_GRAPH_DIR" \
  --outdir "$INITIAL_COMPONENTS_DIR" \
  --target-component-size "${INITIAL_COMPONENT_SIZE:-3}"
