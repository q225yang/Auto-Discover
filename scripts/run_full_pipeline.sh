#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${METHOD:-pruning}"

"$SCRIPT_DIR/01_build_join_graph.sh"
"$SCRIPT_DIR/02_build_union_graph.sh"
"$SCRIPT_DIR/03_merge_join_union_graph.sh"
"$SCRIPT_DIR/04_transform_prepare_search_graph.sh"

if [[ "$METHOD" == "auto_discover" ]]; then
  : "${OUT_ROOT:?Set OUT_ROOT}"
  export INITIAL_COMPONENTS_DIR="${INITIAL_COMPONENTS_DIR:-$OUT_ROOT/initial_components}"
  "$SCRIPT_DIR/05_build_initial_components.sh"
elif [[ "$METHOD" != "pruning" ]]; then
  echo "METHOD must be pruning or auto_discover" >&2
  exit 2
fi

"$SCRIPT_DIR/06_run_search_dp.sh"
