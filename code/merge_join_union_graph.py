#!/usr/bin/env python3
from __future__ import annotations

import argparse
from transform_join_union_graph import run_merge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge join and union graphs, score edges, and assign node utility weights."
    )
    parser.add_argument("--join-dir", required=True, help="Join graph directory from build_graph.py.")
    parser.add_argument("--union-dir", required=True, help="Union graph directory from build_graph.py.")
    parser.add_argument("--dataset-cache", required=True, help="Directory containing preprocessed source tables.")
    parser.add_argument("--outdir", required=True, help="Output directory for the merged graph.")
    parser.add_argument("--node-top-k", type=int, default=5, help="Incident top-k edge utilities used for node utility.")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_merge(
        join_dir=args.join_dir,
        union_dir=args.union_dir,
        dataset_cache=args.dataset_cache,
        outdir=args.outdir,
        node_top_k=args.node_top_k,
        progress_every=args.progress_every,
        sample_debug_rows=0,
        graph_format_weight_mode="utility",
    )


if __name__ == "__main__":
    main()
