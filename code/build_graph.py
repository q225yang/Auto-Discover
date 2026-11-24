#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from qdrant_client import QdrantClient

from graph_index import GraphIndex


LOG_EVERY_N_TABLES = 500
QUERY_BATCH_SIZE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct a join or union table graph from an existing Qdrant vector database."
    )
    qdrant = parser.add_mutually_exclusive_group(required=True)
    qdrant.add_argument("--qdrant-url", help="Qdrant server URL.")
    qdrant.add_argument("--qdrant-path", help="Local Qdrant storage path.")
    parser.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY"))
    parser.add_argument("--collection", required=True, help="Qdrant collection name.")
    parser.add_argument("--graph-mode", choices=["join", "union"], required=True)
    parser.add_argument("--tau-sim", type=float, required=True, help="Similarity threshold.")
    parser.add_argument("--limit-per-col", type=int, default=500, help="Maximum retrieved neighbors per column.")
    parser.add_argument("--output-dir", required=True, help="Directory for graph files and construction logs.")
    parser.add_argument("--join-topk-cols", type=int, default=1, help="Strongest matched columns used for join scoring.")
    parser.add_argument("--query-hnsw-ef", type=int, help="Optional Qdrant HNSW search ef value.")
    return parser.parse_args()


def make_client(args: argparse.Namespace) -> QdrantClient:
    if args.qdrant_url:
        return QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key, timeout=120)
    return QdrantClient(path=args.qdrant_path)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    index = GraphIndex(
        client=make_client(args),
        collection_name=args.collection,
        graph_mode=args.graph_mode,
        tau_sim=args.tau_sim,
        limit_per_col=args.limit_per_col,
        verbose=True,
        log_every_n_tables=LOG_EVERY_N_TABLES,
        dump_edges_path=str(outdir / "table_edges.tsv.gz"),
        dump_nodes_path=str(outdir / "table_nodes.tsv.gz"),
        dump_summary_path=str(outdir / "summary.json"),
        join_topk_cols=args.join_topk_cols,
        query_batch_size=QUERY_BATCH_SIZE,
        query_hnsw_ef=args.query_hnsw_ef,
    )

    started = time.perf_counter()
    stats = index.build(build_graph=False, count_only=False)
    elapsed = time.perf_counter() - started
    print(f"elapsed_wall_sec={elapsed:.3f}", flush=True)
    print(f"output_dir={outdir}", flush=True)
    print(stats, flush=True)


if __name__ == "__main__":
    main()
