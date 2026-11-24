#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


NODES_FILE = "table_nodes.tsv.gz"
EDGES_FILE = "table_edges.tsv.gz"
NODE_UTILITY_COL = "weight"
EDGE_SIMILARITY_COL = "edge_weight"
TARGET_MIN = 0.25
TARGET_MAX = 0.75
FLOAT_FORMAT = "%.12g"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the merged graph for search by converting similarities into edge costs."
    )
    parser.add_argument("--graph-dir", required=True, help="Merged graph directory.")
    parser.add_argument("--outdir", required=True, help="Output directory for the search-ready graph.")
    return parser.parse_args()


def finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def stats(series: pd.Series) -> Dict[str, Optional[float]]:
    values = finite_numeric(series)
    if len(values) == 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def normalize_to_range(series: pd.Series) -> pd.Series:
    if not math.isfinite(TARGET_MIN) or not math.isfinite(TARGET_MAX):
        raise ValueError("Target range must be finite.")
    if TARGET_MAX < TARGET_MIN:
        raise ValueError("Target maximum must be at least the target minimum.")

    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    finite = values.replace([np.inf, -np.inf], np.nan).notna()
    if not finite.any():
        return out

    src_min = float(values.loc[finite].min())
    src_max = float(values.loc[finite].max())
    denom = src_max - src_min
    if denom <= 0.0:
        out.loc[finite] = TARGET_MIN
    else:
        out.loc[finite] = TARGET_MIN + ((values.loc[finite] - src_min) / denom) * (TARGET_MAX - TARGET_MIN)
    return out


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    graph_dir = Path(args.graph_dir)
    outdir = Path(args.outdir)
    nodes_path = graph_dir / NODES_FILE
    edges_path = graph_dir / EDGES_FILE
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing input nodes file: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing input edges file: {edges_path}")
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(nodes_path, sep="\t", compression="gzip")
    edges = pd.read_csv(edges_path, sep="\t", compression="gzip")
    if NODE_UTILITY_COL not in nodes.columns:
        raise ValueError(f"{nodes_path} missing required column: {NODE_UTILITY_COL}")
    if EDGE_SIMILARITY_COL not in edges.columns:
        raise ValueError(f"{edges_path} missing required column: {EDGE_SIMILARITY_COL}")

    node_utility = pd.to_numeric(nodes[NODE_UTILITY_COL], errors="coerce").fillna(0.0)
    edge_similarity = pd.to_numeric(edges[EDGE_SIMILARITY_COL], errors="coerce")
    finite_similarity = finite_numeric(edge_similarity)
    if len(finite_similarity) == 0:
        raise ValueError("No finite edge similarities found.")
    if finite_similarity.lt(0).any():
        raise ValueError("Edge similarities must be nonnegative.")

    log1p_similarity = np.log1p(edge_similarity)
    normalized = normalize_to_range(log1p_similarity)
    squared = normalized ** 2
    edge_cost = 1.0 - squared

    nodes["utility_score"] = node_utility
    nodes["penalty_score"] = (1.0 - node_utility).clip(lower=0.0, upper=1.0)
    nodes["weight"] = node_utility

    if "original_edge_weight" not in edges.columns:
        edges["original_edge_weight"] = edge_similarity
    edges["edge_weight_before_log1p_square"] = edge_similarity
    edges["log1p_edge_similarity"] = log1p_similarity
    edges["normalized_log1p_similarity"] = normalized
    edges["squared_normalized_log1p_similarity"] = squared
    edges["edge_cost"] = edge_cost
    edges["edge_weight"] = edge_cost

    nodes_out = outdir / NODES_FILE
    edges_out = outdir / EDGES_FILE
    nodes.to_csv(nodes_out, sep="\t", index=False, compression="gzip", na_rep="", float_format=FLOAT_FORMAT)
    edges.to_csv(edges_out, sep="\t", index=False, compression="gzip", na_rep="", float_format=FLOAT_FORMAT)

    summary = {
        "input_graph_dir": str(graph_dir),
        "output_graph_dir": str(outdir),
        "node_weight_formula": "weight = utility_score",
        "node_penalty_formula": "penalty_score = clip(1 - utility_score, 0, 1)",
        "edge_cost_formula": "edge_weight = 1 - normalized_log1p_similarity ** 2",
        "edge_normalization_range": [TARGET_MIN, TARGET_MAX],
        "num_nodes": int(len(nodes)),
        "num_edges": int(len(edges)),
        "elapsed_sec": time.perf_counter() - started,
        "before": {
            "node_utility": stats(node_utility),
            "edge_similarity": stats(edge_similarity),
        },
        "after": {
            "node_weight": stats(nodes["weight"]),
            "edge_cost": stats(edges["edge_weight"]),
        },
        "outputs": {
            "nodes": str(nodes_out),
            "edges": str(edges_out),
            "summary": str(outdir / "search_graph_summary.json"),
        },
    }
    (outdir / "search_graph_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
