#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from graph_search import choose_cap, clip_score, fmt_sec, read_edges, read_nodes, safe_float


class BoundedDsu:
    def __init__(self, nodes: List[str]):
        self.parent: Dict[str, str] = {v: v for v in nodes}
        self.size: Dict[str, int] = {v: 1 for v in nodes}
        self.members: Dict[str, Set[str]] = {v: {v} for v in nodes}
        self.seed_edges: Dict[str, List[int]] = {v: [] for v in nodes}

    def find(self, v: str) -> str:
        path = []
        while self.parent[v] != v:
            path.append(v)
            v = self.parent[v]
        root = v
        for x in path:
            self.parent[x] = root
        return root

    def union_if_room(self, u: str, v: str, edge_id: int, max_size: int) -> bool:
        ru = self.find(u)
        rv = self.find(v)
        if ru == rv:
            return False
        if self.size[ru] + self.size[rv] > max_size:
            return False

        if self.size[ru] < self.size[rv]:
            ru, rv = rv, ru

        self.parent[rv] = ru
        self.size[ru] += self.size[rv]
        self.members[ru].update(self.members[rv])
        self.seed_edges[ru].extend(self.seed_edges[rv])
        self.seed_edges[ru].append(edge_id)

        del self.size[rv]
        del self.members[rv]
        del self.seed_edges[rv]
        return True

    def roots(self) -> List[str]:
        return list(self.members.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build connected initial components of bounded "
            "size for graph_search.py."
        )
    )
    p.add_argument("--graph-dir", required=True)
    p.add_argument("--nodes-path", default=None)
    p.add_argument("--edges-path", default=None)
    p.add_argument("--outdir", required=True)

    p.add_argument("--node-id-col", default="table_key")
    p.add_argument("--node-weight-col", default="weight")
    p.add_argument("--edge-src-col", default="src_table_key")
    p.add_argument("--edge-tgt-col", default="tgt_table_key")
    p.add_argument("--edge-weight-col", default="edge_weight")

    p.add_argument("--target-component-size", type=int, default=3)
    p.add_argument("--component-id-prefix", default="ic")
    p.add_argument("--cap", type=float, default=None)
    p.add_argument("--clip-min", type=float, default=0.0)
    p.add_argument("--clip-max", type=float, default=None)
    p.add_argument("--float-format", default="%.12g")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_component_size <= 0:
        raise ValueError("--target-component-size must be positive.")

    t0 = time.perf_counter()
    graph_dir = Path(args.graph_dir)
    nodes_path = Path(args.nodes_path) if args.nodes_path else graph_dir / "table_nodes.tsv.gz"
    edges_path = Path(args.edges_path) if args.edges_path else graph_dir / "table_edges.tsv.gz"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Loading graph ===", flush=True)
    nodes_df = read_nodes(nodes_path, args.node_id_col, args.node_weight_col)
    edges_df = read_edges(edges_path, args.edge_src_col, args.edge_tgt_col, args.edge_weight_col)

    cap = choose_cap(nodes_df, edges_df, args.node_weight_col, args.edge_weight_col, args.cap)
    clip_max = args.clip_max if args.clip_max is not None else cap

    node_ids = list(dict.fromkeys(nodes_df[args.node_id_col].astype(str).tolist()))
    valid_nodes = set(node_ids)
    dsu = BoundedDsu(node_ids)

    edge_records: List[Tuple[float, int, str, str]] = []
    edge_lookup: Dict[int, Tuple[str, str, float]] = {}
    for row in edges_df.itertuples(index=False):
        u = str(getattr(row, args.edge_src_col))
        v = str(getattr(row, args.edge_tgt_col))
        if u == v or u not in valid_nodes or v not in valid_nodes:
            continue
        raw_weight = safe_float(getattr(row, args.edge_weight_col), default=0.0)
        cost = clip_score(raw_weight, args.clip_min, clip_max)
        if cost < 0 and abs(cost) <= 1e-12:
            cost = 0.0
        if cost < -1e-12:
            raise ValueError(f"Edge {(u, v)!r} has negative algorithm cost={cost}.")
        edge_id = len(edge_records)
        edge_records.append((float(max(cost, 0.0)), edge_id, u, v))
        edge_lookup[edge_id] = (u, v, float(max(cost, 0.0)))

    print(
        f"loaded nodes={len(node_ids)} valid_edges={len(edge_records)} "
        f"target_component_size={args.target_component_size}",
        flush=True,
    )

    print("=== Building bounded initial components ===", flush=True)
    num_unions = 0
    for _cost, edge_id, u, v in sorted(edge_records, key=lambda x: (x[0], x[1])):
        if dsu.union_if_room(u, v, edge_id, args.target_component_size):
            num_unions += 1

    component_items = []
    for root in dsu.roots():
        members = sorted(dsu.members[root])
        seed_edges = list(dsu.seed_edges[root])
        component_items.append((members[0], members, seed_edges))
    component_items.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)

    component_id_by_index = {
        i: f"{args.component_id_prefix}{i:07d}"
        for i in range(len(component_items))
    }

    node_rows = []
    edge_rows = []
    for i, (_sort_key, members, seed_edges) in enumerate(component_items):
        component_id = component_id_by_index[i]
        for order, node_id in enumerate(members):
            node_rows.append(
                {
                    "initial_component_id": component_id,
                    args.node_id_col: node_id,
                    "component_node_order": order,
                    "component_size": len(members),
                }
            )
        for order, edge_id in enumerate(seed_edges):
            u, v, cost = edge_lookup[edge_id]
            edge_rows.append(
                {
                    "initial_component_id": component_id,
                    "edge_id_internal": int(edge_id),
                    args.edge_src_col: u,
                    args.edge_tgt_col: v,
                    "edge_weight": cost,
                    "cost": cost,
                    "component_edge_order": order,
                }
            )

    components_path = outdir / "initial_components.tsv.gz"
    component_edges_path = outdir / "initial_component_edges.tsv.gz"
    pd.DataFrame(node_rows).to_csv(
        components_path,
        sep="	",
        index=False,
        compression="gzip",
        na_rep="",
        float_format=args.float_format,
    )
    pd.DataFrame(edge_rows).to_csv(
        component_edges_path,
        sep="	",
        index=False,
        compression="gzip",
        na_rep="",
        float_format=args.float_format,
    )

    elapsed_sec = time.perf_counter() - t0
    size_hist = Counter(len(members) for _sort_key, members, _seed_edges in component_items)
    summary = {
        "graph_dir": str(graph_dir),
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "target_component_size": int(args.target_component_size),
        "cap": float(cap),
        "clip_min": float(args.clip_min),
        "clip_max": float(clip_max),
        "num_nodes": int(len(node_ids)),
        "num_valid_edges": int(len(edge_records)),
        "num_components": int(len(component_items)),
        "num_unions": int(num_unions),
        "num_seed_edges": int(len(edge_rows)),
        "num_non_singleton_components": int(sum(1 for _k, members, _e in component_items if len(members) > 1)),
        "component_size_histogram": {str(k): int(v) for k, v in sorted(size_hist.items())},
        "max_component_size": int(max(size_hist) if size_hist else 0),
        "elapsed_sec": float(elapsed_sec),
        "elapsed": fmt_sec(elapsed_sec),
        "outputs": {
            "initial_components": str(components_path),
            "initial_component_edges": str(component_edges_path),
        },
    }
    summary_path = outdir / "initial_components_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Initial component summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
