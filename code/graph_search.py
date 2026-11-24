#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

@dataclass
class Component:
    idx: int
    nodes: Set[str]
    selected_edges: Set[int] = field(default_factory=set)
    boundary_edges: Set[int] = field(default_factory=set)

    active: int = 1
    deleted: int = 0
    d: float = 0.0
    has_root: bool = False

@dataclass
class EdgeEvent:
    edge_id: int
    from_node: str
    to_node: str
    epsilon: float

@dataclass
class DeactivateEvent:
    component_idx: int
    epsilon: float

@dataclass
class BatchEvent:
    event_type: str
    epsilon: float
    edge_id: Optional[int] = None
    component_idx: Optional[int] = None
    from_node: Optional[str] = None
    to_node: Optional[str] = None

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run moat-growing graph pruning / root-subgraph extraction on "
            "table_nodes.tsv.gz and table_edges.tsv.gz."
        )
    )

    p.add_argument(
        "--graph-dir",
        required=True,
        help="Directory containing table_nodes.tsv.gz and table_edges.tsv.gz.",
    )
    p.add_argument(
        "--nodes-path",
        default=None,
        help="Optional explicit path to table_nodes.tsv.gz.",
    )
    p.add_argument(
        "--edges-path",
        default=None,
        help="Optional explicit path to table_edges.tsv.gz.",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )

    p.add_argument(
        "--root-node-id",
        required=True,
        help="Root table_key / node id whose final component should be extracted.",
    )

    p.add_argument(
        "--initial-components-path",
        default=None,
        help=(
            "Optional TSV/TSV.GZ assigning nodes to initial connected components. "
            "When provided, the moat search starts from these components instead "
            "of singleton nodes."
        ),
    )
    p.add_argument(
        "--initial-component-edges-path",
        default=None,
        help=(
            "Optional TSV/TSV.GZ listing seed edge_id_internal values for each "
            "initial component. Required when any initial component has >1 node."
        ),
    )
    p.add_argument(
        "--initial-component-id-col",
        default="initial_component_id",
        help="Component id column in --initial-components-path and --initial-component-edges-path.",
    )
    p.add_argument(
        "--initial-component-node-col",
        default=None,
        help="Node id column in --initial-components-path. Default: --node-id-col.",
    )

    p.add_argument(
        "--node-id-col",
        default="table_key",
    )
    p.add_argument(
        "--node-weight-col",
        default="weight",
        help="Algorithm-ready node weight/prize column.",
    )

    p.add_argument(
        "--node-weight-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to already-converted node weights before search.",
    )

    p.add_argument(
        "--edge-src-col",
        default="src_table_key",
    )
    p.add_argument(
        "--edge-tgt-col",
        default="tgt_table_key",
    )
    p.add_argument(
        "--edge-weight-col",
        "--edge-utility-col",
        dest="edge_weight_col",
        default="edge_weight",
        help="Algorithm-ready edge weight/cost column.",
    )

    p.add_argument(
        "--cap",
        type=float,
        default=None,
        help=(
            "Optional cap for clipping already-converted weights. "
            "If omitted, use max finite value over node and edge weights."
        ),
    )
    p.add_argument(
        "--clip-min",
        type=float,
        default=0.0,
        help="Minimum allowed already-converted weight.",
    )
    p.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help=(
            "Maximum allowed already-converted weight. "
            "If omitted, clip max is cap."
        ),
    )

    p.add_argument(
        "--stop-when-one-active",
        action="store_true",
        help=(
            "Stop when only one active component remains. "
            "Default is to continue until no active components remain."
        ),
    )
    p.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Optional safety limit on number of algorithm iterations.",
    )
    p.add_argument(
        "--event-batch-size",
        type=int,
        default=0,
        help=(
            "Approximate mixed-event batch size. When >1, first apply the "
            "safe deactivation-before-next-merge batch when available, then "
            "collect this many lowest-epsilon merge/deactivate candidates and "
            "revalidate each before applying. Default 0 keeps only the safer "
            "deactivation-before-merge batching path."
        ),
    )
    p.add_argument(
        "--eps-tol",
        type=float,
        default=1e-12,
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )
    p.add_argument(
        "--search-log-path",
        default=None,
        help="Optional JSONL log path. Default: <outdir>/search_progress.jsonl.",
    )

    p.add_argument(
        "--expand-singleton-root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If the final root component has exactly one node, expand the "
            "written root subgraph using cheapest frontier edges."
        ),
    )
    p.add_argument(
        "--singleton-root-target-size",
        type=int,
        default=50,
        help="Target node count for --expand-singleton-root.",
    )

    p.add_argument(
        "--run-topk",
        action="store_true",
        help="Run rooted_topk_subtree.py after root-subgraph outputs are written.",
    )
    p.add_argument(
        "--topk-script",
        default="rooted_topk_subtree.py",
        help="Path to rooted_topk_subtree.py used by --run-topk.",
    )
    p.add_argument(
        "--topk-outdir",
        default=None,
        help="Optional output directory for rooted top-k subtree results.",
    )
    p.add_argument("--topk-target-size", type=int, default=300)
    p.add_argument("--topk-max-nodes", type=int, default=300)
    p.add_argument("--topk-mode", choices=["exact", "at-most"], default="exact")
    p.add_argument(
        "--topk-num-trees",
        "--topk-top-k",
        dest="topk_num_trees",
        type=int,
        default=1,
        help="Number of distinct rooted DP trees to output when --run-topk is set.",
    )
    p.add_argument("--topk-edge-cost-weight", type=float, default=0.0)
    p.add_argument("--topk-edge-reward-weight", type=float, default=0.0)

    return p.parse_args()

def now() -> float:
    return time.perf_counter()

def fmt_sec(x: float) -> str:
    return f"{x:.3f}s"

def ensure_path(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")

def safe_float(x, default: float = 0.0) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
        return default
    except Exception:
        return default

def read_nodes(path: Path, node_id_col: str, node_weight_col: str) -> pd.DataFrame:
    ensure_path(path, "nodes file")
    df = pd.read_csv(path, sep="\t", compression="gzip")

    missing = [c for c in [node_id_col, node_weight_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required node columns: {missing}")

    df[node_id_col] = df[node_id_col].astype(str)
    df[node_weight_col] = pd.to_numeric(df[node_weight_col], errors="coerce")

    agg = {}
    for c in df.columns:
        if c == node_id_col:
            continue
        elif c == node_weight_col:
            agg[c] = "max"
        else:
            agg[c] = "first"

    df = df.groupby(node_id_col, as_index=False).agg(agg)
    return df

def read_edges(
    path: Path,
    edge_src_col: str,
    edge_tgt_col: str,
    edge_weight_col: str,
) -> pd.DataFrame:
    ensure_path(path, "edges file")
    df = pd.read_csv(path, sep="\t", compression="gzip")

    missing = [
        c for c in [edge_src_col, edge_tgt_col, edge_weight_col]
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"{path} missing required edge columns: {missing}")

    df[edge_src_col] = df[edge_src_col].astype(str)
    df[edge_tgt_col] = df[edge_tgt_col].astype(str)
    df[edge_weight_col] = pd.to_numeric(df[edge_weight_col], errors="coerce")

    df = df[df[edge_src_col] != df[edge_tgt_col]].copy()

    weight = pd.to_numeric(df[edge_weight_col], errors="coerce")
    df = df[weight.notna() & np.isfinite(weight)].copy()

    pair_a = np.minimum(df[edge_src_col].astype(str), df[edge_tgt_col].astype(str))
    pair_b = np.maximum(df[edge_src_col].astype(str), df[edge_tgt_col].astype(str))
    df["_pair_key"] = pair_a + "\t" + pair_b

    df = (
        df.sort_values(edge_weight_col, ascending=True, na_position="last")
        .drop_duplicates("_pair_key", keep="first")
        .drop(columns=["_pair_key"])
        .reset_index(drop=True)
    )

    return df

def read_tsv_maybe_gzip(path: Path, desc: str) -> pd.DataFrame:
    ensure_path(path, desc)
    try:
        compression = "gzip" if str(path).endswith(".gz") else None
        return pd.read_csv(path, sep="	", compression=compression)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def read_initial_component_nodes(
    path: Path,
    component_id_col: str,
    node_id_col: str,
) -> Dict[str, Set[str]]:
    df = read_tsv_maybe_gzip(path, "initial components file")
    missing = [c for c in [component_id_col, node_id_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required initial component columns: {missing}")

    comps: Dict[str, Set[str]] = {}
    for row in df[[component_id_col, node_id_col]].itertuples(index=False):
        component_id = str(row[0])
        node_id = str(row[1])
        comps.setdefault(component_id, set()).add(node_id)

    if not comps:
        raise ValueError(f"No initial component assignments found in {path}.")
    return comps

def read_initial_component_edges(
    path: Path,
    component_id_col: str,
) -> Dict[str, Set[int]]:
    df = read_tsv_maybe_gzip(path, "initial component edges file")
    if len(df) == 0:
        return {}

    missing = [c for c in [component_id_col, "edge_id_internal"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required initial component edge columns: {missing}")

    edge_ids = pd.to_numeric(df["edge_id_internal"], errors="coerce")
    if edge_ids.isna().any():
        bad_count = int(edge_ids.isna().sum())
        raise ValueError(f"{path} has {bad_count} non-integer edge_id_internal values.")

    out: Dict[str, Set[int]] = defaultdict(set)
    for component_id, edge_id in zip(df[component_id_col].astype(str), edge_ids.astype(int)):
        out[str(component_id)].add(int(edge_id))
    return dict(out)

def choose_cap(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_weight_col: str,
    edge_weight_col: str,
    user_cap: Optional[float],
) -> float:
    if user_cap is not None:
        if not math.isfinite(user_cap):
            raise ValueError(f"--cap must be finite, got {user_cap}")
        return float(user_cap)

    vals: List[float] = []

    node_vals = pd.to_numeric(nodes[node_weight_col], errors="coerce")
    edge_vals = pd.to_numeric(edges[edge_weight_col], errors="coerce")

    node_vals = node_vals.replace([np.inf, -np.inf], np.nan).dropna()
    edge_vals = edge_vals.replace([np.inf, -np.inf], np.nan).dropna()

    if len(node_vals) > 0:
        vals.append(float(node_vals.max()))
    if len(edge_vals) > 0:
        vals.append(float(edge_vals.max()))

    if not vals:
        raise ValueError("Could not infer cap because no finite node or edge scores exist.")

    return max(vals)

def clip_score(x: float, clip_min: float, clip_max: float) -> float:
    if not math.isfinite(x):
        x = clip_min
    return min(max(x, clip_min), clip_max)

class RootMoatSolver:
    def __init__(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        *,
        root_node_id: str,
        node_id_col: str,
        node_weight_col: str,
        edge_src_col: str,
        edge_tgt_col: str,
        edge_weight_col: str,
        node_weight_scale: float,
        cap: float,
        clip_min: float,
        clip_max: float,
        eps_tol: float,
        initial_components: Optional[Dict[str, Set[str]]] = None,
        initial_component_edges: Optional[Dict[str, Set[int]]] = None,
        initial_component_paths: Optional[Dict[str, str]] = None,
    ):
        self.nodes_df = nodes_df.copy()
        self.edges_df = edges_df.copy()

        self.root_node_id = str(root_node_id)

        self.node_id_col = node_id_col
        self.node_weight_col = node_weight_col
        self.edge_src_col = edge_src_col
        self.edge_tgt_col = edge_tgt_col
        self.edge_weight_col = edge_weight_col
        self.node_weight_scale = float(node_weight_scale)
        if not math.isfinite(self.node_weight_scale) or self.node_weight_scale < 0:
            raise ValueError(f"node_weight_scale must be finite and nonnegative, got {self.node_weight_scale}")

        self.cap = float(cap)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.eps_tol = float(eps_tol)

        self.nodes: List[str] = []
        self.node_pi: Dict[str, float] = {}
        self.node_d: Dict[str, float] = {}
        self.node_to_component: Dict[str, int] = {}

        self.edge_u: List[str] = []
        self.edge_v: List[str] = []
        self.edge_cost: List[float] = []

        self.incident_edges: Dict[str, Set[int]] = {}

        self.components: List[Component] = []
        self.active_count: int = 0
        self.root_component_idx: Optional[int] = None

        self.initial_component_specs = (
            {str(k): set(str(v) for v in vals) for k, vals in initial_components.items()}
            if initial_components
            else None
        )
        self.initial_component_seed_edges = (
            {str(k): set(int(eid) for eid in vals) for k, vals in initial_component_edges.items()}
            if initial_component_edges
            else {}
        )
        self.initial_component_paths = initial_component_paths or {}
        self.initial_component_stats: Dict[str, object] = {
            "enabled": bool(self.initial_component_specs),
            "paths": self.initial_component_paths,
        }

    def build(self) -> None:
        self._build_nodes()
        self._build_edges()
        self._init_components()

    def _build_nodes(self) -> None:
        node_ids = self.nodes_df[self.node_id_col].astype(str).tolist()
        self.nodes = list(dict.fromkeys(node_ids))

        if self.root_node_id not in set(self.nodes):
            raise ValueError(
                f"Root node {self.root_node_id!r} is not present in node file."
            )

        weight_by_node = dict(
            zip(
                self.nodes_df[self.node_id_col].astype(str),
                pd.to_numeric(self.nodes_df[self.node_weight_col], errors="coerce"),
            )
        )

        for v in self.nodes:
            raw_weight = safe_float(weight_by_node.get(v, 0.0), default=0.0)
            scaled_weight = raw_weight * self.node_weight_scale
            weight = clip_score(scaled_weight, self.clip_min, self.clip_max)

            if weight < 0 and abs(weight) <= self.eps_tol:
                weight = 0.0
            if weight < -self.eps_tol:
                raise ValueError(
                    f"Node {v!r} has negative algorithm weight={weight}. "
                    f"Check graph construction. raw_weight={raw_weight}, scale={self.node_weight_scale}"
                )

            self.node_pi[v] = float(max(weight, 0.0))
            self.node_d[v] = 0.0
            self.incident_edges[v] = set()

    def _build_edges(self) -> None:
        valid_nodes = set(self.nodes)

        kept_rows = []
        for row in self.edges_df.itertuples(index=False):
            u = str(getattr(row, self.edge_src_col))
            v = str(getattr(row, self.edge_tgt_col))

            if u == v:
                continue
            if u not in valid_nodes or v not in valid_nodes:
                continue

            raw_weight = safe_float(getattr(row, self.edge_weight_col), default=0.0)
            cost = clip_score(raw_weight, self.clip_min, self.clip_max)

            if cost < 0 and abs(cost) <= self.eps_tol:
                cost = 0.0
            if cost < -self.eps_tol:
                raise ValueError(
                    f"Edge {(u, v)!r} has negative algorithm weight/cost={cost}. "
                    f"Check graph construction. raw_weight={raw_weight}"
                )

            edge_id = len(self.edge_u)
            self.edge_u.append(u)
            self.edge_v.append(v)
            self.edge_cost.append(float(max(cost, 0.0)))

            self.incident_edges[u].add(edge_id)
            self.incident_edges[v].add(edge_id)

            kept_rows.append(True)

        if len(self.edge_u) == 0:
            print("WARNING: no valid edges were kept after filtering.", flush=True)

    def _init_components(self) -> None:
        self.components = []
        self.node_to_component = {}
        self.root_component_idx = None

        if self.initial_component_specs:
            self._init_seed_components()
        else:
            self._init_singleton_components()

    def _component_boundary_edges(self, nodes: Set[str]) -> Set[int]:
        boundary_edges: Set[int] = set()
        for v in nodes:
            for eid in self.incident_edges[v]:
                u = self.edge_u[eid]
                w = self.edge_v[eid]
                if u in nodes and w in nodes:
                    continue
                boundary_edges.add(eid)
        return boundary_edges

    def _validate_seed_connectivity(
        self,
        source_component_id: str,
        nodes: Set[str],
        selected_edges: Set[int],
    ) -> None:
        if len(nodes) <= 1:
            return
        if len(selected_edges) < len(nodes) - 1:
            raise ValueError(
                f"Initial component {source_component_id!r} has {len(nodes)} nodes but only "
                f"{len(selected_edges)} seed edges; non-singleton seed components must be connected."
            )

        adj: Dict[str, Set[str]] = {v: set() for v in nodes}
        for eid in selected_edges:
            u = self.edge_u[eid]
            v = self.edge_v[eid]
            if u not in nodes or v not in nodes:
                raise ValueError(
                    f"Seed edge_id_internal={eid} for initial component {source_component_id!r} "
                    f"does not stay inside that component."
                )
            adj[u].add(v)
            adj[v].add(u)

        start = next(iter(nodes))
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

        if seen != nodes:
            raise ValueError(
                f"Initial component {source_component_id!r} is disconnected under its seed edges."
            )

    def _append_initial_component(
        self,
        idx: int,
        source_component_id: str,
        nodes: Set[str],
        selected_edges: Set[int],
    ) -> None:
        has_root = self.root_node_id in nodes
        initial_prize = sum(self.node_pi.get(v, 0.0) for v in nodes)
        comp = Component(
            idx=idx,
            nodes=set(nodes),
            selected_edges=set(selected_edges),
            boundary_edges=self._component_boundary_edges(nodes),
            active=0 if has_root or initial_prize <= self.eps_tol else 1,
            deleted=0,
            d=0.0,
            has_root=has_root,
        )
        self.components.append(comp)
        for v in comp.nodes:
            self.node_to_component[v] = idx
        if has_root:
            self.root_component_idx = idx

    def _init_singleton_components(self) -> None:
        for idx, v in enumerate(self.nodes):
            self._append_initial_component(
                idx=idx,
                source_component_id=f"singleton:{v}",
                nodes={v},
                selected_edges=set(),
            )

        self.active_count = sum(c.active for c in self.components)
        size_hist = Counter(len(c.nodes) for c in self.components)
        zero_prize_count = sum(
            1 for c in self.components
            if not c.has_root and self.component_prize(c) <= self.eps_tol
        )
        self.initial_component_stats = {
            "enabled": False,
            "num_components_created": int(len(self.components)),
            "component_size_histogram": {str(k): int(v) for k, v in sorted(size_hist.items())},
            "num_zero_prize_components_initialized_inactive": int(zero_prize_count),
            "active_count_initial": int(self.active_count),
            "paths": self.initial_component_paths,
        }

    def _init_seed_components(self) -> None:
        assert self.initial_component_specs is not None
        valid_nodes = set(self.nodes)
        assigned: Dict[str, str] = {}
        component_specs: List[Tuple[str, Set[str]]] = []
        duplicate_nodes: List[Tuple[str, str, str]] = []
        unknown_nodes: List[Tuple[str, str]] = []

        for source_component_id, raw_nodes in self.initial_component_specs.items():
            nodes: Set[str] = set()
            for v in raw_nodes:
                if v not in valid_nodes:
                    unknown_nodes.append((source_component_id, v))
                    continue
                previous_component = assigned.get(v)
                if previous_component is not None and previous_component != source_component_id:
                    duplicate_nodes.append((v, previous_component, source_component_id))
                    continue
                assigned[v] = source_component_id
                nodes.add(v)
            if nodes:
                component_specs.append((source_component_id, nodes))

        if unknown_nodes:
            preview = unknown_nodes[:5]
            raise ValueError(
                f"Initial components contain {len(unknown_nodes)} nodes not present in the graph. "
                f"First examples: {preview}"
            )
        if duplicate_nodes:
            preview = duplicate_nodes[:5]
            raise ValueError(
                f"Initial components assign {len(duplicate_nodes)} nodes to multiple components. "
                f"First examples: {preview}"
            )

        missing_nodes = [v for v in self.nodes if v not in assigned]
        for v in missing_nodes:
            component_specs.append((f"__singleton_missing__{v}", {v}))

        component_ids = {cid for cid, _nodes in component_specs}
        unknown_edge_component_ids = sorted(set(self.initial_component_seed_edges) - component_ids)
        if unknown_edge_component_ids:
            raise ValueError(
                "Initial component edge file contains component ids not present in the node assignments: "
                f"{unknown_edge_component_ids[:10]}"
            )

        num_seed_edges = 0
        for idx, (source_component_id, nodes) in enumerate(component_specs):
            seed_edges = set(self.initial_component_seed_edges.get(source_component_id, set()))
            for eid in seed_edges:
                if eid < 0 or eid >= len(self.edge_u):
                    raise ValueError(
                        f"Seed edge_id_internal={eid} for initial component {source_component_id!r} "
                        f"is outside valid range [0, {len(self.edge_u) - 1}]."
                    )
                u = self.edge_u[eid]
                v = self.edge_v[eid]
                if u not in nodes or v not in nodes:
                    raise ValueError(
                        f"Seed edge_id_internal={eid} for initial component {source_component_id!r} "
                        f"connects {u!r}-{v!r}, not two nodes inside the component."
                    )

            if len(nodes) > 1 and not seed_edges:
                raise ValueError(
                    f"Initial component {source_component_id!r} has {len(nodes)} nodes but no seed edges. "
                    "Pass --initial-component-edges-path generated with the same graph inputs."
                )

            self._validate_seed_connectivity(source_component_id, nodes, seed_edges)
            self._append_initial_component(idx, source_component_id, nodes, seed_edges)
            num_seed_edges += len(seed_edges)

        self.active_count = sum(c.active for c in self.components)
        if self.root_component_idx is None:
            raise RuntimeError("Root node was not assigned to any initial component.")

        size_hist = Counter(len(c.nodes) for c in self.components)
        zero_prize_count = sum(
            1 for c in self.components
            if not c.has_root and self.component_prize(c) <= self.eps_tol
        )
        root_initial_component = self.components[self.root_component_idx]
        self.initial_component_stats = {
            "enabled": True,
            "num_component_specs": int(len(self.initial_component_specs)),
            "num_components_created": int(len(self.components)),
            "num_missing_nodes_added_as_singletons": int(len(missing_nodes)),
            "num_seed_edges_selected": int(num_seed_edges),
            "num_non_singleton_components": int(sum(1 for c in self.components if len(c.nodes) > 1)),
            "component_size_histogram": {str(k): int(v) for k, v in sorted(size_hist.items())},
            "max_component_size": int(max(size_hist) if size_hist else 0),
            "num_zero_prize_components_initialized_inactive": int(zero_prize_count),
            "active_count_initial": int(self.active_count),
            "root_initial_component_idx": int(root_initial_component.idx),
            "root_initial_component_size": int(len(root_initial_component.nodes)),
            "root_initial_component_seed_edges": int(len(root_initial_component.selected_edges)),
            "paths": self.initial_component_paths,
        }

    def current_component(self, node: str) -> Component:
        return self.components[self.node_to_component[node]]

    def active_components(self) -> List[Component]:
        return [
            c for c in self.components
            if c.deleted == 0 and c.active == 1
        ]

    def alive_components(self) -> List[Component]:
        return [
            c for c in self.components
            if c.deleted == 0
        ]

    def component_prize(self, c: Component) -> float:
        return sum(self.node_pi[v] for v in c.nodes)

    def edge_other_node(self, edge_id: int, node: str) -> str:
        u = self.edge_u[edge_id]
        v = self.edge_v[edge_id]
        if node == u:
            return v
        if node == v:
            return u
        raise ValueError(f"Node {node!r} is not incident to edge_id={edge_id}")

    def find_best_edge_event(self) -> Optional[EdgeEvent]:
        best: Optional[EdgeEvent] = None

        for c in self.active_components():
            stale_edges: List[int] = []

            for edge_id in c.boundary_edges:
                u = self.edge_u[edge_id]
                v = self.edge_v[edge_id]

                cu = self.current_component(u)
                cv = self.current_component(v)

                if cu.idx == cv.idx:
                    stale_edges.append(edge_id)
                    continue

                if cu.idx == c.idx:
                    from_node, to_node = u, v
                    other = cv
                elif cv.idx == c.idx:
                    from_node, to_node = v, u
                    other = cu
                else:
                    stale_edges.append(edge_id)
                    continue

                raw_slack = (
                    self.edge_cost[edge_id]
                    - self.node_d[from_node]
                    - self.node_d[to_node]
                )

                denom = 1.0 + float(other.active)
                epsilon = raw_slack / denom

                if epsilon < 0 and abs(epsilon) <= self.eps_tol:
                    epsilon = 0.0

                if best is None or epsilon < best.epsilon:
                    best = EdgeEvent(
                        edge_id=edge_id,
                        from_node=from_node,
                        to_node=to_node,
                        epsilon=float(epsilon),
                    )

            for edge_id in stale_edges:
                c.boundary_edges.discard(edge_id)

        return best

    def find_deactivation_batch_before(self, edge_epsilon: float) -> List[DeactivateEvent]:
        events: List[DeactivateEvent] = []

        for c in self.active_components():
            epsilon = self.component_prize(c) - c.d

            if epsilon < 0 and abs(epsilon) <= self.eps_tol:
                epsilon = 0.0

            if epsilon < -self.eps_tol:
                raise RuntimeError(
                    f"Negative deactivate epsilon={epsilon} for component={c.idx}. "
                    "This usually means some cost/prize conversion is inconsistent."
                )

            if epsilon < 0:
                epsilon = 0.0

            if math.isinf(edge_epsilon) or epsilon < edge_epsilon - self.eps_tol:
                events.append(DeactivateEvent(component_idx=c.idx, epsilon=float(epsilon)))

        events.sort(key=lambda e: (e.epsilon, e.component_idx))
        return events

    def _batch_event_sort_key(self, event: BatchEvent) -> Tuple[float, int, int]:
        event_rank = 0 if event.event_type == "merge" else 1
        event_id = event.edge_id if event.event_type == "merge" else event.component_idx
        return (float(event.epsilon), event_rank, int(event_id if event_id is not None else -1))

    def collect_mixed_event_batch(self, batch_size: int) -> List[BatchEvent]:
        if batch_size <= 0:
            return []

        events: List[BatchEvent] = []
        best_edge_events: Dict[int, BatchEvent] = {}

        for c in self.active_components():
            stale_edges: List[int] = []

            for edge_id in c.boundary_edges:
                u = self.edge_u[edge_id]
                v = self.edge_v[edge_id]

                cu = self.current_component(u)
                cv = self.current_component(v)

                if cu.idx == cv.idx:
                    stale_edges.append(edge_id)
                    continue

                if cu.idx == c.idx:
                    from_node, to_node = u, v
                    other = cv
                elif cv.idx == c.idx:
                    from_node, to_node = v, u
                    other = cu
                else:
                    stale_edges.append(edge_id)
                    continue

                raw_slack = (
                    self.edge_cost[edge_id]
                    - self.node_d[from_node]
                    - self.node_d[to_node]
                )
                denom = 1.0 + float(other.active)
                epsilon = raw_slack / denom

                if epsilon < 0:
                    epsilon = 0.0

                event = BatchEvent(
                    event_type="merge",
                    epsilon=float(epsilon),
                    edge_id=edge_id,
                    from_node=from_node,
                    to_node=to_node,
                )
                previous = best_edge_events.get(edge_id)
                if previous is None or self._batch_event_sort_key(event) < self._batch_event_sort_key(previous):
                    best_edge_events[edge_id] = event

            for edge_id in stale_edges:
                c.boundary_edges.discard(edge_id)

        events.extend(best_edge_events.values())

        for c in self.active_components():
            epsilon = self.component_prize(c) - c.d

            if epsilon < 0:
                epsilon = 0.0

            events.append(
                BatchEvent(
                    event_type="deactivate",
                    epsilon=float(epsilon),
                    component_idx=c.idx,
                )
            )

        return heapq.nsmallest(batch_size, events, key=self._batch_event_sort_key)

    def apply_revalidated_batch_event(self, event: BatchEvent) -> Tuple[bool, str]:
        if event.event_type == "merge":
            if event.edge_id is None or event.edge_id < 0 or event.edge_id >= len(self.edge_u):
                return False, "invalid_edge_id"

            edge_id = event.edge_id
            u = self.edge_u[edge_id]
            v = self.edge_v[edge_id]
            cu = self.current_component(u)
            cv = self.current_component(v)

            if cu.idx == cv.idx:
                return False, "edge_internal_after_prior_event"

            if int(cu.active) + int(cv.active) == 0:
                return False, "edge_has_no_active_endpoint"

            slack = self.edge_cost[edge_id] - self.node_d[u] - self.node_d[v]
            if slack > self.eps_tol:
                return False, "edge_not_tight_after_prior_event"

            self.merge_components(
                EdgeEvent(
                    edge_id=edge_id,
                    from_node=u,
                    to_node=v,
                    epsilon=0.0,
                )
            )
            return True, "applied"

        if event.event_type == "deactivate":
            if event.component_idx is None or event.component_idx < 0 or event.component_idx >= len(self.components):
                return False, "invalid_component_idx"

            c = self.components[event.component_idx]
            if c.deleted != 0:
                return False, "component_deleted_after_prior_event"
            if c.active != 1:
                return False, "component_inactive_after_prior_event"

            residual = self.component_prize(c) - c.d
            if residual > self.eps_tol:
                return False, "component_not_exhausted_after_prior_event"

            self.deactivate_component(c.idx)
            return True, "applied"

        return False, "unknown_event_type"

    def execute_mixed_event_batch(self, events: List[BatchEvent]) -> Dict[str, object]:
        num_edges_merged = 0
        num_components_deactivated = 0
        num_events_skipped = 0
        skip_reasons: Counter[str] = Counter()

        batch_horizon = 0.0
        for event in events:
            if event.epsilon < 0:
                event.epsilon = 0.0
            if event.epsilon > batch_horizon:
                batch_horizon = float(event.epsilon)

        if batch_horizon > 0:
            self.grow_active_components(batch_horizon)

        for event in events:
            applied, reason = self.apply_revalidated_batch_event(event)
            if applied:
                if event.event_type == "merge":
                    num_edges_merged += 1
                else:
                    num_components_deactivated += 1
            else:
                num_events_skipped += 1
                skip_reasons[reason] += 1

        return {
            "epsilon": float(batch_horizon),
            "num_edges_merged": int(num_edges_merged),
            "num_components_deactivated": int(num_components_deactivated),
            "num_events_attempted": int(len(events)),
            "num_events_applied": int(num_edges_merged + num_components_deactivated),
            "num_events_skipped": int(num_events_skipped),
            "skip_reasons": dict(skip_reasons),
        }

    def grow_active_components(self, epsilon: float) -> None:
        for c in self.active_components():
            c.d += epsilon
            for v in c.nodes:
                self.node_d[v] += epsilon

    def grow_for_deactivation_batch(self, events: List[DeactivateEvent]) -> float:
        if not events:
            return 0.0

        batch_horizon = max(e.epsilon for e in events)
        if batch_horizon < -self.eps_tol:
            raise RuntimeError(
                f"Negative batch horizon={batch_horizon}. "
                "This usually means some cost/prize conversion is inconsistent."
            )
        if batch_horizon < 0:
            batch_horizon = 0.0

        deactivation_delta = {
            e.component_idx: max(0.0, float(e.epsilon))
            for e in events
        }

        for c in self.active_components():
            delta = deactivation_delta.get(c.idx, batch_horizon)
            c.d += delta
            for v in c.nodes:
                self.node_d[v] += delta

        for event in events:
            self.deactivate_component(event.component_idx)

        return float(batch_horizon)

    def deactivate_component(self, component_idx: int) -> None:
        c = self.components[component_idx]
        if c.deleted == 0 and c.active == 1:
            c.active = 0
            self.active_count -= 1

    def merge_components(self, event: EdgeEvent) -> None:
        edge_id = event.edge_id
        u = self.edge_u[edge_id]
        v = self.edge_v[edge_id]

        c1 = self.current_component(u)
        c2 = self.current_component(v)

        if c1.idx == c2.idx:
            return

        old_active_count = int(c1.active) + int(c2.active)

        if len(c1.nodes) >= len(c2.nodes):
            big, small = c1, c2
        else:
            big, small = c2, c1

        new_idx = len(self.components)
        has_root = bool(big.has_root or small.has_root)

        new_nodes = set(big.nodes)
        new_nodes.update(small.nodes)

        new_selected_edges = set(big.selected_edges)
        new_selected_edges.update(small.selected_edges)
        new_selected_edges.add(edge_id)

        new_boundary_edges = set(big.boundary_edges)
        new_boundary_edges.update(small.boundary_edges)

        cleaned_boundary_edges: Set[int] = set()
        for eid in new_boundary_edges:
            a = self.edge_u[eid]
            b = self.edge_v[eid]
            ca = self.current_component(a)
            cb = self.current_component(b)

            a_inside = ca.idx == c1.idx or ca.idx == c2.idx
            b_inside = cb.idx == c1.idx or cb.idx == c2.idx

            if not (a_inside and b_inside):
                cleaned_boundary_edges.add(eid)

        if has_root:
            new_active = 0
        else:
            new_active = 1

        new_component = Component(
            idx=new_idx,
            nodes=new_nodes,
            selected_edges=new_selected_edges,
            boundary_edges=cleaned_boundary_edges,
            active=new_active,
            deleted=0,
            d=0.0,
            has_root=has_root,
        )

        self.components.append(new_component)

        for x in new_component.nodes:
            self.node_to_component[x] = new_idx

        c1.deleted = 1
        c2.deleted = 1

        c1.nodes = set()
        c1.selected_edges = set()
        c1.boundary_edges = set()

        c2.nodes = set()
        c2.selected_edges = set()
        c2.boundary_edges = set()

        self.active_count = self.active_count - old_active_count + new_active

        if has_root:
            self.root_component_idx = new_idx

    def should_continue(self, stop_when_one_active: bool) -> bool:
        if stop_when_one_active:
            return self.active_count > 1
        return self.active_count > 0

    def run(
        self,
        *,
        stop_when_one_active: bool = False,
        max_iters: Optional[int] = None,
        progress_every: int = 100,
        progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
        event_batch_size: int = 0,
    ) -> Dict[str, object]:
        if event_batch_size < 0:
            raise ValueError(f"event_batch_size must be nonnegative, got {event_batch_size}")

        t0 = now()
        it = 0
        num_merge_events_total = 0
        num_deactivation_batches_total = 0
        num_mixed_event_batches_total = 0
        num_components_deactivated_total = 0
        num_batch_events_attempted_total = 0
        num_batch_events_applied_total = 0
        num_batch_events_skipped_total = 0

        while self.should_continue(stop_when_one_active):
            if max_iters is not None and it >= max_iters:
                print(f"WARNING: reached --max-iters={max_iters}; stopping early.", flush=True)
                break

            it += 1

            num_edges_merged = 0
            num_components_deactivated = 0
            num_events_attempted = 0
            num_events_applied = 0
            num_events_skipped = 0
            skip_reasons: Dict[str, int] = {}

            if event_batch_size > 1:
                edge_event = self.find_best_edge_event()
                if edge_event is not None:
                    epsilon_edge = edge_event.epsilon
                else:
                    epsilon_edge = math.inf

                deactivation_batch = self.find_deactivation_batch_before(epsilon_edge)

                if deactivation_batch:
                    event_type = "deactivate_batch"
                    epsilon_min = self.grow_for_deactivation_batch(deactivation_batch)
                    num_components_deactivated = len(deactivation_batch)
                    num_deactivation_batches_total += 1
                    num_components_deactivated_total += num_components_deactivated
                else:
                    batch_events = self.collect_mixed_event_batch(event_batch_size)
                    if not batch_events:
                        break

                    event_type = "mixed_batch"
                    batch_stats = self.execute_mixed_event_batch(batch_events)
                    epsilon_min = float(batch_stats["epsilon"])
                    num_edges_merged = int(batch_stats["num_edges_merged"])
                    num_components_deactivated = int(batch_stats["num_components_deactivated"])
                    num_events_attempted = int(batch_stats["num_events_attempted"])
                    num_events_applied = int(batch_stats["num_events_applied"])
                    num_events_skipped = int(batch_stats["num_events_skipped"])
                    skip_reasons = dict(batch_stats["skip_reasons"])

                    if num_events_applied == 0:
                        raise RuntimeError(
                            "Mixed-event batch made no progress; all collected events were skipped. "
                            f"skip_reasons={skip_reasons}"
                        )

                    num_mixed_event_batches_total += 1
                    num_merge_events_total += num_edges_merged
                    num_components_deactivated_total += num_components_deactivated
                    num_batch_events_attempted_total += num_events_attempted
                    num_batch_events_applied_total += num_events_applied
                    num_batch_events_skipped_total += num_events_skipped
            else:
                edge_event = self.find_best_edge_event()

                if edge_event is not None:
                    epsilon_edge = edge_event.epsilon
                else:
                    epsilon_edge = math.inf

                deactivation_batch = self.find_deactivation_batch_before(epsilon_edge)

                if deactivation_batch:
                    event_type = "deactivate_batch"
                    epsilon_min = max(e.epsilon for e in deactivation_batch)
                elif edge_event is not None:
                    event_type = "merge"
                    epsilon_min = epsilon_edge
                else:
                    break

                if epsilon_min < -self.eps_tol:
                    raise RuntimeError(
                        f"Negative epsilon={epsilon_min} at iteration={it}. "
                        "This usually means some cost/prize conversion is inconsistent."
                    )

                if epsilon_min < 0:
                    epsilon_min = 0.0

                if event_type == "deactivate_batch":
                    epsilon_min = self.grow_for_deactivation_batch(deactivation_batch)
                    num_components_deactivated = len(deactivation_batch)
                    num_deactivation_batches_total += 1
                    num_components_deactivated_total += num_components_deactivated
                else:
                    assert edge_event is not None
                    self.grow_active_components(epsilon_min)
                    self.merge_components(edge_event)
                    num_edges_merged = 1
                    num_merge_events_total += 1

            if progress_every > 0 and (it == 1 or it % progress_every == 0):
                root_size = 0
                root_edges = 0
                if self.root_component_idx is not None:
                    rc = self.components[self.root_component_idx]
                    if rc.deleted == 0:
                        root_size = len(rc.nodes)
                        root_edges = len(rc.selected_edges)

                largest_active_size = max(
                    (len(c.nodes) for c in self.active_components()),
                    default=0,
                )

                elapsed_sec = now() - t0
                print(
                    f"iter={it} event={event_type} eps={epsilon_min:.6g} "
                    f"active={self.active_count} total_components={len(self.components)} "
                    f"merged={num_edges_merged} deactivated={num_components_deactivated} "
                    f"root_size={root_size} root_edges={root_edges} "
                    f"largest_active_size={largest_active_size} "
                    f"elapsed={elapsed_sec:.2f}s",
                    flush=True,
                )

                record = {
                    "iter": it,
                    "event": event_type,
                    "epsilon": epsilon_min,
                    "active_count": self.active_count,
                    "num_components_total": len(self.components),
                    "num_edges_merged": num_edges_merged,
                    "num_components_deactivated": num_components_deactivated,
                    "num_events_attempted": num_events_attempted,
                    "num_events_applied": num_events_applied,
                    "num_events_skipped": num_events_skipped,
                    "skip_reasons": skip_reasons,
                    "root_component_size": root_size,
                    "root_selected_edges": root_edges,
                    "largest_active_size": largest_active_size,
                    "elapsed_sec": elapsed_sec,
                    "elapsed": fmt_sec(elapsed_sec),
                }
                if progress_callback is not None:
                    progress_callback(record)

        return {
            "iterations": it,
            "active_count": self.active_count,
            "num_components_total": len(self.components),
            "num_alive_components": len(self.alive_components()),
            "num_merge_events": int(num_merge_events_total),
            "num_deactivation_batches": int(num_deactivation_batches_total),
            "num_mixed_event_batches": int(num_mixed_event_batches_total),
            "num_components_deactivated_total": int(num_components_deactivated_total),
            "num_batch_events_attempted": int(num_batch_events_attempted_total),
            "num_batch_events_applied": int(num_batch_events_applied_total),
            "num_batch_events_skipped": int(num_batch_events_skipped_total),
            "elapsed_sec": now() - t0,
        }

    def get_root_component(self) -> Component:
        if self.root_component_idx is None:
            raise RuntimeError("root_component_idx is None.")
        c = self.components[self.root_component_idx]
        if c.deleted != 0:
            raise RuntimeError("Root component points to a deleted component.")
        return c

def validate_postprocess_args(args: argparse.Namespace) -> None:
    if args.singleton_root_target_size <= 0:
        raise ValueError("--singleton-root-target-size must be positive.")

def _edge_other_endpoint(solver: RootMoatSolver, edge_id: int, node: str) -> Optional[str]:
    u = solver.edge_u[edge_id]
    v = solver.edge_v[edge_id]
    if node == u:
        return v
    if node == v:
        return u
    return None

def expand_root_output(
    solver: RootMoatSolver,
    start_nodes: Set[str],
    start_edges: Set[int],
    target_size: int,
) -> Tuple[Set[str], Set[int], Dict[str, object]]:
    nodes = set(start_nodes)
    selected_edges = set(start_edges)
    target_size = min(target_size, len(solver.nodes))
    frontier: List[Tuple[float, float, float, int, str, str]] = []

    def push_frontier_edges(node: str, base_path_cost: float) -> None:
        for eid in solver.incident_edges.get(node, set()):
            other = _edge_other_endpoint(solver, eid, node)
            if other is None or other in nodes:
                continue
            edge_cost = solver.edge_cost[eid]
            path_cost = base_path_cost + edge_cost
            heapq.heappush(
                frontier,
                (
                    path_cost,
                    edge_cost,
                    -solver.node_pi.get(other, 0.0),
                    eid,
                    node,
                    other,
                ),
            )

    for node in sorted(nodes):
        push_frontier_edges(node, 0.0)

    added_nodes = 0
    added_edges = 0
    while len(nodes) < target_size and frontier:
        path_cost, _edge_cost, _neg_prize, eid, from_node, to_node = heapq.heappop(frontier)
        if from_node not in nodes or to_node in nodes:
            continue
        nodes.add(to_node)
        selected_edges.add(eid)
        added_nodes += 1
        added_edges += 1
        push_frontier_edges(to_node, path_cost)

    return nodes, selected_edges, {
        "method": "cheapest_frontier_expansion",
        "target_size": int(target_size),
        "added_nodes": int(added_nodes),
        "added_edges": int(added_edges),
        "frontier_exhausted": bool(len(nodes) < target_size),
    }

def postprocess_root_output(
    solver: RootMoatSolver,
    *,
    enabled: bool,
    target_size: int,
) -> Tuple[Set[str], Set[int], Dict[str, object]]:
    root_c = solver.get_root_component()
    output_nodes = set(root_c.nodes)
    output_edges = set(root_c.selected_edges)
    input_num_nodes = len(output_nodes)
    input_num_edges = len(output_edges)

    stats: Dict[str, object] = {
        "enabled": bool(enabled),
        "input_num_nodes": int(input_num_nodes),
        "input_num_edges": int(input_num_edges),
        "target_size": int(target_size),
        "action": "none",
    }

    if not enabled:
        stats["action"] = "disabled"
    elif input_num_nodes == 1:
        output_nodes, output_edges, detail = expand_root_output(
            solver,
            output_nodes,
            output_edges,
            target_size,
        )
        stats.update(detail)
        stats["action"] = "expanded_singleton_root"
    else:
        stats["reason"] = "root_component_not_singleton"

    stats.update(
        {
            "output_num_nodes": int(len(output_nodes)),
            "output_num_edges": int(len(output_edges)),
        }
    )
    return output_nodes, output_edges, stats

def write_outputs(
    outdir: Path,
    solver: RootMoatSolver,
    run_stats: Dict[str, object],
    root_output_nodes: Optional[Set[str]] = None,
    root_output_edges: Optional[Set[int]] = None,
    postprocess_stats: Optional[Dict[str, object]] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    root_c = solver.get_root_component()
    if root_output_nodes is None:
        root_output_nodes = set(root_c.nodes)
    else:
        root_output_nodes = set(root_output_nodes)
    if root_output_edges is None:
        root_output_edges = set(root_c.selected_edges)
    else:
        root_output_edges = set(root_output_edges)

    root_nodes = solver.nodes_df[
        solver.nodes_df[solver.node_id_col].astype(str).isin(root_output_nodes)
    ].copy()

    root_nodes["pi"] = root_nodes[solver.node_id_col].astype(str).map(solver.node_pi)
    root_nodes["node_d"] = root_nodes[solver.node_id_col].astype(str).map(solver.node_d)

    root_nodes.to_csv(
        outdir / "root_subgraph_nodes.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
        na_rep="",
    )

    selected_edge_ids = sorted(root_output_edges)

    selected_rows = []
    for eid in selected_edge_ids:
        selected_rows.append(
            {
                "edge_id_internal": eid,
                solver.edge_src_col: solver.edge_u[eid],
                solver.edge_tgt_col: solver.edge_v[eid],
                "edge_weight": solver.edge_cost[eid],
                "cost": solver.edge_cost[eid],
            }
        )

    selected_edges_simple = pd.DataFrame(selected_rows)
    selected_edges_simple.to_csv(
        outdir / "root_subgraph_selected_edges.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
        na_rep="",
    )

    comp_rows = []
    for c in solver.alive_components():
        comp_rows.append(
            {
                "component_idx": c.idx,
                "num_nodes": len(c.nodes),
                "num_selected_edges": len(c.selected_edges),
                "num_boundary_edges": len(c.boundary_edges),
                "active": c.active,
                "deleted": c.deleted,
                "d": c.d,
                "has_root": c.has_root,
            }
        )

    comp_df = pd.DataFrame(comp_rows).sort_values(
        ["has_root", "num_nodes"],
        ascending=[False, False],
    )
    comp_df.to_csv(outdir / "final_components_summary.csv", index=False)

    with (outdir / "root_subgraph_node_ids.txt").open("w", encoding="utf-8") as f:
        for v in sorted(root_output_nodes):
            f.write(f"{v}\n")

    summary = {
        "root_node_id": solver.root_node_id,
        "cap": solver.cap,
        "clip_min": solver.clip_min,
        "clip_max": solver.clip_max,
        "node_weight_scale": solver.node_weight_scale,
        "num_input_nodes": int(len(solver.nodes_df)),
        "num_internal_nodes": int(len(solver.nodes)),
        "num_input_edges": int(len(solver.edges_df)),
        "num_internal_edges": int(len(solver.edge_u)),
        "root_component_idx": int(root_c.idx),
        "root_component_num_nodes": int(len(root_c.nodes)),
        "root_component_num_selected_edges": int(len(root_c.selected_edges)),
        "root_output_num_nodes": int(len(root_output_nodes)),
        "root_output_num_selected_edges": int(len(root_output_edges)),
        "root_component_active": int(root_c.active),
        "root_component_d": float(root_c.d),
        "initial_component_stats": solver.initial_component_stats,
        "root_postprocess_stats": postprocess_stats or {"enabled": False},
        "run_stats": run_stats,
        "outputs": {
            "root_nodes": str(outdir / "root_subgraph_nodes.tsv.gz"),
            "root_selected_edges": str(outdir / "root_subgraph_selected_edges.tsv.gz"),
            "root_node_ids": str(outdir / "root_subgraph_node_ids.txt"),
            "components_summary": str(outdir / "final_components_summary.csv"),
        },
    }

    with (outdir / "root_subgraph_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Root component summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

def topk_default_outdir(args: argparse.Namespace, outdir: Path) -> Path:
    dirname = (
        f"topk_rooted_subtree_{args.topk_mode}_k{args.topk_target_size}"
        f"_cap{args.topk_max_nodes}"
    )
    if args.topk_num_trees != 1:
        dirname += f"_top{args.topk_num_trees}"
    return outdir / dirname

def run_topk_subtree(args: argparse.Namespace, outdir: Path) -> Dict[str, object]:
    if args.topk_num_trees <= 0:
        raise ValueError("--topk-num-trees must be positive.")

    topk_out = Path(args.topk_outdir) if args.topk_outdir else topk_default_outdir(args, outdir)
    cmd = [
        sys.executable,
        args.topk_script,
        "--subgraph-dir", str(outdir),
        "--outdir", str(topk_out),
        "--root-node-id", str(args.root_node_id),
        "--target-size", str(args.topk_target_size),
        "--max-nodes", str(args.topk_max_nodes),
        "--mode", args.topk_mode,
        "--top-k", str(args.topk_num_trees),
        "--edge-cost-weight", str(args.topk_edge_cost_weight),
        "--edge-reward-weight", str(args.topk_edge_reward_weight),
    ]

    topk_out.mkdir(parents=True, exist_ok=True)
    print("topk_cmd: " + " ".join(cmd), flush=True)
    t0 = now()
    completed = subprocess.run(cmd)
    elapsed_sec = now() - t0
    if completed.returncode != 0:
        raise RuntimeError(
            f"rooted top-k subtree command failed with exit_code={completed.returncode}: "
            + " ".join(cmd)
        )
    return {
        "cmd": cmd,
        "outdir": str(topk_out),
        "exit_code": int(completed.returncode),
        "elapsed_sec": elapsed_sec,
        "elapsed": fmt_sec(elapsed_sec),
    }

def main() -> None:
    args = parse_args()
    validate_postprocess_args(args)
    if args.initial_component_edges_path and not args.initial_components_path:
        raise ValueError(
            "--initial-component-edges-path requires --initial-components-path."
        )

    graph_dir = Path(args.graph_dir)
    nodes_path = Path(args.nodes_path) if args.nodes_path else graph_dir / "table_nodes.tsv.gz"
    edges_path = Path(args.edges_path) if args.edges_path else graph_dir / "table_edges.tsv.gz"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = now()
    if args.search_log_path == "":
        search_log_path = None
        search_log_fh = None
    else:
        search_log_path = Path(args.search_log_path) if args.search_log_path else outdir / "search_progress.jsonl"
        search_log_path.parent.mkdir(parents=True, exist_ok=True)
        search_log_fh = search_log_path.open("w", encoding="utf-8")

    def log_event(kind: str, **fields) -> None:
        if search_log_fh is None:
            return
        record = {
            "kind": kind,
            "wall_elapsed_sec": now() - t0,
        }
        record.update(fields)
        search_log_fh.write(json.dumps(record, sort_keys=True) + "\n")
        search_log_fh.flush()

    print("=== Config ===", flush=True)
    print(f"graph_dir: {graph_dir}", flush=True)
    print(f"nodes_path: {nodes_path}", flush=True)
    print(f"edges_path: {edges_path}", flush=True)
    print(f"outdir: {outdir}", flush=True)
    print(f"root_node_id: {args.root_node_id}", flush=True)
    print(f"node_weight_scale: {args.node_weight_scale}", flush=True)
    print(f"progress_every: {args.progress_every}", flush=True)
    print(f"event_batch_size: {args.event_batch_size}", flush=True)
    print(f"stop_when_one_active: {args.stop_when_one_active}", flush=True)
    print(f"initial_components_path: {args.initial_components_path}", flush=True)
    print(f"initial_component_edges_path: {args.initial_component_edges_path}", flush=True)
    print(f"max_iters: {args.max_iters}", flush=True)
    print(f"eps_tol: {args.eps_tol}", flush=True)
    print(f"search_log_path: {search_log_path}", flush=True)
    print(f"expand_singleton_root: {args.expand_singleton_root}", flush=True)
    print(f"singleton_root_target_size: {args.singleton_root_target_size}", flush=True)
    print(f"run_topk: {args.run_topk}", flush=True)
    if args.run_topk:
        print(f"topk_target_size: {args.topk_target_size}", flush=True)
        print(f"topk_max_nodes: {args.topk_max_nodes}", flush=True)
        print(f"topk_mode: {args.topk_mode}", flush=True)
        print(f"topk_num_trees: {args.topk_num_trees}", flush=True)

    log_event(
        "config",
        nodes_path=str(nodes_path),
        edges_path=str(edges_path),
        outdir=str(outdir),
        root_node_id=str(args.root_node_id),
        progress_every=int(args.progress_every),
        event_batch_size=int(args.event_batch_size),
        stop_when_one_active=bool(args.stop_when_one_active),
        initial_components_path=str(args.initial_components_path) if args.initial_components_path else None,
        initial_component_edges_path=str(args.initial_component_edges_path) if args.initial_component_edges_path else None,
        node_weight_scale=float(args.node_weight_scale),
        expand_singleton_root=bool(args.expand_singleton_root),
        singleton_root_target_size=int(args.singleton_root_target_size),
        run_topk=bool(args.run_topk),
        topk_target_size=int(args.topk_target_size),
        topk_max_nodes=int(args.topk_max_nodes),
        topk_mode=str(args.topk_mode),
        topk_num_trees=int(args.topk_num_trees),
    )

    print("\n=== Loading graph ===", flush=True)
    nodes = read_nodes(nodes_path, args.node_id_col, args.node_weight_col)
    edges = read_edges(
        edges_path,
        args.edge_src_col,
        args.edge_tgt_col,
        args.edge_weight_col,
    )

    cap = choose_cap(
        nodes,
        edges,
        args.node_weight_col,
        args.edge_weight_col,
        args.cap,
    )

    clip_max = args.clip_max
    if clip_max is None:
        clip_max = cap

    load_elapsed_sec = now() - t0
    load_stats = {
        "num_nodes": int(len(nodes)),
        "num_edges": int(len(edges)),
        "cap": cap,
        "clip_min": args.clip_min,
        "clip_max": clip_max,
        "node_weight_scale": float(args.node_weight_scale),
        "load_elapsed": fmt_sec(load_elapsed_sec),
        "load_elapsed_sec": load_elapsed_sec,
    }
    print(f"loaded nodes={len(nodes)} edges={len(edges)} cap={cap}", flush=True)
    log_event("load_done", **load_stats)

    initial_components = None
    initial_component_edges = None
    initial_component_paths: Dict[str, str] = {}
    if args.initial_components_path:
        initial_component_node_col = args.initial_component_node_col or args.node_id_col
        initial_components_path = Path(args.initial_components_path)
        initial_components = read_initial_component_nodes(
            initial_components_path,
            args.initial_component_id_col,
            initial_component_node_col,
        )
        initial_component_paths["components"] = str(initial_components_path)

        if args.initial_component_edges_path:
            initial_component_edges_path = Path(args.initial_component_edges_path)
            initial_component_edges = read_initial_component_edges(
                initial_component_edges_path,
                args.initial_component_id_col,
            )
            initial_component_paths["component_edges"] = str(initial_component_edges_path)
        else:
            initial_component_edges = {}

        initial_size_hist = Counter(len(v) for v in initial_components.values())
        initial_load_stats = {
            "num_component_specs": int(len(initial_components)),
            "component_size_histogram": {str(k): int(v) for k, v in sorted(initial_size_hist.items())},
            "num_seed_edge_components": int(len(initial_component_edges)),
            "num_seed_edges": int(sum(len(v) for v in initial_component_edges.values())),
            "components_path": str(initial_components_path),
            "component_edges_path": str(args.initial_component_edges_path) if args.initial_component_edges_path else None,
        }
        print(
            f"loaded initial components: specs={initial_load_stats['num_component_specs']} "
            f"size_hist={initial_load_stats['component_size_histogram']} "
            f"seed_edges={initial_load_stats['num_seed_edges']}",
            flush=True,
        )
        log_event("initial_components_loaded", **initial_load_stats)

    print("\n=== Building internal graph ===", flush=True)
    solver = RootMoatSolver(
        nodes,
        edges,
        root_node_id=args.root_node_id,
        node_id_col=args.node_id_col,
        node_weight_col=args.node_weight_col,
        edge_src_col=args.edge_src_col,
        edge_tgt_col=args.edge_tgt_col,
        edge_weight_col=args.edge_weight_col,
        node_weight_scale=args.node_weight_scale,
        cap=cap,
        clip_min=args.clip_min,
        clip_max=clip_max,
        eps_tol=args.eps_tol,
        initial_components=initial_components,
        initial_component_edges=initial_component_edges,
        initial_component_paths=initial_component_paths,
    )
    solver.build()

    build_elapsed_sec = now() - t0
    build_stats = {
        "internal_nodes": len(solver.nodes),
        "internal_edges": len(solver.edge_u),
        "active_count": solver.active_count,
        "initial_component_stats": solver.initial_component_stats,
        "build_elapsed": fmt_sec(build_elapsed_sec),
        "build_elapsed_sec": build_elapsed_sec,
    }
    print(
        f"built internal graph: nodes={len(solver.nodes)} "
        f"edges={len(solver.edge_u)} active={solver.active_count}",
        flush=True,
    )
    if solver.initial_component_stats.get("enabled"):
        print(
            "initial component stats: "
            f"created={solver.initial_component_stats.get('num_components_created')} "
            f"size_hist={solver.initial_component_stats.get('component_size_histogram')} "
            f"seed_edges={solver.initial_component_stats.get('num_seed_edges_selected')}",
            flush=True,
        )
    log_event("build_done", **build_stats)

    print("\n=== Running moat-growing algorithm ===", flush=True)
    log_event("run_start", active_count=int(solver.active_count))
    run_stats = solver.run(
        stop_when_one_active=args.stop_when_one_active,
        max_iters=args.max_iters,
        progress_every=args.progress_every,
        progress_callback=lambda record: log_event("iteration", **record),
        event_batch_size=int(args.event_batch_size),
    )
    run_stats["search_log_path"] = str(search_log_path) if search_log_path is not None else None
    log_event("run_done", **run_stats)

    print("\n=== Expanding singleton root output if needed ===", flush=True)
    root_output_nodes, root_output_edges, postprocess_stats = postprocess_root_output(
        solver,
        enabled=bool(args.expand_singleton_root),
        target_size=int(args.singleton_root_target_size),
    )
    print(
        f"root postprocess action={postprocess_stats['action']} "
        f"nodes={postprocess_stats['input_num_nodes']}->{postprocess_stats['output_num_nodes']} "
        f"edges={postprocess_stats['input_num_edges']}->{postprocess_stats['output_num_edges']}",
        flush=True,
    )
    log_event("root_postprocess_done", **postprocess_stats)

    print("\n=== Writing outputs ===", flush=True)
    write_outputs(
        outdir,
        solver,
        run_stats,
        root_output_nodes=root_output_nodes,
        root_output_edges=root_output_edges,
        postprocess_stats=postprocess_stats,
    )
    log_event("outputs_written", outdir=str(outdir))

    if args.run_topk:
        print("\n=== Running rooted top-k subtree DP ===", flush=True)
        topk_stats = run_topk_subtree(args, outdir)
        log_event("topk_done", **topk_stats)

    if search_log_fh is not None:
        search_log_fh.close()

    print("\n=== Done ===", flush=True)

if __name__ == "__main__":
    main()
