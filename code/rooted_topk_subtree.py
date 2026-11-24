#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select a best connected rooted subtree from graph_search root-subgraph "
            "outputs using capped tree dynamic programming."
        )
    )
    p.add_argument("--subgraph-dir", required=True)
    p.add_argument("--outdir", default=None)
    p.add_argument("--nodes-file", default="root_subgraph_nodes.tsv.gz")
    p.add_argument("--edges-file", default="root_subgraph_selected_edges.tsv.gz")
    p.add_argument("--summary-file", default="root_subgraph_run_summary.json")
    p.add_argument("--root-node-id", default=None)

    p.add_argument("--node-id-col", default="table_key")
    p.add_argument("--edge-src-col", default="src_table_key")
    p.add_argument("--edge-tgt-col", default="tgt_table_key")
    p.add_argument(
        "--node-score-col",
        default="auto",
        help="Node reward column. auto tries utility_score, weight, pi, then node_d.",
    )
    p.add_argument(
        "--edge-cost-col",
        default="auto",
        help="Edge cost column. auto tries cost, then edge_weight.",
    )
    p.add_argument(
        "--edge-cost-weight",
        type=float,
        default=0.0,
        help="Objective subtracts this times edge cost for each selected edge.",
    )
    p.add_argument(
        "--edge-reward-weight",
        type=float,
        default=0.0,
        help="Objective adds this times (1 - edge cost) for each selected edge.",
    )
    p.add_argument(
        "--top-k",
        "--num-trees",
        dest="top_k",
        type=int,
        default=1,
        help="Number of distinct rooted subtree results to materialize from the DP.",
    )

    p.add_argument("--target-size", type=int, default=300)
    p.add_argument("--max-nodes", type=int, default=300)
    p.add_argument("--mode", choices=["exact", "at-most"], default="exact")
    p.add_argument(
        "--fallback-smaller",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For exact mode, use the largest feasible size <= target if exact target is unavailable.",
    )
    p.add_argument("--float-format", default="%.12g")
    return p.parse_args()

def now() -> float:
    return time.perf_counter()

def ensure_path(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")

def read_table(path: Path) -> pd.DataFrame:
    ensure_path(path, "table")
    try:
        if str(path).endswith(".gz"):
            return pd.read_csv(path, sep="\t", compression="gzip")
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def write_table(df: pd.DataFrame, path: Path, float_format: str) -> None:
    if str(path).endswith(".gz"):
        df.to_csv(
            path,
            sep="\t",
            index=False,
            compression="gzip",
            na_rep="",
            float_format=float_format,
        )
    else:
        df.to_csv(path, sep="\t", index=False, na_rep="", float_format=float_format)

def load_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def choose_col(df: pd.DataFrame, requested: str, candidates: List[str], desc: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Missing requested {desc} column {requested!r}.")
        return requested
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer {desc} column. Tried: {candidates}")

def safe_float(x, default: float = 0.0) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
    except Exception:
        pass
    return default

def validate_args(args: argparse.Namespace) -> None:
    if args.target_size <= 0:
        raise ValueError("--target-size must be positive.")
    if args.max_nodes <= 0:
        raise ValueError("--max-nodes must be positive.")
    if args.target_size > args.max_nodes:
        raise ValueError("--target-size must be <= --max-nodes.")
    if not math.isfinite(args.edge_cost_weight) or args.edge_cost_weight < 0:
        raise ValueError("--edge-cost-weight must be finite and nonnegative.")
    if not math.isfinite(args.edge_reward_weight) or args.edge_reward_weight < 0:
        raise ValueError("--edge-reward-weight must be finite and nonnegative.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

@dataclass(frozen=True)
class Candidate:
    score: float
    choices: Tuple[Tuple[str, int, int], ...]

@dataclass(frozen=True)
class RootSelection:
    rank: int
    size: int
    state_rank: int
    score: float
    reason: str

def infer_root(args: argparse.Namespace, summary: Dict[str, object]) -> str:
    if args.root_node_id:
        return str(args.root_node_id)
    root = summary.get("root_node_id")
    if root is None:
        raise ValueError("Missing --root-node-id and no root_node_id in summary file.")
    return str(root)

def orient_tree(
    node_ids: List[str],
    edges: pd.DataFrame,
    root: str,
    src_col: str,
    tgt_col: str,
) -> Tuple[Dict[str, List[str]], Dict[str, Optional[str]], Dict[str, int], Dict[str, int], List[str], List[str]]:
    node_set = set(node_ids)
    adj: Dict[str, List[Tuple[str, int]]] = {v: [] for v in node_ids}
    warnings: List[str] = []

    for eid, row in edges.iterrows():
        u = str(row[src_col])
        v = str(row[tgt_col])
        if u == v:
            warnings.append(f"ignored self-loop edge row {eid}: {u}")
            continue
        if u not in node_set or v not in node_set:
            warnings.append(f"ignored edge row {eid} with endpoint outside node table: {u}, {v}")
            continue
        adj[u].append((v, int(eid)))
        adj[v].append((u, int(eid)))

    if root not in node_set:
        raise ValueError(f"Root {root!r} is not present in node table.")

    parent: Dict[str, Optional[str]] = {root: None}
    parent_edge: Dict[str, int] = {}
    depth: Dict[str, int] = {root: 0}
    children: Dict[str, List[str]] = {v: [] for v in node_ids}
    order: List[str] = []
    skipped_cycle_edges = 0

    stack = [root]
    while stack:
        u = stack.pop()
        order.append(u)
        for v, eid in adj.get(u, []):
            if v == parent.get(u):
                continue
            if v in parent:
                skipped_cycle_edges += 1
                continue
            parent[v] = u
            parent_edge[v] = eid
            depth[v] = depth[u] + 1
            children[u].append(v)
            stack.append(v)

    unreachable = sorted(node_set - set(parent))
    if unreachable:
        warnings.append(f"{len(unreachable)} nodes are unreachable from root and were ignored.")
    if skipped_cycle_edges:
        warnings.append(f"ignored {skipped_cycle_edges} already-visited adjacency entries while rooting tree.")

    return children, parent, parent_edge, depth, order, warnings

def prune_candidates(candidates: List[Candidate], top_k: int) -> List[Candidate]:
    best_by_signature: Dict[Tuple[Tuple[str, int, int], ...], Candidate] = {}
    for candidate in candidates:
        previous = best_by_signature.get(candidate.choices)
        if previous is None or candidate.score > previous.score:
            best_by_signature[candidate.choices] = candidate

    ordered = sorted(
        best_by_signature.values(),
        key=lambda candidate: (-candidate.score, candidate.choices),
    )
    return ordered[:top_k]

def run_dp(
    root: str,
    order: List[str],
    children: Dict[str, List[str]],
    parent_edge: Dict[str, int],
    node_score: Dict[str, float],
    edge_value: Dict[int, float],
    max_nodes: int,
    top_k: int,
) -> Dict[str, Dict[int, List[Candidate]]]:
    dp: Dict[str, Dict[int, List[Candidate]]] = {}

    for u in reversed(order):
        states: Dict[int, List[Candidate]] = {
            1: [Candidate(score=node_score.get(u, 0.0), choices=())]
        }

        for child in children.get(u, []):
            child_states = dp[child]
            edge_bonus = edge_value.get(parent_edge[child], 0.0)

            candidates_by_size: Dict[int, List[Candidate]] = {
                size: list(candidates)
                for size, candidates in states.items()
            }

            for parent_size, parent_candidates in states.items():
                for child_size, child_candidates in child_states.items():
                    total_size = parent_size + child_size
                    if total_size > max_nodes:
                        continue
                    bucket = candidates_by_size.setdefault(total_size, [])
                    for parent_candidate in parent_candidates:
                        for child_rank, child_candidate in enumerate(child_candidates):
                            bucket.append(
                                Candidate(
                                    score=(
                                        parent_candidate.score
                                        + child_candidate.score
                                        + edge_bonus
                                    ),
                                    choices=parent_candidate.choices
                                    + ((child, child_size, child_rank),),
                                )
                            )

            states = {
                size: prune_candidates(candidates, top_k)
                for size, candidates in candidates_by_size.items()
                if candidates
            }

        dp[u] = states

    return dp

def choose_root_selections(
    root_states: Dict[int, List[Candidate]],
    target_size: int,
    mode: str,
    fallback_smaller: bool,
    top_k: int,
) -> List[RootSelection]:
    feasible = sorted(
        size for size, candidates in root_states.items()
        if size <= target_size and candidates
    )
    if not feasible:
        raise ValueError(f"No feasible rooted subtree size <= {target_size}.")

    if mode == "at-most":
        ranked: List[Tuple[int, int, Candidate]] = []
        for size in feasible:
            for state_rank, candidate in enumerate(root_states[size]):
                ranked.append((size, state_rank, candidate))
        ranked.sort(key=lambda item: (-item[2].score, -item[0], item[1]))
        return [
            RootSelection(
                rank=i + 1,
                size=size,
                state_rank=state_rank,
                score=candidate.score,
                reason="best_score_at_most_target",
            )
            for i, (size, state_rank, candidate) in enumerate(ranked[:top_k])
        ]

    if target_size in root_states and root_states[target_size]:
        size = target_size
        reason = "exact_target"
    elif fallback_smaller:
        size = max(feasible)
        reason = "fallback_largest_feasible_under_target"
    else:
        raise ValueError(f"Exact target size {target_size} is not feasible.")

    return [
        RootSelection(
            rank=state_rank + 1,
            size=size,
            state_rank=state_rank,
            score=candidate.score,
            reason=reason,
        )
        for state_rank, candidate in enumerate(root_states[size][:top_k])
    ]

def reconstruct_candidate(
    u: str,
    candidate: Candidate,
    dp: Dict[str, Dict[int, List[Candidate]]],
    parent_edge: Dict[str, int],
) -> Tuple[List[str], List[int]]:
    nodes = [u]
    edges: List[int] = []

    for child, child_size, child_rank in candidate.choices:
        try:
            child_candidate = dp[child][child_size][child_rank]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Missing child candidate for node={u} child={child} "
                f"size={child_size} rank={child_rank}"
            ) from exc

        child_nodes, child_edges = reconstruct_candidate(
            child,
            child_candidate,
            dp,
            parent_edge,
        )
        edges.append(parent_edge[child])
        edges.extend(child_edges)
        nodes.extend(child_nodes)

    return nodes, edges

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)

def main() -> None:
    args = parse_args()
    validate_args(args)
    t0 = now()

    subgraph_dir = Path(args.subgraph_dir)
    nodes_path = subgraph_dir / args.nodes_file
    edges_path = subgraph_dir / args.edges_file
    summary_path = subgraph_dir / args.summary_file
    summary = load_summary(summary_path)
    root = infer_root(args, summary)

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        dirname = f"topk_rooted_subtree_{args.mode}_k{args.target_size}_cap{args.max_nodes}"
        if args.top_k != 1:
            dirname += f"_top{args.top_k}"
        outdir = subgraph_dir / dirname
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Loading root subgraph ===", flush=True)
    print(f"nodes_path={nodes_path}", flush=True)
    print(f"edges_path={edges_path}", flush=True)
    print(f"root_node_id={root}", flush=True)

    nodes = read_table(nodes_path)
    edges = read_table(edges_path)
    if edges.empty:
        edges = pd.DataFrame(columns=[args.edge_src_col, args.edge_tgt_col, "cost"])

    if args.node_id_col not in nodes.columns:
        raise ValueError(f"{nodes_path} missing node id column {args.node_id_col!r}")
    for c in [args.edge_src_col, args.edge_tgt_col]:
        if c not in edges.columns:
            raise ValueError(f"{edges_path} missing edge endpoint column {c!r}")

    nodes = nodes.copy()
    edges = edges.copy().reset_index(drop=True)
    node_ids = nodes[args.node_id_col].astype(str).tolist()

    node_score_col = choose_col(
        nodes,
        args.node_score_col,
        ["utility_score", "weight", "pi", "node_d"],
        "node score",
    )
    edge_cost_col = choose_col(edges, args.edge_cost_col, ["cost", "edge_weight"], "edge cost")

    node_score = {
        str(node_id): safe_float(score, 0.0)
        for node_id, score in zip(node_ids, pd.to_numeric(nodes[node_score_col], errors="coerce"))
    }
    edge_cost = {
        int(i): safe_float(x, 0.0)
        for i, x in pd.to_numeric(edges[edge_cost_col], errors="coerce").items()
    }
    edge_value = {
        eid: (-args.edge_cost_weight * cost) + (args.edge_reward_weight * (1.0 - cost))
        for eid, cost in edge_cost.items()
    }

    children, parent, parent_edge, depth, order, warnings = orient_tree(
        node_ids,
        edges,
        root,
        args.edge_src_col,
        args.edge_tgt_col,
    )

    reachable_count = len(order)
    target_size = min(args.target_size, args.max_nodes, reachable_count)

    print("=== Running capped tree DP ===", flush=True)
    print(
        f"reachable_nodes={reachable_count} target_size={target_size} max_nodes={args.max_nodes} "
        f"mode={args.mode} top_k={args.top_k}",
        flush=True,
    )
    dp = run_dp(
        root=root,
        order=order,
        children=children,
        parent_edge=parent_edge,
        node_score=node_score,
        edge_value=edge_value,
        max_nodes=args.max_nodes,
        top_k=args.top_k,
    )

    selections = choose_root_selections(
        dp[root],
        target_size,
        args.mode,
        args.fallback_smaller,
        args.top_k,
    )

    print("=== Writing top-k subtrees ===", flush=True)
    nodes_out_path = outdir / "topk_subtree_nodes.tsv.gz"
    edges_out_path = outdir / "topk_subtree_edges.tsv.gz"
    node_ids_path = outdir / "topk_subtree_node_ids.txt"
    nodes_all_path = outdir / "topk_subtrees_nodes.tsv.gz"
    edges_all_path = outdir / "topk_subtrees_edges.tsv.gz"
    node_ids_all_path = outdir / "topk_subtrees_node_ids.tsv"
    results_table_path = outdir / "topk_subtree_results.tsv"
    summary_out_path = outdir / "topk_subtree_summary.json"

    all_nodes_out: List[pd.DataFrame] = []
    all_edges_out: List[pd.DataFrame] = []
    node_id_rows: List[Dict[str, object]] = []
    result_summaries: List[Dict[str, object]] = []
    result_table_rows: List[Dict[str, object]] = []
    previous_node_sets: List[set[str]] = []
    first_nodes_out: Optional[pd.DataFrame] = None
    first_edges_out: Optional[pd.DataFrame] = None
    first_node_order: Optional[List[str]] = None

    for selection in selections:
        candidate = dp[root][selection.size][selection.state_rank]
        selected_nodes_order, selected_edge_ids = reconstruct_candidate(
            root,
            candidate,
            dp,
            parent_edge,
        )

        selected_node_set = set(selected_nodes_order)
        order_map = {v: i for i, v in enumerate(selected_nodes_order)}
        edge_order_map = {eid: i for i, eid in enumerate(selected_edge_ids)}

        selected_parent = {root: ""}
        selected_depth = {root: 0}
        for v in selected_nodes_order[1:]:
            selected_parent[v] = parent.get(v) or ""
            selected_depth[v] = depth.get(v, -1)

        nodes_out = nodes[nodes[args.node_id_col].astype(str).isin(selected_node_set)].copy()
        nodes_out["topk_order"] = nodes_out[args.node_id_col].astype(str).map(order_map)
        nodes_out["topk_parent"] = nodes_out[args.node_id_col].astype(str).map(selected_parent)
        nodes_out["topk_depth"] = nodes_out[args.node_id_col].astype(str).map(selected_depth)
        nodes_out["topk_node_score"] = nodes_out[args.node_id_col].astype(str).map(node_score)
        nodes_out = nodes_out.sort_values("topk_order")

        edges_out = edges.loc[selected_edge_ids].copy()
        edges_out["topk_edge_order"] = edges_out.index.map(edge_order_map)
        edges_out["topk_edge_cost"] = edges_out.index.map(edge_cost)
        edges_out["topk_edge_contribution"] = edges_out.index.map(edge_value)
        edges_out = edges_out.sort_values("topk_edge_order")

        node_score_sum = float(sum(node_score.get(v, 0.0) for v in selected_nodes_order))
        edge_cost_sum = float(sum(edge_cost.get(eid, 0.0) for eid in selected_edge_ids))
        edge_reward_sum = float(sum(1.0 - edge_cost.get(eid, 0.0) for eid in selected_edge_ids))

        if previous_node_sets:
            node_jaccard_vs_rank1: Optional[float] = jaccard(selected_node_set, previous_node_sets[0])
            min_node_jaccard_vs_previous: Optional[float] = min(
                jaccard(selected_node_set, prev) for prev in previous_node_sets
            )
        else:
            node_jaccard_vs_rank1 = None
            min_node_jaccard_vs_previous = None
        previous_node_sets.append(selected_node_set)

        nodes_ranked = nodes_out.copy()
        nodes_ranked.insert(0, "topk_rank", selection.rank)
        nodes_ranked.insert(1, "topk_result_score", float(selection.score))
        all_nodes_out.append(nodes_ranked)

        edges_ranked = edges_out.copy()
        edges_ranked.insert(0, "topk_rank", selection.rank)
        edges_ranked.insert(1, "topk_result_score", float(selection.score))
        all_edges_out.append(edges_ranked)

        for i, node_id in enumerate(selected_nodes_order):
            node_id_rows.append(
                {
                    "topk_rank": selection.rank,
                    "topk_order": i,
                    args.node_id_col: node_id,
                }
            )

        rank_outputs: Dict[str, str] = {}
        if args.top_k > 1:
            rank_nodes_path = outdir / f"topk_subtree_rank{selection.rank:03d}_nodes.tsv.gz"
            rank_edges_path = outdir / f"topk_subtree_rank{selection.rank:03d}_edges.tsv.gz"
            rank_node_ids_path = outdir / f"topk_subtree_rank{selection.rank:03d}_node_ids.txt"
            write_table(nodes_out, rank_nodes_path, args.float_format)
            write_table(edges_out, rank_edges_path, args.float_format)
            with rank_node_ids_path.open("w", encoding="utf-8") as f:
                for v in selected_nodes_order:
                    f.write(f"{v}\n")
            rank_outputs = {
                "nodes": str(rank_nodes_path),
                "edges": str(rank_edges_path),
                "node_ids": str(rank_node_ids_path),
            }

        if selection.rank == 1:
            first_nodes_out = nodes_out
            first_edges_out = edges_out
            first_node_order = selected_nodes_order

        result_summary = {
            "rank": int(selection.rank),
            "selection_reason": selection.reason,
            "dp_size": int(selection.size),
            "dp_state_rank": int(selection.state_rank + 1),
            "selected_num_nodes": int(len(selected_nodes_order)),
            "selected_num_edges": int(len(selected_edge_ids)),
            "selected_objective": float(selection.score),
            "selected_node_score_sum": node_score_sum,
            "selected_edge_cost_sum": edge_cost_sum,
            "selected_edge_reward_sum": edge_reward_sum,
            "node_jaccard_vs_rank1": node_jaccard_vs_rank1,
            "min_node_jaccard_vs_previous": min_node_jaccard_vs_previous,
            "outputs": rank_outputs,
        }
        result_summaries.append(result_summary)

        result_table_rows.append(
            {
                k: v
                for k, v in result_summary.items()
                if k != "outputs"
            }
        )

    if first_nodes_out is None or first_edges_out is None or first_node_order is None:
        raise RuntimeError("No top-k rooted subtree candidates were selected.")

    write_table(first_nodes_out, nodes_out_path, args.float_format)
    write_table(first_edges_out, edges_out_path, args.float_format)
    with node_ids_path.open("w", encoding="utf-8") as f:
        for v in first_node_order:
            f.write(f"{v}\n")

    write_table(pd.concat(all_nodes_out, ignore_index=True), nodes_all_path, args.float_format)
    write_table(pd.concat(all_edges_out, ignore_index=True), edges_all_path, args.float_format)
    pd.DataFrame(node_id_rows).to_csv(node_ids_all_path, sep="\t", index=False)
    pd.DataFrame(result_table_rows).to_csv(results_table_path, sep="\t", index=False)

    first_result = result_summaries[0]
    result = {
        "subgraph_dir": str(subgraph_dir),
        "outdir": str(outdir),
        "root_node_id": root,
        "input_num_nodes": int(len(nodes)),
        "input_num_edges": int(len(edges)),
        "reachable_nodes_from_root": int(reachable_count),
        "target_size_requested": int(args.target_size),
        "max_nodes_cap": int(args.max_nodes),
        "mode": args.mode,
        "fallback_smaller": bool(args.fallback_smaller),
        "top_k_requested": int(args.top_k),
        "num_results": int(len(result_summaries)),
        "root_dp_num_feasible_sizes": int(len(dp[root])),
        "selection_reason": first_result["selection_reason"],
        "selected_num_nodes": first_result["selected_num_nodes"],
        "selected_num_edges": first_result["selected_num_edges"],
        "node_score_col": node_score_col,
        "edge_cost_col": edge_cost_col,
        "edge_cost_weight": float(args.edge_cost_weight),
        "edge_reward_weight": float(args.edge_reward_weight),
        "objective_formula": "sum(node_score) - edge_cost_weight * sum(edge_cost) + edge_reward_weight * sum(1 - edge_cost)",
        "selected_objective": first_result["selected_objective"],
        "selected_node_score_sum": first_result["selected_node_score_sum"],
        "selected_edge_cost_sum": first_result["selected_edge_cost_sum"],
        "selected_edge_reward_sum": first_result["selected_edge_reward_sum"],
        "results": result_summaries,
        "warnings": warnings,
        "elapsed_sec": now() - t0,
        "outputs": {
            "nodes": str(nodes_out_path),
            "edges": str(edges_out_path),
            "node_ids": str(node_ids_path),
            "all_nodes": str(nodes_all_path),
            "all_edges": str(edges_all_path),
            "all_node_ids": str(node_ids_all_path),
            "results_table": str(results_table_path),
            "summary": str(summary_out_path),
        },
    }

    if result_summaries:
        result_summaries[0]["outputs"] = {
            "nodes": str(nodes_out_path),
            "edges": str(edges_out_path),
            "node_ids": str(node_ids_path),
            **result_summaries[0].get("outputs", {}),
        }

    with summary_out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2), flush=True)
    print("=== Done ===", flush=True)

if __name__ == "__main__":
    main()
