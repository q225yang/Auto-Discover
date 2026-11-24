#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


CODE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run graph search and rooted top-k DP for a list of query roots."
    )
    parser.add_argument("--graph-dir", required=True, help="Directory with table_nodes.tsv.gz and table_edges.tsv.gz.")
    parser.add_argument("--roots-file", required=True, help="Text file with one root table key per line.")
    parser.add_argument("--outbase", required=True, help="Output directory for per-root results.")
    parser.add_argument("--top-k", type=int, default=50, help="Number of DP results per root.")
    parser.add_argument("--max-nodes", type=int, default=20, help="Maximum nodes in each DP result.")
    parser.add_argument("--dp-mode", choices=["exact", "at-most"], default="at-most")
    parser.add_argument("--edge-cost-weight", type=float, default=1.0)
    parser.add_argument("--singleton-root-target-size", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--initial-components-dir", help="Optional initial-component directory.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing graph-search and DP outputs.")
    return parser.parse_args()


def read_roots(path: Path) -> List[str]:
    roots: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        root = line.strip()
        if root and not root.startswith("#"):
            roots.append(root)
    if not roots:
        raise ValueError(f"No roots found in {path}")
    return roots


def root_dir_name(root: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._=-]+", "_", root).strip("_")
    digest = hashlib.sha1(root.encode("utf-8")).hexdigest()[:10]
    return f"root_{slug[:80]}_{digest}" if slug else f"root_{digest}"


def topk_dir_name(mode: str, max_nodes: int, top_k: int, edge_cost_weight: float) -> str:
    name = f"topk_rooted_subtree_{mode}_k{max_nodes}_cap{max_nodes}"
    if edge_cost_weight:
        name += f"_edgecost{edge_cost_weight:g}"
    if top_k != 1:
        name += f"_top{top_k}"
    return name


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": repr(exc)}


def run_root(args: argparse.Namespace, root: str) -> Dict[str, Any]:
    root_out = Path(args.outbase) / root_dir_name(root)
    root_out.mkdir(parents=True, exist_ok=True)

    graph_summary = root_out / "root_subgraph_run_summary.json"
    topk_out = root_out / topk_dir_name(args.dp_mode, args.max_nodes, args.top_k, args.edge_cost_weight)
    topk_summary = topk_out / "topk_subtree_summary.json"
    topk_out.mkdir(parents=True, exist_ok=True)

    profile: Dict[str, Any] = {
        "root": root,
        "outdir": str(root_out),
        "graph_summary": str(graph_summary),
        "topk_summary": str(topk_summary),
        "exit_code": 0,
        "graph_reused": False,
        "topk_reused": False,
        "graph_wall_sec": 0.0,
        "topk_wall_sec": 0.0,
    }
    total_t0 = time.perf_counter()

    if not (args.reuse_existing and graph_summary.exists()):
        graph_cmd = [
            sys.executable,
            str(CODE_DIR / "graph_search.py"),
            "--graph-dir",
            args.graph_dir,
            "--outdir",
            str(root_out),
            "--root-node-id",
            root,
            "--singleton-root-target-size",
            str(args.singleton_root_target_size),
            "--progress-every",
            str(args.progress_every),
            "--search-log-path",
            "",
        ]
        if args.initial_components_dir:
            initial_dir = Path(args.initial_components_dir)
            graph_cmd.extend(
                [
                    "--initial-components-path",
                    str(initial_dir / "initial_components.tsv.gz"),
                    "--initial-component-edges-path",
                    str(initial_dir / "initial_component_edges.tsv.gz"),
                ]
            )

        graph_t0 = time.perf_counter()
        with (root_out / "graph_search.log").open("w", encoding="utf-8") as log:
            log.write("CMD: " + " ".join(graph_cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(graph_cmd, stdout=log, stderr=subprocess.STDOUT)
        profile["graph_wall_sec"] = time.perf_counter() - graph_t0
        profile["graph_exit_code"] = proc.returncode
        if proc.returncode != 0:
            profile["exit_code"] = proc.returncode
            profile["total_wall_sec"] = time.perf_counter() - total_t0
            return profile
    else:
        profile["graph_reused"] = True
        profile["graph_exit_code"] = 0

    if not (args.reuse_existing and topk_summary.exists()):
        topk_cmd = [
            sys.executable,
            str(CODE_DIR / "rooted_topk_subtree.py"),
            "--subgraph-dir",
            str(root_out),
            "--outdir",
            str(topk_out),
            "--root-node-id",
            root,
            "--target-size",
            str(args.max_nodes),
            "--max-nodes",
            str(args.max_nodes),
            "--mode",
            args.dp_mode,
            "--top-k",
            str(args.top_k),
            "--edge-cost-weight",
            str(args.edge_cost_weight),
        ]
        topk_t0 = time.perf_counter()
        with (topk_out / "topk_dp.log").open("w", encoding="utf-8") as log:
            log.write("CMD: " + " ".join(topk_cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(topk_cmd, stdout=log, stderr=subprocess.STDOUT)
        profile["topk_wall_sec"] = time.perf_counter() - topk_t0
        profile["topk_exit_code"] = proc.returncode
        profile["exit_code"] = proc.returncode
    else:
        profile["topk_reused"] = True
        profile["topk_exit_code"] = 0

    graph_json = read_json(graph_summary)
    topk_json = read_json(topk_summary)
    run_stats = graph_json.get("run_stats", {})
    profile["graph_elapsed_sec"] = run_stats.get("elapsed_sec")
    profile["graph_iterations"] = run_stats.get("iterations")
    profile["root_output_num_nodes"] = graph_json.get("root_output_num_nodes")
    profile["root_output_num_selected_edges"] = graph_json.get("root_output_num_selected_edges")
    profile["topk_elapsed_sec"] = topk_json.get("elapsed_sec")
    profile["topk_num_results"] = topk_json.get("num_results")
    profile["total_wall_sec"] = time.perf_counter() - total_t0
    return profile


def main() -> None:
    args = parse_args()
    roots = read_roots(Path(args.roots_file))
    outbase = Path(args.outbase)
    outbase.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    profile_jsonl = outbase / "profile.jsonl"
    profile_jsonl.write_text("", encoding="utf-8")

    for root in roots:
        row = run_root(args, root)
        rows.append(row)
        with profile_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        status = "OK" if row.get("exit_code") == 0 else "FAIL"
        print(f"[{status}] {row['root']}", flush=True)

    summary = {
        "num_roots": len(rows),
        "num_success": sum(1 for row in rows if row.get("exit_code") == 0),
        "num_failures": sum(1 for row in rows if row.get("exit_code") != 0),
        "total_graph_wall_sec": sum(float(row.get("graph_wall_sec") or 0.0) for row in rows),
        "total_topk_wall_sec": sum(float(row.get("topk_wall_sec") or 0.0) for row in rows),
        "total_wall_sec_sum": sum(float(row.get("total_wall_sec") or 0.0) for row in rows),
        "profile_jsonl": str(profile_jsonl),
    }
    (outbase / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
