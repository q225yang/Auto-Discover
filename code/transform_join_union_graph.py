#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score join/union edges, merge graphs, and assign node utility weights."
    )

    p.add_argument("--join-dir", required=True)
    p.add_argument("--union-dir", required=True)
    p.add_argument("--dataset-cache", required=True)
    p.add_argument(
        "--outdir",
        required=True,
    )

    p.add_argument("--max-join-edges", type=int, default=None)
    p.add_argument("--max-union-edges", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--sample-debug-rows", type=int, default=30)
    p.add_argument(
        "--node-top-k",
        type=int,
        default=5,
        help="Use the mean of each node's top-K incident edge utility scores before converting to graph node weight.",
    )

    p.add_argument(
        "--graph-format-weight-mode",
        choices=["utility", "algorithm"],
        default="utility",
        help=(
            "utility writes graph-format node weight=utility_score and keeps edge_weight as the "
            "original edge strength; algorithm writes m - score costs for both nodes and edges."
        ),
    )

    return p.parse_args()

def ensure_path(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")

def read_edges(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    ensure_path(path, "edge file")

    df = pd.read_csv(path, sep="\t", compression="gzip", nrows=max_rows)

    required = [
        "src_table_key",
        "src_table_id",
        "tgt_table_key",
        "tgt_table_id",
        "edge_weight",
        "support_cols",
        "best_src_col_idx",
        "best_tgt_col_idx",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df["src_table_key"] = df["src_table_key"].astype(str)
    df["tgt_table_key"] = df["tgt_table_key"].astype(str)

    df["original_edge_weight"] = pd.to_numeric(df["edge_weight"], errors="coerce")
    df["edge_weight"] = pd.to_numeric(df["edge_weight"], errors="coerce")
    df["support_cols"] = pd.to_numeric(df["support_cols"], errors="coerce")
    df["best_src_col_idx"] = pd.to_numeric(df["best_src_col_idx"], errors="coerce")
    df["best_tgt_col_idx"] = pd.to_numeric(df["best_tgt_col_idx"], errors="coerce")

    return df

def read_nodes(path: Path) -> pd.DataFrame:
    ensure_path(path, "node file")

    df = pd.read_csv(path, sep="\t", compression="gzip")

    required = ["table_key", "table_id", "num_cols"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df["table_key"] = df["table_key"].astype(str)
    df["num_cols"] = pd.to_numeric(df["num_cols"], errors="coerce")

    return df

def pair_key_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [tuple(sorted((a, b))) for a, b in zip(df["src_table_key"], df["tgt_table_key"])],
        index=df.index,
    )

def load_node_col_counts(join_nodes_path: Path, union_nodes_path: Path) -> Dict[str, int]:
    jn = read_nodes(join_nodes_path)
    un = read_nodes(union_nodes_path)

    nodes = pd.concat([jn, un], ignore_index=True)
    nodes = nodes.dropna(subset=["num_cols"])

    out: Dict[str, int] = {}
    for key, sub in nodes.groupby("table_key"):
        out[str(key)] = int(sub["num_cols"].max())

    return out

def count_data_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return max(n - 1, 0)

def read_header_cols(path: Path) -> Tuple[List[str], str]:
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                header = f.readline().rstrip("\r\n")
            return header.split("\t"), enc
        except Exception:
            pass
    return [], "unknown"

def normalize_value_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.strip().str.lower()
    s = s.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "none": np.nan,
            "null": np.nan,
            "na": np.nan,
            "n/a": np.nan,
            "not available": np.nan,
        }
    )
    return s.dropna()

NULL_VALUE_TOKENS = ["", "nan", "none", "null", "na", "n/a", "not available"]

def arrow_column_value_counts(
    path: Path,
    *,
    col_idx: int,
    expected_cols: int,
    sep: str,
    encoding: str,
) -> Optional[pd.Series]:
    if pa is None or pc is None or pacsv is None:
        return None
    if len(sep) != 1:
        return None

    col_name = f"f{int(col_idx)}"
    try:
        read_options = pacsv.ReadOptions(
            use_threads=False,
            skip_rows=1,
            column_names=[f"f{i}" for i in range(int(expected_cols))],
            encoding=encoding,
        )
        parse_options = pacsv.ParseOptions(delimiter=sep)
        convert_options = pacsv.ConvertOptions(
            include_columns=[col_name],
            column_types={col_name: pa.string()},
            strings_can_be_null=True,
            null_values=NULL_VALUE_TOKENS,
        )
        table = pacsv.read_csv(
            path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
        arr = table.column(0).combine_chunks()
        arr = pc.utf8_lower(pc.utf8_trim_whitespace(arr))
        tokens = pa.array(NULL_VALUE_TOKENS, type=pa.string())
        valid = pc.and_(
            pc.invert(pc.is_null(arr)),
            pc.invert(pc.is_in(arr, value_set=tokens)),
        )
        arr = pc.filter(arr, valid)
        counts = pc.value_counts(arr)
        values = counts.field("values").to_pylist()
        freqs = counts.field("counts").to_numpy(zero_copy_only=False)
        return pd.Series(freqs, index=pd.Index(values, dtype="object"))
    except Exception:
        return None

def union_custom_score(Ra: float, Rb: float, Ca: float, Cb: float, m: float) -> float:
    if any(pd.isna(x) for x in [Ra, Rb, Ca, Cb, m]):
        return np.nan
    if Ra <= 0 or Rb <= 0 or Ca <= 0 or Cb <= 0:
        return np.nan

    m = min(float(m), float(Ca), float(Cb))

    row_balance = 2.0 * min(Ra, Rb) / (Ra + Rb)

    numerator = Ra * (Cb - m) + Rb * (Ca - m)
    denominator = Ra * (Ca - m) + Rb * (Cb - m) + 1.0
    penalty = 1.0 / ((numerator / denominator) + 1.0)

    return row_balance * penalty

def join_explosion_penalty(Rj: float, Ra: float, Rb: float) -> float:
    if any(pd.isna(x) for x in [Rj, Ra, Rb]):
        return np.nan
    if Ra <= 0 or Rb <= 0 or Rj < 0:
        return np.nan

    base = max(Ra, Rb)
    excess = max((Rj / base) - 1.0, 0.0)
    return 1.0 / (1.0 + excess)

def join_custom_score(
    Ra: float,
    Rb: float,
    Ca: float,
    Cb: float,
    ma: float,
    mb: float,
    Rj: float,
) -> float:
    if any(pd.isna(x) for x in [Ra, Rb, Ca, Cb, ma, mb, Rj]):
        return np.nan
    if Ra <= 0 or Rb <= 0 or Ca <= 0 or Cb <= 0:
        return np.nan
    if ma < 0 or mb < 0 or Rj < 0:
        return np.nan

    ma = min(ma, Ra)
    mb = min(mb, Rb)

    coverage = math.sqrt((ma * mb) / (Ra * Rb))

    numerator = Ca * (Ra - ma) + Cb * (Rb - mb)
    denominator = ma * (Ca - 1.0) + mb * (Cb - 1.0) + 1.0
    sparsity_penalty = 1.0 / ((numerator / denominator) + 1.0)

    explosion = join_explosion_penalty(Rj, Ra, Rb)

    return coverage * sparsity_penalty * explosion

def stat_summary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []

    for c in cols:
        if c not in df.columns:
            continue

        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        if len(x) == 0:
            rows.append(
                {
                    "metric": c,
                    "count": 0,
                    "min": np.nan,
                    "p25": np.nan,
                    "median": np.nan,
                    "mean": np.nan,
                    "p75": np.nan,
                    "p90": np.nan,
                    "p95": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "metric": c,
                    "count": int(len(x)),
                    "min": float(x.min()),
                    "p25": float(x.quantile(0.25)),
                    "median": float(x.median()),
                    "mean": float(x.mean()),
                    "p75": float(x.quantile(0.75)),
                    "p90": float(x.quantile(0.90)),
                    "p95": float(x.quantile(0.95)),
                    "p99": float(x.quantile(0.99)),
                    "max": float(x.max()),
                }
            )

    return pd.DataFrame(rows)

class JoinUnionScoreAnalyzer:
    def __init__(
        self,
        join_dir: Path,
        union_dir: Path,
        dataset_cache: Path,
    ):
        self.join_dir = join_dir
        self.union_dir = union_dir
        self.dataset_cache = dataset_cache
        self.tables_dir = dataset_cache / "tables"
        self.manifest_path = dataset_cache / "manifest_tables.jsonl"

        ensure_path(self.manifest_path, "manifest_tables.jsonl")
        ensure_path(self.tables_dir, "tables directory")

        self.manifest_df = pd.read_json(self.manifest_path, lines=True)

        if "table_key" not in self.manifest_df.columns:
            raise ValueError(f"{self.manifest_path} missing table_key column")

        self.manifest_df["table_key"] = self.manifest_df["table_key"].astype(str)

        self.manifest_by_key = {
            str(row["table_key"]): row for _, row in self.manifest_df.iterrows()
        }

        self.node_num_cols = load_node_col_counts(
            join_dir / "table_nodes.tsv.gz",
            union_dir / "table_nodes.tsv.gz",
        )
        self._value_count_cache: Dict[Tuple[str, int], Tuple[Optional[pd.Series], str]] = {}
        self._value_count_cache_keys: set[Tuple[str, int]] = set()

    def resolve_table_path(self, table_key: str) -> Optional[Path]:
        table_key = str(table_key)

        row = self.manifest_by_key.get(table_key)
        if row is None:
            return None

        candidates: List[str] = []

        for col in ["table_path", "csv_path", "table_id"]:
            if col in row.index and pd.notna(row[col]):
                candidates.append(str(row[col]))

        for cand in candidates:
            p = Path(cand)
            if p.exists():
                return p

            p_cache = self.dataset_cache / cand
            if p_cache.exists():
                return p_cache

            p2 = self.tables_dir / cand
            if p2.exists():
                return p2

            base = Path(cand).name
            matches = list(self.tables_dir.rglob(base))
            if matches:
                return matches[0]

        return None

    @lru_cache(maxsize=None)
    def get_table_info(self, table_key: str) -> Dict[str, object]:
        table_key = str(table_key)

        path = self.resolve_table_path(table_key)
        graph_num_cols = self.node_num_cols.get(table_key, np.nan)

        if path is None:
            return {
                "table_key": table_key,
                "path": None,
                "num_rows": np.nan,
                "num_cols": graph_num_cols,
                "encoding": None,
                "columns": [],
                "status": "missing_table_path",
            }

        cols, enc = read_header_cols(path)

        try:
            nrows = count_data_rows(path)
        except Exception:
            nrows = np.nan

        if pd.isna(graph_num_cols):
            num_cols = len(cols) if cols else np.nan
        else:
            num_cols = int(graph_num_cols)

        return {
            "table_key": table_key,
            "path": str(path),
            "num_rows": nrows,
            "num_cols": num_cols,
            "encoding": enc,
            "columns": cols,
            "status": "ok",
        }

    def read_table_column_values(self, table_key: str, col_idx: int) -> Tuple[Optional[pd.Series], str]:
        table_key = str(table_key)

        info = self.get_table_info(table_key)
        path = info["path"]

        if path is None:
            return None, "missing_table_path"

        if pd.isna(info["num_cols"]):
            return None, "unknown_num_cols"

        col_idx = int(col_idx)

        if col_idx < 0 or col_idx >= int(info["num_cols"]):
            return None, "column_index_out_of_range"

        encodings = []
        if info.get("encoding"):
            encodings.append(str(info["encoding"]))

        encodings.extend(["utf-8", "utf-8-sig", "latin1", "cp1252"])
        encodings = list(dict.fromkeys(encodings))

        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            sep_candidates = [",", "\t", ";", "|"]
        elif suffix in {".tsv", ".tab"}:
            sep_candidates = ["\t", ",", ";", "|"]
        else:
            sep_candidates = ["\t", ",", ";", "|"]

        expected_cols = int(info["num_cols"])

        for sep in sep_candidates:
            for enc in encodings:
                try:
                    with open(path, "r", encoding=enc, errors="replace", newline="") as header_f:
                        header_row = next(csv.reader(header_f, delimiter=sep), [])

                    if expected_cols > 1 and len(header_row) < expected_cols:
                        continue
                    if col_idx >= len(header_row):
                        continue

                    df = pd.read_csv(
                        path,
                        sep=sep,
                        encoding=enc,
                        usecols=[col_idx],
                        dtype=str,
                        keep_default_na=False,
                        on_bad_lines="skip",
                        engine="c",
                    )

                    s = normalize_value_series(df.iloc[:, 0])
                    return s, "ok"

                except Exception:
                    continue

        return None, "unreadable_column"

    def read_table_column_value_counts(self, table_key: str, col_idx: int) -> Tuple[Optional[pd.Series], str]:
        key = (str(table_key), int(col_idx))
        if key in self._value_count_cache:
            return self._value_count_cache[key]

        status = "ok"
        result: Tuple[Optional[pd.Series], str]

        info = self.get_table_info(key[0])
        path = info.get("path")
        expected_cols = info.get("num_cols")
        if path is not None and pd.notna(expected_cols):
            path_obj = Path(str(path))
            suffix = path_obj.suffix.lower()
            if suffix == ".csv":
                sep_candidates = [",", "\t", ";", "|"]
            elif suffix in {".tsv", ".tab"}:
                sep_candidates = ["\t", ",", ";", "|"]
            else:
                sep_candidates = ["\t", ",", ";", "|"]

            encodings = []
            if info.get("encoding"):
                encodings.append(str(info["encoding"]))
            encodings.extend(["utf-8", "utf-8-sig", "latin1", "cp1252"])
            encodings = list(dict.fromkeys(encodings))

            for sep in sep_candidates:
                for enc in encodings:
                    try:
                        with open(path_obj, "r", encoding=enc, errors="replace", newline="") as header_f:
                            header_row = next(csv.reader(header_f, delimiter=sep), [])
                        if int(expected_cols) > 1 and len(header_row) < int(expected_cols):
                            continue
                        if key[1] >= len(header_row):
                            continue
                    except Exception:
                        continue

                    vc = arrow_column_value_counts(
                        path_obj,
                        col_idx=key[1],
                        expected_cols=int(expected_cols),
                        sep=sep,
                        encoding=enc,
                    )
                    if vc is not None:
                        result = (vc, "ok")
                        if key in self._value_count_cache_keys:
                            self._value_count_cache[key] = result
                        return result

        values, status = self.read_table_column_values(key[0], key[1])
        if status != "ok" or values is None:
            result = (None, status)
        else:
            result = (values.value_counts(), "ok")

        if key in self._value_count_cache_keys:
            self._value_count_cache[key] = result

        return result

    def add_table_size_columns(self, edges: pd.DataFrame) -> pd.DataFrame:
        out = edges.copy()

        src_infos = [self.get_table_info(k) for k in out["src_table_key"].astype(str)]
        tgt_infos = [self.get_table_info(k) for k in out["tgt_table_key"].astype(str)]

        out["Ra"] = [x["num_rows"] for x in src_infos]
        out["Rb"] = [x["num_rows"] for x in tgt_infos]
        out["Ca"] = [x["num_cols"] for x in src_infos]
        out["Cb"] = [x["num_cols"] for x in tgt_infos]

        out["src_table_path"] = [x["path"] for x in src_infos]
        out["tgt_table_path"] = [x["path"] for x in tgt_infos]
        out["src_table_info_status"] = [x["status"] for x in src_infos]
        out["tgt_table_info_status"] = [x["status"] for x in tgt_infos]

        return out

    def score_union(self, union_edges: pd.DataFrame) -> pd.DataFrame:
        df = self.add_table_size_columns(union_edges)

        df["m"] = pd.to_numeric(df["support_cols"], errors="coerce")
        df["m_clipped"] = df[["m", "Ca", "Cb"]].min(axis=1)

        df["unmatched_cols_a"] = df["Ca"] - df["m_clipped"]
        df["unmatched_cols_b"] = df["Cb"] - df["m_clipped"]

        df["row_balance"] = [
            2.0 * min(Ra, Rb) / (Ra + Rb)
            if pd.notna(Ra) and pd.notna(Rb) and Ra > 0 and Rb > 0
            else np.nan
            for Ra, Rb in df[["Ra", "Rb"]].itertuples(index=False, name=None)
        ]

        df["row_ratio_min_over_max"] = [
            min(Ra, Rb) / max(Ra, Rb)
            if pd.notna(Ra) and pd.notna(Rb) and Ra > 0 and Rb > 0
            else np.nan
            for Ra, Rb in df[["Ra", "Rb"]].itertuples(index=False, name=None)
        ]

        df["column_match_ratio"] = df["m_clipped"] / df[["Ca", "Cb"]].min(axis=1)

        df["custom_union_score"] = [
            union_custom_score(Ra, Rb, Ca, Cb, m)
            for Ra, Rb, Ca, Cb, m in df[
                ["Ra", "Rb", "Ca", "Cb", "m_clipped"]
            ].itertuples(index=False, name=None)
        ]

        df["relation_type"] = "union"
        df["final_score"] = df["custom_union_score"]

        return df

    def estimate_join_ma_mb_rj(
        self,
        src_key: str,
        tgt_key: str,
        src_col_idx: int,
        tgt_col_idx: int,
    ) -> Tuple[float, float, float, str]:
        if pd.isna(src_col_idx) or pd.isna(tgt_col_idx):
            return np.nan, np.nan, np.nan, "missing_column_index"

        vc_a, status_a = self.read_table_column_value_counts(str(src_key), int(src_col_idx))
        if status_a != "ok" or vc_a is None:
            return np.nan, np.nan, np.nan, f"src_{status_a}"

        vc_b, status_b = self.read_table_column_value_counts(str(tgt_key), int(tgt_col_idx))
        if status_b != "ok" or vc_b is None:
            return np.nan, np.nan, np.nan, f"tgt_{status_b}"

        if len(vc_a) == 0 or len(vc_b) == 0:
            return 0.0, 0.0, 0.0, "ok"
        common_values = vc_a.index.intersection(vc_b.index)

        if len(common_values) == 0:
            return 0.0, 0.0, 0.0, "ok"

        ma = float(vc_a.loc[common_values].sum())
        mb = float(vc_b.loc[common_values].sum())
        Rj = float((vc_a.loc[common_values] * vc_b.loc[common_values]).sum())

        return ma, mb, Rj, "ok"

    def score_join(self, join_edges: pd.DataFrame, progress_every: int = 500) -> pd.DataFrame:
        df = self.add_table_size_columns(join_edges)

        ma_list: List[float] = []
        mb_list: List[float] = []
        rj_list: List[float] = []
        status_list: List[str] = []

        n = len(df)

        endpoint_counts: Counter[Tuple[str, int]] = Counter()
        for src_key, tgt_key, src_col_idx, tgt_col_idx in df[
            ["src_table_key", "tgt_table_key", "best_src_col_idx", "best_tgt_col_idx"]
        ].itertuples(index=False, name=None):
            if pd.notna(src_col_idx):
                endpoint_counts[(str(src_key), int(src_col_idx))] += 1
            if pd.notna(tgt_col_idx):
                endpoint_counts[(str(tgt_key), int(tgt_col_idx))] += 1

        cache_min_uses = 50
        self._value_count_cache = {}
        self._value_count_cache_keys = {
            key for key, count in endpoint_counts.items() if count >= cache_min_uses
        }
        print(
            f"value-count cache endpoints: {len(self._value_count_cache_keys)}/{len(endpoint_counts)} "
            f"with min_uses={cache_min_uses}",
            flush=True,
        )

        for i, row in enumerate(df.itertuples(index=False), start=1):
            ma, mb, Rj, status = self.estimate_join_ma_mb_rj(
                row.src_table_key,
                row.tgt_table_key,
                row.best_src_col_idx,
                row.best_tgt_col_idx,
            )

            ma_list.append(ma)
            mb_list.append(mb)
            rj_list.append(Rj)
            status_list.append(status)

            if i == 1 or i == n or (progress_every > 0 and i % progress_every == 0):
                print(f"processed join edges: {i}/{n}", flush=True)

        df["ma"] = ma_list
        df["mb"] = mb_list
        df["Rj"] = rj_list
        df["join_eval_status"] = status_list
        df["join_eval_valid"] = df["join_eval_status"].eq("ok")

        df["ma_clipped"] = np.nan
        df["mb_clipped"] = np.nan

        valid = df["join_eval_valid"]

        df.loc[valid, "ma_clipped"] = df.loc[valid, ["ma", "Ra"]].min(axis=1)
        df.loc[valid, "mb_clipped"] = df.loc[valid, ["mb", "Rb"]].min(axis=1)

        df["ma_over_Ra"] = np.nan
        df["mb_over_Rb"] = np.nan

        df.loc[valid, "ma_over_Ra"] = df.loc[valid, "ma_clipped"] / df.loc[valid, "Ra"]
        df.loc[valid, "mb_over_Rb"] = df.loc[valid, "mb_clipped"] / df.loc[valid, "Rb"]

        df["row_overlap_geom"] = np.nan
        df.loc[valid, "row_overlap_geom"] = np.sqrt(
            (df.loc[valid, "ma_clipped"] * df.loc[valid, "mb_clipped"])
            / (df.loc[valid, "Ra"] * df.loc[valid, "Rb"])
        )

        df["join_explosion_ratio"] = np.nan
        df.loc[valid, "join_explosion_ratio"] = (
            df.loc[valid, "Rj"] / df.loc[valid, ["Ra", "Rb"]].max(axis=1)
        )

        df["join_explosion_penalty"] = [
            join_explosion_penalty(Rj, Ra, Rb) if is_valid else np.nan
            for Rj, Ra, Rb, is_valid in df[
                ["Rj", "Ra", "Rb", "join_eval_valid"]
            ].itertuples(index=False, name=None)
        ]

        df["custom_join_score"] = [
            join_custom_score(Ra, Rb, Ca, Cb, ma, mb, Rj) if is_valid else np.nan
            for Ra, Rb, Ca, Cb, ma, mb, Rj, is_valid in df[
                ["Ra", "Rb", "Ca", "Cb", "ma_clipped", "mb_clipped", "Rj", "join_eval_valid"]
            ].itertuples(index=False, name=None)
        ]

        df["relation_type"] = "join"
        df["final_score"] = df["custom_join_score"]

        return df

def build_merged_nodes(
    join_nodes: pd.DataFrame,
    union_nodes: pd.DataFrame,
) -> pd.DataFrame:
    join_nodes = join_nodes.copy()
    union_nodes = union_nodes.copy()

    join_nodes["present_in_join_graph"] = True
    join_nodes["present_in_union_graph"] = False

    union_nodes["present_in_join_graph"] = False
    union_nodes["present_in_union_graph"] = True

    nodes = pd.concat([join_nodes, union_nodes], ignore_index=True, sort=False)

    agg = {}
    for c in nodes.columns:
        if c == "table_key":
            continue
        elif c in ["present_in_join_graph", "present_in_union_graph"]:
            agg[c] = "max"
        elif c == "num_cols":
            agg[c] = "max"
        else:
            agg[c] = "first"

    out = nodes.groupby("table_key", as_index=False).agg(agg)

    out["present_in_join_graph"] = out["present_in_join_graph"].astype(bool)
    out["present_in_union_graph"] = out["present_in_union_graph"].astype(bool)

    return out

def build_merged_edges(
    union_scored: pd.DataFrame,
    join_scored: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    u = union_scored.copy()
    j = join_scored.copy()

    u["pair_key"] = pair_key_series(u)
    j["pair_key"] = pair_key_series(j)

    union_pairs = set(u["pair_key"])

    j_only = j[~j["pair_key"].isin(union_pairs)].copy()

    merged = pd.concat([u, j_only], ignore_index=True, sort=False)
    merged = add_edge_utility_scores(merged)

    stats = {
        "union_edges_kept": int(len(u)),
        "join_edges_input": int(len(j)),
        "join_edges_removed_because_union_overlap": int(len(j) - len(j_only)),
        "join_edges_kept_nonoverlap": int(len(j_only)),
        "merged_edges": int(len(merged)),
        "overlap_pairs_treated_as_union": int(len(j) - len(j_only)),
        "merged_edges_missing_utility_score": int(merged["utility_score"].isna().sum()),
    }

    return merged, stats

def add_edge_utility_scores(edges: pd.DataFrame) -> pd.DataFrame:
    out = edges.copy()

    if "relation_type" not in out.columns:
        if "edge_type" in out.columns:
            out["relation_type"] = out["edge_type"]
        else:
            out["relation_type"] = ""

    rel = out["relation_type"].astype(str).str.lower()
    out["utility_score"] = np.nan

    if "custom_union_score" in out.columns:
        mask = rel.eq("union")
        out.loc[mask, "utility_score"] = pd.to_numeric(
            out.loc[mask, "custom_union_score"], errors="coerce"
        )

    if "custom_join_score" in out.columns:
        mask = rel.eq("join")
        out.loc[mask, "utility_score"] = pd.to_numeric(
            out.loc[mask, "custom_join_score"], errors="coerce"
        )

    missing = out["utility_score"].isna()
    for fallback in ["final_score", "edge_weight", "original_edge_weight"]:
        if not missing.any() or fallback not in out.columns:
            continue
        out.loc[missing, "utility_score"] = pd.to_numeric(
            out.loc[missing, fallback], errors="coerce"
        )
        missing = out["utility_score"].isna()

    out["final_score"] = out["utility_score"]

    return out

def build_node_utility_weights(
    nodes: pd.DataFrame,
    merged_edges: pd.DataFrame,
    top_k: int = 5,
    score_col: str = "utility_score",
) -> pd.DataFrame:
    out = nodes.copy()

    scores_by_node: Dict[str, List[float]] = {}

    if score_col not in merged_edges.columns:
        return out.assign(utility_score=0.0, weight=0.0, utility_score_degree=0)

    edge_score_cols = ["src_table_key", "tgt_table_key", score_col]
    for src_key, tgt_key, raw_score in merged_edges[edge_score_cols].itertuples(index=False, name=None):
        score = pd.to_numeric(raw_score, errors="coerce")
        if pd.isna(score) or not math.isfinite(float(score)):
            continue

        score_f = float(score)
        scores_by_node.setdefault(str(src_key), []).append(score_f)
        scores_by_node.setdefault(str(tgt_key), []).append(score_f)

    utility_scores: List[float] = []
    degrees: List[int] = []

    for table_key in out["table_key"].astype(str):
        vals = sorted(scores_by_node.get(table_key, []), reverse=True)
        degrees.append(len(vals))
        if vals:
            utility_scores.append(float(np.mean(vals[:top_k])))
        else:
            utility_scores.append(0.0)

    out["utility_score"] = utility_scores
    out["utility_score_degree"] = degrees

    return out

def apply_algorithm_weights(
    merged_edges: pd.DataFrame,
    merged_nodes: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    edges = merged_edges.copy()
    nodes = merged_nodes.copy()

    original_edge_col = "original_edge_weight" if "original_edge_weight" in edges.columns else "edge_weight"
    if original_edge_col not in edges.columns:
        raise ValueError("Cannot compute algorithm weights without original_edge_weight or edge_weight.")

    original_edge_score = pd.to_numeric(edges[original_edge_col], errors="coerce")
    finite_edge_score = original_edge_score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_edge_score) == 0:
        raise ValueError("Cannot compute algorithm weights: no finite original edge scores.")

    m = float(finite_edge_score.max())

    if "original_edge_weight" not in edges.columns:
        edges["original_edge_weight"] = original_edge_score

    converted_edge_weight = m - original_edge_score
    converted_edge_weight = converted_edge_weight.mask(converted_edge_weight.abs() <= 1e-12, 0.0)
    edges["edge_weight"] = converted_edge_weight.clip(lower=0.0)

    if "utility_score" not in nodes.columns:
        nodes["utility_score"] = pd.to_numeric(nodes.get("weight", 0.0), errors="coerce")

    node_utility = pd.to_numeric(nodes["utility_score"], errors="coerce").fillna(0.0)
    converted_node_weight = m - node_utility
    converted_node_weight = converted_node_weight.mask(converted_node_weight.abs() <= 1e-12, 0.0)
    nodes["weight"] = converted_node_weight.clip(lower=0.0)

    edge_weight_numeric = pd.to_numeric(edges["edge_weight"], errors="coerce")
    node_weight_numeric = pd.to_numeric(nodes["weight"], errors="coerce")

    stats = {
        "algorithm_weight_cap_m": m,
        "algorithm_weight_source_col": original_edge_col,
        "edge_original_score_min": float(finite_edge_score.min()),
        "edge_original_score_max": float(finite_edge_score.max()),
        "edge_weight_min": float(edge_weight_numeric.min()),
        "edge_weight_max": float(edge_weight_numeric.max()),
        "node_utility_min": float(node_utility.min()),
        "node_utility_max": float(node_utility.max()),
        "node_weight_min": float(node_weight_numeric.min()),
        "node_weight_max": float(node_weight_numeric.max()),
    }

    return edges, nodes, stats

GRAPH_EDGE_COLUMNS = [
    "src_table_key",
    "src_table_id",
    "tgt_table_key",
    "tgt_table_id",
    "edge_type",
    "src_cols",
    "candidate_edges",
    "support_cols",
    "edge_weight",
    "best_col_score",
    "avg_support_weight",
    "best_src_col_idx",
    "best_tgt_col_idx",
    "meta_json",
    "utility_score",
]

GRAPH_NODE_COLUMNS = [
    "table_key",
    "table_id",
    "num_cols",
    "utility_score",
    "weight",
]

def build_graph_format_edges(merged_edges: pd.DataFrame) -> pd.DataFrame:
    out = merged_edges.copy()

    if "edge_type" not in out.columns:
        out["edge_type"] = out.get("relation_type", "")

    if "utility_score" not in out.columns:
        out = add_edge_utility_scores(out)

    for col in GRAPH_EDGE_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    out["edge_type"] = out["edge_type"].fillna(out.get("relation_type", "")).astype(str)
    out["meta_json"] = out["meta_json"].fillna("").astype(str)

    for col in [
        "src_cols",
        "candidate_edges",
        "support_cols",
        "best_src_col_idx",
        "best_tgt_col_idx",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round().astype("Int64")

    for col in [
        "edge_weight",
        "best_col_score",
        "avg_support_weight",
        "utility_score",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out[GRAPH_EDGE_COLUMNS]

def build_graph_format_nodes(weighted_nodes: pd.DataFrame) -> pd.DataFrame:
    out = weighted_nodes.copy()

    for col in GRAPH_NODE_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col not in {"num_cols", "utility_score", "weight"} else np.nan

    out["num_cols"] = pd.to_numeric(out["num_cols"], errors="coerce").round().astype("Int64")
    out["utility_score"] = pd.to_numeric(out["utility_score"], errors="coerce").fillna(0.0)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)

    return out[GRAPH_NODE_COLUMNS]

def write_graph_format_outputs(
    outdir: Path,
    merged_edges: pd.DataFrame,
    merged_nodes: pd.DataFrame,
) -> Dict[str, int]:
    graph_edges = build_graph_format_edges(merged_edges)
    graph_nodes = build_graph_format_nodes(merged_nodes)

    graph_edges.to_csv(
        outdir / "table_edges.tsv.gz",
        sep="	",
        index=False,
        compression="gzip",
        na_rep="",
        float_format="%.6f",
    )
    graph_nodes.to_csv(
        outdir / "table_nodes.tsv.gz",
        sep="	",
        index=False,
        compression="gzip",
        na_rep="",
        float_format="%.6f",
    )

    return {
        "table_edges_rows": int(len(graph_edges)),
        "table_nodes_rows": int(len(graph_nodes)),
        "table_edges_missing_utility_score": int(graph_edges["utility_score"].isna().sum()),
        "table_nodes_zero_weight": int(graph_nodes["weight"].eq(0.0).sum()),
    }

def build_overlap_table(union_scored: pd.DataFrame, join_scored: pd.DataFrame) -> pd.DataFrame:
    u = union_scored.copy()
    j = join_scored.copy()

    u["pair_key"] = pair_key_series(u)
    j["pair_key"] = pair_key_series(j)

    u_keep = [
        "pair_key",
        "src_table_key",
        "src_table_id",
        "tgt_table_key",
        "tgt_table_id",
        "original_edge_weight",
        "custom_union_score",
        "row_balance",
        "column_match_ratio",
        "support_cols",
        "m_clipped",
        "Ra",
        "Rb",
        "Ca",
        "Cb",
    ]

    j_keep = [
        "pair_key",
        "original_edge_weight",
        "custom_join_score",
        "join_eval_status",
        "join_eval_valid",
        "ma",
        "mb",
        "Rj",
        "row_overlap_geom",
        "join_explosion_ratio",
        "join_explosion_penalty",
    ]

    u_keep = [c for c in u_keep if c in u.columns]
    j_keep = [c for c in j_keep if c in j.columns]

    out = u[u_keep].merge(
        j[j_keep],
        on="pair_key",
        how="inner",
        suffixes=("_union", "_join"),
    )

    out["custom_join_minus_union"] = out["custom_join_score"] - out["custom_union_score"]
    out["custom_join_over_union"] = out["custom_join_score"] / out[
        "custom_union_score"
    ].replace(0, np.nan)

    return out

def write_samples(
    outdir: Path,
    union_scored: pd.DataFrame,
    join_scored: pd.DataFrame,
    merged_edges: pd.DataFrame,
    n: int,
) -> None:
    if n <= 0:
        return

    sample_dir = outdir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if len(union_scored) > 0:
        cols = [
            "src_table_key",
            "src_table_id",
            "tgt_table_key",
            "tgt_table_id",
            "original_edge_weight",
            "custom_union_score",
            "row_balance",
            "column_match_ratio",
            "support_cols",
            "m_clipped",
            "Ra",
            "Rb",
            "Ca",
            "Cb",
            "unmatched_cols_a",
            "unmatched_cols_b",
        ]
        cols = [c for c in cols if c in union_scored.columns]

        union_scored.sort_values(
            "custom_union_score", ascending=False, na_position="last"
        )[cols].head(n).to_csv(sample_dir / "union_top_custom_score.csv", index=False)

        union_scored.sort_values(
            "custom_union_score", ascending=True, na_position="last"
        )[cols].head(n).to_csv(sample_dir / "union_bottom_custom_score.csv", index=False)

    if len(join_scored) > 0:
        cols = [
            "src_table_key",
            "src_table_id",
            "tgt_table_key",
            "tgt_table_id",
            "original_edge_weight",
            "custom_join_score",
            "join_eval_status",
            "join_eval_valid",
            "best_src_col_idx",
            "best_tgt_col_idx",
            "ma",
            "mb",
            "Rj",
            "ma_over_Ra",
            "mb_over_Rb",
            "row_overlap_geom",
            "join_explosion_ratio",
            "join_explosion_penalty",
            "Ra",
            "Rb",
            "Ca",
            "Cb",
            "src_table_path",
            "tgt_table_path",
        ]
        cols = [c for c in cols if c in join_scored.columns]

        join_scored.sort_values(
            "custom_join_score", ascending=False, na_position="last"
        )[cols].head(n).to_csv(sample_dir / "join_top_custom_score.csv", index=False)

        join_scored[join_scored["join_eval_status"].ne("ok")][cols].head(n).to_csv(
            sample_dir / "join_invalid_eval_examples.csv", index=False
        )

        join_scored[
            join_scored["join_eval_valid"]
            & join_scored["ma"].eq(0)
            & join_scored["mb"].eq(0)
            & join_scored["Rj"].eq(0)
        ][cols].head(n).to_csv(sample_dir / "join_valid_zero_overlap_examples.csv", index=False)

    if len(merged_edges) > 0:
        cols = [
            "relation_type",
            "src_table_key",
            "tgt_table_key",
            "original_edge_weight",
            "utility_score",
            "final_score",
            "custom_union_score",
            "custom_join_score",
            "Ra",
            "Rb",
            "Ca",
            "Cb",
        ]
        cols = [c for c in cols if c in merged_edges.columns]

        merged_edges.sort_values(
            "final_score", ascending=False, na_position="last"
        )[cols].head(n).to_csv(sample_dir / "merged_top_final_score.csv", index=False)

def _run(args: argparse.Namespace) -> None:

    join_dir = Path(args.join_dir)
    union_dir = Path(args.union_dir)
    dataset_cache = Path(args.dataset_cache)
    outdir = Path(args.outdir)

    if args.node_top_k <= 0:
        raise ValueError(f"--node-top-k must be positive, got: {args.node_top_k}")

    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Config ===", flush=True)
    print(f"join_dir={join_dir}", flush=True)
    print(f"union_dir={union_dir}", flush=True)
    print(f"dataset_cache={dataset_cache}", flush=True)
    print(f"outdir={outdir}", flush=True)
    print(f"max_join_edges={args.max_join_edges}", flush=True)
    print(f"max_union_edges={args.max_union_edges}", flush=True)
    print(f"node_top_k={args.node_top_k}", flush=True)

    analyzer = JoinUnionScoreAnalyzer(
        join_dir=join_dir,
        union_dir=union_dir,
        dataset_cache=dataset_cache,
    )

    print("\n=== Loading nodes ===", flush=True)

    join_nodes_raw = read_nodes(join_dir / "table_nodes.tsv.gz")
    union_nodes_raw = read_nodes(union_dir / "table_nodes.tsv.gz")

    merged_nodes = build_merged_nodes(join_nodes_raw, union_nodes_raw)

    print(f"merged nodes before utility weighting: {len(merged_nodes)}", flush=True)

    print("\n=== Loading/scoring union edges ===", flush=True)

    union_edges_raw = read_edges(
        union_dir / "table_edges.tsv.gz",
        max_rows=args.max_union_edges,
    )

    union_scored = analyzer.score_union(union_edges_raw)

    union_scored.to_csv(
        outdir / "union_edges_scored.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    union_summary = stat_summary(
        union_scored,
        [
            "original_edge_weight",
            "custom_union_score",
            "final_score",
            "row_balance",
            "column_match_ratio",
            "support_cols",
            "m",
            "m_clipped",
            "Ra",
            "Rb",
            "Ca",
            "Cb",
            "unmatched_cols_a",
            "unmatched_cols_b",
            "row_ratio_min_over_max",
        ],
    )
    union_summary.to_csv(outdir / "union_score_summary.csv", index=False)

    print(f"union edges scored: {len(union_scored)}", flush=True)

    print("\n=== Loading/scoring join edges ===", flush=True)

    join_edges_raw = read_edges(
        join_dir / "table_edges.tsv.gz",
        max_rows=args.max_join_edges,
    )

    join_scored = analyzer.score_join(
        join_edges_raw,
        progress_every=args.progress_every,
    )

    join_scored.to_csv(
        outdir / "join_edges_scored.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    join_summary_all = stat_summary(
        join_scored,
        [
            "original_edge_weight",
            "custom_join_score",
            "final_score",
            "row_overlap_geom",
            "join_explosion_ratio",
            "join_explosion_penalty",
            "ma",
            "mb",
            "Rj",
            "ma_clipped",
            "mb_clipped",
            "ma_over_Ra",
            "mb_over_Rb",
            "Ra",
            "Rb",
            "Ca",
            "Cb",
        ],
    )
    join_summary_all.to_csv(outdir / "join_score_summary.csv", index=False)

    join_valid = join_scored[join_scored["join_eval_valid"]].copy()
    join_summary_valid = stat_summary(
        join_valid,
        [
            "original_edge_weight",
            "custom_join_score",
            "final_score",
            "row_overlap_geom",
            "join_explosion_ratio",
            "join_explosion_penalty",
            "ma",
            "mb",
            "Rj",
            "ma_clipped",
            "mb_clipped",
            "ma_over_Ra",
            "mb_over_Rb",
            "Ra",
            "Rb",
            "Ca",
            "Cb",
        ],
    )
    join_summary_valid.to_csv(outdir / "join_score_summary_valid_only.csv", index=False)

    join_status_counts = (
        join_scored["join_eval_status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="count")
    )
    join_status_counts.to_csv(outdir / "join_eval_status_counts.csv", index=False)

    valid_zero = int(
        (
            join_scored["join_eval_valid"]
            & join_scored["ma"].eq(0)
            & join_scored["mb"].eq(0)
            & join_scored["Rj"].eq(0)
        ).sum()
    )

    print(f"join edges scored: {len(join_scored)}", flush=True)
    print("\n=== Join eval status ===", flush=True)
    print(join_status_counts.to_string(index=False), flush=True)
    print(f"valid_zero_overlap={valid_zero}", flush=True)

    print("\n=== Merging edges: union wins overlaps ===", flush=True)

    merged_edges, merge_stats = build_merged_edges(union_scored, join_scored)

    merged_edges.to_csv(
        outdir / "merged_edges_scored.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    merged_nodes = build_node_utility_weights(
        merged_nodes,
        merged_edges,
        top_k=args.node_top_k,
        score_col="utility_score",
    )

    if args.graph_format_weight_mode == "algorithm":
        graph_edges, graph_nodes, algorithm_weight_stats = apply_algorithm_weights(
            merged_edges=merged_edges,
            merged_nodes=merged_nodes,
        )
    else:
        graph_edges = merged_edges.copy()
        graph_nodes = merged_nodes.copy()
        graph_nodes["weight"] = pd.to_numeric(
            graph_nodes["utility_score"], errors="coerce"
        ).fillna(0.0)
        algorithm_weight_stats = {}

    graph_nodes.to_csv(
        outdir / "merged_table_nodes.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    node_weight_summary = stat_summary(
        graph_nodes,
        ["utility_score", "weight", "utility_score_degree", "num_cols"],
    )
    node_weight_summary.to_csv(outdir / "node_weight_summary.csv", index=False)

    if algorithm_weight_stats:
        with (outdir / "algorithm_weight_stats.json").open("w", encoding="utf-8") as f:
            json.dump(algorithm_weight_stats, f, indent=2)

    graph_dump_stats = write_graph_format_outputs(
        outdir=outdir,
        merged_edges=graph_edges,
        merged_nodes=graph_nodes,
    )

    if algorithm_weight_stats:
        print("algorithm weight stats:", json.dumps(algorithm_weight_stats, indent=2), flush=True)
    else:
        print("graph-format weight mode: utility", flush=True)
    print("graph-format dump stats:", json.dumps(graph_dump_stats, indent=2), flush=True)

    merged_score_summary = stat_summary(
        merged_edges,
        [
            "original_edge_weight",
            "utility_score",
            "final_score",
            "custom_union_score",
            "custom_join_score",
            "Ra",
            "Rb",
            "Ca",
            "Cb",
            "row_balance",
            "column_match_ratio",
            "row_overlap_geom",
            "join_explosion_ratio",
            "join_explosion_penalty",
        ],
    )
    merged_score_summary.to_csv(outdir / "merged_score_summary.csv", index=False)

    relation_counts = (
        merged_edges["relation_type"]
        .value_counts(dropna=False)
        .rename_axis("relation_type")
        .reset_index(name="count")
    )
    relation_counts.to_csv(outdir / "merged_relation_type_counts.csv", index=False)

    print("merge stats:", json.dumps(merge_stats, indent=2), flush=True)

    print("\n=== Building overlap diagnostic table ===", flush=True)

    overlap = build_overlap_table(union_scored, join_scored)

    overlap.to_csv(
        outdir / "overlap_edges_scored.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    overlap_summary = stat_summary(
        overlap,
        [
            "custom_union_score",
            "custom_join_score",
            "custom_join_minus_union",
            "custom_join_over_union",
            "original_edge_weight_union",
            "original_edge_weight_join",
            "row_overlap_geom",
            "join_explosion_ratio",
            "join_explosion_penalty",
        ],
    )
    overlap_summary.to_csv(outdir / "overlap_score_summary.csv", index=False)

    print(f"overlap pairs: {len(overlap)}", flush=True)

    write_samples(
        outdir=outdir,
        union_scored=union_scored,
        join_scored=join_scored,
        merged_edges=merged_edges,
        n=args.sample_debug_rows,
    )

    run_summary = {
        "join_dir": str(join_dir),
        "union_dir": str(union_dir),
        "dataset_cache": str(dataset_cache),
        "outdir": str(outdir),
        "max_join_edges": args.max_join_edges,
        "max_union_edges": args.max_union_edges,
        "node_top_k": args.node_top_k,
        "graph_format_weight_mode": args.graph_format_weight_mode,
        "num_join_edges_raw": int(len(join_edges_raw)),
        "num_union_edges_raw": int(len(union_edges_raw)),
        "num_join_edges_scored": int(len(join_scored)),
        "num_union_edges_scored": int(len(union_scored)),
        "num_merged_edges": int(len(merged_edges)),
        "num_merged_nodes": int(len(merged_nodes)),
        "merge_stats": merge_stats,
        "algorithm_weight_stats": algorithm_weight_stats,
        "graph_dump_stats": graph_dump_stats,
        "join_eval": {
            "num_join_eval_valid": int(join_scored["join_eval_valid"].sum()),
            "num_join_eval_invalid": int((~join_scored["join_eval_valid"]).sum()),
            "num_valid_zero_overlap": valid_zero,
            "status_counts": join_status_counts.to_dict(orient="records"),
        },
    }

    with (outdir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("\n=== Done ===", flush=True)
    print(f"wrote scored merged edges: {outdir / 'merged_edges_scored.tsv.gz'}", flush=True)
    print(f"wrote weighted merged nodes: {outdir / 'merged_table_nodes.tsv.gz'}", flush=True)
    print(f"wrote graph-format edges: {outdir / 'table_edges.tsv.gz'}", flush=True)
    print(f"wrote graph-format nodes: {outdir / 'table_nodes.tsv.gz'}", flush=True)
    print(f"wrote run summary: {outdir / 'run_summary.json'}", flush=True)


def run_merge(
    join_dir: str | Path,
    union_dir: str | Path,
    dataset_cache: str | Path,
    outdir: str | Path,
    node_top_k: int = 5,
    progress_every: int = 500,
    sample_debug_rows: int = 30,
    graph_format_weight_mode: str = "utility",
    max_join_edges: Optional[int] = None,
    max_union_edges: Optional[int] = None,
) -> None:
    _run(
        argparse.Namespace(
            join_dir=str(join_dir),
            union_dir=str(union_dir),
            dataset_cache=str(dataset_cache),
            outdir=str(outdir),
            max_join_edges=max_join_edges,
            max_union_edges=max_union_edges,
            progress_every=progress_every,
            sample_debug_rows=sample_debug_rows,
            node_top_k=node_top_k,
            graph_format_weight_mode=graph_format_weight_mode,
        )
    )


def main() -> None:
    _run(parse_args())

if __name__ == "__main__":
    main()
