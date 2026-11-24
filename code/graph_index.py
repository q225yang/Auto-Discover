from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import gzip
import json
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, QueryRequest, SearchParams

@dataclass
class BuildStats:
    num_tables: int
    num_col_queries: int
    num_table_pairs: int
    num_edges_final: int
    num_self_table_skipped: int

@dataclass
class ProfileStats:
    total_sec: float = 0.0
    list_tables_sec: float = 0.0
    fetch_columns_sec: float = 0.0
    query_sec: float = 0.0
    aggregate_sec: float = 0.0
    matching_sec: float = 0.0
    add_graph_sec: float = 0.0

    num_tables_seen: int = 0
    num_tables_processed: int = 0
    num_col_queries: int = 0
    num_hits_total: int = 0
    num_candidate_edges_total: int = 0
    num_matching_calls: int = 0
    num_queries_at_limit: int = 0

    num_self_table_skipped: int = 0

class GraphIndex:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        tau_sim: float = 0.98,
        limit_per_col: int = 2000,
        scroll_page_size: int = 10000,
        table_group_key: str = "table_key",
        table_label_key: str = "table_id",
        col_idx_key: str = "col_idx",
        verbose: bool = True,
        log_every_n_tables: int = 500,
        debug_first_tables: Optional[int] = None,
        dump_edges_path: Optional[str] = None,
        dump_nodes_path: Optional[str] = None,
        dump_summary_path: Optional[str] = None,
        dump_flush_every_tables: int = 100,
        dump_mode: str = "w",
        store_edge_meta: bool = False,
        graph_mode: str = "union",
        join_topk_cols: int = 1,
        query_batch_size: int = 1,
        query_hnsw_ef: Optional[int] = None,
        exact_search: bool = False,
        max_col_queries: Optional[int] = None,
    ):
        if graph_mode not in {"union", "join"}:
            raise ValueError(f"graph_mode must be 'union' or 'join', got: {graph_mode}")
        if join_topk_cols <= 0:
            raise ValueError(f"join_topk_cols must be >= 1, got: {join_topk_cols}")
        if query_batch_size <= 0:
            raise ValueError(f"query_batch_size must be >= 1, got: {query_batch_size}")
        if query_hnsw_ef is not None and query_hnsw_ef <= 0:
            raise ValueError(f"query_hnsw_ef must be positive, got: {query_hnsw_ef}")
        if max_col_queries is not None and max_col_queries <= 0:
            raise ValueError(f"max_col_queries must be positive, got: {max_col_queries}")

        self.client = client
        self.collection_name = collection_name
        self.tau_sim = tau_sim
        self.limit_per_col = limit_per_col
        self.scroll_page_size = scroll_page_size

        self.table_group_key = table_group_key
        self.table_label_key = table_label_key
        self.col_idx_key = col_idx_key

        self.verbose = verbose
        self.log_every_n_tables = log_every_n_tables
        self.debug_first_tables = debug_first_tables

        self.dump_edges_path = dump_edges_path
        self.dump_nodes_path = dump_nodes_path
        self.dump_summary_path = dump_summary_path
        self.dump_flush_every_tables = dump_flush_every_tables
        self.dump_mode = dump_mode
        self.store_edge_meta = store_edge_meta

        self.graph_mode = graph_mode
        self.join_topk_cols = join_topk_cols
        self.query_batch_size = query_batch_size
        self.query_hnsw_ef = query_hnsw_ef
        self.exact_search = exact_search
        self.max_col_queries = max_col_queries

        self.graph = None
        self.stats: Optional[BuildStats] = None

        self.table_col_counts: Dict[str, int] = {}
        self.table_labels: Dict[str, str] = {}
        self.num_table_pairs: int = 0
        self.profile: ProfileStats = ProfileStats()

    def _open_text_maybe_gz(self, path: str, mode: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".gz"):
            return gzip.open(path, mode + "t", encoding="utf-8")
        return open(path, mode, encoding="utf-8")

    def _open_dumps(self):
        edge_fh = self._open_text_maybe_gz(self.dump_edges_path, self.dump_mode) if self.dump_edges_path else None
        if edge_fh and self.dump_mode == "w":
            edge_fh.write(
                "src_table_key\tsrc_table_id\ttgt_table_key\ttgt_table_id\t"
                "edge_type\tsrc_cols\tcandidate_edges\tsupport_cols\t"
                "edge_weight\tbest_col_score\tavg_support_weight\t"
                "best_src_col_idx\tbest_tgt_col_idx\tmeta_json\n"
            )
            edge_fh.flush()

        return edge_fh

    def _dump_nodes(self):
        if not self.dump_nodes_path:
            return
        with self._open_text_maybe_gz(self.dump_nodes_path, "w") as f:
            f.write("table_key\ttable_id\tnum_cols\n")
            for t in sorted(self.table_col_counts):
                table_id = self.table_labels.get(t, "")
                f.write(f"{t}\t{table_id}\t{self.table_col_counts[t]}\n")
            f.flush()

    def _max_weight_bipartite_result(
        self,
        edge_map: Dict[Tuple[int, int], float],
    ) -> Optional[dict]:
        if not edge_map:
            return None
        import networkx as nx

        G = nx.Graph()
        best_src_col_idx = -1
        best_tgt_col_idx = -1
        best_col_score = 0.0

        for (src_col_idx, tgt_col_idx), sim in edge_map.items():
            if sim <= 0.0:
                continue
            s = float(sim)
            if s > best_col_score:
                best_col_score = s
                best_src_col_idx = src_col_idx
                best_tgt_col_idx = tgt_col_idx
            G.add_edge(("L", src_col_idx), ("R", tgt_col_idx), weight=s)

        if G.number_of_edges() == 0:
            return None

        t0 = time.perf_counter()
        matching = nx.max_weight_matching(G, maxcardinality=False, weight="weight")
        self.profile.matching_sec += time.perf_counter() - t0
        self.profile.num_matching_calls += 1

        score = 0.0
        matched_pairs = []
        for u, v in matching:
            w = float(G[u][v]["weight"])
            score += w
            if u[0] == "L":
                src_col_idx = int(u[1])
                tgt_col_idx = int(v[1])
            else:
                src_col_idx = int(v[1])
                tgt_col_idx = int(u[1])
            matched_pairs.append(
                {
                    "src_col_idx": src_col_idx,
                    "tgt_col_idx": tgt_col_idx,
                    "score": w,
                }
            )

        support_cols = len(matching)
        if support_cols == 0:
            return None

        avg_support_weight = score / support_cols

        return {
            "edge_type": "union",
            "edge_weight": score,
            "support_cols": support_cols,
            "best_col_score": best_col_score,
            "avg_support_weight": avg_support_weight,
            "best_src_col_idx": best_src_col_idx,
            "best_tgt_col_idx": best_tgt_col_idx,
            "meta": {
                "matched_colpairs": sorted(matched_pairs, key=lambda x: x["score"], reverse=True),
            },
        }

    def _topk_join_result(
        self,
        edge_map: Dict[Tuple[int, int], float],
    ) -> Optional[dict]:
        if not edge_map:
            return None

        ranked = sorted(
            (
                (src_col_idx, tgt_col_idx, float(sim))
                for (src_col_idx, tgt_col_idx), sim in edge_map.items()
                if sim > 0.0
            ),
            key=lambda x: x[2],
            reverse=True,
        )
        if not ranked:
            return None

        chosen = ranked[: self.join_topk_cols]
        support_cols = len(chosen)
        if support_cols == 0:
            return None

        best_src_col_idx, best_tgt_col_idx, best_col_score = chosen[0]

        if support_cols == 1:
            edge_weight = best_col_score
        else:
            edge_weight = sum(x[2] for x in chosen) / support_cols

        return {
            "edge_type": "join",
            "edge_weight": edge_weight,
            "support_cols": support_cols,
            "best_col_score": best_col_score,
            "avg_support_weight": edge_weight,
            "best_src_col_idx": best_src_col_idx,
            "best_tgt_col_idx": best_tgt_col_idx,
            "meta": {
                "topk_colpairs": [
                    {
                        "src_col_idx": src_col_idx,
                        "tgt_col_idx": tgt_col_idx,
                        "score": score,
                    }
                    for src_col_idx, tgt_col_idx, score in chosen
                ]
            },
        }

    def _reduce_pair(
        self,
        edge_map: Dict[Tuple[int, int], float],
    ) -> Optional[dict]:
        if self.graph_mode == "union":
            return self._max_weight_bipartite_result(edge_map)
        return self._topk_join_result(edge_map)

    def build(self, build_graph: bool = False, count_only: bool = False) -> BuildStats:
        if count_only and build_graph:
            raise ValueError("count_only=True cannot be combined with build_graph=True")
        build_t0 = time.perf_counter()

        self.table_col_counts.clear()
        self.table_labels.clear()
        self.num_table_pairs = 0
        self.stats = None
        self.profile = ProfileStats()

        t0 = time.perf_counter()
        table_keys = self._list_table_keys()
        self.profile.list_tables_sec += time.perf_counter() - t0

        if self.debug_first_tables is not None:
            table_keys = table_keys[: self.debug_first_tables]

        self.profile.num_tables_seen = len(table_keys)

        if build_graph:
            import networkx as nx

            t_graph = time.perf_counter()
            self.graph = nx.Graph()
            for tk in table_keys:
                tid = self.table_labels.get(tk, tk)
                self.graph.add_node(
                    tk,
                    table_id=tid,
                )
            self.profile.add_graph_sec += time.perf_counter() - t_graph

        edge_fh = self._open_dumps()

        try:
            if self.verbose:
                print(
                    f"[config] collection={self.collection_name} "
                    f"tau_sim={self.tau_sim} limit_per_col={self.limit_per_col} "
                    f"num_tables={len(table_keys)} build_graph={build_graph} "
                    f"count_only={count_only} store_edge_meta={self.store_edge_meta} "
                    f"mode={self.graph_mode} "
                    f"join_topk_cols={self.join_topk_cols} "
                    f"query_batch_size={self.query_batch_size} "
                    f"query_hnsw_ef={self.query_hnsw_ef} "
                    f"exact_search={self.exact_search} "
                    f"max_col_queries={self.max_col_queries}",
                    flush=True,
                )

            for i, src_table_key in enumerate(table_keys, start=1):
                src_label = self.table_labels.get(src_table_key, src_table_key)
                cols_in_table = 0
                table_hits = 0

                tgt_tables: Set[str] = set()
                edges_by_tgt: Dict[str, Dict[Tuple[int, int], float]] = {}
                t_fetch = time.perf_counter()
                src_columns = list(self._fetch_columns(src_table_key))
                if self.max_col_queries is not None:
                    remaining = self.max_col_queries - self.profile.num_col_queries
                    if remaining <= 0:
                        break
                    src_columns = src_columns[:remaining]
                self.profile.fetch_columns_sec += time.perf_counter() - t_fetch

                for _, src_col_idx, hits in self._find_by_threshold_for_columns(src_columns, src_table_key):
                    cols_in_table += 1
                    self.profile.num_col_queries += 1
                    if len(hits) >= self.limit_per_col:
                        self.profile.num_queries_at_limit += 1

                    table_hits += len(hits)
                    self.profile.num_hits_total += len(hits)

                    t_agg = time.perf_counter()
                    for _, sim, tgt_table_key, tgt_col_idx in hits:
                        if sim <= 0.0:
                            continue

                        if tgt_table_key <= src_table_key:
                            if tgt_table_key == src_table_key:
                                self.profile.num_self_table_skipped += 1
                            continue

                        if count_only:
                            tgt_tables.add(tgt_table_key)
                        else:
                            tgt_map = edges_by_tgt.setdefault(tgt_table_key, {})
                            key = (src_col_idx, tgt_col_idx)
                            old = tgt_map.get(key)
                            if old is None or sim > old:
                                tgt_map[key] = sim
                    self.profile.aggregate_sec += time.perf_counter() - t_agg

                self.table_col_counts[src_table_key] = cols_in_table
                table_pair_count = 0
                table_candidate_edges = 0
                table_edge_weight_sum = 0.0
                unique_tgts = len(tgt_tables) if count_only else len(edges_by_tgt)

                if count_only:
                    table_pair_count = len(tgt_tables)
                    self.num_table_pairs += table_pair_count
                else:
                    for tgt_table_key, edge_map in edges_by_tgt.items():
                        candidate_edges = len(edge_map)
                        self.profile.num_candidate_edges_total += candidate_edges
                        table_candidate_edges += candidate_edges

                        reduced = self._reduce_pair(edge_map)
                        if reduced is None:
                            continue

                        self.num_table_pairs += 1
                        table_pair_count += 1
                        table_edge_weight_sum += reduced["edge_weight"]

                        tgt_label = self.table_labels.get(tgt_table_key, tgt_table_key)

                        if build_graph:
                            t_graph = time.perf_counter()
                            attrs = {
                                "weight": reduced["edge_weight"],
                                "edge_type": reduced["edge_type"],
                            }
                            if self.store_edge_meta:
                                attrs["support_cols"] = reduced["support_cols"]
                                attrs["candidate_edges"] = candidate_edges
                                attrs["best_col_score"] = reduced["best_col_score"]
                                attrs["avg_support_weight"] = reduced["avg_support_weight"]
                                attrs["best_src_col_idx"] = reduced["best_src_col_idx"]
                                attrs["best_tgt_col_idx"] = reduced["best_tgt_col_idx"]
                                attrs["meta"] = reduced["meta"]
                            self.graph.add_edge(src_table_key, tgt_table_key, **attrs)
                            self.profile.add_graph_sec += time.perf_counter() - t_graph

                        if edge_fh is not None:
                            meta_json = json.dumps(reduced["meta"], ensure_ascii=False) if self.store_edge_meta else ""
                            edge_fh.write(
                                f"{src_table_key}\t{src_label}\t"
                                f"{tgt_table_key}\t{tgt_label}\t"
                                f"{reduced['edge_type']}\t"
                                f"{cols_in_table}\t{candidate_edges}\t{reduced['support_cols']}\t"
                                f"{reduced['edge_weight']:.6f}\t{reduced['best_col_score']:.6f}\t"
                                f"{reduced['avg_support_weight']:.6f}\t"
                                f"{reduced['best_src_col_idx']}\t{reduced['best_tgt_col_idx']}\t"
                                f"{meta_json}\n"
                            )

                elapsed = time.perf_counter() - build_t0
                avg_hits_per_col = (table_hits / cols_in_table) if cols_in_table else 0.0

                if edge_fh is not None and self.dump_flush_every_tables > 0 and i % self.dump_flush_every_tables == 0:
                    edge_fh.flush()

                if self.verbose and (
                    i == 1
                    or i == len(table_keys)
                    or (self.log_every_n_tables > 0 and i % self.log_every_n_tables == 0)
                ):
                    print(
                        f"[progress] mode={self.graph_mode} {i}/{len(table_keys)} "
                        f"src={src_table_key} cols={cols_in_table} "
                        f"hits={table_hits} avg_hits_per_col={avg_hits_per_col:.2f} "
                        f"unique_tgts={unique_tgts} candidate_edges={table_candidate_edges} "
                        f"pairs_added={table_pair_count} pairs_so_far={self.num_table_pairs} "
                        f"col_queries_so_far={self.profile.num_col_queries} "
                        f"queries_at_limit={self.profile.num_queries_at_limit} "
                        f"edge_weight_sum={table_edge_weight_sum:.4f} "
                        f"elapsed_sec={elapsed:.1f}",
                        flush=True,
                    )

            self.profile.num_tables_processed = len(self.table_col_counts)
            num_edges_final = self.graph.number_of_edges() if build_graph else self.num_table_pairs

            self.stats = BuildStats(
                num_tables=self.profile.num_tables_processed,
                num_col_queries=self.profile.num_col_queries,
                num_table_pairs=self.num_table_pairs,
                num_edges_final=num_edges_final,
                num_self_table_skipped=self.profile.num_self_table_skipped,
            )

            self.profile.total_sec = time.perf_counter() - build_t0
            self._dump_nodes()

            total_possible_pairs = self.stats.num_tables * (self.stats.num_tables - 1) // 2
            pair_density = (self.stats.num_table_pairs / total_possible_pairs) if total_possible_pairs else 0.0

            summary = {
                "build_stats": {
                    "num_tables": self.stats.num_tables,
                    "num_col_queries": self.stats.num_col_queries,
                    "num_table_pairs": self.stats.num_table_pairs,
                    "num_edges_final": self.stats.num_edges_final,
                    "num_self_table_skipped": self.stats.num_self_table_skipped,
                },
                "profile": self.profile.__dict__,
                "graph": {
                    "built": bool(build_graph),
                    "num_nodes": self.graph.number_of_nodes() if build_graph else 0,
                    "num_edges": self.graph.number_of_edges() if build_graph else 0,
                },
                "derived": {
                    "total_possible_table_pairs": total_possible_pairs,
                    "pair_density_vs_all_pairs": pair_density,
                },
                "config": {
                    "collection_name": self.collection_name,
                    "graph_mode": self.graph_mode,
                    "tau_sim": self.tau_sim,
                    "limit_per_col": self.limit_per_col,
                    "scroll_page_size": self.scroll_page_size,
                    "table_group_key": self.table_group_key,
                    "table_label_key": self.table_label_key,
                    "col_idx_key": self.col_idx_key,
                    "debug_first_tables": self.debug_first_tables,
                    "build_graph": bool(build_graph),
                    "count_only": bool(count_only),
                    "store_edge_meta": self.store_edge_meta,
                    "join_topk_cols": self.join_topk_cols,
                    "query_batch_size": self.query_batch_size,
                    "query_hnsw_ef": self.query_hnsw_ef,
                    "exact_search": bool(self.exact_search),
                    "max_col_queries": self.max_col_queries,
                },
                "artifacts": {
                    "dump_edges_path": self.dump_edges_path,
                    "dump_nodes_path": self.dump_nodes_path,
                    "dump_summary_path": self.dump_summary_path,
                },
            }
            self._dump_summary(summary)

            if self.verbose:
                print("\n=== Build summary ===", flush=True)
                print(self.stats, flush=True)
                print(
                    f"timing_sec: total={self.profile.total_sec:.3f} "
                    f"list_tables={self.profile.list_tables_sec:.3f} "
                    f"fetch_columns={self.profile.fetch_columns_sec:.3f} "
                    f"query={self.profile.query_sec:.3f} "
                    f"aggregate={self.profile.aggregate_sec:.3f} "
                    f"matching={self.profile.matching_sec:.3f} "
                    f"add_graph={self.profile.add_graph_sec:.3f}",
                    flush=True,
                )
                print(
                    f"counts: tables={self.profile.num_tables_seen} "
                    f"tables_processed={self.profile.num_tables_processed} "
                    f"col_queries={self.profile.num_col_queries} "
                    f"hits_total={self.profile.num_hits_total} "
                    f"queries_at_limit={self.profile.num_queries_at_limit} "
                    f"candidate_edges_total={self.profile.num_candidate_edges_total} "
                    f"matching_calls={self.profile.num_matching_calls} "
                    f"self_table_skipped={self.profile.num_self_table_skipped}",
                    flush=True,
                )

            return self.stats

        finally:
            if edge_fh is not None:
                edge_fh.flush()
                edge_fh.close()

    def num_bipartite_computations_needed(self) -> int:
        return self.num_table_pairs

    def _same_table_filter(self, src_table_key: str) -> Filter:
        return Filter(
            must_not=[
                FieldCondition(
                    key=self.table_group_key,
                    match=MatchValue(value=src_table_key),
                )
            ]
        )

    def _search_params(self) -> Optional[SearchParams]:
        if self.query_hnsw_ef is None and not self.exact_search:
            return None
        return SearchParams(hnsw_ef=self.query_hnsw_ef, exact=self.exact_search)

    def _hits_from_points(self, points) -> List[Tuple[str, float, str, int]]:
        out = []
        for p in points:
            payload = p.payload or {}
            tgt_table_key = payload.get(self.table_group_key)
            tgt_col_idx = payload.get(self.col_idx_key)
            if tgt_table_key is None or tgt_col_idx is None:
                continue

            out.append(
                (
                    str(p.id),
                    float(p.score),
                    str(tgt_table_key),
                    int(tgt_col_idx),
                )
            )
        return out

    def _find_by_threshold(self, query_vec, src_table_key: str) -> List[Tuple[str, float, str, int]]:
        filt = self._same_table_filter(src_table_key)

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=filt,
            search_params=self._search_params(),
            limit=self.limit_per_col,
            score_threshold=self.tau_sim,
            with_payload=[self.table_group_key, self.col_idx_key],
            with_vectors=False,
        )

        return self._hits_from_points(res.points)

    def _find_by_threshold_for_columns(
        self,
        src_columns: Sequence[Tuple[str, object, int]],
        src_table_key: str,
    ) -> List[Tuple[str, int, List[Tuple[str, float, str, int]]]]:
        out = []

        if self.query_batch_size == 1:
            for point_id, src_vec, src_col_idx in src_columns:
                t_query = time.perf_counter()
                hits = self._find_by_threshold(src_vec, src_table_key)
                self.profile.query_sec += time.perf_counter() - t_query
                out.append((point_id, src_col_idx, hits))
            return out

        for start in range(0, len(src_columns), self.query_batch_size):
            chunk = src_columns[start : start + self.query_batch_size]
            filt = self._same_table_filter(src_table_key)
            search_params = self._search_params()
            requests = [
                QueryRequest(
                    query=src_vec,
                    filter=filt,
                    params=search_params,
                    limit=self.limit_per_col,
                    score_threshold=self.tau_sim,
                    with_payload=[self.table_group_key, self.col_idx_key],
                    with_vector=False,
                )
                for _, src_vec, _ in chunk
            ]

            t_query = time.perf_counter()
            responses = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=requests,
            )
            self.profile.query_sec += time.perf_counter() - t_query

            for (point_id, _, src_col_idx), res in zip(chunk, responses):
                out.append((point_id, src_col_idx, self._hits_from_points(res.points)))

        return out

    def _list_table_keys(self) -> List[str]:
        seen = set()
        next_offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                with_payload=[self.table_group_key, self.table_label_key],
                with_vectors=False,
                limit=self.scroll_page_size,
                offset=next_offset,
            )
            if not points:
                break

            for p in points:
                payload = p.payload or {}
                table_key = payload.get(self.table_group_key)
                if table_key is None:
                    continue
                table_key = str(table_key)
                seen.add(table_key)

                table_label = payload.get(self.table_label_key)
                if table_label is not None and table_key not in self.table_labels:
                    self.table_labels[table_key] = str(table_label)

            if next_offset is None:
                break

        return sorted(seen)

    def _fetch_columns(self, table_key: str):
        filt = Filter(
            must=[
                FieldCondition(
                    key=self.table_group_key,
                    match=MatchValue(value=table_key),
                )
            ]
        )

        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filt,
                with_payload=[self.col_idx_key],
                with_vectors=True,
                limit=self.scroll_page_size,
                offset=next_offset,
            )
            if not points:
                break

            for p in points:
                if p.vector is None:
                    continue
                payload = p.payload or {}
                col_idx = payload.get(self.col_idx_key)
                if col_idx is None:
                    continue
                yield str(p.id), p.vector, int(col_idx)

            if next_offset is None:
                break
