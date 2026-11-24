# Instructions

This project contains the main Auto-Discover pipeline. It starts after tables have been preprocessed and column embeddings have been loaded into Qdrant.

The preprocessing code, embedding generation code, and vector database build code are not included here.

## Data Sources

- **SANTOS Large**
  - Dataset record: https://zenodo.org/records/7758091
  - Project repository: https://github.com/northeastern-datalab/santos

- **YADL Full**
  - Dataset record: https://zenodo.org/records/10624396
  - Project page: https://soda-inria.github.io/retrieve-merge-predict/

- **LakeBench OpenData**
  - Repository: https://github.com/BIT-DataLab/LakeBench
  - Benchmark record: https://zenodo.org/records/15499092

## Upstream Components

The table columns should already be embedded before this pipeline runs. We use Starmie and DeepJoin style embeddings depending on the experiment setting.

- Starmie: https://github.com/megagonlabs/starmie
- DeepJoin reproduction in LakeBench: https://github.com/BIT-DataLab/LakeBench/tree/main/join/Deepjoin

Qdrant is used as the vector database for column embeddings.

- Qdrant documentation: https://qdrant.tech/documentation/

The Qdrant collection must contain one point per embedded column. The required payload fields are `table_key`, `table_id`, and `col_idx`.

## Required External Inputs

### Qdrant Collection

The join and union graph construction scripts query an existing Qdrant collection. Use either a local Qdrant storage directory or a running Qdrant server.

### Preprocessed Dataset Cache

The merge and utility scoring stage needs the preprocessed tables. The cache should have this relative structure:

```text
DATASET_CACHE/
  manifest_tables.jsonl
  tables/
    <preprocessed table files>
```

`manifest_tables.jsonl` should contain one JSON object per table. The important fields are:

- `table_key`: stable table id used in Qdrant payloads and graph files.
- `table_id`: original dataset relative table filename or display name.
- `table_path` or `csv_path`: relative path or filename for the preprocessed table file.

The files under `tables/` should be cleaned CSV tables with a header row. The column order must match the `col_idx` values used when creating embeddings.

### Query Roots

`ROOTS_FILE` is a text file with one query root `table_key` per line. This repository includes 100 SANTOS query tables, 100 LakeBench query tables, and 99 YADL query tables under `main_query_tables/`. Their original dataset filenames are listed in `main_query_tables/query_table_file_mapping.csv`.

## Pipeline Stages

Run the full pipeline with:

```bash
scripts/run_full_pipeline.sh
```

### Offline Phase

1. `scripts/01_build_join_graph.sh` builds the join graph from Qdrant.
2. `scripts/02_build_union_graph.sh` builds the union graph from Qdrant.
3. `scripts/03_merge_join_union_graph.sh` merges both graphs and assigns utility weights.
4. `scripts/04_transform_prepare_search_graph.sh` prepares the merged graph for search and DP.
5. `scripts/05_build_initial_components.sh` builds initial components for Auto-Discover after the search graph is ready.

### Query Phase

6. `scripts/06_run_search_dp.sh` runs root search and top-k DP for the query roots.

The default output layout is:

```text
OUT_ROOT/
  join_graph/
  union_graph/
  merged_graph/
  search_graph/
  initial_components/
  search_results/
```

## Main Settings

```bash
export PYTHON=python
export COLLECTION=<qdrant_collection_name>
export OUT_ROOT=<pipeline_output_directory>
export DATASET_CACHE=<preprocessed_dataset_cache>
export ROOTS_FILE=<query_roots_file>

export QDRANT_PATH=<qdrant_storage_directory>
# or
export QDRANT_URL=http://localhost:6333
export QDRANT_API_KEY=<optional_api_key>

export JOIN_TAU=0.80
export UNION_TAU=0.85
export LIMIT_PER_COL=500
export NODE_TOP_K=5
export TOP_K=50
export MAX_NODES=20
export DP_MODE=at-most
export METHOD=pruning
export INITIAL_COMPONENT_SIZE=3
```

## Experiment Variants

This pipeline covers Pruning and Auto-Discover.

**Pruning** runs graph search and rooted top-k DP directly on the prepared search graph:

```bash
export METHOD=pruning
scripts/run_full_pipeline.sh
```

**Auto-Discover** first builds initial components in the offline phase, then runs graph search and rooted top-k DP using those components:

```bash
export METHOD=auto_discover
export INITIAL_COMPONENT_SIZE=3
scripts/run_full_pipeline.sh
```

To build or reuse initial components separately, run `scripts/05_build_initial_components.sh` and set `INITIAL_COMPONENTS_DIR` before running `scripts/06_run_search_dp.sh`.

## Code Files

- `code/build_graph.py`: command-line entry point for join or union graph construction.
- `code/graph_index.py`: Qdrant graph construction logic.
- `code/merge_join_union_graph.py`: command-line entry point for graph merge and utility scoring.
- `code/transform_join_union_graph.py`: merge and utility scoring implementation.
- `code/prepare_search_graph.py`: prepares the merged graph for search and DP.
- `code/build_initial_components.py`: builds bounded initial components for Auto-Discover.
- `code/run_search_dp.py`: runs root search and DP for query roots.
- `code/graph_search.py`: root-conditioned graph search.
- `code/rooted_topk_subtree.py`: rooted top-k DP.

## Monitoring

During offline stages, follow terminal output or redirect each script to a log file. During the query phase, follow `search_results/profile.jsonl` under the selected output directory.
