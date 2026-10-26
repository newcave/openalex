"""Align two cluster partitions by maximum shared-term overlap."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.optimize import linear_sum_assignment


def load_partition(path: Path) -> dict[int, set[str]]:
    frame = pd.read_parquet(path)
    term_column = "용어" if "용어" in frame.columns else frame.columns[1]
    return {int(cluster): set(group[term_column]) for cluster, group in frame.groupby("cluster")}


def align(raw: dict[int, set[str]], fdr: dict[int, set[str]]) -> pd.DataFrame:
    raw_ids = sorted(raw)
    fdr_ids = sorted(fdr)
    overlap = [[len(raw[r] & fdr[f]) for f in fdr_ids] for r in raw_ids]
    rows, cols = linear_sum_assignment(-pd.DataFrame(overlap).to_numpy())
    records = []
    for row, col in zip(rows, cols):
        raw_id, fdr_id = raw_ids[row], fdr_ids[col]
        shared = overlap[row][col]
        union = len(raw[raw_id] | fdr[fdr_id])
        records.append({
            "raw_cluster": raw_id,
            "fdr_cluster": fdr_id,
            "raw_terms": len(raw[raw_id]),
            "fdr_terms": len(fdr[fdr_id]),
            "shared_terms": shared,
            "jaccard": shared / union if union else 0.0,
            "raw_retained_pct": shared / len(raw[raw_id]) * 100 if raw[raw_id] else 0.0,
            "fdr_supported_pct": shared / len(fdr[fdr_id]) * 100 if fdr[fdr_id] else 0.0,
        })
    return pd.DataFrame(records).sort_values("raw_cluster").reset_index(drop=True)


def compute(raw_path: Path, fdr_path: Path) -> pd.DataFrame:
    return align(load_partition(raw_path), load_partition(fdr_path))