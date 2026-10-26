"""Compare trend and Louvain results for several minimum document frequencies."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import build_dataset, normalization, trend_tags


SOURCE = ROOT / "analysis" / "output_임계15"
LONG_PATH = SOURCE / "unigram_long.parquet"
CLUSTER_OUTPUT = ROOT / "reports" / "min_count_sensitivity.csv"
RISING_OUTPUT = ROOT / "reports" / "min_count_sensitivity_rising.csv"
THRESHOLDS = (10, 15, 20)
MIN_COOCCUR = 2
LOUVAIN_SEED = 0


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    corpus = build_dataset.load_analyzable_corpus(columns=["id", "year"])
    long = pd.read_parquet(LONG_PATH, columns=["id", "year", "term"])
    mapping = normalization.load_dict()
    long["term"] = normalization.apply_dict(long["term"], mapping)
    long = long.drop_duplicates(["id", "term"])
    year_totals = corpus.groupby("year")["id"].nunique().sort_index()
    term_year_counts = long.groupby(["term", "year"])["id"].nunique().unstack(fill_value=0)
    term_year_counts = term_year_counts.reindex(columns=year_totals.index, fill_value=0)
    return corpus, long, term_year_counts, year_totals


def compute_tags(term_year_counts: pd.DataFrame, year_totals: pd.Series, threshold: int) -> pd.DataFrame:
    overall_total = term_year_counts.sum(axis=1)
    candidates = list(overall_total[overall_total >= threshold].sort_values(ascending=False).index)
    shares = term_year_counts.div(year_totals, axis=1) * 100
    tags = trend_tags.compute_tags(term_year_counts, shares, candidates)
    return tags.sort_values("전체기간_평균비중(%)", ascending=False).reset_index(drop=True)


def cluster_rising(long: pd.DataFrame, tags: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rising = tags.loc[tags["추세(MK)"].eq("Rising"), "용어"].tolist()
    rising_set = set(rising)
    graph_long = long[long["term"].isin(rising_set)]
    doc_ids = graph_long["id"].astype("category")
    term_ids = graph_long["term"].astype("category")
    matrix = sparse.csr_matrix(
        (np.ones(len(graph_long)), (doc_ids.cat.codes, term_ids.cat.codes)),
        shape=(len(doc_ids.cat.categories), len(term_ids.cat.categories)),
    )
    co = (matrix.T @ matrix).tocoo()
    terms_index = term_ids.cat.categories
    graph = nx.Graph()
    graph.add_nodes_from(rising)
    mask = (co.row < co.col) & (co.data >= MIN_COOCCUR)
    for i, j, weight in zip(co.row[mask], co.col[mask], co.data[mask]):
        graph.add_edge(terms_index[i], terms_index[j], weight=float(weight))

    communities = nx.community.louvain_communities(graph, weight="weight", seed=LOUVAIN_SEED)
    communities = sorted(communities, key=len, reverse=True)
    rows = [{"term": term, "cluster": cid} for cid, members in enumerate(communities) for term in members]
    partition = pd.DataFrame(rows)
    summary = {
        "rising_terms": len(rising),
        "edges": graph.number_of_edges(),
        "isolated_terms": sum(graph.degree(term) == 0 for term in graph.nodes),
        "clusters": len(communities),
        "largest_cluster": max(map(len, communities), default=0),
        "smallest_cluster": min(map(len, communities), default=0),
        "modularity": round(nx.community.modularity(graph, communities, weight="weight"), 6),
        "cluster_sizes": "|".join(map(str, sorted((len(c) for c in communities), reverse=True))),
    }
    return partition, summary


def main() -> None:
    _, long, term_year_counts, year_totals = load_inputs()
    summaries = []
    rising_rows = []
    results = []

    for threshold in THRESHOLDS:
        if threshold == 15:
            tags_path = SOURCE / "trend_tags_unigram_all.parquet"
            tags = pd.read_parquet(tags_path)
        else:
            print(f"[sensitivity] computing tags for threshold={threshold}")
            tags = compute_tags(term_year_counts, year_totals, threshold)
        partition, graph_summary = cluster_rising(long, tags)
        candidate_count = len(tags)
        rising_set = set(tags.loc[tags["추세(MK)"].eq("Rising"), "용어"])
        results.append((threshold, candidate_count, rising_set, graph_summary))

    baseline_rising = next(rising for threshold, _, rising, _ in results if threshold == 15)
    for threshold, candidate_count, rising_set, graph_summary in results:
        overlap = len(rising_set & baseline_rising) / len(rising_set | baseline_rising)
        row = {"min_doc_count": threshold, "candidate_terms": candidate_count, "rising_overlap_with_15_jaccard": round(overlap, 6)}
        row.update(graph_summary)
        summaries.append(row)
        rising_rows.extend({"min_doc_count": threshold, "term": term} for term in sorted(rising_set))
        print(f"[sensitivity] threshold={threshold} candidates={candidate_count} "
              f"rising={graph_summary['rising_terms']} clusters={graph_summary['clusters']}")

    CLUSTER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(CLUSTER_OUTPUT, index=False, encoding="utf-8-sig")
    pd.DataFrame(rising_rows).to_csv(RISING_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[sensitivity] saved={CLUSTER_OUTPUT}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()