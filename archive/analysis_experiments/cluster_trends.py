"""Compute year-by-year document shares for the saved Rising clusters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import trend_tags


SOURCE = ROOT / "analysis" / "output_임계15"
CORPUS_PATH = SOURCE / "filtered_corpus.parquet"
LONG_PATH = SOURCE / "unigram_long.parquet"
CLUSTER_PATH = SOURCE / "rising_unigram_clusters.parquet"
MAPPING_PATH = SOURCE / "term_normalization_dict.csv"
INTERPRETATIONS_PATH = ROOT / "reports" / "rising_cluster_review" / "interpretations.json"
OUTPUT_LONG_PATH = SOURCE / "cluster_trends.parquet"
OUTPUT_SUMMARY_PATH = SOURCE / "cluster_trends_summary.csv"


def _load_labels() -> dict[int, str]:
    data = json.loads(INTERPRETATIONS_PATH.read_text(encoding="utf-8"))
    return {int(item["cluster"]): item["label"] for item in data["clusters"]}


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus = pd.read_parquet(CORPUS_PATH, columns=["id", "year", "language"])
    corpus = corpus[corpus["language"].eq("en")].drop_duplicates("id")
    years = sorted(corpus["year"].unique())
    denominator = corpus.groupby("year")["id"].nunique().reindex(years, fill_value=0)

    partition = pd.read_parquet(CLUSTER_PATH)
    term_column = "용어" if "용어" in partition.columns else partition.columns[1]
    partition = partition.rename(columns={term_column: "term"})
    term_to_cluster = partition.set_index("term")["cluster"].to_dict()

    mapping_df = pd.read_csv(MAPPING_PATH)
    mapping = dict(zip(mapping_df["variant"], mapping_df["canonical"]))
    long = pd.read_parquet(LONG_PATH, columns=["id", "year", "term"])
    long = long[long["id"].isin(corpus["id"])]
    long["term"] = long["term"].map(lambda term: mapping.get(term, term))
    long = long[long["term"].isin(term_to_cluster)]
    long = long.drop_duplicates(["id", "term"])
    long["cluster"] = long["term"].map(term_to_cluster)

    document_clusters = long.drop_duplicates(["id", "year", "cluster"])
    counts = document_clusters.groupby(["year", "cluster"])["id"].nunique()
    clusters = sorted(set(term_to_cluster.values()))
    rows = []
    for year in years:
        for cluster in clusters:
            documents = int(counts.get((year, cluster), 0))
            total = int(denominator.loc[year])
            rows.append({
                "year": int(year),
                "cluster": int(cluster),
                "documents": documents,
                "total_documents": total,
                "document_share_pct": documents / total * 100 if total else 0.0,
            })
    series = pd.DataFrame(rows)

    labels = _load_labels()
    summary_rows = []
    for cluster in clusters:
        sub = series[series["cluster"].eq(cluster)].sort_values("year")
        values = sub["document_share_pct"].to_numpy(dtype=float)
        stats = trend_tags._mk_stats(values)
        n_edge = min(5, len(values))
        recent_change = float(values[-n_edge:].mean() - values[:n_edge].mean())
        top_terms = partition[partition["cluster"].eq(cluster)].sort_values(
            "전체기간_평균비중(%)", ascending=False
        )["term"].head(10).tolist()
        summary_rows.append({
            "cluster": int(cluster),
            "label": labels.get(int(cluster), f"군집 {cluster}"),
            "term_count": int((partition["cluster"] == cluster).sum()),
            "mean_document_share_pct": round(float(values.mean()), 4),
            "mk_tau": round(stats["tau"], 4) if pd.notna(stats["tau"]) else None,
            "mk_p": round(stats["p"], 6) if pd.notna(stats["p"]) else None,
            "sen_slope_pct_point_per_year": round(stats["slope"], 6) if pd.notna(stats["slope"]) else None,
            "recent5_minus_first5_pct_point": round(recent_change, 4),
            "top_terms": " | ".join(top_terms),
        })
    summary = pd.DataFrame(summary_rows)
    series["label"] = series["cluster"].map(labels)
    return series, summary


def main() -> None:
    series, summary = compute()
    OUTPUT_LONG_PATH.parent.mkdir(parents=True, exist_ok=True)
    series.to_parquet(OUTPUT_LONG_PATH, index=False)
    summary.to_csv(OUTPUT_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"[cluster_trends] years={series.year.nunique()} clusters={series.cluster.nunique()}")
    print(f"[cluster_trends] saved={OUTPUT_LONG_PATH.name}, {OUTPUT_SUMMARY_PATH.name}")
    print(summary[["cluster", "mk_tau", "mk_p", "sen_slope_pct_point_per_year"]].to_string(index=False))


if __name__ == "__main__":
    main()