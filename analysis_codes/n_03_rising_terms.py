"""[분석 3단계] Rising 용어 선별: 제목 용어별 연도 비중에 Mann-Kendall 추세검정 +
Benjamini-Hochberg FDR 보정을 적용해 "유의하게 증가하는(Rising)" 용어를 고른다.

이 모듈은 함수만 제공한다(단독 실행 없음). n_04(전체기간), n_06(구간별), 웹앱이 불러다 쓴다.
  - load_inputs():          01_corpus의 코퍼스/용어 테이블/정규화 사전 로드 -> 용어x연도 문서수 표
  - build_fdr_tags():       MK 검정 + BH 보정 -> 용어별 추세 판정(추세(MK-FDR) 컬럼)
  - build_cluster_trends(): 커뮤니티(n_04 결과)별 연도별 문서 비중 + 추세 요약
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis_codes import n_01_build_corpus as build_dataset, normalization, trend_tags
from config import DEFAULT

SOURCE = DEFAULT.shared_source
FDR_LABELS = {
    0: "수처리·막공정과 미생물 처리",
    1: "하천 수문변동·기후·가뭄 연구",
    2: "도시 물시스템·저수지 운영 최적화",
    3: "홍수·유량 예측과 데이터 기반 수문 모델",
    4: "지하수·연안대수층과 수문환경 상호작용",
}
MIN_TOTAL_COUNT = trend_tags.MIN_TOTAL_COUNT


def load_inputs(source: Path | None = None, corpus_path: Path | None = None,
                dict_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """인자를 안 주면 기본 분야(config.DEFAULT) 경로를 쓴다(웹앱·pipeline.py는 분야별 경로를 넘김)."""
    source = Path(source) if source else SOURCE
    corpus = build_dataset.load_analyzable_corpus(columns=["id", "year"], corpus_path=corpus_path)
    long = pd.read_parquet(source / "unigram_long.parquet", columns=["id", "year", "term"])
    mapping = normalization.load_dict(dict_path)
    long["term"] = normalization.apply_dict(long["term"], mapping)
    long = long.drop_duplicates(["id", "term"])
    year_totals = corpus.groupby("year")["id"].nunique().sort_index()
    term_year_counts = long.groupby(["term", "year"])["id"].nunique().unstack(fill_value=0)
    term_year_counts = term_year_counts.reindex(columns=year_totals.index, fill_value=0)
    return corpus, long, term_year_counts, year_totals


def build_fdr_tags(term_year_counts: pd.DataFrame, year_totals: pd.Series,
                    min_total_count: int = MIN_TOTAL_COUNT) -> pd.DataFrame:
    overall_total = term_year_counts.sum(axis=1)
    candidates = list(overall_total[overall_total >= min_total_count].sort_values(ascending=False).index)
    shares = term_year_counts.div(year_totals, axis=1) * 100
    tags = trend_tags.compute_tags(term_year_counts, shares, candidates)

    # compute_tags rounds MK_p for display, so recalculate the raw p-values for FDR.
    raw_stats = {term: trend_tags._mk_stats(shares.loc[term].to_numpy(dtype=float)) for term in candidates}
    raw_p = pd.Series({term: stats["p"] for term, stats in raw_stats.items()})
    valid = raw_p.notna()
    q_values = pd.Series(np.nan, index=raw_p.index, dtype=float)
    q_values.loc[valid] = false_discovery_control(raw_p.loc[valid].to_numpy(), method="bh")

    tags["MK_p_raw"] = tags["용어"].map(raw_p)
    tags["BH_q"] = tags["용어"].map(q_values)
    tags["추세(MK-FDR)"] = np.select(
        [tags["BH_q"].le(0.05) & tags["MK_tau"].gt(0), tags["BH_q"].le(0.05) & tags["MK_tau"].lt(0)],
        ["Rising", "Declining"],
        default="Not Significant",
    )
    tags.loc[tags["BH_q"].isna(), "추세(MK-FDR)"] = "판정불가"
    return tags.sort_values("전체기간_평균비중(%)", ascending=False).reset_index(drop=True)


def build_cluster_trends(corpus: pd.DataFrame, long: pd.DataFrame, partition: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(corpus["year"].unique())
    denominator = corpus.groupby("year")["id"].nunique().reindex(years, fill_value=0)
    term_to_cluster = partition.set_index("용어")["cluster"].to_dict()
    matched = long[long["term"].isin(term_to_cluster)].copy()
    matched["cluster"] = matched["term"].map(term_to_cluster)
    matched = matched.drop_duplicates(["id", "year", "cluster"])
    counts = matched.groupby(["year", "cluster"])["id"].nunique()
    clusters = sorted(partition["cluster"].unique())
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
    summary_rows = []
    for cluster in clusters:
        values = series.loc[series["cluster"].eq(cluster)].sort_values("year")["document_share_pct"].to_numpy(float)
        stats = trend_tags._mk_stats(values)
        top_terms = partition.loc[partition["cluster"].eq(cluster)].sort_values(
            "전체기간_평균비중(%)", ascending=False
        )["용어"].head(10).tolist()
        summary_rows.append({
            "cluster": int(cluster),
            "label": FDR_LABELS.get(int(cluster), f"FDR 군집 {int(cluster)}"),
            "term_count": int((partition["cluster"] == cluster).sum()),
            "mean_document_share_pct": round(float(values.mean()), 4),
            "mk_tau": round(stats["tau"], 4) if pd.notna(stats["tau"]) else None,
            "mk_p": round(stats["p"], 6) if pd.notna(stats["p"]) else None,
            "sen_slope_pct_point_per_year": round(stats["slope"], 6) if pd.notna(stats["slope"]) else None,
            "recent5_minus_first5_pct_point": round(float(values[-5:].mean() - values[:5].mean()), 4),
            "top_terms": " | ".join(top_terms),
        })
    return series, pd.DataFrame(summary_rows)
