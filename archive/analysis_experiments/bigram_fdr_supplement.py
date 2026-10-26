"""바이그램 Rising 보충표 — 유니그램만으로는 안 보이는 신호를 확인하는 가벼운 로버스트니스 체크.

배경(대화 기록, 2026-09-09): 바이그램도 유니그램과 똑같이 전용 co-occurrence 네트워크·군집화까지
다 만드는 방안(A)과, 유니그램 8군집 스토리는 그대로 두고 바이그램은 보충 증거로만 쓰는 방안(B)을
검토했다. 실측 결과 바이그램 Rising 동시출현 네트워크는 유니그램보다 훨씬 성기고(고립노드 8.8%,
min_cooccur=2 기준— analysis/output_임계15/bigram_long.parquet 기반 실측) 유니그램 8군집 스토리와
별개의 큰 결과 섹션을 또 만들면 8페이지 논문 분량상 부담이 커서 B를 채택했다.

이 스크립트는 바이그램 후보(게이트 통과 3,864개)에도 유니그램과 동일한 Benjamini-Hochberg FDR
보정을 적용한 뒤, "구성 단어 두 개가 모두 이미 유니그램 FDR-Rising 목록(1,080개)에 있는" 바이그램은
제외하고 남는(=유니그램 분석만으로는 안 보이는) 바이그램 상위 목록을 출력한다. 별도 네트워크·군집화는
하지 않는다.

사용법: python -m analysis.bigram_fdr_supplement
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import build_dataset, normalization, trend_tags  # noqa: E402

SOURCE = ROOT / "analysis" / "output_임계15"
OUTPUT = ROOT / "analysis" / "output_fdr_assoc" / "bigram_supplement.csv"
MIN_TOTAL_COUNT = trend_tags.MIN_TOTAL_COUNT


def load_bigram_inputs() -> tuple[pd.DataFrame, pd.Series]:
    corpus = build_dataset.load_analyzable_corpus(columns=["id", "year"])
    long = pd.read_parquet(SOURCE / "bigram_long.parquet", columns=["id", "year", "term"])
    mapping = normalization.load_dict()
    long["term"] = normalization.apply_dict(long["term"], mapping)
    long = long.drop_duplicates(["id", "term"])
    year_totals = corpus.groupby("year")["id"].nunique().sort_index()
    term_year_counts = long.groupby(["term", "year"])["id"].nunique().unstack(fill_value=0)
    term_year_counts = term_year_counts.reindex(columns=year_totals.index, fill_value=0)
    return term_year_counts, year_totals


def build_bigram_fdr_tags(term_year_counts: pd.DataFrame, year_totals: pd.Series) -> pd.DataFrame:
    overall_total = term_year_counts.sum(axis=1)
    candidates = list(overall_total[overall_total >= MIN_TOTAL_COUNT].sort_values(ascending=False).index)
    shares = term_year_counts.div(year_totals, axis=1) * 100
    tags = trend_tags.compute_tags(term_year_counts, shares, candidates)

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
    return tags


def main() -> None:
    counts, year_totals = load_bigram_inputs()
    tags = build_bigram_fdr_tags(counts, year_totals)
    rising = tags.loc[tags["추세(MK-FDR)"].eq("Rising")].copy()
    print(f"바이그램 FDR-Rising: {len(rising)}개 (참고: 미보정 1,045개)")

    uni_tags = pd.read_parquet(SOURCE.parent / "output_fdr_assoc" / "trend_tags_unigram_fdr.parquet")
    uni_fdr_rising = set(uni_tags.loc[uni_tags["추세(MK-FDR)"].eq("Rising"), "용어"])

    def both_covered(term: str) -> bool:
        parts = term.split()
        return len(parts) == 2 and all(p in uni_fdr_rising for p in parts)

    rising["유니그램만으로_이미_보임"] = rising["용어"].map(both_covered)
    new_signal = rising.loc[~rising["유니그램만으로_이미_보임"]].sort_values("전체기간_평균비중(%)", ascending=False)
    print(f"유니그램 FDR-Rising(1,080개)만으로는 안 보이는 바이그램: {len(new_signal)}개 / {len(rising)}개")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    new_signal.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUTPUT}\n")

    cols = ["용어", "전체기간_평균비중(%)", "Sen_slope(%p_연)", "MK_tau", "BH_q"]
    print("=== 상위 20개 (평균비중 기준) ===")
    print(new_signal[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
