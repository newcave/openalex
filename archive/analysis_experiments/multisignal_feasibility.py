"""다중신호 타당성 점검 — 논문 §3.5.1 근거 생성.

문제의식: FDR-Rising 1,080개는 이미 "26년간 유의하게 증가"라는 공통점을 갖는다. 그런데
현재 파이프라인은 Rising 선별에 MK의 p/q와 tau만 쓰고, Sen's slope·z-score·버스트는
표시용으로만 쓴다. 그렇다면 "다중신호"라는 표현이 정당한가? 두 가지를 확인한다.

  (1) Rising 내부에서 신호들이 실제로 흩어지는가, 그리고 서로 독립인가
      -> Spearman 상관행렬. 상관이 1에 가까운 쌍은 '중복 정보'이므로 정직하게 보고해야 한다.
  (2) 부상 '양상'의 차이가 주제 군집과 연관되는가
      -> 군집은 co-occurrence 구조로만 만들었고 시계열 신호는 거기에 관여하지 않으므로
         이 연관은 순환논리가 아니다.

주 검정은 구간화가 필요 없는 Kruskal-Wallis로 하고, 3분위 교차표는 예시로만 제시한다.
n이 1,080이라 p값은 쉽게 유의해지므로 효과크기(Cramér's V)를 반드시 함께 낸다.

원본 파일은 읽기만 한다. 출력은 reports/multisignal_*.csv 4개.

사용법: python -m analysis.multisignal_feasibility
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FINAL = ROOT / "analysis" / "output_fdr_assoc"
OUT = ROOT / "reports"

SIGNALS = ["최근3년_zscore", "최대버스트강도", "버스트연도", "Sen_slope(%p_연)", "전체기간_평균비중(%)"]
MIN_CLUSTER_SIZE = 30  # 기대빈도 5 미만 셀을 피하기 위한 '큰 군집' 기준


def load() -> pd.DataFrame:
    tags = pd.read_parquet(FINAL / "trend_tags_unigram_fdr.parquet")
    rising = tags[tags["추세(MK-FDR)"].eq("Rising")]
    partition = pd.read_parquet(FINAL / "rising_unigram_clusters.parquet", columns=["cluster", "term"])
    merged = partition.merge(rising, left_on="term", right_on="용어", how="inner")
    assert len(merged) == len(rising), f"군집 매칭 누락: {len(merged)} != {len(rising)}"
    return merged


def cramers_v(table: pd.DataFrame) -> tuple[float, float, float, int]:
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return chi2, p, v, int((expected < 5).sum())


def main() -> None:
    df = load()
    print(f"FDR-Rising(군집 매칭 완료): {len(df)}개\n")

    print("=== (가) Rising 내부의 신호 분포 ===")
    print(df[SIGNALS].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
    print("\n결측:", {c: int(df[c].isna().sum()) for c in SIGNALS})

    print("\n=== (나) 신호 간 Spearman 상관 ===")
    corr = df[SIGNALS].corr(method="spearman").round(3)
    print(corr.to_string())
    print("\n※ 상관이 높은 쌍은 '독립 신호'라고 주장할 수 없다 — 논문에 그대로 보고할 것.")

    print("\n=== (다) 군집별 신호 요약 (중앙값) ===")
    profile = df.groupby("cluster").agg(
        n=("term", "size"),
        z_median=("최근3년_zscore", "median"),
        burst_median=("최대버스트강도", "median"),
        burst_year_median=("버스트연도", "median"),
        sen_median=("Sen_slope(%p_연)", "median"),
    ).round(3)
    print(profile.to_string())

    # 3분위 기반 유형 (예시용 — 임계값은 관측 분포의 분위수일 뿐 이론값이 아니다)
    valid = df.dropna(subset=["최근3년_zscore"]).copy()
    low, high = valid["최근3년_zscore"].quantile([1 / 3, 2 / 3])
    valid["유형"] = np.select(
        [valid["최근3년_zscore"] <= low, valid["최근3년_zscore"] <= high],
        ["A_완만", "B_중간"], default="C_최근급등")
    print(f"\nz-score 3분위 경계: {low:.3f}, {high:.3f}")

    crosstab = pd.crosstab(valid["cluster"], valid["유형"])
    print("\n=== 군집 x 유형 교차표 (행 비율 %) ===")
    print((crosstab.div(crosstab.sum(axis=1), axis=0) * 100).round(1).to_string())

    tests = []
    chi2, p, v, small = cramers_v(crosstab)
    tests.append({"test": "chi2_all_clusters", "statistic": round(chi2, 2), "p_value": p,
                  "effect_size": round(v, 4), "effect_name": "Cramer_V",
                  "note": f"기대빈도<5 셀 {small}개"})

    big_ids = crosstab.index[crosstab.sum(axis=1) >= MIN_CLUSTER_SIZE]
    big = crosstab.loc[big_ids]
    chi2b, pb, vb, smallb = cramers_v(big)
    tests.append({"test": f"chi2_clusters_ge{MIN_CLUSTER_SIZE}", "statistic": round(chi2b, 2),
                  "p_value": pb, "effect_size": round(vb, 4), "effect_name": "Cramer_V",
                  "note": f"군집 {list(big_ids)} | 기대빈도<5 셀 {smallb}개"})

    # 주 검정: 구간화 없이 연속 z-score로 군집 간 분포 차이
    groups = [g["최근3년_zscore"].to_numpy()
              for _, g in valid[valid["cluster"].isin(big_ids)].groupby("cluster")]
    h, ph = stats.kruskal(*groups)
    tests.append({"test": f"kruskal_zscore_clusters_ge{MIN_CLUSTER_SIZE}", "statistic": round(h, 2),
                  "p_value": ph, "effect_size": None, "effect_name": None,
                  "note": "주 검정 — 구간화 불필요"})

    tests_df = pd.DataFrame(tests)
    print("\n=== 검정 결과 ===")
    print(tests_df.to_string(index=False))
    print("\n※ n이 크므로 p값만으로 강한 주장을 하지 말 것. Cramer's V를 반드시 병기한다.")

    OUT.mkdir(parents=True, exist_ok=True)
    corr.to_csv(OUT / "multisignal_signal_correlation.csv", encoding="utf-8-sig")
    profile.to_csv(OUT / "multisignal_cluster_profile.csv", encoding="utf-8-sig")
    crosstab.to_csv(OUT / "multisignal_cluster_type_crosstab.csv", encoding="utf-8-sig")
    tests_df.to_csv(OUT / "multisignal_tests.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT / 'multisignal_signal_correlation.csv'} 외 3개")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
