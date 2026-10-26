"""run_fdr_assoc_pipeline.py를 5년 단위 구간으로 반복 실행한다.

배경: 논문 본문은 2000~2025 전체 26년을 하나의 기간으로 놓고 Rising 용어를 뽑았다.
이 스크립트는 같은 통계 절차(Hamed-Rao modified Mann-Kendall + Benjamini-Hochberg FDR,
association-strength 정규화 동시출현 네트워크, Louvain resolution=1.0/seed=0)를 그대로 두고,
분석기간만 5년 구간(2000-2005, 2006-2010, 2011-2015, 2016-2020, 2021-2025)으로 쪼개서
구간별로 독립적으로 반복한다. 각 구간 안에서:
  - MIN_TOTAL_COUNT(15) 이상 등장한 용어만 후보로 삼고
  - 그 구간의 연도별 상대비중만으로 MK 검정 + BH 보정을 다시 수행하고
  - 그 구간에 속한 문서만으로 동시출현 네트워크를 다시 구성하고 Louvain을 돌린다.
즉 "26년 전체에서 Rising으로 뽑힌 용어"가 아니라 "그 5년 구간 안에서만 Rising으로 뽑힌 용어"를
쓴다 — 시기별로 다른 용어 집합과 다른 군집 구조가 나올 수 있다.

주의(전체 기간 분석과의 차이, 해석 시 반드시 감안할 것):
  - MK 검정 표본이 구간당 5~6개 연도뿐이다. 26개 연도로 검정할 때보다 검정력이 훨씬 낮고,
    Hamed-Rao 보정이 쓰는 시차자기상관 추정도 이렇게 짧은 시계열에서는 불안정할 수 있다.
    구간별 Rising 판정을 "26년 분석과 같은 신뢰도"로 취급하면 안 된다.
  - MIN_TOTAL_COUNT=15는 26년 전체 기준으로 사용자가 확정한 값을 그대로 가져온 것이다.
    구간 길이가 26년 대비 1/5 수준으로 줄었으므로, 같은 용어라도 구간 안에서는 이 문턱을
    넘기 더 어렵다 — 연도별 논문 수가 뒤로 갈수록 느는 코퍼스 구조상 특히 초기 구간
    (2000-2005)에서 후보 자체가 적게 남을 수 있다.
  - 구간마다 Rising 용어 집합과 문서 집합이 다르므로 군집 수·군집 구성·modularity를
    구간 간에 직접 비교할 때는 "같은 기준으로 각자 새로 뽑은 결과"라는 점을 감안해야 한다.

사용법: python -m analysis.run_fdr_assoc_pipeline_windowed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.run_fdr_pipeline import load_inputs, build_fdr_tags  # noqa: E402
from analysis.run_fdr_assoc_pipeline import build_clusters_association  # noqa: E402

OUTPUT = ROOT / "analysis" / "output_fdr_assoc_windows"
WINDOWS = [(2000, 2005), (2006, 2010), (2011, 2015), (2016, 2020), (2021, 2025)]
TOP_N_TERMS = 15
# 2026-09-14: 구간(5~6개 연도)당 MIN_TOTAL_COUNT=15 필터를 적용한 첫 실행에서 BH 보정 후
# Rising 용어가 구간당 0~5개로 붕괴함(검정력 부족 + 다중검정 보정의 복합효과, 필터 문턱과는
# 무관 — 오히려 후보를 줄여 m을 줄이므로 필터를 없애면 통계적으로는 더 불리해질 수 있음).
# 사용자 요청으로 문턱을 없애고(=1, 사실상 전체 유니그램) 실측 비교를 위해 남겨둔 파라미터.
MIN_TOTAL_COUNT = 15


def run_window(start: int, end: int, corpus: pd.DataFrame, long: pd.DataFrame,
                term_year_counts: pd.DataFrame, year_totals: pd.Series) -> tuple[dict, list[dict]]:
    label = f"{start}_{end}"
    years = [y for y in year_totals.index if start <= y <= end]
    counts_w = term_year_counts.loc[:, years]
    totals_w = year_totals.loc[years]

    tags_w = build_fdr_tags(counts_w, totals_w, min_total_count=MIN_TOTAL_COUNT)
    long_w = long[long["year"].between(start, end)]
    partition_w, graph_summary = build_clusters_association(long_w, tags_w)

    out_dir = OUTPUT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    tags_w.to_parquet(out_dir / "trend_tags_unigram_fdr.parquet", index=False)
    tags_w.to_csv(out_dir / "trend_tags_unigram_fdr.csv", index=False, encoding="utf-8-sig")

    out_partition = partition_w.rename(columns={"용어": "term"})
    cols = ["cluster", "term"] + [c for c in out_partition.columns if c not in ("cluster", "term")]
    out_partition = out_partition[cols]
    out_partition.to_parquet(out_dir / "rising_unigram_clusters.parquet", index=False)

    n_docs = int(corpus.loc[corpus["year"].between(start, end), "id"].nunique())
    run_summary = {
        "window": label,
        "years": years,
        "n_years": len(years),
        "documents_in_window": n_docs,
        "min_total_count": MIN_TOTAL_COUNT,
        "min_cooccurrence": 2,
        "louvain_seed": 0,
        "mk_tested_terms": int(len(tags_w)),
        "valid_mk_p_values": int(tags_w["MK_p_raw"].notna().sum()),
        "raw_rising_terms": int(tags_w["추세(MK)"].eq("Rising").sum()),
        **graph_summary,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    topterm_rows = []
    for cid, sub in out_partition.groupby("cluster"):
        top = sub.sort_values("전체기간_평균비중(%)", ascending=False)["term"].head(TOP_N_TERMS).tolist()
        topterm_rows.append({
            "window": label,
            "cluster": int(cid),
            "size": int(len(sub)),
            "top_terms": " | ".join(top),
        })
    return run_summary, topterm_rows


def main() -> None:
    corpus, long, term_year_counts, year_totals = load_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_topterms = []
    for start, end in WINDOWS:
        run_summary, topterm_rows = run_window(start, end, corpus, long, term_year_counts, year_totals)
        summary_rows.append(run_summary)
        all_topterms.extend(topterm_rows)
        modularity = run_summary["modularity"]
        mod_str = f"{modularity:.3f}" if pd.notna(modularity) else "nan"
        print(f"[{start}-{end}] docs={run_summary['documents_in_window']} "
              f"rising={run_summary['rising_terms']} clusters={run_summary['clusters']} "
              f"modularity={mod_str} sizes={run_summary['cluster_sizes']}")

    summary_df = pd.DataFrame([
        {k: (v if not isinstance(v, list) else "|".join(map(str, v))) for k, v in row.items()}
        for row in summary_rows
    ])
    summary_df.to_csv(OUTPUT / "windows_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_topterms).to_csv(OUTPUT / "windows_cluster_topterms.csv", index=False, encoding="utf-8-sig")
    print(f"\n[output] {OUTPUT}")


if __name__ == "__main__":
    main()
