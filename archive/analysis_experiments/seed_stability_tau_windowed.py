"""5년 구간별 tau>0 기반 네트워크(run_tau_positive_pipeline_windowed.py)에 대해, seed
0~49로 Louvain을 반복해 군집 수 분포의 안정성을 확인한다.

배경: 구간별 seed=0 결과의 군집 수가 12/7/9/7/12로 서로 다르다. 사용자가 "구간마다 다른
seed를 써서 전부 9개로 맞출 수 있냐"고 물었는데, 이건 원하는 숫자가 나올 때까지 seed를
골라 쓰는 것이라(결과 맞추기) 권하지 않았다. 대신 26년 전체 분석(§3.5, 논문 본문)과 같은
방식 — seed는 고정(0)하되, 0~49 전체를 돌려서 "이 구간에서 군집 수가 얼마나 안정적인가"를
반복분석으로 확인하는 방식을 택했다.

그래프 자체(노드=그 구간의 tau>0 용어, 엣지=association-strength 동시출현)는 seed와
무관하므로 구간당 한 번만 만들고, Louvain만 50번 돌린다.

사용법: python -m analysis.seed_stability_tau_windowed
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.run_fdr_pipeline import load_inputs, build_fdr_tags  # noqa: E402
from analysis.run_tau_positive_pipeline_windowed import (  # noqa: E402
    MIN_TOTAL_COUNT, build_graph_from_terms, compute_windows,
)

OUTPUT = ROOT / "analysis" / "output_tau_positive_windows"
SEEDS = range(50)


def main() -> None:
    corpus, long, term_year_counts, year_totals = load_inputs()
    windows = compute_windows(int(year_totals.index.min()), int(year_totals.index.max()))

    detail_rows = []
    summary_rows = []
    for start, end in windows:
        label = f"{start}_{end}"
        years = [y for y in year_totals.index if start <= y <= end]
        counts_w = term_year_counts.loc[:, years]
        totals_w = year_totals.loc[years]
        tags_w = build_fdr_tags(counts_w, totals_w, min_total_count=MIN_TOTAL_COUNT)
        rising_terms = tags_w.loc[tags_w["MK_tau"] > 0, "용어"].tolist()

        long_w = long[long["year"].between(start, end)]
        graph = build_graph_from_terms(long_w, rising_terms)

        counts = []
        for seed in SEEDS:
            communities = nx.community.louvain_communities(graph, weight="weight", seed=seed)
            n_clusters = len(communities)
            modularity = nx.community.modularity(graph, communities, weight="weight")
            counts.append(n_clusters)
            detail_rows.append({
                "window": label, "seed": seed, "clusters": n_clusters,
                "modularity": round(modularity, 4),
            })

        tally = Counter(counts)
        mode_clusters, mode_freq = tally.most_common(1)[0]
        seed0_clusters = counts[0]
        summary_rows.append({
            "window": label,
            "seed0_clusters": seed0_clusters,
            "mode_clusters": mode_clusters,
            "mode_freq_of_50": mode_freq,
            "cluster_count_range": f"{min(counts)}-{max(counts)}",
            "distribution": " | ".join(f"{k}:{v}" for k, v in sorted(tally.items())),
        })
        print(f"[{label}] seed=0 -> {seed0_clusters}개 | 최빈값 {mode_clusters}개({mode_freq}/50회) "
              f"| 범위 {min(counts)}~{max(counts)}개")

    pd.DataFrame(detail_rows).to_csv(OUTPUT / "seed_stability_detail.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT / "seed_stability_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n[output] {OUTPUT / 'seed_stability_summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
