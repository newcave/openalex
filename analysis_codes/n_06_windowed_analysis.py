"""[분석 6단계] 구간별 분석: 5년 비중복 구간에서, MK+BH 유의성 검정 대신 tau(방향)>0만으로
Rising을 정의해 Louvain을 돌린다. (n_03~n_05 전체기간 분석과는 독립적인 별도 분석)

입력: results/<분야>/01_corpus/
출력: results/<분야>/04_windowed/<시작연도>_<끝연도>/ + windows_summary.csv, windows_cluster_topterms.csv
실행: python -m analysis_codes.n_06_windowed_analysis   (인자 없이 실행하면 기본 분야 config.DEFAULT_DOMAIN)

배경(대화 기록, 2026-09-15): run_fdr_assoc_pipeline_windowed.py(archive/analysis_experiments)로 5년 비중복 구간(00-05,
06-10, ...)마다 독립적으로 MK+BH 검정을 했더니, 4/5 구간에서 BH 보정 후 Rising 용어가
0개였다. 원인을 파보니 "증가 방향(tau>0)"인 용어 자체는 구간당 643~1,086개(후보의 절반
이상)로 많았고, 문제는 "그 증가가 우연이 아니라고 통계적으로 확신할 수 있는가"였다 —
연도 관측치가 5~6개뿐이라 이 확신의 기준(BH 보정 후 유의성)을 넘는 용어가 거의 없었다.

이 스크립트는 사용자 지시대로 그 유의성 검정 단계를 건너뛰고, **tau>0(그냥 그 구간 안에서
방향이 증가)인 용어를 그대로 "Rising"으로 취급**해 association-strength 동시출현 네트워크를
만들고 Louvain을 돌린다. 이때 "Rising"은 더 이상 26년 전체 분석이나 run_fdr_assoc_pipeline_
windowed.py의 통계적 의미(유의 수준 하에서 우연이 아니라고 확인된 증가)가 아니라, "이 5년
구간 안에서 그냥 방향이 위였던 용어"라는 훨씬 약한 의미다 — n=5~6에서는 tau>0인 용어가
전체 후보의 절반 안팎이나 되므로(예: 2000-2005는 1,244개 중 693개), 사실상 "그 구간에서
자주 쓰인 용어 중 최근이 예전보다 조금이라도 더 많이 쓰인 절반"을 모은 것에 가깝다. 이
방식으로 나온 군집을 "통계적으로 유의하게 부상한 연구주제"라고 부르면 안 되고, "그 구간
안에서 상대적으로 증가 방향을 보인 용어들의 구조"로 서술해야 한다.

다른 조건(MIN_TOTAL_COUNT=15, MIN_COOCCUR=2, association-strength 정규화, Louvain
resolution=1.0/seed=0)은 26년 전체 분석·run_fdr_assoc_pipeline_windowed.py와 동일하게
유지했다.

"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis_codes.n_03_rising_terms import load_inputs, build_fdr_tags  # noqa: E402
from config import DEFAULT  # noqa: E402

OUTPUT = DEFAULT.windowed_output
MIN_TOTAL_COUNT = 15
MIN_COOCCUR = 2
LOUVAIN_SEED = 0
TOP_N_TERMS = 15
TARGET_WINDOW_SIZE = 5.5  # "5~6년 단위" 목표 — compute_windows가 실제 연도 범위에 맞춰 나눈다


def compute_windows(year_min: int, year_max: int, target_size: float = TARGET_WINDOW_SIZE) -> list[tuple[int, int]]:
    """전체 연도 범위를 5~6년 단위의 비중복 구간으로 나눈다.

    예전에는 WINDOWS = [(2000,2005), (2006,2010), ...]처럼 연도를 직접 하드코딩했는데,
    이러면 나중에 더 오래된 자료(예: 1900년대)가 추가되거나 매년 최신 연도가 쌓여도 구간이
    그대로 2000-2025에 고정돼버린다. 대신 실제 데이터의 연도 범위(year_min~year_max)를 받아서
    그때그때 5~6년 단위로 다시 나눈다. 나머지 연도는 앞쪽 구간부터 하나씩 더 배분해서(6년짜리가
    되게) 전부 5년 또는 6년 구간이 되도록 한다. year_min=2000, year_max=2025(26년)를 넣으면
    기존과 동일하게 [(2000,2005), (2006,2010), (2011,2015), (2016,2020), (2021,2025)]가 나온다."""
    total_years = year_max - year_min + 1
    if total_years <= 0:
        return []
    n_windows = max(1, round(total_years / target_size))
    base, remainder = divmod(total_years, n_windows)
    windows: list[tuple[int, int]] = []
    start = year_min
    for i in range(n_windows):
        size = base + (1 if i < remainder else 0)
        end = start + size - 1
        windows.append((start, end))
        start = end + 1
    return windows


def build_graph_from_terms(long: pd.DataFrame, terms: list[str]) -> nx.Graph:
    """MIN_COOCCUR 이상, association-strength 가중치의 동시출현 그래프를 만든다.
    Louvain은 이 그래프를 넘겨받은 쪽에서 seed를 골라 돌린다 — seed 반복분석(50회)을
    할 때 그래프 자체는 seed와 무관하므로 한 번만 만들어 재사용하기 위해 분리했다."""
    term_set = set(terms)
    graph_long = long[long["term"].isin(term_set)]
    doc_ids = graph_long["id"].astype("category")
    term_ids = graph_long["term"].astype("category")
    matrix = sparse.csr_matrix(
        (np.ones(len(graph_long)), (doc_ids.cat.codes, term_ids.cat.codes)),
        shape=(len(doc_ids.cat.categories), len(term_ids.cat.categories)),
    )
    co = (matrix.T @ matrix).tocoo()
    terms_index = term_ids.cat.categories
    doc_freq = np.asarray(matrix.sum(axis=0)).ravel()

    mask = (co.row < co.col) & (co.data >= MIN_COOCCUR)
    rows_i, rows_j, weights_raw = co.row[mask], co.col[mask], co.data[mask]
    denom = np.sqrt(doc_freq[rows_i] * doc_freq[rows_j])
    weights_assoc = weights_raw / denom

    graph = nx.Graph()
    graph.add_nodes_from(terms)
    for i, j, w_raw, w_assoc in zip(rows_i, rows_j, weights_raw, weights_assoc):
        graph.add_edge(terms_index[i], terms_index[j], weight=float(w_assoc), raw_cooccurrence=int(w_raw))
    return graph


def build_clusters_from_terms(long: pd.DataFrame, tags: pd.DataFrame, terms: list[str]) -> tuple[pd.DataFrame, dict]:
    """run_fdr_assoc_pipeline.build_clusters_association과 동일한 로직이되, 'Rising' 판정
    컬럼 대신 호출부가 넘겨준 term 목록을 그대로 node 집합으로 쓴다."""
    graph = build_graph_from_terms(long, terms)
    communities = sorted(
        nx.community.louvain_communities(graph, weight="weight", seed=LOUVAIN_SEED),
        key=len,
        reverse=True,
    )
    rows = [{"cluster": cid, "용어": term} for cid, members in enumerate(communities) for term in members]
    partition = pd.DataFrame(rows, columns=["cluster", "용어"]).merge(tags, on="용어", how="left")
    summary = {
        "rising_terms": len(terms),
        "edges": graph.number_of_edges(),
        "isolated_terms": sum(graph.degree(term) == 0 for term in graph.nodes),
        "clusters": len(communities),
        "cluster_sizes": "|".join(map(str, sorted((len(c) for c in communities), reverse=True))),
        "modularity": (
            nx.community.modularity(graph, communities, weight="weight")
            if communities and graph.number_of_edges() > 0
            else np.nan
        ),
        "weight_scheme": "association_strength",
        "rising_definition": "tau>0 (유의성 검정 없음)",
    }
    return partition, summary


def run_window(start: int, end: int, corpus: pd.DataFrame, long: pd.DataFrame,
                term_year_counts: pd.DataFrame, year_totals: pd.Series,
                output_dir: Path | None = None) -> tuple[dict, list[dict]]:
    label = f"{start}_{end}"
    years = [y for y in year_totals.index if start <= y <= end]
    counts_w = term_year_counts.loc[:, years]
    totals_w = year_totals.loc[years]

    tags_w = build_fdr_tags(counts_w, totals_w, min_total_count=MIN_TOTAL_COUNT)
    rising_terms = tags_w.loc[tags_w["MK_tau"] > 0, "용어"].tolist()

    long_w = long[long["year"].between(start, end)]
    partition_w, graph_summary = build_clusters_from_terms(long_w, tags_w, rising_terms)

    out_dir = (Path(output_dir) if output_dir else OUTPUT) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    tags_w.to_parquet(out_dir / "trend_tags_unigram.parquet", index=False)
    tags_w.to_csv(out_dir / "trend_tags_unigram.csv", index=False, encoding="utf-8-sig")

    out_partition = partition_w.rename(columns={"용어": "term"})
    cols = ["cluster", "term"] + [c for c in out_partition.columns if c not in ("cluster", "term")]
    out_partition = out_partition[cols]
    out_partition.to_parquet(out_dir / "tau_positive_unigram_clusters.parquet", index=False)

    n_docs = int(corpus.loc[corpus["year"].between(start, end), "id"].nunique())
    run_summary = {
        "window": label,
        "years": years,
        "n_years": len(years),
        "documents_in_window": n_docs,
        "min_total_count": MIN_TOTAL_COUNT,
        "min_cooccurrence": MIN_COOCCUR,
        "louvain_seed": LOUVAIN_SEED,
        "candidate_terms": int(len(tags_w)),
        "tau_positive_terms": int(len(rising_terms)),
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

    year_min, year_max = int(year_totals.index.min()), int(year_totals.index.max())
    windows = compute_windows(year_min, year_max)
    print(f"[windows] {year_min}-{year_max} ({year_max - year_min + 1}년) -> {len(windows)}개 구간: {windows}")

    summary_rows = []
    all_topterms = []
    for start, end in windows:
        run_summary, topterm_rows = run_window(start, end, corpus, long, term_year_counts, year_totals)
        summary_rows.append(run_summary)
        all_topterms.extend(topterm_rows)
        modularity = run_summary["modularity"]
        mod_str = f"{modularity:.3f}" if pd.notna(modularity) else "nan"
        print(f"[{start}-{end}] docs={run_summary['documents_in_window']} "
              f"tau>0={run_summary['tau_positive_terms']} clusters={run_summary['clusters']} "
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
