"""[분석 4단계] 전체기간 커뮤니티 도출: n_03이 고른 Rising 용어로 동시출현 네트워크를 만들고
(association strength 정규화) Louvain으로 연구주제 커뮤니티를 찾은 뒤 연도별 추이를 계산한다.

입력: results/<분야>/01_corpus/
출력: results/<분야>/02_full_period/ (trend_tags_unigram_fdr, rising_unigram_clusters, cluster_trends_*)
실행: python -m analysis_codes.n_04_cluster_network   (인자 없이 실행하면 기본 분야. 웹앱은 build_clusters_association만 호출)

배경(대화 기록, 2026-09-09):
- 이전 방식(원시 동시출현 횟수 가중치, archive/analysis_experiments 참고)까지는 Benjamini-Hochberg FDR 보정으로 다중검정 문제를 해결했지만,
  Rising 용어 동시출현 네트워크의 modularity가 0.19 안팎으로 낮았다(min_count_sensitivity.csv에서도
  10/15/20 어느 값에서나 동일하게 낮음). resolution 파라미터를 0.5~10.0까지 실측으로 훑어봤지만
  detection resolution을 올릴수록 표준 modularity(resolution=1 기준)는 오히려 떨어졌다
  (resolution=1.0에서 0.192로 최댓값, 10.0에서 0.050) — "더 잘게 쪼개면 modularity가 오른다"는
  추정은 이 데이터에서는 틀렸음을 실측으로 확인.
- 대신 엣지 가중치를 원시 동시출현 횟수 대신 association strength로 정규화(Van Eck & Waltman,
  "How to normalize cooccurrence data?", JASIST 2009 — VOSviewer가 기본으로 쓰는 정규화 방식)했더니
  min_cooccur=2, resolution=1.0에서 modularity가 0.192 → 0.300으로 실측 개선됨(군집 5→8개,
  고립노드 0개). min_cooccur=3으로 올리면 modularity는 더 오르지만(0.32) 군집이 16~22개로
  과도하게 쪼개지고 고립노드가 생겨 해석 가능성이 오히려 나빠져 채택하지 않음.
- association strength: w'_ij = c_ij / sqrt(freq_i * freq_j). 허브 용어(예: water)가 등장 빈도가
  높다는 이유만으로 아무 용어와나 강하게 연결되는 걸 보정한다.

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

from analysis_codes.n_03_rising_terms import load_inputs, build_fdr_tags, build_cluster_trends, FDR_LABELS  # noqa: E402
from config import DEFAULT  # noqa: E402

OUTPUT = DEFAULT.full_output
MIN_COOCCUR = 2  # 실측 결과 채택값 (min_cooccur=3은 과분할이라 기각)
LOUVAIN_SEED = 0


def build_clusters_association(long: pd.DataFrame, tags: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rising = tags.loc[tags["추세(MK-FDR)"].eq("Rising"), "용어"].tolist()
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
    doc_freq = np.asarray(matrix.sum(axis=0)).ravel()

    mask = (co.row < co.col) & (co.data >= MIN_COOCCUR)
    rows_i, rows_j, weights_raw = co.row[mask], co.col[mask], co.data[mask]
    denom = np.sqrt(doc_freq[rows_i] * doc_freq[rows_j])
    weights_assoc = weights_raw / denom

    graph = nx.Graph()
    graph.add_nodes_from(rising)
    for i, j, w_raw, w_assoc in zip(rows_i, rows_j, weights_raw, weights_assoc):
        graph.add_edge(terms_index[i], terms_index[j], weight=float(w_assoc), raw_cooccurrence=int(w_raw))

    communities = sorted(
        nx.community.louvain_communities(graph, weight="weight", seed=LOUVAIN_SEED),
        key=len,
        reverse=True,
    )
    rows = [{"cluster": cid, "용어": term} for cid, members in enumerate(communities) for term in members]
    partition = pd.DataFrame(rows, columns=["cluster", "용어"]).merge(tags, on="용어", how="left")
    summary = {
        "rising_terms": len(rising),
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
        "min_cooccurrence": MIN_COOCCUR,
    }
    return partition, summary


def main() -> None:
    corpus, long, counts, year_totals = load_inputs()
    tags = build_fdr_tags(counts, year_totals)
    partition, graph_summary = build_clusters_association(long, tags)
    series, summary = build_cluster_trends(corpus, long, partition)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    tags.to_parquet(OUTPUT / "trend_tags_unigram_fdr.parquet", index=False)
    tags.to_csv(OUTPUT / "trend_tags_unigram_fdr.csv", index=False, encoding="utf-8-sig")
    # review_rising_clusters류 스크립트가 기대하는 형식(cluster, term)에 맞춤 — 두 번째 컬럼이 term
    out_partition = partition.rename(columns={"용어": "term"})
    cols = ["cluster", "term"] + [c for c in out_partition.columns if c not in ("cluster", "term")]
    out_partition = out_partition[cols]
    out_partition.to_parquet(OUTPUT / "rising_unigram_clusters.parquet", index=False)
    series.to_parquet(OUTPUT / "cluster_trends_fdr_assoc.parquet", index=False)
    summary.to_csv(OUTPUT / "cluster_trends_summary_fdr_assoc.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "run_summary.json").write_text(json.dumps({
        "source": f"{DEFAULT.shared_source.relative_to(ROOT).as_posix()} (unigram_long/filtered_corpus/term_normalization_dict)",
        "fdr_method": "Benjamini-Hochberg",
        "fdr_alpha": 0.05,
        "louvain_seed": LOUVAIN_SEED,
        "raw_rising_terms": int(tags["추세(MK)"].eq("Rising").sum()),
        "fdr_rising_terms": int(tags["추세(MK-FDR)"].eq("Rising").sum()),
        **graph_summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **graph_summary}, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
