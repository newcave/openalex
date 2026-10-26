"""엣지 가중치 3종(raw / cosine / association strength) ablation — 논문 §5.5.1 근거 생성.

배경: run_fdr_assoc_pipeline.py가 쓰는 가중치 c_ij / sqrt(f_i*f_j)는 코드 주석과 달리
association strength가 아니라 **cosine(Salton)** 이다. van Eck & Waltman(2009)의
association strength는 c_ij / (f_i*f_j) 이며, 그 논문은 cosine보다 association strength를
권한다. 이 스크립트는 세 가중치를 동일 조건에서 비교한다.

핵심 설계 — 분할과 평가를 분리한다:
  modularity Q는 엣지 가중치를 바꾸면 '다른 그래프의 목적함수 값'이 된다. Louvain은 주어진
  가중치에서 Q를 최대화하므로, 각 분할이 자기 가중치에서 1등인 것은 정의상 당연하다.
  따라서 가중치 A로 찾은 분할을 가중치 B로도 평가하는 3x3 교차표를 만들어야
  "정규화가 구조를 개선했는가"를 동일 잣대로 판정할 수 있다.

조건 통제: 노드(FDR-Rising)·엣지(min_cooccur=2)·Louvain 시드가 셋 다 동일하고 가중치만 다르다.

원본 파일은 읽기만 한다. 출력은 reports/weight_scheme_*.csv 4개.

사용법: python -m analysis.weight_scheme_ablation
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from math import lgamma, log
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import build_dataset, normalization  # noqa: E402

SOURCE = ROOT / "analysis" / "output_임계15"
FINAL = ROOT / "analysis" / "output_fdr_assoc"
OUT = ROOT / "reports"
MIN_COOCCUR = 2
LOUVAIN_SEED = 0
TOP_TERMS = 12


# ---------------------------------------------------------------- 입력
def load_graph_inputs():
    """run_fdr_pipeline.load_inputs()와 동일한 방식으로 long 테이블을 만들고,
    이미 저장된 FDR 태그에서 Rising 용어를 읽는다(MK 재계산 비용 절약, 결과는 동일)."""
    long = pd.read_parquet(SOURCE / "unigram_long.parquet", columns=["id", "year", "term"])
    long["term"] = normalization.apply_dict(long["term"], normalization.load_dict())
    long = long.drop_duplicates(["id", "term"])

    tags = pd.read_parquet(FINAL / "trend_tags_unigram_fdr.parquet")
    rising = tags.loc[tags["추세(MK-FDR)"].eq("Rising"), "용어"].tolist()

    graph_long = long[long["term"].isin(set(rising))]
    doc_ids = graph_long["id"].astype("category")
    term_ids = graph_long["term"].astype("category")
    matrix = sparse.csr_matrix(
        (np.ones(len(graph_long)), (doc_ids.cat.codes, term_ids.cat.codes)),
        shape=(len(doc_ids.cat.categories), len(term_ids.cat.categories)),
    )
    co = (matrix.T @ matrix).tocoo()
    doc_freq = np.asarray(matrix.sum(axis=0)).ravel()
    mask = (co.row < co.col) & (co.data >= MIN_COOCCUR)
    return rising, term_ids.cat.categories, doc_freq, co.row[mask], co.col[mask], co.data[mask], matrix.shape[0]


def weight_schemes(cooc, f_i, f_j):
    """세 가중치. association strength는 VOSviewer 관례대로 2m배 스케일하지만,
    modularity는 전역 상수배에 불변이므로 분할 결과는 스케일과 무관하다(main에서 검증)."""
    return {
        "raw": cooc.astype(float),
        "cosine": cooc / np.sqrt(f_i * f_j),
        "assoc_strength": (cooc / (f_i * f_j)) * (2.0 * cooc.sum()),
    }


# ---------------------------------------------------------------- 분할 비교 지표
def _labels(communities, nodes):
    lookup = {t: cid for cid, members in enumerate(communities) for t in members}
    return np.array([lookup[t] for t in nodes])


def _contingency(a, b):
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    for x, y in zip(a, b):
        table[x, y] += 1
    return table


def _comb2(x):
    return x * (x - 1) / 2.0


def adjusted_rand(a, b) -> float:
    """Hubert & Arabie(1985) adjusted Rand index."""
    n = _contingency(a, b)
    total = n.sum()
    index = _comb2(n).sum()
    exp_a, exp_b = _comb2(n.sum(1)).sum(), _comb2(n.sum(0)).sum()
    expected = exp_a * exp_b / _comb2(total)
    maximum = 0.5 * (exp_a + exp_b)
    return (index - expected) / (maximum - expected)


def adjusted_mutual_info(a, b) -> float:
    """Vinh, Epps & Bailey(2010) adjusted mutual information (max 정규화)."""
    n = _contingency(a, b)
    total = int(n.sum())
    rows, cols = n.sum(1).astype(int), n.sum(0).astype(int)

    mi = 0.0
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            if n[i, j]:
                mi += (n[i, j] / total) * log(n[i, j] * total / (rows[i] * cols[j]))

    def logfact(x):
        return lgamma(x + 1)

    emi = 0.0
    for i in range(len(rows)):
        for j in range(len(cols)):
            lo = max(1, rows[i] + cols[j] - total)
            hi = min(rows[i], cols[j])
            for nij in range(lo, hi + 1):
                term = (nij / total) * log(nij * total / (rows[i] * cols[j]))
                logp = (logfact(rows[i]) + logfact(cols[j])
                        + logfact(total - rows[i]) + logfact(total - cols[j])
                        - logfact(total) - logfact(nij) - logfact(rows[i] - nij)
                        - logfact(cols[j] - nij) - logfact(total - rows[i] - cols[j] + nij))
                emi += term * np.exp(logp)

    def entropy(counts):
        p = counts[counts > 0] / total
        return -(p * np.log(p)).sum()

    h_a, h_b = entropy(rows), entropy(cols)
    return (mi - emi) / (max(h_a, h_b) - emi)


# ---------------------------------------------------------------- main
def main() -> None:
    rising, terms_index, doc_freq, rows_i, rows_j, cooc, n_docs = load_graph_inputs()
    schemes = weight_schemes(cooc, doc_freq[rows_i], doc_freq[rows_j])
    print(f"FDR-Rising 용어 {len(rising)}개 | 엣지 {len(cooc):,}개 | 문서 {n_docs:,}건")

    graphs = {}
    for name, weights in schemes.items():
        graph = nx.Graph()
        graph.add_nodes_from(rising)
        for i, j, w, c in zip(rows_i, rows_j, weights, cooc):
            graph.add_edge(terms_index[i], terms_index[j], weight=float(w), raw=int(c))
        graphs[name] = graph

    partitions = {}
    for name, graph in graphs.items():
        communities = sorted(
            nx.community.louvain_communities(graph, weight="weight", seed=LOUVAIN_SEED),
            key=len, reverse=True,
        )
        partitions[name] = communities
        isolated = sum(1 for node in graph.nodes if graph.degree(node) == 0)
        sizes = "|".join(str(len(c)) for c in communities)
        print(f"[{name:15s}] 군집 {len(communities):2d}개 | 고립 {isolated} | 크기 {sizes}")

    # association strength의 2m 스케일이 결과에 영향을 주지 않는지 확인
    unscaled = nx.Graph()
    unscaled.add_nodes_from(rising)
    for i, j, w in zip(rows_i, rows_j, cooc / (doc_freq[rows_i] * doc_freq[rows_j])):
        unscaled.add_edge(terms_index[i], terms_index[j], weight=float(w))
    same = ([set(c) for c in sorted(nx.community.louvain_communities(
        unscaled, weight="weight", seed=LOUVAIN_SEED), key=len, reverse=True)]
        == [set(c) for c in partitions["assoc_strength"]])
    print(f"\nassociation strength 스케일 불변성(2m배 vs 원본): 동일 분할 = {same}")

    # 3x3 교차 평가 — 이 표가 이 스크립트의 핵심 산출물
    cross_rows = []
    for p_name, communities in partitions.items():
        row = {
            "partition_from": p_name,
            "n_clusters": len(communities),
            "cluster_sizes": "|".join(str(len(c)) for c in communities),
        }
        for g_name, graph in graphs.items():
            row[f"Q_under_{g_name}"] = round(
                nx.community.modularity(graph, communities, weight="weight"), 6)
        cross_rows.append(row)
    cross = pd.DataFrame(cross_rows)
    print("\n=== 분할(행) x 평가 가중치(열) modularity ===")
    print(cross.to_string(index=False))

    agree = pd.DataFrame([
        {"a": a, "b": b,
         "ARI": round(adjusted_rand(_labels(partitions[a], rising), _labels(partitions[b], rising)), 4),
         "AMI": round(adjusted_mutual_info(_labels(partitions[a], rising), _labels(partitions[b], rising)), 4)}
        for a, b in combinations(partitions, 2)
    ])
    print("\n=== 분할 간 일치도 ===")
    print(agree.to_string(index=False))

    freq = dict(zip(terms_index, doc_freq))
    cluster_rows = [
        {"scheme": name, "cluster": cid, "size": len(members),
         "top_terms": " | ".join(sorted(members, key=lambda t: -freq.get(t, 0))[:TOP_TERMS])}
        for name, communities in partitions.items()
        for cid, members in enumerate(communities)
    ]

    crosstab = pd.DataFrame(_contingency(_labels(partitions["cosine"], rising),
                                         _labels(partitions["assoc_strength"], rising)))
    crosstab.index.name, crosstab.columns.name = "cosine", "assoc"

    OUT.mkdir(parents=True, exist_ok=True)
    cross.to_csv(OUT / "weight_scheme_ablation.csv", index=False, encoding="utf-8-sig")
    agree.to_csv(OUT / "weight_scheme_agreement.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cluster_rows).to_csv(OUT / "weight_scheme_clusters.csv", index=False, encoding="utf-8-sig")
    crosstab.to_csv(OUT / "weight_scheme_crosstab_cosine_assoc.csv", encoding="utf-8-sig")

    print("\n" + json.dumps({
        "fdr_rising_terms": len(rising), "edges": int(len(cooc)), "documents": int(n_docs),
        "min_cooccurrence": MIN_COOCCUR, "louvain_seed": LOUVAIN_SEED,
        "assoc_scale_invariant": bool(same),
    }, ensure_ascii=False, indent=2))
    print(f"\n저장: {OUT / 'weight_scheme_ablation.csv'} 외 3개")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
