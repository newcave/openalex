"""FDR+association strength 최종 8개 군집을 pyvis로 시각화 (before/2606_statics_all/network_community.html
스타일 재현: 단일 spring layout + matplotlib tab20 팔레트 + 진한 남색 배경).

노드 선택 규칙(대화 기록, 2026-09-10):
- 군집 크기가 SMALL_CLUSTER_MAX 이하인 군집(여기선 16/4/2개짜리)은 전부 표시.
- 나머지 큰 군집은 sqrt(군집 크기)에 비례해 예산을 배분하되 최소 MIN_PER_BIG_CLUSTER개는 보장
  (정비례로 하면 가장 큰 군집이 화면을 거의 다 차지해 작은 군집의 세부 용어가 안 보이게 됨).
- 각 군집 안에서는 document_count(빈도) 상위 순으로 뽑는다.

레이아웃: 전체를 하나의 networkx spring_layout으로 배치한다(대시보드의 군집별 강제분리 레이아웃과
다른 방식). 끌어당기는 힘(가중치)은 association strength로 계산해 정규화 효과가 레이아웃에도
반영되게 하고, 엣지 두께는 원시 동시출현 횟수로 표시한다(사람이 "실제 몇 번 같이 나왔는지" 감을
잡기엔 원시 횟수가 더 직관적이라 표시값과 배치용 가중치를 분리했다).

사용법: python -m analysis.visualize_fdr_assoc_network
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from pyvis.network import Network
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TERMS_CSV = ROOT / "reports" / "rising_cluster_review_fdr_assoc" / "terms.csv"
INTERPRETATIONS = ROOT / "reports" / "rising_cluster_review_fdr_assoc" / "interpretations.json"
SOURCE = ROOT / "analysis" / "output_임계15"
OUT_HTML = ROOT / "reports" / "rising_cluster_review_fdr_assoc" / "network.html"

SMALL_CLUSTER_MAX = 20  # 이 이하 크기의 군집은 전부 표시
MIN_PER_BIG_CLUSTER = 15  # 큰 군집이라도 최소 이만큼은 보여줌
BIG_CLUSTER_BUDGET = 150  # 큰 군집들에 배분할 노드 수 총합(대략치, sqrt 비례 배분의 기준)
MIN_COOCCUR = 2
SAME_CLUSTER_BOOST = 20  # 레이아웃 인력 계산 시 같은 군집 엣지에 곱하는 배율.
# 실측(scratchpad/tune_cluster_boost.py, 2026-09-10): boost=1(가중치 그대로)이면 "최근접 이웃이
# 같은 군집일 비율"이 33.9%에 불과해 군집끼리 뭉쳐 보이지 않았다. boost를 올릴수록 이 비율이
# 올라가다 20 근방부터 둔화됐다(20→90.2%, 30→92.5%, 이득 대비 왜곡 증가). 대시보드의 강제 원형
# 분리 배치(사용자가 별로라고 평가) 대신, 같은 단일 spring layout 안에서 군집 내부 인력만
# 세게 줘서 자연스럽게 뭉치게 하는 방식 — 노드 위치를 강제로 특정 좌표에 고정하지 않는다.


def select_nodes(terms: pd.DataFrame) -> pd.DataFrame:
    sizes = terms.groupby("cluster").size()
    big_clusters = sizes[sizes > SMALL_CLUSTER_MAX]
    weights = np.sqrt(big_clusters.to_numpy(dtype=float))
    raw_alloc = BIG_CLUSTER_BUDGET * weights / weights.sum()
    n_per_big = {cid: max(MIN_PER_BIG_CLUSTER, round(n)) for cid, n in zip(big_clusters.index, raw_alloc)}

    picked = []
    for cid, group in terms.groupby("cluster"):
        n = len(group) if cid not in n_per_big else min(n_per_big[cid], len(group))
        picked.append(group.sort_values("document_count", ascending=False).head(n))
    result = pd.concat(picked, ignore_index=True)
    print("군집별 선택 노드 수:")
    print(result.groupby("cluster").size().to_string())
    print(f"전체 노드 수: {len(result)}")
    return result


def build_subgraph(selected_terms: list[str], doc_freq: dict[str, int]) -> nx.Graph:
    corpus = pd.read_parquet(SOURCE / "filtered_corpus.parquet", columns=["id", "language"])
    analyzable_ids = set(corpus.loc[corpus["language"] == "en", "id"])
    mapping_df = pd.read_csv(SOURCE / "term_normalization_dict.csv")
    mapping = dict(zip(mapping_df["variant"], mapping_df["canonical"]))

    long = pd.read_parquet(SOURCE / "unigram_long.parquet", columns=["id", "term"])
    long = long[long["id"].isin(analyzable_ids)]
    long["term"] = long["term"].map(lambda t: mapping.get(t, t))
    long = long[long["term"].isin(selected_terms)].drop_duplicates(["id", "term"])

    doc_ids = long["id"].astype("category")
    term_ids = long["term"].astype("category")
    matrix = sparse.csr_matrix(
        (np.ones(len(long)), (doc_ids.cat.codes, term_ids.cat.codes)),
        shape=(len(doc_ids.cat.categories), len(term_ids.cat.categories)),
    )
    co = (matrix.T @ matrix).tocoo()
    terms_index = term_ids.cat.categories

    graph = nx.Graph()
    graph.add_nodes_from(selected_terms)
    mask = (co.row < co.col) & (co.data >= MIN_COOCCUR)
    for i, j, w in zip(co.row[mask], co.col[mask], co.data[mask]):
        a, b = terms_index[i], terms_index[j]
        assoc = w / math.sqrt(doc_freq[a] * doc_freq[b])
        graph.add_edge(a, b, raw=int(w), assoc=float(assoc))
    return graph


def cluster_palette(n_clusters: int) -> list[str]:
    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "gist_rainbow")
    return [mcolors.to_hex(cmap(i / max(n_clusters - 1, 1))) for i in range(n_clusters)]


def main() -> None:
    terms = pd.read_csv(TERMS_CSV)
    interp = json.loads(INTERPRETATIONS.read_text(encoding="utf-8"))
    label_by_cluster = {c["cluster"]: c["label"] for c in interp["clusters"]}

    selected = select_nodes(terms)
    doc_freq = dict(zip(terms["term"], terms["document_count"]))
    cluster_by_term = dict(zip(selected["term"], selected["cluster"]))
    graph = build_subgraph(selected["term"].tolist(), doc_freq)
    n_isolated = sum(1 for n in graph.nodes if graph.degree(n) == 0)
    print(f"엣지 {graph.number_of_edges():,}개, 고립노드 {n_isolated}개")

    # 레이아웃 전용 가중치: 같은 군집 엣지는 SAME_CLUSTER_BOOST배로 세게 당겨 뭉쳐 보이게 하고,
    # 실제 데이터값(assoc/raw)은 그대로 둔다(툴팁·엣지 두께는 원본 값을 쓴다).
    for u, v, data in graph.edges(data=True):
        same = cluster_by_term[u] == cluster_by_term[v]
        data["layout_weight"] = data["assoc"] * (SAME_CLUSTER_BOOST if same else 1.0)

    pos = nx.spring_layout(graph, weight="layout_weight", k=2.2, seed=42, iterations=300)
    scale = 900

    cluster_ids = sorted(selected["cluster"].unique())
    palette = dict(zip(cluster_ids, cluster_palette(len(cluster_ids))))
    doc_count_by_term = dict(zip(selected["term"], selected["document_count"]))
    max_doc = max(doc_count_by_term.values())

    net = Network(height="850px", width="100%", bgcolor="#1a1a2e", font_color="white", cdn_resources="in_line")
    net.toggle_physics(False)
    for node in graph.nodes():
        x, y = pos[node]
        cid = cluster_by_term[node]
        count = doc_count_by_term[node]
        size = 10 + 32 * (count / max_doc) ** 0.5
        net.add_node(
            node, label=node, size=float(size),
            x=float(x * scale), y=float(y * scale), physics=False,
            color=palette[cid],
            title=f"<b>{node}</b><br>군집 {cid}: {label_by_cluster.get(cid, '')}<br>문서 수: {count:,}건",
        )
    max_raw = max((d["raw"] for _, _, d in graph.edges(data=True)), default=1)
    for u, v, data in graph.edges(data=True):
        w = data["raw"]
        net.add_edge(u, v, width=min(w / max_raw * 6, 6), color="rgba(180,180,180,0.35)",
                     title=f"동시출현 {w:,}건")

    legend = " &nbsp;·&nbsp; ".join(
        f'<span style="color:{palette[c]}">●</span> 군집{c}: {label_by_cluster.get(c, "")}' for c in cluster_ids
    )
    html = net.generate_html(notebook=False)
    html = html.replace(
        "<body>",
        f'<body><div style="padding:10px 16px;background:#1a1a2e;color:#eee;'
        f'font-family:\'Segoe UI\',\'Noto Sans KR\',sans-serif;font-size:13px;'
        f'border-bottom:1px solid #333;">{legend}</div>',
        1,
    )
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"저장: {OUT_HTML}")


if __name__ == "__main__":
    main()
