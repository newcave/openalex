# analysis/cooccurrence_cluster.py
# Rising 유니그램 용어들의 동시출현 네트워크를 만들고 Louvain 커뮤니티 탐지로 클러스터링한다.
# 방법론(동시출현 네트워크 + modularity 기반 커뮤니티 탐지로 "연구주제 클러스터" 찾기)은
# 계량서지학 co-word analysis 표준 기법(VOSviewer가 쓰는 방식, 원조는 Callon의 co-word
# analysis) — 대화 기록에서 근거·트레이드오프 논의.
#
# 엣지 임계값(동시출현 2회 이상만 연결로 인정)은 감으로 정한 게 아니라 실제 분포를 보고
# 정함(analysis/explore_cooccurrence_threshold.py 결과, 2026-09-01) — 1회성 우연 조합은
# 걸러내면서 Rising 용어 1,471개를 하나도 고립시키지 않는 지점.
#
# modularity 자체가 "허브 용어(water 등)는 원래 뭐랑도 잘 엮인다"는 것을 감안해 기대값을
# 빼고 계산하므로, 동시출현 횟수를 따로 정규화(association strength 등)하지 않고 원시
# 가중치 그래프에 그대로 적용한다 (대화 기록 참고).
#
# 사용법: python -m analysis.cooccurrence_cluster

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

SOURCE_DIR = r"analysis\output_임계15"
MIN_COOCCUR = 2  # explore_cooccurrence_threshold.py 결과로 확정한 임계값
LOUVAIN_SEED = 0  # 재현성 (Louvain은 초기화에 따라 결과가 살짝 달라질 수 있어 시드 고정)

OUT_PATH = r"analysis\output_임계15\rising_unigram_clusters.parquet"


def build_graph():
    tags = pd.read_parquet(rf"{SOURCE_DIR}\trend_tags_unigram_all.parquet")
    rising = tags[tags["추세(MK)"] == "Rising"].set_index("용어")
    rising_terms = rising.index.tolist()

    mapping_df = pd.read_csv(rf"{SOURCE_DIR}\term_normalization_dict.csv")
    mapping = dict(zip(mapping_df["variant"], mapping_df["canonical"]))

    corpus = pd.read_parquet(rf"{SOURCE_DIR}\filtered_corpus.parquet", columns=["id", "language"])
    analyzable_ids = set(corpus.loc[corpus["language"] == "en", "id"])

    long_df = pd.read_parquet(rf"{SOURCE_DIR}\unigram_long.parquet", columns=["id", "term"])
    long_df = long_df[long_df["id"].isin(analyzable_ids)]
    long_df = long_df.assign(term=long_df["term"].map(lambda t: mapping.get(t, t)))
    long_df = long_df[long_df["term"].isin(rising_terms)].drop_duplicates(["id", "term"])

    doc_ids = long_df["id"].astype("category")
    term_ids = long_df["term"].astype("category")
    M = sparse.csr_matrix(
        (np.ones(len(long_df)), (doc_ids.cat.codes, term_ids.cat.codes)),
        shape=(len(doc_ids.cat.categories), len(term_ids.cat.categories)),
    )
    co = (M.T @ M).tocoo()
    terms_index = term_ids.cat.categories

    G = nx.Graph()
    G.add_nodes_from(rising_terms)  # 엣지가 하나도 없는 용어도 노드로는 남겨둠 (고립 노드로 표시)

    mask = (co.row < co.col) & (co.data >= MIN_COOCCUR)
    for i, j, w in zip(co.row[mask], co.col[mask], co.data[mask]):
        G.add_edge(terms_index[i], terms_index[j], weight=float(w))

    return G, rising


def main():
    G, rising = build_graph()
    n_isolated = sum(1 for n in G.nodes if G.degree(n) == 0)
    print(f"노드(Rising 용어) {G.number_of_nodes():,}개, 엣지 {G.number_of_edges():,}개, "
          f"고립 노드 {n_isolated}개")

    communities = nx.community.louvain_communities(G, weight="weight", seed=LOUVAIN_SEED)
    communities = sorted(communities, key=len, reverse=True)
    print(f"\n클러스터 {len(communities)}개 발견")

    sizes = sorted((len(c) for c in communities), reverse=True)
    print(f"클러스터 크기 분포: 최대 {sizes[0]}, 최소 {sizes[-1]}, "
          f"1개짜리(고립) 클러스터 {sum(1 for s in sizes if s == 1)}개")

    rows = []
    for cid, members in enumerate(communities):
        for term in members:
            rows.append({"cluster": cid, "용어": term})
    cluster_df = pd.DataFrame(rows).merge(
        rising[["전체기간_평균비중(%)", "Sen_slope(%p_연)", "MK_tau"]].reset_index(), on="용어"
    )
    cluster_df.to_parquet(OUT_PATH, index=False)
    print(f"\n저장 완료: {OUT_PATH}")

    print("\n=== 상위 15개 클러스터 (크기 순), 각 클러스터의 평균비중 상위 10개 용어 ===")
    for cid in range(min(15, len(communities))):
        sub = cluster_df[cluster_df["cluster"] == cid].sort_values("전체기간_평균비중(%)", ascending=False)
        top_terms = ", ".join(sub["용어"].head(10).tolist())
        print(f"\n[클러스터 {cid}] {len(sub)}개 용어 — 대표: {top_terms}")


if __name__ == "__main__":
    main()
