# analysis/explore_louvain_resolution.py
# cooccurrence_cluster.py에서 resolution=1(기본값)로 돌렸더니 클러스터 4개가 전체 1,471개
# 용어 중 1,447개(98%)를 먹어버리는 문제가 있어(대화 기록 참고), resolution을 감으로
# 올리지 않고 실제로 값을 바꿔가며 클러스터 개수·크기 분포가 어떻게 변하는지 보고 정하기
# 위한 1회성 탐색 스크립트.
#
# 사용법: python -m analysis.explore_louvain_resolution

import networkx as nx
import pandas as pd

from analysis.cooccurrence_cluster import build_graph, LOUVAIN_SEED


def summarize(communities: list[set]) -> dict:
    sizes = sorted((len(c) for c in communities), reverse=True)
    n = len(sizes)
    return {
        "클러스터수": n,
        "최대크기": sizes[0],
        "상위3개_합": sum(sizes[:3]),
        "상위3개_비율": f"{sum(sizes[:3]) / sum(sizes):.1%}",
        "크기1~2(사실상노이즈)": sum(1 for s in sizes if s <= 2),
        "크기3~9": sum(1 for s in sizes if 3 <= s <= 9),
        "크기10~49": sum(1 for s in sizes if 10 <= s <= 49),
        "크기50~199": sum(1 for s in sizes if 50 <= s <= 199),
        "크기200이상": sum(1 for s in sizes if s >= 200),
    }


def main() -> None:
    G, _ = build_graph()
    print(f"그래프: 노드 {G.number_of_nodes():,}개, 엣지 {G.number_of_edges():,}개\n")

    rows = []
    for res in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
        communities = nx.community.louvain_communities(G, weight="weight", resolution=res, seed=LOUVAIN_SEED)
        row = {"resolution": res}
        row.update(summarize(communities))
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
