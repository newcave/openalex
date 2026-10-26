"""군집별 하위구조 진단 — 세분화 후보를 안정성으로 판정한다 (논문 §5.10 근거).

왜 필요한가: 시드 안정성(§5.9)은 군집의 '경계'를 믿을 수 있는지를 재는 것이지,
'내부가 여러 주제로 섞여 있는지'를 재는 것이 아니다. 둘은 다른 질문이다. 낮은 안정성의
원인은 두 가지이고 처방이 정반대다.

  (a) 뚜렷한 하위 덩어리가 여럿  -> 시드마다 갈라지는 위치가 '일관됨'  -> 세분화가 옳다
  (b) 연결이 사방으로 흩어진 잔여 군집 -> 하위 분할도 매번 다름       -> 세분화하면 쓰레기가 나온다

따라서 세분화 여부는 "그 군집의 유도 부분그래프에서 찾은 하위 분할이 시드에 걸쳐
재현되는가"로 판정해야 한다. 재현되면 실재하는 하위구조이고, 아니면 잔여 군집이다.

주의: 부분그래프의 modularity는 그 부분그래프의 총 가중치를 기준으로 정규화되므로
전체 그래프의 Q와 비교할 수 없다. 여기서는 '세분화 정당성'의 근거로 Q를 쓰지 않고
하위 분할의 재현성(ARI)만 쓴다.

원본 파일은 읽기만 한다. 출력은 reports/subcluster_stability*.csv 2개.

사용법: python -m analysis.subcluster_stability
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.weight_scheme_ablation import (  # noqa: E402
    adjusted_rand, load_graph_inputs, weight_schemes,
)

OUT = ROOT / "reports"
N_SEEDS = 50
REFERENCE_SEED = 0
SCHEME = "cosine"       # 주 결과의 가중치
MIN_SIZE = 10           # 이보다 작은 군집은 하위 분할 진단이 무의미
TOP_TERMS = 8


def main() -> None:
    rising, terms_index, doc_freq, rows_i, rows_j, cooc, _ = load_graph_inputs()
    weights = weight_schemes(cooc, doc_freq[rows_i], doc_freq[rows_j])[SCHEME]

    graph = nx.Graph()
    graph.add_nodes_from(rising)
    for i, j, w in zip(rows_i, rows_j, weights):
        graph.add_edge(terms_index[i], terms_index[j], weight=float(w))

    reference = sorted(
        nx.community.louvain_communities(graph, weight="weight", seed=REFERENCE_SEED),
        key=len, reverse=True,
    )
    print(f"기준 분할({SCHEME}, seed={REFERENCE_SEED}): {len(reference)}개 군집\n")

    freq = dict(zip(terms_index, doc_freq))
    summary_rows, member_rows = [], []

    for cid, members in enumerate(reference):
        if len(members) < MIN_SIZE:
            summary_rows.append({
                "cluster": cid, "size": len(members), "verdict": "too_small",
                "sub_ari_mean": None, "sub_clusters_mode": None,
                "note": f"{MIN_SIZE}개 미만 — 하위 분할 진단 대상 아님",
            })
            continue

        sub = graph.subgraph(members).copy()
        nodes = sorted(sub.nodes)
        index = {t: k for k, t in enumerate(nodes)}

        partitions, counts = [], []
        for seed in range(N_SEEDS):
            comms = sorted(nx.community.louvain_communities(sub, weight="weight", seed=seed),
                           key=len, reverse=True)
            partitions.append(comms)
            counts.append(len(comms))

        labels = []
        for comms in partitions:
            lookup = {t: k for k, mem in enumerate(comms) for t in mem}
            labels.append(np.array([lookup[t] for t in nodes]))

        pair_ari = [adjusted_rand(labels[a], labels[b])
                    for a, b in combinations(range(N_SEEDS), 2)]
        ari_mean = float(np.mean(pair_ari))

        # 하위 분할이 안정적이면 '실재하는 하위구조', 불안정하면 '잔여 군집'
        if ari_mean >= 0.70:
            verdict = "세분화 타당 (하위구조 재현됨)"
        elif ari_mean >= 0.50:
            verdict = "경계선 — 추가 근거 필요"
        else:
            verdict = "세분화 부적절 (잔여 군집)"

        co = np.zeros((len(nodes), len(nodes)), dtype=np.int32)
        for lab in labels:
            co += (lab[:, None] == lab[None, :])

        ref_sub = partitions[REFERENCE_SEED]
        for sid, sub_members in enumerate(ref_sub):
            idx = np.array([index[t] for t in sub_members])
            if len(idx) > 1:
                block = co[np.ix_(idx, idx)]
                iu = np.triu_indices(len(idx), k=1)
                cohesion = float(block[iu].mean()) / N_SEEDS
            else:
                cohesion = float("nan")
            top = sorted(sub_members, key=lambda t: -freq.get(t, 0))[:TOP_TERMS]
            member_rows.append({
                "cluster": cid, "subcluster": sid, "size": len(sub_members),
                "internal_comembership": round(cohesion, 4) if cohesion == cohesion else None,
                "top_terms": " | ".join(top),
            })

        summary_rows.append({
            "cluster": cid, "size": len(members), "verdict": verdict,
            "sub_ari_mean": round(ari_mean, 4),
            "sub_ari_sd": round(float(np.std(pair_ari)), 4),
            "sub_ari_min": round(float(np.min(pair_ari)), 4),
            "sub_clusters_mode": int(pd.Series(counts).mode().iloc[0]),
            "sub_clusters_min": min(counts), "sub_clusters_max": max(counts),
            "note": "",
        })
        print(f"[군집 {cid}] n={len(members):3d} 하위군집 {min(counts)}~{max(counts)}개 "
              f"(최빈 {pd.Series(counts).mode().iloc[0]}) | 하위분할 ARI {ari_mean:.4f} → {verdict}")

    summary = pd.DataFrame(summary_rows)
    members_df = pd.DataFrame(member_rows)

    print("\n=== 세분화 판정 요약 ===")
    print(summary.to_string(index=False))
    print(f"\n=== 기준 시드의 하위군집 구성 (상위 {TOP_TERMS} 용어) ===")
    for cid, grp in members_df.groupby("cluster"):
        print(f"\n[군집 {cid}]")
        print(grp[["subcluster", "size", "internal_comembership", "top_terms"]].to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "subcluster_stability_summary.csv", index=False, encoding="utf-8-sig")
    members_df.to_csv(OUT / "subcluster_stability_members.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT / 'subcluster_stability_summary.csv'} 외 1개")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
