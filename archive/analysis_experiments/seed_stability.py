"""Louvain 시드 안정성 검정 — 논문 §5.9 근거 생성.

문제의식: Louvain은 노드 방문 순서에 따라 결과가 달라지므로 코드가 seed=0으로 고정돼 있다.
즉 현재 8개 군집은 "시드 0일 때의 답"이다. 특히 작은 군집(16/4/2개)이 시드를 바꾸면
사라지는지 확인하지 않으면, 심사자가 시드를 바꿔 재현이 안 되는 순간 결과 전체가 무너진다.
interpretations.json도 군집 5(해수침투)에 대해 같은 우려를 스스로 기록해 뒀다.

측정 지표 3종:
  (1) 군집 개수 분포          — 시드마다 몇 개가 나오는가
  (2) 분할 간 ARI             — 시드 0 대비, 그리고 모든 시드 쌍
  (3) 군집별 재현율 두 가지
      - best-match Jaccard : 기준 분할의 군집 R이 다른 시드에서 얼마나 통째로 살아남는가
      - 내부 동반확률       : R의 용어 쌍이 같은 군집에 함께 묶일 확률 (군집 응집도)
      두 지표를 함께 보는 이유 — Jaccard는 '군집이 쪼개지거나 흡수되는' 것을 잡고,
      동반확률은 '핵심 용어들이 흩어지지 않고 붙어 다니는가'를 잡는다. 군집이 다른 군집에
      통째로 흡수되면 Jaccard는 떨어지지만 동반확률은 높게 유지된다.

원본 파일은 읽기만 한다. 출력은 reports/seed_stability_*.csv 3개.

사용법: python -m analysis.seed_stability
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.weight_scheme_ablation import (  # noqa: E402
    adjusted_rand, load_graph_inputs, weight_schemes, _labels,
)

OUT = ROOT / "reports"
N_SEEDS = 50
REFERENCE_SEED = 0          # 현재 논문 결과가 나온 시드
FOCUS = "cosine"            # 주 결과의 가중치 (군집별 재현율을 자세히 볼 대상)


def build_graph(nodes, terms_index, rows_i, rows_j, weights) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for i, j, w in zip(rows_i, rows_j, weights):
        graph.add_edge(terms_index[i], terms_index[j], weight=float(w))
    return graph


def best_match_jaccard(reference: set, communities: list[set]) -> float:
    return max((len(reference & c) / len(reference | c)) for c in communities)


def main() -> None:
    rising, terms_index, doc_freq, rows_i, rows_j, cooc, n_docs = load_graph_inputs()
    schemes = weight_schemes(cooc, doc_freq[rows_i], doc_freq[rows_j])
    index = {t: k for k, t in enumerate(rising)}
    print(f"노드 {len(rising)}개 | 엣지 {len(cooc):,}개 | 시드 {N_SEEDS}개\n")

    summary_rows, cluster_rows, seed_rows = [], [], []
    # 모든 가중치 그래프를 미리 만들어 둔다 — 각 시드의 분할을 '모든 가중치로' 평가하기 위함.
    # (§5.5.1의 raw 기준 재평가 차이가 시드 잡음보다 큰지 판정해야 하므로 필수)
    all_graphs = {n: build_graph(rising, terms_index, rows_i, rows_j, w) for n, w in schemes.items()}

    for scheme_name, weights in schemes.items():
        graph = all_graphs[scheme_name]

        partitions = []
        for seed in range(N_SEEDS):
            communities = sorted(
                nx.community.louvain_communities(graph, weight="weight", seed=seed),
                key=len, reverse=True,
            )
            partitions.append(communities)
        print(f"[{scheme_name}] Louvain {N_SEEDS}회 완료")

        counts = [len(p) for p in partitions]
        labels = [_labels(p, rising) for p in partitions]

        # 각 시드 분할을 세 가중치 전부로 평가 — 시드 잡음의 폭을 재기 위함
        q_by_eval = {
            eval_name: [nx.community.modularity(g, p, weight="weight") for p in partitions]
            for eval_name, g in all_graphs.items()
        }

        # (2) ARI — 모든 시드 쌍, 그리고 기준 시드 대비
        pair_ari = [adjusted_rand(labels[a], labels[b]) for a, b in combinations(range(N_SEEDS), 2)]
        ref_ari = [adjusted_rand(labels[REFERENCE_SEED], labels[s])
                   for s in range(N_SEEDS) if s != REFERENCE_SEED]

        summary_rows.append({
            "scheme": scheme_name,
            "n_seeds": N_SEEDS,
            "clusters_min": min(counts), "clusters_median": int(np.median(counts)),
            "clusters_max": max(counts),
            "clusters_mode": int(pd.Series(counts).mode().iloc[0]),
            "clusters_distinct_values": len(set(counts)),
            "ARI_allpairs_mean": round(float(np.mean(pair_ari)), 4),
            "ARI_allpairs_sd": round(float(np.std(pair_ari)), 4),
            "ARI_allpairs_min": round(float(np.min(pair_ari)), 4),
            f"ARI_vs_seed{REFERENCE_SEED}_mean": round(float(np.mean(ref_ari)), 4),
            f"ARI_vs_seed{REFERENCE_SEED}_min": round(float(np.min(ref_ari)), 4),
            **{f"Q_under_{e}_mean": round(float(np.mean(v)), 6) for e, v in q_by_eval.items()},
            **{f"Q_under_{e}_sd": round(float(np.std(v)), 6) for e, v in q_by_eval.items()},
            **{f"Q_under_{e}_min": round(float(np.min(v)), 6) for e, v in q_by_eval.items()},
            **{f"Q_under_{e}_max": round(float(np.max(v)), 6) for e, v in q_by_eval.items()},
        })
        for seed, c in enumerate(counts):
            seed_rows.append({"scheme": scheme_name, "seed": seed, "n_clusters": c,
                              **{f"Q_under_{e}": round(v[seed], 6) for e, v in q_by_eval.items()}})

        # (3) 군집별 재현율 — 기준 시드 분할의 각 군집이 얼마나 살아남는가
        reference = partitions[REFERENCE_SEED]
        co_member = np.zeros((len(rising), len(rising)), dtype=np.int32)
        for lab in labels:
            co_member += (lab[:, None] == lab[None, :])

        for cid, members in enumerate(reference):
            idx = np.array([index[t] for t in members])
            if len(idx) > 1:
                block = co_member[np.ix_(idx, idx)]
                iu = np.triu_indices(len(idx), k=1)
                cohesion = float(block[iu].mean()) / N_SEEDS
            else:
                cohesion = float("nan")
            jac = [best_match_jaccard(set(members), p) for p in partitions]
            top = sorted(members, key=lambda t: -doc_freq[list(terms_index).index(t)])[:6] \
                if len(members) <= 6 else sorted(members)[:6]
            cluster_rows.append({
                "scheme": scheme_name,
                "reference_cluster": cid,
                "size": len(members),
                "best_match_jaccard_mean": round(float(np.mean(jac)), 4),
                "best_match_jaccard_min": round(float(np.min(jac)), 4),
                "internal_comembership": round(cohesion, 4) if cohesion == cohesion else None,
                "sample_terms": " | ".join(top),
            })

    summary = pd.DataFrame(summary_rows)
    clusters = pd.DataFrame(cluster_rows)
    seeds = pd.DataFrame(seed_rows)

    print("\n=== 가중치별 안정성 요약 (군집 수 / ARI) ===")
    base_cols = [c for c in summary.columns if not c.startswith("Q_under_")]
    print(summary[base_cols].to_string(index=False))

    print("\n=== 시드 50개에 걸친 modularity 분포 ===")
    q_cols = ["scheme"] + [c for c in summary.columns if c.startswith("Q_under_")]
    print(summary[q_cols].to_string(index=False))

    print("\n=== 핵심 판정: raw 기준 재평가 차이가 시드 잡음보다 큰가? ===")
    # 분포 범위가 겹치는지만 보는 것은 판정 기준으로 너무 조악하다(꼬리 하나만 겹쳐도 '겹침'이
    # 되어버린다). 평균 차이를 시드 산포로 나눈 효과크기와 검정을 함께 본다.
    raw_q = seeds.loc[seeds.scheme.eq("raw"), "Q_under_raw"].to_numpy()
    cos_q = seeds.loc[seeds.scheme.eq("cosine"), "Q_under_raw"].to_numpy()
    t_stat, p_val = stats.ttest_ind(raw_q, cos_q, equal_var=False)
    cohen_d = (raw_q.mean() - cos_q.mean()) / np.sqrt((raw_q.var(ddof=1) + cos_q.var(ddof=1)) / 2)
    print(f"raw 분할의 Q_under_raw   : {raw_q.mean():.6f} ± {raw_q.std(ddof=1):.6f} "
          f"[{raw_q.min():.6f}, {raw_q.max():.6f}]")
    print(f"cosine 분할의 Q_under_raw: {cos_q.mean():.6f} ± {cos_q.std(ddof=1):.6f} "
          f"[{cos_q.min():.6f}, {cos_q.max():.6f}]")
    print(f"평균 차이 {raw_q.mean() - cos_q.mean():.6f} | Welch t={t_stat:.2f}, p={p_val:.3e}, "
          f"Cohen's d={cohen_d:.2f}")
    print(f"분포 겹침: cosine 중 raw 최솟값을 넘는 시드 {int((cos_q > raw_q.min()).sum())}/{N_SEEDS}개")

    print("\n=== 보고된 시드 0 값이 50개 중 몇 등인가 (낙관 편향 점검) ===")
    for sch in schemes:
        for evalw in schemes:
            v = seeds.loc[seeds.scheme.eq(sch), f"Q_under_{evalw}"].to_numpy()
            print(f"  분할 {sch:15s} / 평가 {evalw:15s} seed0={v[0]:.6f} "
                  f"순위 {int((v > v[0]).sum()) + 1}/{N_SEEDS}")
    print(f"\n=== 시드 {REFERENCE_SEED} 기준 군집별 재현율 ({FOCUS}) ===")
    print(clusters[clusters.scheme.eq(FOCUS)].to_string(index=False))
    print("\n=== 군집 개수 분포 ===")
    print(seeds.groupby(["scheme", "n_clusters"]).size().to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "seed_stability_summary.csv", index=False, encoding="utf-8-sig")
    clusters.to_csv(OUT / "seed_stability_clusters.csv", index=False, encoding="utf-8-sig")
    seeds.to_csv(OUT / "seed_stability_counts.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT / 'seed_stability_summary.csv'} 외 2개")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
