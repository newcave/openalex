# analysis/explore_cooccurrence_threshold.py
# Rising 용어 동시출현 네트워크(Louvain 클러스터링용)를 만들기 전에, 엣지 가중치(동시출현
# 횟수) 임계값을 감으로 정하지 않고 실제 분포를 보고 정하기 위한 1회성 탐색 스크립트
# (explore_min_count_threshold.ipynb와 같은 목적, 같은 방식).
#
# 사용법: python -m analysis.explore_cooccurrence_threshold

import numpy as np
import pandas as pd
from scipy import sparse

SOURCE_DIR_TAGS = r"analysis\output_임계15"


def main() -> None:
    tags = pd.read_parquet(rf"{SOURCE_DIR_TAGS}\trend_tags_unigram_all.parquet")
    rising_terms = tags[tags["추세(MK)"] == "Rising"]["용어"].tolist()
    print(f"Rising 유니그램: {len(rising_terms):,}개")

    mapping_df = pd.read_csv(rf"{SOURCE_DIR_TAGS}\term_normalization_dict.csv")
    mapping = dict(zip(mapping_df["variant"], mapping_df["canonical"]))

    corpus = pd.read_parquet(rf"{SOURCE_DIR_TAGS}\filtered_corpus.parquet", columns=["id", "language"])
    analyzable_ids = set(corpus.loc[corpus["language"] == "en", "id"])

    long_df = pd.read_parquet(rf"{SOURCE_DIR_TAGS}\unigram_long.parquet", columns=["id", "term"])
    long_df = long_df[long_df["id"].isin(analyzable_ids)]
    long_df = long_df.assign(term=long_df["term"].map(lambda t: mapping.get(t, t)))
    long_df = long_df[long_df["term"].isin(rising_terms)].drop_duplicates(["id", "term"])

    n_docs_with_any = long_df["id"].nunique()
    print(f"Rising 용어를 하나라도 포함한 논문 수: {n_docs_with_any:,}건")

    # 문서 x 용어 0/1 희소행렬을 만들고, 그 전치곱(M^T @ M)으로 용어x용어 동시출현 행렬을 구한다.
    doc_ids = long_df["id"].astype("category")
    term_ids = long_df["term"].astype("category")
    M = sparse.csr_matrix(
        (np.ones(len(long_df)), (doc_ids.cat.codes, term_ids.cat.codes)),
        shape=(len(doc_ids.cat.categories), len(term_ids.cat.categories)),
    )
    co = (M.T @ M).tocoo()
    terms_index = term_ids.cat.categories

    # 대각선(자기 자신과의 "동시출현" = 그냥 등장횟수) 제외, i<j만 남겨서 중복 제거.
    mask = co.row < co.col
    weights = co.data[mask]
    print(f"\n엣지(동시출현 쌍) 총 개수(가중치>=1): {len(weights):,}개")

    print("\n임계값별 남는 엣지 수 / 그 엣지에 걸리는 고유 용어 수:")
    rows_i, rows_j = co.row[mask], co.col[mask]
    for t in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]:
        keep = weights >= t
        n_edges = int(keep.sum())
        involved_terms = set(rows_i[keep]) | set(rows_j[keep])
        isolated = len(rising_terms) - len(involved_terms)
        print(f"  >= {t:>3}회 동시출현: 엣지 {n_edges:>7,}개 | 네트워크에 남는 용어 {len(involved_terms):>5,}개 "
              f"| 고립되는 용어 {isolated:>5,}개")

    print("\n가중치 분포 기술통계:")
    print(pd.Series(weights).describe())


if __name__ == "__main__":
    main()
