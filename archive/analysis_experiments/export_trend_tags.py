# analysis/export_trend_tags.py
# 대시보드 '신호 태그' 탭의 기본 화면(유니그램/바이그램 × 전체 저널, 최소빈도 필터 통과분)을
# 미리 계산해서 저장해둔다. 데이터(코퍼스)가 안 바뀌는 한 대시보드를 켤 때마다 다시 계산할
# 필요가 없으므로, dashboard_view.py는 이 스크립트가 만든 parquet 파일을 읽기만 한다
# (analysis/trend_tags.py의 load_precomputed 참고). csv/xlsx는 엑셀에서 직접 열어보는 용도.
#
# 데이터를 다시 수집했거나 정규화 사전을 바꿨으면 이 스크립트를 다시 돌려야 한다:
#   python -m analysis.export_trend_tags

import pandas as pd

from analysis import build_dataset, normalization, trend_tags
from analysis.term_frequency import UNIGRAM_PATH, BIGRAM_PATH

NGRAM_PATHS = {"유니그램": UNIGRAM_PATH, "바이그램": BIGRAM_PATH}


def _compute_for(ngram_type: str) -> pd.DataFrame:
    # dashboard_view.py의 _load_doc_counts()와 동일한 이유로 언어 서브셋을 씀 —
    # 한글 논문은 용어 기여가 0이라 분모에서도 빼야 비율이 정확해짐 (2026-08-26).
    doc_counts = build_dataset.load_analyzable_corpus(columns=["id", "year", "journal"])
    long_df = pd.read_parquet(NGRAM_PATHS[ngram_type])

    mapping = normalization.load_dict()
    long_df = long_df.assign(term=normalization.apply_dict(long_df["term"], mapping))

    year_doc_totals = doc_counts.groupby("year")["id"].nunique().sort_index()
    term_year_counts = long_df.groupby(["term", "year"])["id"].nunique().unstack(fill_value=0)
    term_year_counts = term_year_counts.reindex(columns=year_doc_totals.index, fill_value=0)
    term_year_share = term_year_counts.div(year_doc_totals, axis=1) * 100

    overall_total = term_year_counts.sum(axis=1)
    candidate_terms = list(
        overall_total[overall_total >= trend_tags.MIN_TOTAL_COUNT].sort_values(ascending=False).index
    )
    print(f"[{ngram_type}] 분석대상 용어 수: {len(candidate_terms):,}개 (최소빈도 {trend_tags.MIN_TOTAL_COUNT}건 이상)")

    tags_df = trend_tags.compute_tags(term_year_counts, term_year_share, candidate_terms)
    return tags_df.sort_values("전체기간_평균비중(%)", ascending=False).reset_index(drop=True)


def main() -> None:
    trend_tags.OUTPUT_DIR.mkdir(exist_ok=True)

    for ngram_type, parquet_path in trend_tags.PRECOMPUTED_PATHS.items():
        tags_df = _compute_for(ngram_type)

        tags_df.to_parquet(parquet_path, index=False)  # 대시보드가 읽는 파일 (dtype 그대로 보존)
        csv_path = parquet_path.with_suffix(".csv")
        xlsx_path = parquet_path.with_suffix(".xlsx")
        tags_df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # BOM: 엑셀에서 한글 안 깨짐
        tags_df.to_excel(xlsx_path, index=False)

        print(f"저장 완료: {parquet_path.name}, {csv_path.name}, {xlsx_path.name}")


if __name__ == "__main__":
    main()
