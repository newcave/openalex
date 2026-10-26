# analysis/export_term_charts_binned.py
# 임계값 없이 전체 용어(유니그램 42,791개 + 바이그램 232,129개)를 다 그래프로 뽑는 건 무리라서
# (549,840장, ~25시간 — 자세한 계산은 대화 기록 참고), 대신 "전체기간 총건수" 기준 5개 구간
# (로그스케일로 대략 5배씩)으로 나누고, 각 구간에서 상위 25개(건수 많은 순)와 랜덤 25개를
# 따로 뽑아 대표 샘플로 그래프를 만든다.
#
# 25개인 이유: 바이그램 5구간(500건+)이 정확히 25개뿐이라, 그 최소 구간 크기를 전체 구간·양쪽
# ngram타입에 동일하게 적용하면 "구간마다 개수가 달라서 비교가 안 되는" 문제가 없어진다
# (2026-09-01 사용자 확인). 다만 바이그램 5구간은 모집단 자체가 25개라 상위25/랜덤25가
# 결국 똑같은 25개가 된다 — 코드에서 따로 예외처리하지 않고 그대로 두되 실행 로그에 표시한다.
#
# 원본 데이터는 analysis/output_5262개/ 에서 읽는다 (예전 필터링 버전을 보존하려고 사용자가
# analysis/output을 그 이름으로 옮겨둠 — 안의 filtered_corpus/unigram_long/bigram_long/
# 정규화사전은 내용이 동일한 원본 데이터라 그대로 재사용). 분모(연도별 전체 논문 수)는
# build_dataset.load_analyzable_corpus()와 동일하게 language=="en"만 남긴 서브셋을 쓴다
# (2026-08-26 확정된 방식 — 한글 논문은 아직 토큰화가 안 돼 분자 기여가 0이라 분모에서도 빼야
# 비율이 정확해짐).
#
# 등장 이전 연도(0으로 채워진 구간)는 자르고, 실제로 등장한 첫 해부터 통계(MK/Sen's slope/
# zscore/버스트)를 다시 계산 + 그래프도 그 구간만 그린다 — "태어나기 전은 0살"이라고 우기며
# 통계를 내면 등장 자체가 항상 강한 Rising 신호로 잡히는 문제를 피하기 위함
# (2026-09-01 사용자 확인, 기존 export_term_charts.py의 26년 고정 방식과는 다름).
#
# 저장 구조: output_full/{unigram,bigram}/{구간}/{상위25,랜덤25}/{Rising,Declining,
#            Not_Significant,판정불가}/{용어}_trend_band.png, _burst.png
#
# 사용법:
#   python -m analysis.export_term_charts_binned --test    # 구간마다 2개씩만 (동작 확인용)
#   python -m analysis.export_term_charts_binned            # 전체 (5구간 x 2선정방식 x 25개 x 2ngram = 최대 500용어 x 2장)

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis import normalization, trend_tags
from analysis.dashboard_view import _term_burst_chart, _term_trend_band_chart
from analysis.export_term_charts import _append_metrics_footer, _fmt, _safe_filename

# --- 원본 데이터 경로 (예전 결과 보존을 위해 옮겨진 폴더) ---
SOURCE_DIR = Path(__file__).resolve().parent / "output_5262개"
CORPUS_PATH = SOURCE_DIR / "filtered_corpus.parquet"
NORM_DICT_PATH = SOURCE_DIR / "term_normalization_dict.csv"
NGRAM_LONG_PATHS = {
    "unigram": ("유니그램", SOURCE_DIR / "unigram_long.parquet"),
    "bigram": ("바이그램", SOURCE_DIR / "bigram_long.parquet"),
}

ANALYZABLE_LANGUAGES = {"en"}  # build_dataset.ANALYZABLE_LANGUAGES와 동일 (2026-08-26 확정)

OUT_ROOT = Path(__file__).resolve().parent / "output_full"

# 구간 경계 (lo <= 총건수 < hi). 마지막 구간만 hi=None(무제한). 실제 분포 확인 후 로그스케일로
# 대략 5배씩 늘어나게 설계 (대화 기록의 임계값 테이블 참고, 임의로 정하지 않음).
BINS = [
    (1, 5, "1_초희귀_1-4건"),
    (5, 20, "2_희귀_5-19건"),
    (20, 100, "3_중간_20-99건"),
    (100, 500, "4_흔함_100-499건"),
    (500, None, "5_핵심어_500건이상"),
]
TOP_N_PER_BIN = 25
RANDOM_SEED = 0  # 랜덤25 재현성용


def _load_norm_mapping() -> dict[str, str]:
    if not NORM_DICT_PATH.exists():
        return {}
    df = pd.read_csv(NORM_DICT_PATH)
    if df.empty:
        return {}
    return dict(zip(df["variant"], df["canonical"]))


def _load_year_doc_totals() -> pd.Series:
    """build_dataset.load_analyzable_corpus()와 동일한 언어 필터(en만)를 적용한 연도별 전체 논문 수."""
    doc_counts = pd.read_parquet(CORPUS_PATH, columns=["id", "year", "language"])
    doc_counts = doc_counts[doc_counts["language"].isin(ANALYZABLE_LANGUAGES)]
    return doc_counts.groupby("year")["id"].nunique().sort_index()


def _build_matrices(long_path: Path, mapping: dict[str, str], year_doc_totals: pd.Series):
    """index=용어, columns=연도(2000~2025 전체) 형태의 (건수, 비중%) 행렬 한 쌍을 만든다."""
    long_df = pd.read_parquet(long_path)
    long_df = long_df.assign(term=normalization.apply_dict(long_df["term"], mapping))

    term_year_counts = long_df.groupby(["term", "year"])["id"].nunique().unstack(fill_value=0)
    term_year_counts = term_year_counts.reindex(columns=year_doc_totals.index, fill_value=0)
    term_year_share = term_year_counts.div(year_doc_totals, axis=1) * 100
    return term_year_counts, term_year_share


def _select_bin_terms(overall_total: pd.Series, lo: int, hi: int | None, top_n: int):
    """구간 [lo, hi)에 속하는 용어 중 상위 top_n개(건수 많은 순)와 랜덤 top_n개를 뽑는다.
    모집단이 top_n 이하면 상위/랜덤이 똑같아진다 (호출부에서 표시)."""
    if hi is None:
        bucket = overall_total[overall_total >= lo]
    else:
        bucket = overall_total[(overall_total >= lo) & (overall_total < hi)]
    bucket = bucket.sort_values(ascending=False)

    top_terms = bucket.head(top_n).index.tolist()
    n_random = min(top_n, len(bucket))
    random_terms = bucket.sample(n=n_random, random_state=RANDOM_SEED).index.tolist() if n_random else []
    return bucket, top_terms, random_terms


def _trimmed_series(term: str, counts: pd.DataFrame, shares: pd.DataFrame):
    """등장 이전(리딩 제로) 연도를 잘라낸 (건수 시리즈, 비중 시리즈) 한 쌍을 반환한다."""
    count_row = counts.loc[term]
    nonzero_years = count_row[count_row > 0].index
    first_year = int(nonzero_years.min())  # 총건수>=1인 용어만 다루므로 항상 존재
    return count_row.loc[first_year:], shares.loc[term].loc[first_year:], first_year


def _compute_tags_for_trimmed(share_series: pd.Series, count_series: pd.Series, first_year: int) -> dict:
    """trend_tags의 개별 계산 함수를 그대로 재사용 (compute_tags()는 26년 고정 배치용이라
    여기선 안 씀 — 용어마다 등장기간 길이가 달라서 배치로 못 묶음)."""
    values = share_series.to_numpy(dtype=float)
    stats = trend_tags._mk_stats(values)
    trend_label = trend_tags._trend_label(stats["tau"], stats["p"])
    z = trend_tags._zscore(values, recent_years=trend_tags.ZSCORE_RECENT_YEARS)
    burst_series = trend_tags.macd_burstiness_series(share_series)
    peak_burst = burst_series.max()
    peak_year = int(burst_series.idxmax()) if pd.notna(peak_burst) else None
    return {
        "추세(MK)": trend_label,
        "MK_tau": stats["tau"],
        "MK_p": stats["p"],
        "Sen_slope": stats["slope"],
        "zscore": z,
        "최대버스트강도": peak_burst,
        "버스트연도": peak_year,
        "등장기간_평균비중": float(share_series.mean()),
        "등장기간_총건수": int(count_series.sum()),
        "신규진입(2015~)": first_year >= trend_tags.INTRO_YEAR_CUTOFF,
        "첫등장연도": first_year,
    }


def _metrics_lines_trimmed(tag: dict) -> list[str]:
    is_new = "O" if tag["신규진입(2015~)"] else "X"
    return [
        f"등장기간: {tag['첫등장연도']}~2025   추세(MK): {tag['추세(MK)']}   "
        f"MK_tau: {_fmt(tag['MK_tau'])}   MK_p: {_fmt(tag['MK_p'])}",
        f"Sen_slope: {_fmt(tag['Sen_slope'])} %p/년   "
        f"최근{trend_tags.ZSCORE_RECENT_YEARS}년_zscore: {_fmt(tag['zscore'])}",
        f"등장기간_평균비중: {_fmt(tag['등장기간_평균비중'], '%')}   "
        f"등장기간_총건수: {_fmt(tag['등장기간_총건수'], '건')}",
        f"신규진입(2015~): {is_new}",
    ]


def _save_term_charts(term: str, share_series: pd.Series, tag: dict, out_dir: Path, used_names: set[str]) -> None:
    label_dir = out_dir / _safe_filename(tag["추세(MK)"])
    label_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(term)
    if safe_name in used_names:
        n = 2
        while f"{safe_name}_{n}" in used_names:
            n += 1
        safe_name = f"{safe_name}_{n}"
    used_names.add(safe_name)

    metrics_lines = _metrics_lines_trimmed(tag)
    # 밴드차트의 "최근 N년" 강조는 등장기간이 N년보다 짧으면 인덱싱이 깨지므로 클램프.
    recent_years = min(trend_tags.ZSCORE_RECENT_YEARS, len(share_series))

    trend_chart = _term_trend_band_chart(share_series, recent_years=recent_years).properties(
        title=f"{term} — 추세·급등확인 ({tag['첫등장연도']}~2025)"
    )
    trend_path = label_dir / f"{safe_name}_trend_band.png"
    trend_chart.save(str(trend_path))
    _append_metrics_footer(trend_path, metrics_lines)

    burst_chart = _term_burst_chart(share_series).properties(
        title=f"{term} — 버스트 강도 ({tag['첫등장연도']}~2025)"
    )
    burst_path = label_dir / f"{safe_name}_burst.png"
    burst_chart.save(str(burst_path))
    _append_metrics_footer(burst_path, metrics_lines)


def _export_for(ngram_key: str, top_n: int) -> int:
    ngram_label, long_path = NGRAM_LONG_PATHS[ngram_key]
    mapping = _load_norm_mapping()
    year_doc_totals = _load_year_doc_totals()
    counts, shares = _build_matrices(long_path, mapping, year_doc_totals)
    overall_total = counts.sum(axis=1)

    total_saved = 0
    for lo, hi, bin_label in BINS:
        bucket, top_terms, random_terms = _select_bin_terms(overall_total, lo, hi, top_n)
        if len(bucket) <= top_n:
            print(f"[{ngram_label}/{bin_label}] 모집단이 {len(bucket)}개뿐이라 상위/랜덤 선택이 동일합니다.")

        for selection_name, terms in [("상위25", top_terms), ("랜덤25", random_terms)]:
            out_dir = OUT_ROOT / ngram_key / bin_label / selection_name
            used_names: set[str] = set()
            desc = f"{ngram_label}/{bin_label}/{selection_name}"
            for term in tqdm(terms, desc=desc, unit="term"):
                count_series, share_series, first_year = _trimmed_series(term, counts, shares)
                tag = _compute_tags_for_trimmed(share_series, count_series, first_year)
                _save_term_charts(term, share_series, tag, out_dir, used_names)
                total_saved += 1

    return total_saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="구간마다 2개씩만 뽑아서 확인용으로 실행")
    args = parser.parse_args()

    top_n = 2 if args.test else TOP_N_PER_BIN

    total = 0
    for ngram_key in NGRAM_LONG_PATHS:
        total += _export_for(ngram_key, top_n)

    print(f"완료: 용어 {total:,}개 x 2장 = {total * 2:,}장 저장. 저장 위치: {OUT_ROOT}")


if __name__ == "__main__":
    main()
