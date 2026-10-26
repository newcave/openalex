# analysis/export_term_charts.py
# 대시보드 '신호 태그' 탭에서 용어 하나를 고르면 보여주는 그래프 2종
# (analysis/dashboard_view.py의 _term_trend_band_chart / _term_burst_chart)을,
# 최소빈도(analysis.trend_tags.MIN_TOTAL_COUNT) 필터를 통과한 유니그램/바이그램 전체 용어에
# 대해 PNG 파일로 뽑아둔다. 대시보드를 열지 않고 이미지만 훑어보고 싶을 때 씀.
#
# 사전 조건: analysis/output/trend_tags_{unigram,bigram}_all.parquet가 이미 있어야 한다
# (없으면 `python -m analysis.export_trend_tags`를 먼저 실행). 태그 자체(MK검정 등, 느림)는
# 이미 계산된 그 캐시를 그대로 쓰고, 여기서는 그래프에 필요한 원본 시계열(term_year_share)만
# 같은 방식으로 다시 만든다 — export_trend_tags._compute_for와 동일한 로직.
#
# 저장 구조: term_charts/{unigram,bigram}/{Rising,Declining,Not_Significant,판정불가}/ 아래에
# 용어별로 담긴다 (trend_tags.TREND_LABELS 기준 4분류 — 용어 하나가 여러 조건에 걸치지 않고
# 정확히 하나의 라벨/폴더에만 속함).
#
# 사용법:
#   python -m analysis.export_term_charts --test        # 유니그램/바이그램 각 5개만 (평균비중 상위) — 먼저 이걸로 확인
#   python -m analysis.export_term_charts                # 전체 (유니그램 5,262개 + 바이그램 3,864개) x 2장
#   python -m analysis.export_term_charts --limit 100    # 각 ngram 타입별 앞에서부터 N개만 (평균비중 상위 순)

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from analysis import build_dataset, normalization, trend_tags
from analysis.dashboard_view import _term_burst_chart, _term_trend_band_chart
from analysis.term_frequency import BIGRAM_PATH, UNIGRAM_PATH

# ngram_key(폴더명, 영문) -> (long포맷 parquet 경로, trend_tags.load_precomputed에 넘길 한글 라벨)
NGRAM_TYPES = {
    "unigram": ("유니그램", UNIGRAM_PATH),
    "bigram": ("바이그램", BIGRAM_PATH),
}

OUT_ROOT = trend_tags.OUTPUT_DIR / "term_charts"

# Windows 파일명에 못 쓰는 문자. 바이그램의 공백은 언더스코어로 먼저 바꾸고 나머지를 제거한다.
_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|]')

# 그래프 아래에 태그 테이블 값을 같이 적으려면 한글(추세, 비중 등)이 깨지지 않는 폰트가 필요하다.
# 맑은 고딕은 한국어 Windows에 기본 포함되어 있음.
_FONT_PATH = Path(r"C:\Windows\Fonts\malgun.ttf")
_FOOTER_FONT_SIZE = 15
_FOOTER_PADDING = 14


def _load_footer_font() -> ImageFont.ImageFont:
    if _FONT_PATH.exists():
        return ImageFont.truetype(str(_FONT_PATH), size=_FOOTER_FONT_SIZE)
    print(f"경고: 한글 폰트({_FONT_PATH})를 찾지 못해 기본 폰트로 대체합니다 — 한글이 깨져 보일 수 있습니다.")
    return ImageFont.load_default()


_FOOTER_FONT = _load_footer_font()


def _safe_filename(term: str) -> str:
    """용어를 파일명으로 안전하게 바꾼다 (공백 -> '_', 그 외 금지문자는 제거)."""
    name = term.strip().replace(" ", "_")
    name = _UNSAFE_CHARS.sub("", name)
    return name or "unknown_term"


def _fmt(value, unit: str = "") -> str:
    """태그 테이블 값 하나를 사람이 읽기 좋게 정리. 결측치는 '-'로."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:g}{unit}"
    return f"{value}{unit}"


def _metrics_lines(row: pd.Series, zscore_col: str) -> list[str]:
    """이미지 아래에 같이 적을 태그 테이블 값들. 그래프만 봐서는 안 보이는 검정 통계치·크기
    정보를 같이 붙여야 해석이 된다는 사용자 피드백에 따라 추가.
    최대버스트강도(양수쪽 피크만 요약해 그래프와 안 맞을 수 있음)/최근5년_평균비중/첫등장연도는
    사용자 요청으로 제외."""
    is_new = "O" if row["신규진입(2015~)"] else "X"
    return [
        f"추세(MK): {row['추세(MK)']}   MK_tau: {_fmt(row['MK_tau'])}   "
        f"MK_p: {_fmt(row['MK_p'])}   Sen_slope: {_fmt(row['Sen_slope(%p_연)'])} %p/년",
        f"{zscore_col}: {_fmt(row[zscore_col])}",
        f"전체기간_평균비중: {_fmt(row['전체기간_평균비중(%)'], '%')}   "
        f"전체기간_총건수: {_fmt(row['전체기간_총건수'], '건')}",
        f"신규진입(2015~): {is_new}",
    ]


def _append_metrics_footer(png_path: Path, lines: list[str]) -> None:
    """altair가 저장한 PNG 아래에 흰 여백을 늘려 태그 테이블 값을 텍스트로 붙인다."""
    img = Image.open(png_path).convert("RGB")
    line_height = _FOOTER_FONT.size + 6
    footer_height = _FOOTER_PADDING * 2 + line_height * len(lines)

    canvas = Image.new("RGB", (img.width, img.height + footer_height), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = img.height + _FOOTER_PADDING
    for line in lines:
        draw.text((_FOOTER_PADDING, y), line, fill="black", font=_FOOTER_FONT)
        y += line_height
    canvas.save(png_path)


def _build_term_year_share(long_path: Path) -> pd.DataFrame:
    """export_trend_tags._compute_for와 동일한 방식으로 term_year_share(용어 x 연도, 상대빈도 %)를
    만든다. index=용어, columns=연도(정수)."""
    # export_trend_tags.py / dashboard_view.py와 동일한 이유로 언어 서브셋을 씀 —
    # 한글 논문은 용어 기여가 0이라 분모에서도 빼야 비율이 정확해짐 (2026-08-26).
    doc_counts = build_dataset.load_analyzable_corpus(columns=["id", "year", "journal"])
    long_df = pd.read_parquet(long_path)

    mapping = normalization.load_dict()
    long_df = long_df.assign(term=normalization.apply_dict(long_df["term"], mapping))

    year_doc_totals = doc_counts.groupby("year")["id"].nunique().sort_index()
    term_year_counts = long_df.groupby(["term", "year"])["id"].nunique().unstack(fill_value=0)
    term_year_counts = term_year_counts.reindex(columns=year_doc_totals.index, fill_value=0)
    return term_year_counts.div(year_doc_totals, axis=1) * 100


def _export_for(ngram_key: str, limit: int | None) -> int:
    ngram_label, long_path = NGRAM_TYPES[ngram_key]

    tags_df = trend_tags.load_precomputed(ngram_label)
    if tags_df is None:
        raise SystemExit(
            f"[{ngram_label}] 미리 계산된 태그 파일이 없습니다. "
            "먼저 `python -m analysis.export_trend_tags`를 실행해주세요."
        )

    # load_precomputed 결과는 export_trend_tags.py에서 이미 "전체기간_평균비중(%)" 내림차순으로 정렬돼 있다.
    terms = tags_df["용어"].tolist()
    if limit is not None:
        terms = terms[:limit]

    # dashboard_view.py가 슬라이더로 다시 계산하는 zscore 컬럼명 규칙과 동일하게 찾는다
    # (하드코딩된 "최근3년_zscore"가 아니라 "…년_zscore"로 끝나는 컬럼을 찾음 — trend_tags.ZSCORE_RECENT_YEARS가
    # 바뀌어도 그대로 맞도록).
    zscore_col = next(c for c in tags_df.columns if c.endswith("년_zscore"))
    tags_by_term = tags_df.set_index("용어")

    term_year_share = _build_term_year_share(long_path)

    missing = [t for t in terms if t not in term_year_share.index]
    if missing:
        print(f"[{ngram_label}] 경고: term_year_share에 없는 용어 {len(missing)}개는 건너뜁니다 (예: {missing[:5]})")
        terms = [t for t in terms if t not in missing]

    out_dir = OUT_ROOT / ngram_key
    label_dirs = {}
    for label in trend_tags.TREND_LABELS:  # Rising/Declining/Not Significant/판정불가 폴더를 미리 만들어둠
        label_dir = out_dir / _safe_filename(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        label_dirs[label] = label_dir

    used_names: set[str] = set()
    for term in tqdm(terms, desc=f"{ngram_label} 차트 저장", unit="term"):
        safe_name = _safe_filename(term)
        if safe_name in used_names:  # 서로 다른 용어가 같은 파일명으로 정리되는 드문 충돌 대비
            n = 2
            while f"{safe_name}_{n}" in used_names:
                n += 1
            safe_name = f"{safe_name}_{n}"
        used_names.add(safe_name)

        row = tags_by_term.loc[term]
        term_dir = label_dirs[row["추세(MK)"]]
        share_series = term_year_share.loc[term]
        metrics_lines = _metrics_lines(row, zscore_col)

        trend_chart = _term_trend_band_chart(
            share_series, recent_years=trend_tags.ZSCORE_RECENT_YEARS
        ).properties(title=f"{term} — 추세·급등확인")
        trend_path = term_dir / f"{safe_name}_trend_band.png"
        trend_chart.save(str(trend_path))
        _append_metrics_footer(trend_path, metrics_lines)

        burst_chart = _term_burst_chart(share_series).properties(title=f"{term} — 버스트 강도")
        burst_path = term_dir / f"{safe_name}_burst.png"
        burst_chart.save(str(burst_path))
        _append_metrics_footer(burst_path, metrics_lines)

    return len(terms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test", action="store_true",
        help="유니그램/바이그램 각 5개만 (평균비중 상위) 뽑아서 확인용으로 실행"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="각 ngram 타입별로 앞에서부터 N개만 (평균비중 상위 순). --test와 동시에 쓰면 --test가 우선."
    )
    args = parser.parse_args()

    limit = 5 if args.test else args.limit

    total = 0
    for ngram_key in NGRAM_TYPES:
        total += _export_for(ngram_key, limit)

    print(f"완료: 용어 {total:,}개 x 2장 = {total * 2:,}장 저장. 저장 위치: {OUT_ROOT}")


if __name__ == "__main__":
    main()
