# analysis_codes/trend_tags.py  (도구: n_03의 Mann-Kendall 추세검정 계산에서 사용)
# lifecycle.py(5구간+tau임계값+4단계강제분류)의 대체안 프로토타입.
# 용어 하나를 "도입/성장/성숙/쇠퇴" 중 하나로 억지로 합치지 않고, 서로 독립적인
# 지표 4개를 컬럼으로 따로 둔다 — 근거와 설계 논의는 대화 기록 참고, 요약하면:
#   - 하나로 합치면 서로 다른 신호(느리지만 꾸준한 성장 vs 최근 급등)를 구분 못 하고,
#     충돌 시(예: 장기추세는 안정인데 최근만 급등) 뭘 우선할지 임의로 정해야 하는 문제가 있었음.
#   - "water"처럼 이미 포화된 핵심 용어는 MK 검정상 "유의미한 추세 없음(Not Significant)"이
#     나오는 게 맞는 판정이고, 이걸 "중요하지 않다"로 취급하면 안 됨 — 그래서 크기(평균비중)
#     정보를 추세 라벨과 별개로 항상 같이 보여준다.
#   - 라벨 이름 "Stable"은 "안 변한다고 증명됨"으로 오해되기 쉬워(가설검정 일반 원칙상
#     "유의하지 않다" ≠ "변화가 없다는 증명") "Not Significant"로 정정(2026-08-20). 계산 자체가
#     실패해 p가 안 나오는 "판정불가"와는 원인이 다른 별개 상태 — 자세한 구분은 대화 기록 참고.
#
# 지표별 출처:
#   - Mann-Kendall 검정 + Sen's slope:
#     Marrone, M. (2020). "Application of entity linking to identify research fronts
#     and trends." Scientometrics 122:357-379. Hamed & Rao(1998)의 자기상관 보정판을
#     그대로 사용(원 논문도 이 보정판을 씀). p<0.05를 유의수준으로 쓰는 것은 논문 관행 그대로.
#   - z-score(최근 급등/Hot): 같은 논문에서 함께 쓰는 지표. "최근이 자기 과거 대비
#     표준편차 몇 배 튀는가"로, MK 검정(26년 전체를 보는 지표)이 놓칠 수 있는
#     "최근에서야 움직이기 시작한" 용어를 잡기 위한 것.
#   - MACD 기반 burstiness:
#     Tattershall, E., Nenadic, G., & Stevens, R. D. (2020). "Detecting bursty terms
#     in computer science research." Scientometrics 122:681-699. 정식 Kleinberg(2003)
#     버스트 탐지 알고리즘 대신 이 방식을 쓰는 이유: (1) pandas만으로 계산 가능해 외부
#     의존성이 가볍고, (2) 원 논문이 지적하듯 Kleinberg는 "연속적으로 발생하는 이벤트"를
#     가정하는데 학술논문은 저널 발행 단위로 "배치"로 쏟아지기 때문에 이 도메인에 덜 맞음.
#     나중에 필요하면 Kleinberg 배치형 구현(예: burst_detection 패키지)으로 교체 가능.

from __future__ import annotations

import numpy as np
import pandas as pd
import pymannkendall as mk

from config import DATA_ROOT

# 게이트2(최소빈도) 값. "26년 통틀어 N건 이상"만 분석대상으로 본다.
# analysis/explore_min_count_threshold.ipynb로 분포를 보고 사용자가 15로 확정(2026-08-19).
# (여기 둔 이유: dashboard_view.py와 export_trend_tags.py 양쪽이 같은 값을 쓰도록 —
# export_trend_tags.py가 dashboard_view.py를 import하면 순환참조가 생겨서 더 하위
# 모듈인 여기로 옮겼다.)
MIN_TOTAL_COUNT = 15

# 예전 Streamlit 대시보드(archive/analysis_experiments/dashboard_view.py)가 읽던 미리 계산해둔 파일 위치
# (archive/analysis_experiments/export_trend_tags.py가 만든다). 현재 웹앱은 이 파일을 쓰지 않는다.
OUTPUT_DIR = DATA_ROOT / "archive" / "results_수자원_실험산출물" / "01_corpus"
PRECOMPUTED_PATHS = {
    "유니그램": OUTPUT_DIR / "trend_tags_unigram_all.parquet",
    "바이그램": OUTPUT_DIR / "trend_tags_bigram_all.parquet",
}

MK_ALPHA = 0.05            # Mann-Kendall 유의수준 (관행값, p<0.05)
ZSCORE_RECENT_YEARS = 3    # "최근"으로 볼 연도 수 — 근거는 README/대화 참고, 잠정치
MACD_SPANS = (6, 12, 3)    # (단기, 장기, 시그널) EMA span. Tattershall(2020)이 31개년
                            # 데이터에 쓴 값 그대로 채택 — 우리 데이터도 26개년으로 비슷한 스케일.
INTRO_YEAR_CUTOFF = 2015   # lifecycle.py의 도입기 기준과 동일하게 맞춤

TREND_LABELS = ["Rising", "Declining", "Not Significant", "판정불가"]


def _mk_stats(values: np.ndarray) -> dict:
    """연도별 값 배열 하나(용어 1개)에 대해 MK 검정 + Sen's slope + intercept."""
    if np.allclose(values, values[0]):
        # 값이 전부 동일(대부분 전 구간 0)하면 검정 자체가 무의미 → 안정으로 취급
        return {"tau": 0.0, "p": 1.0, "slope": 0.0, "intercept": float(values[0])}
    try:
        result = mk.hamed_rao_modification_test(values)
        return {
            "tau": float(result.Tau),
            "p": float(result.p),
            "slope": float(result.slope),
            "intercept": float(result.intercept),
        }
    except Exception:
        return {"tau": np.nan, "p": np.nan, "slope": np.nan, "intercept": np.nan}


def _trend_label(tau: float, p: float) -> str:
    if pd.isna(p):
        return "판정불가"
    if p < MK_ALPHA and tau > 0:
        return "Rising"
    if p < MK_ALPHA and tau < 0:
        return "Declining"
    return "Not Significant"


def _zscore(values: np.ndarray, recent_years: int = ZSCORE_RECENT_YEARS) -> float:
    """최근 N년 평균이 그 이전 과거 평균 대비 표준편차 몇 배인지."""
    if len(values) <= recent_years:
        return np.nan
    recent = values[-recent_years:]
    history = values[:-recent_years]
    hist_std = history.std(ddof=0)
    if hist_std == 0 or np.isnan(hist_std):
        return np.nan
    return float((recent.mean() - history.mean()) / hist_std)


def compute_zscore_column(term_year_share: pd.DataFrame, terms, recent_years: int) -> pd.Series:
    """MK/Sen's slope처럼 검정이 필요한 계산(compute_tags, 느림)과 분리해둔 이유: z-score는
    단순 평균/표준편차 계산이라 가벼워서, 사용자가 '최근 N년' 슬라이더를 움직일 때마다
    전체 재계산(용어당 통계검정) 없이 이 함수만 다시 불러도 된다.

    term_year_share: index=용어, columns=연도. 반환값: index=용어, value=z-score."""
    result = {}
    for term in terms:
        share_row = term_year_share.loc[term].to_numpy(dtype=float)
        result[term] = _zscore(share_row, recent_years=recent_years)
    return pd.Series(result)


def macd_burstiness_series(series: pd.Series, spans=MACD_SPANS) -> pd.Series:
    """연도별 시계열 하나를 받아 연도별 burstiness 시계열을 반환한다 (그래프용).
    Tattershall et al.(2020) 식(5): histogram을 '역대 최고값의 제곱근'으로 정규화 —
    절대 빈도가 큰 용어(예: water)의 사소한 변동이 희귀어의 진짜 버스트보다
    부풀려지는 걸 막기 위한 스케일링이다."""
    n1, n2, n3 = spans
    ema_short = series.ewm(span=n1, adjust=False).mean()
    ema_long = series.ewm(span=n2, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=n3, adjust=False).mean()
    histogram = macd_line - signal_line
    peak = series.max()
    if not peak or np.isnan(peak):
        return histogram * np.nan
    return histogram / np.sqrt(peak)


def trend_line_for_term(share_series: pd.Series) -> pd.Series:
    """MK/Sen's slope의 intercept+slope로 만든 직선을 실제 연도 축에 맞춰 반환 (그래프 오버레이용)."""
    values = share_series.to_numpy(dtype=float)
    stats = _mk_stats(values)
    idx = np.arange(len(values))
    predicted = stats["intercept"] + stats["slope"] * idx
    return pd.Series(predicted, index=share_series.index)


def compute_tags(term_year_counts: pd.DataFrame, term_year_share: pd.DataFrame,
                  candidate_terms) -> pd.DataFrame:
    """용어별 태그 테이블을 만든다.

    term_year_counts / term_year_share: index=용어, columns=연도(정수), 두 데이터프레임의
        컬럼(연도) 구성이 같아야 한다.
    candidate_terms: 게이트2(최소빈도)를 통과해 이미 걸러진 용어 목록. 이 함수 자체는
        추가로 용어를 걸러내지 않는다 — 필터링은 이 함수를 부르기 전에 끝나 있어야 한다.
    """
    years = sorted(term_year_share.columns)
    counts = term_year_counts.reindex(index=list(candidate_terms), columns=years, fill_value=0)
    shares = term_year_share.reindex(index=list(candidate_terms), columns=years, fill_value=0.0)

    rows = []
    for term in candidate_terms:
        count_row = counts.loc[term].to_numpy(dtype=float)
        share_row = shares.loc[term].to_numpy(dtype=float)

        stats = _mk_stats(share_row)
        trend = _trend_label(stats["tau"], stats["p"])
        z = _zscore(share_row)

        burst_series = macd_burstiness_series(shares.loc[term])
        peak_burst = burst_series.max()
        peak_year = int(burst_series.idxmax()) if pd.notna(peak_burst) else None

        nonzero_years = [y for y, c in zip(years, count_row) if c > 0]
        first_year = min(nonzero_years) if nonzero_years else None
        is_new = first_year is not None and first_year >= INTRO_YEAR_CUTOFF

        recent_n = min(5, len(years))
        rows.append({
            "용어": term,
            "신규진입(2015~)": is_new,
            "첫등장연도": first_year,
            "추세(MK)": trend,
            "MK_tau": round(stats["tau"], 3) if pd.notna(stats["tau"]) else np.nan,
            "MK_p": round(stats["p"], 4) if pd.notna(stats["p"]) else np.nan,
            # 컬럼명에 아포스트로피(')를 쓰면 altair/Vega-Lite가 필드 경로 파싱에 실패해
            # 그래프가 빈 화면으로 나오는 문제가 있어(브라우저 콘솔에만 에러가 찍히고
            # 파이썬 쪽은 조용히 통과함) "Sen's" 대신 "Sen"으로 표기한다.
            "Sen_slope(%p_연)": round(stats["slope"], 4) if pd.notna(stats["slope"]) else np.nan,
            "최근3년_zscore": round(z, 2) if pd.notna(z) else np.nan,
            "최대버스트강도": round(peak_burst, 2) if pd.notna(peak_burst) else np.nan,
            "버스트연도": peak_year,
            "전체기간_평균비중(%)": round(float(share_row.mean()), 4),
            "최근5년_평균비중(%)": round(float(share_row[-recent_n:].mean()), 4),
            "전체기간_총건수": int(count_row.sum()),
        })

    return pd.DataFrame(rows)


def load_precomputed(ngram_type: str) -> pd.DataFrame | None:
    """export_trend_tags.py가 미리 계산해둔 (유니그램|바이그램)×전체 저널 결과를 읽는다.
    파일이 없으면(아직 export를 안 돌렸으면) None을 반환 — 호출부에서 실시간 계산으로
    대체해야 한다."""
    path = PRECOMPUTED_PATHS.get(ngram_type)
    if path is None or not path.exists():
        return None
    return pd.read_parquet(path)
