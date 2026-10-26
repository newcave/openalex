# analysis_codes/n_01_build_corpus.py
# [분석 1단계] 코퍼스 만들기: dataset/<분야>/ 수집 원본 parquet에 type/language 필터링 규칙을 적용해
# 이후 분석 단계(n_02~n_06)가 공통으로 재사용할 기준 데이터셋(filtered_corpus.parquet)을 만든다.
#
# 입력: dataset/<분야>/*.parquet
# 출력: results/<분야>/01_corpus/filtered_corpus.parquet
# 실행: python -m analysis_codes.n_01_build_corpus   (인자 없이 실행하면 기본 분야 config.DEFAULT_DOMAIN)

import glob
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import DEFAULT
from pipeline_codes.utils import logger

OUTPUT_PATH = DEFAULT.corpus_path

COLUMNS = [
    "id", "journal", "year", "type", "language",
    "display_name", "cited_by_count", "concepts", "keywords",
]

# 대화에서 코퍼스 진단 후 확정한 규칙:
#   - type: article/review/letter만 남기고 editorial/erratum/retraction/book-chapter 등은 제외
#   - letter 중 comment/reply류(원논문에 대한 응답)는 제외 — type 필드가 아니라 title 텍스트로 판별
#     (실제로 'article'로 잘못 분류된 comment/reply도 있어 type만으로는 못 걸러냄)
#   - language: en/ko만 남김. ja/fa는 doi 100%에 가깝게 결측·인용 0에 몰림·특정 연도 집중 등
#     번역 스텁/이상치 패턴이 확인되어 제외 (misc/explore_corpus.ipynb 참고)
INCLUDE_TYPES = {"article", "review", "letter"}
INCLUDE_LANGUAGES = {"en", "ko"}

# 제목 맨 앞에서만 검사하는 패턴. "discussion of"/"response to"는 실제 연구제목에도
# 흔히 등장하므로("~ in response to climate change") 문장 시작일 때만 comment/reply로 간주.
START_PATTERNS = [
    r"^reply\b",
    r"^repl(y|ies) to\b",
    r"^author'?s?'?\s+repl(y|ies)\b",   # Author's reply / Authors' reply
    r"^response to\b",
    r"^discussion (on|of)\b",
    r"^rebuttal to\b",
    r"^letter to the editor\b",
    r"^correspondence letter\b",         # "Correspondence Letter" 스텁만. "Correspondence between/of..."는 정상 연구제목이라 제외
]
# 문장 어디에 있어도 comment/reply로 봐도 안전한 패턴
ANYWHERE_PATTERNS = [
    r"\bcomments? on\b",
    r"\breply to comment\b",
]

_START_RX = re.compile("|".join(START_PATTERNS), re.IGNORECASE)
_ANYWHERE_RX = re.compile("|".join(ANYWHERE_PATTERNS), re.IGNORECASE)


def is_comment_reply(title) -> bool:
    if not isinstance(title, str):
        return False
    t = title.strip()
    return bool(_START_RX.match(t) or _ANYWHERE_RX.search(t))


def load_raw_corpus(cache_dir: Path | None = None) -> pd.DataFrame:
    """cache_dir를 안 주면 기본 분야(config.DEFAULT)의 dataset 폴더를 읽는다."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT.dataset_dir
    files = sorted(glob.glob(str(cache_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"{cache_dir}에 parquet 캐시가 없습니다.")
    dfs = [pd.read_parquet(f, columns=COLUMNS) for f in files]
    return pd.concat(dfs, ignore_index=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """type -> comment/reply -> language 순서로 순차 필터링 (대화에서 확정한 순서)."""
    step0 = df
    logger.info(f"[build_dataset] 원본: {len(step0):,}건")

    step1 = step0[step0["type"].isin(INCLUDE_TYPES)]
    logger.info(
        f"[build_dataset] type 필터({sorted(INCLUDE_TYPES)}) 후: {len(step1):,}건 "
        f"(제외 {len(step0) - len(step1):,}건)"
    )

    is_cr = step1["display_name"].apply(is_comment_reply)
    step2 = step1[~is_cr]
    logger.info(f"[build_dataset] comment/reply 제외 후: {len(step2):,}건 (제외 {int(is_cr.sum()):,}건)")

    step3 = step2[step2["language"].isin(INCLUDE_LANGUAGES)]
    logger.info(
        f"[build_dataset] language 필터({sorted(INCLUDE_LANGUAGES)}) 후: {len(step3):,}건 "
        f"(제외 {len(step2) - len(step3):,}건)"
    )

    return step3.reset_index(drop=True)


# ---------------------------------------------------------------- 용어빈도 분모용 서브셋
#
# text_utils.py의 토크나이저가 영문 전용(한글 형태소 분석 미지원)이라, language=='ko' 논문은
# unigram_long/bigram_long에 용어를 하나도 만들지 못한다(분자 기여 0). 그런데 상대빈도(%)
# 계산의 분모(그 해 전체 논문 수)는 filtered_corpus.parquet 전체를 그대로 쓰고 있어서, 한글
# 논문이 있는 연도/저널의 비율이 실제보다 미세하게 낮게(과소) 계산되는 문제가 있었다
# (2026-08-26 발견 — 153건, 필터후 코퍼스의 0.17%).
#
# 근본 해결(한글 형태소 분석기 추가)은 나중 작업으로 미루고, 우선 분모도 "지금 실제로
# 토큰화 가능한" 언어로 맞춰서 분자-분모 정합성만 바로잡는다. 한글 지원이 추가되면
# ANALYZABLE_LANGUAGES에 "ko"를 넣으면 된다.
ANALYZABLE_LANGUAGES = {"en"}


def load_analyzable_corpus(columns: list[str] | None = None, corpus_path: Path | None = None) -> pd.DataFrame:
    """용어빈도/신호태그 계산의 분모로 쓸 서브셋을 읽는다.

    filtered_corpus.parquet 자체(INCLUDE_LANGUAGES=en/ko)는 그대로 두고, 여기서만
    ANALYZABLE_LANGUAGES로 한 번 더 좁힌다 — "언어적으로 포함할 논문"과 "지금 토큰화
    가능한 논문"은 서로 다른 질문이라 필터를 분리해뒀다.
    """
    read_cols = list(columns) if columns else None
    if read_cols and "language" not in read_cols:
        read_cols = read_cols + ["language"]

    df = pd.read_parquet(corpus_path or OUTPUT_PATH, columns=read_cols)
    df = df[df["language"].isin(ANALYZABLE_LANGUAGES)]
    if columns:
        df = df[list(columns)]
    return df.reset_index(drop=True)


def main(cache_dir: Path | None = None, output_path: Path | None = None) -> pd.DataFrame:
    output_path = Path(output_path) if output_path else OUTPUT_PATH
    df = load_raw_corpus(cache_dir)
    filtered = apply_filters(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path, index=False)
    logger.info(f"[build_dataset] 저장 완료: {output_path} ({len(filtered):,}건)")
    return filtered


if __name__ == "__main__":
    main()
