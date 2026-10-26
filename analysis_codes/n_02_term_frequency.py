# analysis_codes/n_02_term_frequency.py
# [분석 2단계] 제목 용어 추출: n_01이 만든 필터링 코퍼스의 제목을 유니그램/바이그램으로 쪼개
# (문서, 연도, 저널, 용어) 롱포맷 테이블로 저장한다.
#
# 여기서는 "용어 정규화"를 아직 적용하지 않은 원시 빈도다 — 정규화(term_normalization_dict.csv)는
# 다음 단계(n_03)에서 읽을 때 적용한다. 무거운 토큰화를 미리 끝내두면 이후 단계와 대시보드는
# 이 롱포맷 테이블을 groupby만 하면 되므로 빠르다.
#
# 입력: results/<분야>/01_corpus/filtered_corpus.parquet
# 출력: results/<분야>/01_corpus/unigram_long.parquet, bigram_long.parquet
# 실행: python -m analysis_codes.n_02_term_frequency   (인자 없이 실행하면 기본 분야 config.DEFAULT_DOMAIN)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from analysis_codes import n_01_build_corpus as build_dataset
from analysis_codes.text_utils import unigrams, bigrams
from config import DEFAULT
from pipeline_codes.utils import logger

OUTPUT_DIR = DEFAULT.shared_source
UNIGRAM_PATH = OUTPUT_DIR / "unigram_long.parquet"
BIGRAM_PATH = OUTPUT_DIR / "bigram_long.parquet"


def _explode(df: pd.DataFrame, extractor, ngram_type: str) -> pd.DataFrame:
    """df의 display_name마다 extractor로 용어 집합을 뽑아 (문서 1건당 용어 1행)로 펼친다."""
    terms = df["display_name"].apply(extractor)
    long_df = df[["id", "year", "journal"]].copy()
    long_df["term"] = terms
    long_df = long_df.explode("term").dropna(subset=["term"])
    long_df["ngram_type"] = ngram_type
    return long_df.reset_index(drop=True)


def build_long_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"[term_frequency] 유니그램/바이그램 추출 대상: {len(df):,}건")
    uni_long = _explode(df, unigrams, "unigram")
    bi_long = _explode(df, bigrams, "bigram")
    logger.info(f"[term_frequency] 유니그램 롱테이블: {len(uni_long):,}행 (고유 용어 {uni_long['term'].nunique():,}개)")
    logger.info(f"[term_frequency] 바이그램 롱테이블: {len(bi_long):,}행 (고유 용어 {bi_long['term'].nunique():,}개)")
    return uni_long, bi_long


def main(corpus_path: Path | None = None, output_dir: Path | None = None) -> None:
    """인자를 안 주면 기본 분야(config.DEFAULT)의 01_corpus를 쓴다."""
    corpus_path = Path(corpus_path) if corpus_path else build_dataset.OUTPUT_PATH
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path}가 없습니다. 먼저 analysis_codes/n_01_build_corpus.py를 실행하세요."
        )
    df = pd.read_parquet(corpus_path, columns=["id", "year", "journal", "display_name"])

    uni_long, bi_long = build_long_tables(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    uni_long.to_parquet(output_dir / UNIGRAM_PATH.name, index=False)
    bi_long.to_parquet(output_dir / BIGRAM_PATH.name, index=False)
    logger.info(f"[term_frequency] 저장 완료: {UNIGRAM_PATH.name}, {BIGRAM_PATH.name}")


if __name__ == "__main__":
    main()
