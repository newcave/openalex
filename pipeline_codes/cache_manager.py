# pipeline_codes/cache_manager.py
# 수집 원본(parquet) 저장/조회. 파일 하나 = 저널 하나의 1년치 (저널명_ISSN_연도.parquet)
#   ISSN + 연도 범위로 저장 폴더를 확인해서
#   -> 이미 있는 연도: 파일에서 읽어오기
#   -> 없는 연도: OpenAlex API로 새로 수집해야 할 연도로 반환
import re
from pathlib import Path
import numpy as np
import pandas as pd
from config import DEFAULT
from pipeline_codes.utils import logger

# 기본 저장 위치 = 기본 분야(config.DEFAULT)의 dataset 폴더 (실제 수집은 항상 cache_dir 인자로 분야 폴더를 넘김)
CACHE_DIR = DEFAULT.dataset_dir


## 캐시 로드 & 저장
def _safe_filename_part(name: str | None) -> str:
    """파일명에 쓸 수 없는 문자를 정리 (저널명용)"""
    if not name or not str(name).strip():
        return "unknown"
    cleaned = re.sub(r'[\\/:*?"<>|]', '_', str(name).strip())
    cleaned = re.sub(r'\s+', '_', cleaned)
    return cleaned or "unknown"


def _cache_path(journal_name: str | None, issns: list[str], year: int, cache_dir: Path | None = None) -> Path:
    """
    저널명_ISSN_연도 형식 캐시 파일 경로.
    issns는 "실제로 resolve에 성공한 ISSN" 리스트를 받아서, 그 중 첫 번째로 이름을 붙인다
    (예: 1번이 안 되고 2번이 됐으면 issns=[2]로 넘어와 A_2_2020.parquet로 저장됨).
    """
    safe_issn = issns[0].replace('/', '_')
    safe_name = _safe_filename_part(journal_name)
    return (cache_dir or CACHE_DIR) / f"{safe_name}_{safe_issn}_{year}.parquet"


def _find_cache_path(issns: list[str], year: int, cache_dir: Path | None = None) -> Path | None:
    """
    ISSN 그룹 + 연도로 캐시 파일 검색.
    저장 시점엔 실제 성공한 ISSN 하나로만 파일명이 정해지므로, 조회 시점엔
    (아직 어떤 ISSN이 성공할지 모르니) 그룹 내 ISSN을 순서대로 다 시도해서
    먼저 찾아지는 파일을 반환한다.
    """
    cache_dir = cache_dir or CACHE_DIR
    for issn in issns:
        safe_issn = issn.replace('/', '_')
        matches = list(cache_dir.glob(f"*_{safe_issn}_{year}.parquet"))
        if matches:
            return matches[0]
        legacy = cache_dir / f"{safe_issn}_{year}.parquet"
        if legacy.exists():
            return legacy
    return None


def save(issns: list[str], df: pd.DataFrame, cache_dir: Path | None = None):
    """
    새로 가져온 데이터를 연도별로 분리해서 저장.
    ISSN 그룹(Print/Electronic 등) 전체를 저널 하나로 취급해 연도당 파일 1개만 저장한다.
    파일명: {저널명}_{그룹 대표 ISSN}_{연도}.parquet
    """
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if 'publication_date' not in df.columns:
        logger.warning("publication_date 컬럼 없음 - 캐시 저장 스킵")
        return

    if df.empty:
        return

    # 연도 컬럼 추출
    df['year'] = pd.to_datetime(
        df['publication_date'], errors='coerce'
    ).dt.year

    journal_name = None
    if 'journal' in df.columns:
        non_null = df['journal'].dropna()
        if not non_null.empty:
            journal_name = non_null.iloc[0]

    for year, year_df in df.groupby('year'):
        year = int(year)
        path = _cache_path(journal_name, issns, year, cache_dir)

        # 기존(다른 이름) 캐시 파일이 있으면 정리 후 신규 이름으로 저장
        old_path = _find_cache_path(issns, year, cache_dir)
        if old_path is not None and old_path != path:
            old_path.unlink(missing_ok=True)

        year_df.to_parquet(path, index=False)
        logger.info(f"캐시 저장: {path.name} ({len(year_df)}건)")


## 캐시 조회 함수
def _convert(obj):
    """중첩된 구조에서 numpy 타입을 Python 기본 타입으로 재귀 변환"""
    if isinstance(obj, np.ndarray):
        return [_convert(i) for i in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(i) for i in obj]
    return obj


def _parquet_to_df(path: Path) -> pd.DataFrame:
    """parquet 읽고 중첩된 numpy 타입 → Python 기본 타입 변환"""
    df = pd.read_parquet(path)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(_convert)
    return df


def get(issns: list[str],
        year_start: int,
        year_end: int,
        cache_dir: Path | None = None) -> tuple[pd.DataFrame, list[int]]:
    """
    ISSN + 연도 범위로 캐시 조회.
    cache_dir를 안 주면 기본 분야(config.DEFAULT)의 dataset 폴더를 쓴다.
    반환: (캐시된 DataFrame, API 호출 필요한 연도 리스트)
    """
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_dfs = []
    missing_years = []

    for year in range(year_start, year_end + 1):
        path = _find_cache_path(issns, year, cache_dir)

        if path is not None:
            try:
                cached_dfs.append(_parquet_to_df(path))
                logger.info(f"캐시 히트: {path.name}")
                continue
            except Exception as e:
                logger.warning(f"캐시 로드 실패 {path.name}: {e}")

        missing_years.append(year)
        logger.info(f"{year}년 캐시 없음 - API 호출 필요")

    if cached_dfs:
        cached_df = pd.concat(cached_dfs, ignore_index=True)
        cached_df = cached_df.drop_duplicates(subset='id', keep='last')
    else:
        cached_df = pd.DataFrame()

    return cached_df, missing_years