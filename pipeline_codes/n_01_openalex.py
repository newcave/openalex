# pipeline_codes/n_01_openalex.py
# 기존 real1 + real2 기능을 통합한 개선된 버전
# ISSN 리스트와 출판 연도 범위를 받아 OpenAlex에서 논문 데이터를 수집하는 파이프라인
# DOI 없는 논문에 대해 제목 기반 역검색으로 DOI 보강 기능 포함

import asyncio
import os
import re
import time
import random
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import aiohttp
import pandas as pd
import requests
from pyalex import Sources, Works, config

from pipeline_codes import event_log
from pipeline_codes.utils import logger, collect_metrics

# OpenAlex에서 가져올 필드
FIELDS = [
    'id',
    'doi',
    'display_name',
    'publication_date',
    'authorships',
    'primary_location',
    'abstract_inverted_index',
    'cited_by_count',
    'concepts',
    'keywords',
    'language',
    'type',
    'open_access',
]

BATCH_SIZE = 5  # DOI 역검색 동시 요청 수

# OpenAlex 저널(source) 정보 자체가 틀린 경우의 보정표 (2026-10-15 실측).
# 한국물환경학회지: OpenAlex가 이 ISSN을 '신경정신의학'(S4306495044) 저널에 합쳐둬서 ISSN으로 받으면
# 정신의학 논문이 섞여 들어온다(1,440건 중 학회지 논문은 201건). 그래서 저널 조회 없이 학회지 고유
# DOI 접두어로 받는다. OpenAlex에 있는 건 2014~2021년 383편뿐이다.
DOI_PREFIX_SOURCES = {
    "2289-0971": ("10.15681/kswe", "한국물환경학회지"),
    "2289-098X": ("10.15681/kswe", "한국물환경학회지"),
}
# Environmental Engineering Research: 같은 저널 안에 일본 환경공학연구논문집(DOI 10.11532/proes,
# 2000~2008년 501편)이 섞여 있어 수집할 때 뺀다.
EXCLUDED_DOI_PREFIXES = {
    "1226-1025": ("10.11532/proes",),
    "2005-968X": ("10.11532/proes",),
}
_DOI_PREFIX_ID = "doi_prefix:"  # resolve_journal_ids가 저널 ID 대신 넘기는 표시


class OpenAlexUnavailable(RuntimeError):
    """재시도를 다 써도 OpenAlex 응답을 못 받은 경우(대부분 429 = 일일 요청 한도 초과).
    이건 "데이터가 없다"가 아니라 "못 받았다"이므로, 빈 결과나 중간까지 받은 일부를 캐시에 저장하면
    안 된다(저장하면 다음 수집 때 '이미 있음'으로 건너뛰어 영영 덜 받은 채로 남는다).
    2026-10-06 상하수도 수집에서 Journal of Hydrology 2020년이 600건(실제 약 1,300건)만 저장된 사례로 확인."""

    def __init__(self, message: str, rate_limited: bool):
        super().__init__(message)
        self.rate_limited = rate_limited


def _is_rate_limit(e: Exception) -> bool:
    return "429" in str(e)


class OpenAlexBudgetExhausted(OpenAlexUnavailable):
    """429 중에서도 "일일 요청 한도(크레딧) 소진"이 확인된 경우. 몇 초~몇 분 재시도해도 풀리지 않고
    reset_at(UTC)에야 초기화되므로, 재시도 없이 즉시 멈춰야 한다."""

    def __init__(self, message: str, reset_at: datetime):
        super().__init__(message, rate_limited=True)
        self.reset_at = reset_at


def check_budget() -> dict:
    """OpenAlex에 가장 가벼운 요청(저널 1건 조회, 1크레딧)을 보내 응답 헤더로 오늘 남은 한도를 확인한다.

    2026-10-08 실측: 목록/조회 요청은 1크레딧($0.0001), 제목 검색(DOI 역검색)은 10크레딧, 하루 한도
    10,000크레딧($1). 따라서 남은 크레딧이 1 미만이면 어떤 요청도 할 수 없는 "소진" 상태다.
    반환: {"status", "remaining", "reset_at"(UTC datetime), "exhausted"}. 확인 자체가 실패하면
    exhausted=False로 돌려 기존 재시도 흐름에 맡긴다(모르는 걸 소진으로 단정하지 않음)."""
    params = {"filter": "issn:0043-1397", "select": "id", "per-page": 1}
    if config.api_key:
        params["api_key"] = config.api_key
    if config.email:
        params["mailto"] = config.email
    now = datetime.now(timezone.utc)
    try:
        r = requests.get("https://api.openalex.org/sources", params=params, timeout=20)
    except Exception as e:
        logger.warning(f"OpenAlex 한도 확인 실패: {e}")
        return {"status": None, "remaining": None, "reset_at": None, "exhausted": False}

    def _num(name):
        try:
            return float(r.headers.get(name))
        except (TypeError, ValueError):
            return None

    remaining = _num("X-RateLimit-Remaining")
    prepaid = _num("X-RateLimit-Prepaid-Remaining-USD") or 0.0
    reset_s = _num("X-RateLimit-Reset")
    if reset_s is not None:
        reset_at = now + timedelta(seconds=reset_s)
    else:  # 헤더가 없으면 OpenAlex 일일 한도 기준 시각(자정 UTC)으로
        reset_at = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    body = r.text[:300].lower() if r.status_code != 200 else ""
    exhausted = (
        (remaining is not None and remaining < 1 and prepaid <= 0)
        or (r.status_code == 429 and "budget" in body)
    )
    logger.info(f"OpenAlex 한도 확인: status={r.status_code}, 남은 크레딧={remaining}, 소진={exhausted}")
    return {"status": r.status_code, "remaining": remaining, "reset_at": reset_at, "exhausted": exhausted}


# ---------------------------------------------------------------- API 키 여러 개 돌려 쓰기
# .env에 키를 여러 개 넣어 두면(OPENALEX_API_KEY, OPENALEX_API_KEY_2, OPENALEX_API_KEY_3 ...), 앞 키의 하루
# 한도가 다 차는 순간 다음 키로 바꿔 끼우고 받던 요청부터 이어 간다. pyalex는 요청마다 config.api_key를
# 새로 읽으므로 config.api_key만 바꾸면 된다.
_api_keys: list[str] = []
_key_index = 0


def load_api_keys() -> list[str]:
    """환경변수(.env)의 OpenAlex 키 목록. OPENALEX_API_KEY(콤마로 여러 개도 가능) 다음에 OPENALEX_API_KEY_2, _3 ... 순서."""
    keys = [k.strip() for k in os.environ.get("OPENALEX_API_KEY", "").split(",")]
    numbered = sorted((int(m.group(1)), v) for name, v in os.environ.items()
                      if (m := re.fullmatch(r"OPENALEX_API_KEY_(\d+)", name)))
    keys += [v.strip() for _, v in numbered]
    return list(dict.fromkeys(k for k in keys if k))


def use_api_keys(keys, restart: bool = False) -> None:
    """수집에 쓸 키 목록을 정한다. 같은 목록이면 지금 쓰던 키를 유지한다(restart=True면 첫 키부터 다시)."""
    global _api_keys, _key_index
    keys = [keys] if isinstance(keys, str) else [k for k in (keys or []) if k]
    if keys != _api_keys or restart:
        _api_keys, _key_index = keys, 0
    config.api_key = _api_keys[_key_index] if _api_keys else None


def _switch_to_next_key() -> bool:
    global _key_index
    if _key_index + 1 >= len(_api_keys):
        return False
    _key_index += 1
    config.api_key = _api_keys[_key_index]
    logger.warning(f"OpenAlex 키 {_key_index}번 하루 한도 소진 -> {_key_index + 1}번 키로 전환 ({_key_index + 1}/{len(_api_keys)})")
    return True


def raise_if_budget_exhausted(context: str) -> bool:
    """지금 키의 하루 한도가 다 찼으면 다음 키로 바꾸고 True를 돌려준다(호출한 쪽은 같은 요청을 다시 하면 된다).
    더 바꿀 키가 없으면 OpenAlexBudgetExhausted를 던진다. 한도가 남아 있으면 False."""
    budget = check_budget()
    switched = False
    while budget["exhausted"]:
        if not _switch_to_next_key():
            raise OpenAlexBudgetExhausted(f"{context} - OpenAlex 일일 요청 한도 소진(키 {len(_api_keys) or 0}개 모두)",
                                          budget["reset_at"])
        switched = True
        budget = check_budget()
    return switched

def reconstruct_abstract(inverted_index: dict) -> str:
    """
    OpenAlex abstract_inverted_index를 평문으로 복원.
    {단어: [위치1, 위치2, ...]} 형태를 위치 순서대로 재조립.
    """
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    pairs = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    pairs.append((pos, word))
    pairs.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pairs)


def sanitize_title(raw: str) -> str:
    """
    논문 제목 정제.
    대괄호 내용, 비ASCII 문자, 다중 공백 제거.
    DOI 역검색 시 제목 비교에도 재사용됨.
    """
    if not raw:
        return ""
    t = re.sub(r"[\[【].*?[】\]]", "", raw)
    t = re.sub(r"[^\x00-\x7F]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_doi(doi: str) -> str:
    """
    DOI 정규화: URL 형태 → 순수 DOI 형태.
    예: 'https://doi.org/10.1016/j.xxx' → '10.1016/j.xxx'
    """
    return doi.replace("https://doi.org/", "").strip() if doi else ""


##### DOI 보강
# DOI 없는 논문을 제목으로 OpenAlex에서 역검색하여 DOI 보강 (정규화된 제목 완전일치 + 출판연도 일치 시에만 업데이트).
async def search_doi_by_title(session: aiohttp.ClientSession,
                               work: dict,
                               sem: asyncio.Semaphore) -> dict | None:
    """
    DOI 없는 논문을 제목으로 OpenAlex에서 역검색.
    정규화된 제목 완전일치 + 출판연도 일치 시에만 DOI 업데이트.
    """
    title = work.get('display_name', '')
    pub_date = work.get('publication_date', '')
    if not title or not pub_date:
        return None

    original_year = pub_date[:4]  # '2021-03-15' → '2021'
    sanitized = sanitize_title(title).lower()

    async with sem:
        backoff = 2.0
        for attempt in range(4):
            try:
                results = Works().search_filter(
                    display_name=title
                ).select(['id', 'doi', 'display_name', 'publication_date']).get()

                for candidate in results:
                    # 연도 일치 확인
                    candidate_year = (candidate.get('publication_date') or '')[:4]
                    if candidate_year != original_year:
                        continue

                    # 제목 완전일치 확인
                    candidate_title = sanitize_title(
                        candidate.get('display_name') or ''
                    ).lower()
                    if candidate_title != sanitized:
                        continue

                    # DOI 있는지 확인
                    doi = candidate.get('doi')
                    if not doi:
                        continue

                    logger.info(f"DOI 역검색 성공: {work['id']} → {doi}")
                    return {'work_id': work['id'], 'doi': doi}

                return None  # 검색은 됐지만 일치하는 후보 없음

            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    logger.warning(f"DOI 역검색 429 - {backoff:.0f}초 후 재시도 [{title[:30]}...]")
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                logger.warning(f"DOI 역검색 실패 [{title[:30]}...]: {e}")
                # "일치 후보 없음"(None)과 구분해 실패 사유를 돌려준다 - 모니터링 집계용.
                return {'work_id': work['id'], 'error': str(e)}

    return None


async def enrich_doi_by_title(works: list[dict], failures: dict | None = None) -> int:
    """
    DOI 없는 논문 목록에 대해 제목 기반 역검색 일괄 수행.
    반환: 보강 성공 건수. failures(dict)를 넘기면 검색 실패 건수를 사유별로 채운다
    (event_log.classify 기준, 예: {"budget": 3, "server": 12}).
    """
    missing = [w for w in works if not w.get('doi')]
    if not missing:
        logger.info("DOI 결측 없음 - 역검색 스킵")
        return 0

    logger.info(f"DOI 역검색 시작: {len(missing)}건")
    sem = asyncio.Semaphore(BATCH_SIZE)
    enriched_count = 0

    async with aiohttp.ClientSession() as session:
        tasks = [search_doi_by_title(session, w, sem) for w in missing]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # 결과 반영
    result_map = {}
    for r in results:
        if isinstance(r, dict) and r.get('work_id') and r.get('doi'):
            result_map[r['work_id']] = r['doi']
        elif failures is not None and (isinstance(r, BaseException) or (isinstance(r, dict) and r.get('error'))):
            reason = event_log.classify(str(r if isinstance(r, BaseException) else r['error']))
            failures[reason] = failures.get(reason, 0) + 1

    for w in works:
        if w['id'] in result_map:
            w['doi'] = result_map[w['id']]
            enriched_count += 1

    logger.info(f"DOI 역검색 완료: {enriched_count}/{len(missing)}건 보강")
    return enriched_count

def fetch_journal_id(issn: str) -> tuple[str, str] | tuple[None, None]:
    """
    ISSN으로 OpenAlex 저널 ID 조회.
    반환: (journal_id, display_name) 또는 (None, None) = OpenAlex에 그 ISSN 저널이 없음.
    응답 자체를 못 받으면(429 등) "없음"과 구분되도록 OpenAlexUnavailable을 던진다.
    """
    backoff = 1.0
    max_retry = 7

    while True:
        try:
            src_list = next(
                Sources()
                .filter(issn=[issn])
                .select(['id', 'display_name'])
                .paginate(per_page=5)
            )
            if not src_list:
                logger.warning(f"ISSN={issn} 에 해당하는 저널 없음")
                return None, None

            first = src_list[0] if isinstance(src_list[0], dict) else {}
            journal_id = first.get('id')
            display_name = first.get('display_name')

            if not journal_id:
                logger.warning(f"ISSN={issn} 저널 ID 없음")
                return None, None

            logger.info(f"저널 확인: {display_name} (ID={journal_id})")
            return journal_id, display_name

        except Exception as e:
            if "429" in str(e) and max_retry == 7:
                # 첫 429에서 한 번만: 하루 한도 소진이면 다음 키로 바꿔 다시 하고, 남은 키가 없으면 바로 멈춘다.
                if raise_if_budget_exhausted(f"ISSN {issn} 저널 조회"):
                    continue
            if "429" in str(e) and max_retry > 0:
                sleep_s = backoff + random.uniform(0, 0.5)
                logger.warning(f"ISSN={issn} 저널 조회 429 - {sleep_s:.1f}초 후 재시도")
                time.sleep(sleep_s)
                backoff = min(backoff * 2.0, 30.0)
                max_retry -= 1
                continue
            logger.error(f"ISSN={issn} 저널 조회 실패: {e}")
            reason = "OpenAlex 요청 한도 초과(429)" if _is_rate_limit(e) else f"OpenAlex 응답 오류: {e}"
            raise OpenAlexUnavailable(f"ISSN {issn} 저널 조회 실패 - {reason}", _is_rate_limit(e)) from e


def fetch_works_for_year(journal_id: str, year: int,
                          include_only_with_abstract: bool = False) -> list[dict]:
    """
    저널 ID + 단일 연도로 논문 수집 (페이지네이션 + 재시도 포함).
    재시도를 다 쓰면 중간까지 받은 것을 버리고 OpenAlexUnavailable을 던진다(일부만 저장 방지).
    """
    yearly = []
    if journal_id.startswith(_DOI_PREFIX_ID):
        source_filter = {"doi_starts_with": journal_id[len(_DOI_PREFIX_ID):]}
    else:
        source_filter = {"journal": journal_id}
    works_q = Works().filter(
        **source_filter,
        from_publication_date=f"{year}-01-01",
        to_publication_date=f"{year}-12-31",
    )

    if include_only_with_abstract:
        works_q = works_q.filter(has_abstract='true')

    backoff = 1.0
    max_retry = 7
    budget_checked = False
    pg = works_q.select(FIELDS).paginate(per_page=200)

    while True:
        try:
            page = next(pg)
            yearly.extend(page)
            backoff = 1.0
        except StopIteration:
            break
        except Exception as e:
            if _is_rate_limit(e) and not budget_checked:
                # 첫 429에서 한 번만: 하루 한도 소진이면 다음 키로 바꿔 이 페이지부터 다시 받는다.
                # 남은 키가 없으면 재시도 없이 바로 멈춘다(받던 연도는 버림).
                budget_checked = True
                if raise_if_budget_exhausted(f"{year}년 수집"):
                    budget_checked = False
                    continue
            if max_retry <= 0:
                logger.error(f"{year}년 수집 중 재시도 초과(받은 {len(yearly)}건 버림): {e}")
                reason = "OpenAlex 요청 한도 초과(429)" if _is_rate_limit(e) else f"OpenAlex 응답 오류: {e}"
                raise OpenAlexUnavailable(f"{year}년 수집 실패 - {reason}", _is_rate_limit(e)) from e
            sleep_s = backoff + random.uniform(0, 0.5)
            logger.warning(f"{year}년 수집 오류, {sleep_s:.1f}초 후 재시도: {e}")
            time.sleep(sleep_s)
            backoff = min(backoff * 2.0, 30.0)
            max_retry -= 1

    logger.info(f"{year}년 [{journal_id}] 수집: {len(yearly)}건")
    return yearly


def fetch_works(journal_id: str, year_start: int, year_end: int,
                include_only_with_abstract: bool = False) -> list[dict]:
    """
    저널 ID + 연도 범위로 논문 수집.
    연도별로 나눠서 조회 (OpenAlex 페이지 제한 대응).
    """
    works = []
    for year in range(year_start, year_end + 1):
        works.extend(fetch_works_for_year(journal_id, year, include_only_with_abstract))
    return works


def resolve_journal_ids(issns: list[str]) -> tuple[list[str], list[str]]:
    """
    ISSN 리스트 → (OpenAlex 저널 ID 리스트, 실제 resolve에 성공한 ISSN 리스트).
    존재하지 않는 ISSN은 matched_issns에서 빠지므로, 캐시 파일명에
    "실제로 써서 성공한 ISSN"만 쓸 수 있게 된다.
    """
    journal_ids = []
    matched_issns = []
    for issn in issns:
        if issn in DOI_PREFIX_SOURCES:
            prefix, name = DOI_PREFIX_SOURCES[issn]
            logger.info(f"ISSN={issn}: OpenAlex 저널 정보가 틀려 DOI 접두어 {prefix}로 수집 ({name})")
            journal_ids.append(_DOI_PREFIX_ID + prefix)
            matched_issns.append(issn)
            continue
        journal_id, _ = fetch_journal_id(issn)
        if journal_id:
            journal_ids.append(journal_id)
            matched_issns.append(issn)

    before_dedup = len(journal_ids)
    journal_ids = list(dict.fromkeys(journal_ids))
    if len(journal_ids) < before_dedup:
        logger.info(f"저널 ID 중복 제거: {before_dedup}건 -> {len(journal_ids)}건")

    return journal_ids, matched_issns


def _finalize_df(unique_works: list[dict], issns: list[str]) -> pd.DataFrame:
    """
    OpenAlex works 원본 리스트 → 정제된 DataFrame 변환.
    abstract 복원, 결측 행 제거, journal/issn 컬럼 정리.
    """
    for w in unique_works:
        aii = w.get('abstract_inverted_index')
        if isinstance(aii, dict) and aii:
            w['abstract'] = reconstruct_abstract(aii)
        else:
            w['abstract'] = ""
        w.pop('abstract_inverted_index', None)

    df = pd.json_normalize(unique_works, sep='_')
    if df.empty:
        return df

    drop_cols = [c for c in df.columns if 'abstract_inverted_index' in c]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    if 'authorships' in df.columns:
        before = len(df)
        df = df[df['authorships'].apply(
            lambda x: bool(x) if isinstance(x, list) else str(x).strip() != '[]'
        )]
        logger.info(f"authorships 없는 행 제거: {before - len(df)}건")

    before = len(df)
    df = df[df['id'].str.match(r'^https?://openalex\.org/W\d+', na=False)]
    logger.info(f"id 패턴 필터: {before - len(df)}건 제거")

    if 'primary_location_source_display_name' in df.columns:
        df['journal'] = df['primary_location_source_display_name']
    else:
        df['journal'] = None

    drop_cols = [c for c in df.columns if c.startswith('primary_location')]
    df.drop(columns=drop_cols, inplace=True)

    df['issn'] = ','.join(issns)
    return df


async def collect_year(journal_ids: list[str], issns: list[str], year: int,
                        include_only_with_abstract: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    단일 연도치 논문 수집 + DOI 역검색 + DataFrame 정제.
    연도 단위로 실행되므로, 중간에 API 예산/네트워크 문제가 생겨도
    이미 끝난 연도까지는 온전히 처리된 상태로 남는다.
    반환: (DataFrame, metrics 딕셔너리)
    """
    all_works = []
    for journal_id in journal_ids:
        all_works.extend(fetch_works_for_year(journal_id, year, include_only_with_abstract))

    unique_works = list({w['id']: w for w in all_works}.values())

    excluded = tuple(p for issn in issns for p in EXCLUDED_DOI_PREFIXES.get(issn, ()))
    if excluded:
        before = len(unique_works)
        unique_works = [w for w in unique_works
                        if not any(f"doi.org/{p}" in (w.get('doi') or '').lower() for p in excluded)]
        if before > len(unique_works):
            logger.info(f"{year}년 다른 저널 논문 제외(DOI {', '.join(excluded)}): {before - len(unique_works)}건")

    doi_missing_before = sum(1 for w in unique_works if not w.get('doi'))
    doi_failures: dict = {}
    # [2026-10-14 일시 중단] DOI 역검색(제목 검색, 1건당 10크레딧)이 하루 한도 대부분을 써서 수집이 자주
    # 막힘. 상하수도 실적도 3,703건 중 8건(0.2%)만 성공. 다시 쓰려면 아래 줄의 주석을 풀고 그 다음 줄을 지운다.
    # doi_enriched = await enrich_doi_by_title(unique_works, doi_failures)
    doi_enriched = 0
    doi_missing_after = sum(1 for w in unique_works if not w.get('doi'))

    df = _finalize_df(unique_works, issns)
    # DOI 접두어로 받은 저널은 OpenAlex 저널명이 틀렸거나('신경정신의학') 비어 있으므로 바로잡는다.
    override_name = next((DOI_PREFIX_SOURCES[i][1] for i in issns if i in DOI_PREFIX_SOURCES), None)
    if override_name and not df.empty:
        df['journal'] = override_name

    metrics = collect_metrics(
        total_collected=len(unique_works),
        doi_missing_before=doi_missing_before,
        doi_enriched=doi_enriched,
        doi_missing_after=doi_missing_after,
        doi_search_failed=sum(doi_failures.values()),
        doi_fail_reasons=doi_failures,
    )

    logger.info(f"{year}년 파이프라인 완료: {len(df)}건")
    return df, metrics


#### main 파이프라인 (연도별 collect_year를 순차 실행하는 올인원 헬퍼)
## issns: ISSN 리스트 (예: ['0043-1354', '1234-5678'])
## year_start, year_end: 수집할 출판 연도 범위
## email: OpenAlex API 사용을 위한 이메일 (config.email에 설정됨)
## 반환: (DataFrame, metrics 딕셔너리) - metrics는 연도별 값을 합산

async def main(issns: list[str],
         year_start: int,
         year_end: int,
         email: str,
         include_only_with_abstract: bool = False) -> tuple[pd.DataFrame, dict]:
    config.email = email

    journal_ids, matched_issns = resolve_journal_ids(issns)
    if not journal_ids:
        logger.error("유효한 저널 ID 없음 - 종료")
        return pd.DataFrame(), {}

    year_dfs = []
    total_metrics: dict = {}
    for year in range(year_start, year_end + 1):
        year_df, year_metrics = await collect_year(
            journal_ids, matched_issns, year, include_only_with_abstract
        )
        if not year_df.empty:
            year_dfs.append(year_df)
        for k, v in year_metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v

    df = pd.concat(year_dfs, ignore_index=True) if year_dfs else pd.DataFrame()
    logger.info(f"파이프라인 완료: {len(df)}건")
    return df, total_metrics


if __name__ == "__main__":
    df, metrics = main(
        issns=['0043-1354'],
        year_start=2020,
        year_end=2022,
        email='your@email.com',
    )
    print(df.head())
    print(metrics)