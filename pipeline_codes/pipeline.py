# pipeline_codes/pipeline.py
# 수집 실행. 웹앱 "데이터 수집" 화면이 run()을 부르고, 명령줄로도 분야 전체를 한 번에 받을 수 있다:
#   python -m pipeline_codes.pipeline --domain ai water_env            (2000~2025, 이미 받은 연도는 건너뜀)
#   python -m pipeline_codes.pipeline --domain ai --start 2010 --end 2025
# 명령줄 수집은 .env의 OpenAlex 키를 순서대로 쓰고, 모든 키의 하루 한도가 차면 초기화 시각까지 기다렸다가 이어 받는다.
import argparse
import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd

from pipeline_codes import n_01_openalex as openalex
from pipeline_codes import n_02_institution_extract as institutions
from pipeline_codes import n_04_visualize as visualize
from pipeline_codes import cache_manager
from pipeline_codes import event_log
from pipeline_codes.utils import logger


async def run(issns: list[str],
              year_start: int,
              year_end: int,
              email: str,
              api_key: str | list[str] | None = None,
              html_path: str = "network.html",
              cache_dir: Path | None = None,
              domain: str | None = None) -> AsyncGenerator:
    """
    데이터 수집 파이프라인 실행 (시각화 제외).
    시각화는 사용자가 버튼을 클릭했을 때 run_visualize()로 별도 실행.

    연도 단위로 [논문 수집+DOI 보강] -> [기관 정보 파싱] -> [캐시 저장]을
    순차 반복한다(cache_dir를 안 주면 기본 분야 dataset 폴더에 저장). 중간에 API 예산 소진, 네트워크 장애 등으로 중단되더라도
    이미 완료된 연도는 온전히 캐시에 저장되어 있어 다음 실행에서 재수집할 필요가 없다.

    기관 정보(ROR/기관명)는 OpenAlex가 authorships에 이미 매칭해서 제공하므로
    네트워크 호출 없이 파싱만 한다 (pipeline_codes.n_02_institution_extract). OpenAlex가
    매칭하지 못한 나머지는 ROR 로컬 덤프로 보강할 수 있는데, 이는 전체 캐시에
    대해 별도로(reprocess_institutions.py --local-match) 1회 실행하는 것이
    효율적이라 이 수집 루프에는 포함하지 않는다.

    yield 구조:
      - step  : 단계 메시지
      - done  : 완료 여부
      - log   : (선택) UI 로그 패널에 표시할 메시지
      - level : (선택) "info" | "warning" | "error"
      - df    : (완료 시) 최종 정제된 DataFrame
    """
    all_metrics: dict = {}

    yield {"step": "📡 논문 수집 준비 중...", "done": False}

    cached_df, missing_years = cache_manager.get(issns, year_start, year_end, cache_dir)
    collected_dfs = [cached_df] if not cached_df.empty else []

    # 모니터링용 구조화 기록(logs/events). domain은 수집을 시킨 분야 키(config.py).
    journal_name = None
    if not cached_df.empty and "journal" in cached_df.columns and cached_df["journal"].notna().any():
        journal_name = str(cached_df["journal"].dropna().iloc[0])

    def rec(event: str, **fields):
        event_log.record(event, domain=domain, issns=issns, journal=journal_name, **fields)

    # OpenAlex 응답을 못 받아(대부분 429 요청 한도 초과) 이번에 받지 못한 연도. 캐시에 저장하지 않으므로
    # 다음 수집 때 자동으로 다시 시도된다. 결과 metrics에 담아 화면이 "성공"과 구분해 보여주게 한다.
    failed_years: list[int] = []
    error_msg: str | None = None
    rate_limited = False
    reset_at = None  # 하루 한도 소진이 확인되면 초기화 시각(UTC)

    if missing_years:
        openalex.config.email = email
        if api_key:
            # 키가 여러 개면 앞 키의 하루 한도가 찼을 때 다음 키로 넘어간다(n_01_openalex.use_api_keys).
            openalex.use_api_keys(api_key)
        try:
            # 시작 전 확인: 이미 하루 한도를 다 썼으면 저널 조회부터 429로 헛돌지 않게 바로 멈춘다.
            openalex.raise_if_budget_exhausted("수집 시작 전 확인")
            journal_ids, matched_issns = openalex.resolve_journal_ids(issns)
        except openalex.OpenAlexUnavailable as e:
            error_msg, rate_limited = str(e), e.rate_limited
            reset_at = getattr(e, "reset_at", None)
            failed_years = sorted(missing_years)
            logger.error(error_msg)
            rec("lookup", status="fail", reason=event_log.classify(error_msg), message=error_msg,
                years=failed_years)
            yield {"step": "", "done": False, "log": error_msg, "level": "error"}
            journal_ids, matched_issns = None, []

        if journal_ids is not None and not journal_ids:
            msg = f"ISSN {', '.join(issns)} 에 해당하는 저널을 찾을 수 없습니다. ISSN을 확인해주세요."
            logger.error(msg)
            rec("lookup", status="fail", reason="not_found", message=msg, years=sorted(missing_years))
            yield {"step": "", "done": False, "log": msg, "level": "error"}
            yield {"step": "오류", "done": True, "html_path": "",
                   "metrics": {"total_collected": len(cached_df), "not_found": True, "error": msg,
                               "failed_years": sorted(missing_years), "rate_limited": False}}
            return

        for year in sorted(missing_years) if journal_ids else []:
            yield {"step": f"📡 {year}년 논문 수집 중...", "done": False}
            try:
                year_df, metrics_year = await openalex.collect_year(journal_ids, matched_issns, year)
            except openalex.OpenAlexUnavailable as e:
                # 한도 초과는 몇 분 안에 풀리지 않으므로 남은 연도도 시도하지 않고 멈춘다(이미 받은 연도는 저장돼 있음).
                error_msg, rate_limited = str(e), e.rate_limited
                reset_at = getattr(e, "reset_at", None)
                failed_years = [y for y in sorted(missing_years) if y >= year]
                msg = f"{error_msg} → {year}년부터 수집 중단(이미 받은 연도는 저장됨, 다음 수집 때 빠진 연도만 다시 받음)"
                logger.error(msg)
                rec("collect", status="fail", year=year, reason=event_log.classify(error_msg), message=error_msg,
                    years=failed_years)
                yield {"step": "", "done": False, "log": msg, "level": "error"}
                break

            if metrics_year.get("doi_search_failed"):
                rec("doi", status="fail", year=year, n=int(metrics_year["doi_search_failed"]),
                    reasons=metrics_year.get("doi_fail_reasons") or {})

            if year_df.empty:
                msg = f"{year}년: 수집된 논문 없음"
                logger.info(msg)
                rec("collect", status="empty", year=year, n=0)
                yield {"step": "", "done": False, "log": msg, "level": "info"}
                continue

            doi_before = metrics_year.get("doi_missing_before", 0)
            doi_after  = metrics_year.get("doi_missing_after", 0)
            msg = f"{year}년 논문 {len(year_df)}건 수집, DOI 결측 {doi_before}건 중 {doi_before - doi_after}건 보강"
            logger.info(msg)
            yield {"step": "", "done": False, "log": msg, "level": "info"}

            # ── 해당 연도만 기관 정보 파싱 (네트워크 호출 없음) ──
            year_df, metrics_inst = institutions.main(year_df, unmatched_out=None)
            with_ror = metrics_inst.get("rows_with_ror", 0)
            rate = metrics_inst.get("affiliation_match_rate")
            msg = f"{year}년 기관 파싱: {with_ror}/{len(year_df)}건 ROR 존재, affiliation 매칭률 {rate}"
            logger.info(msg)
            yield {"step": "", "done": False, "log": msg, "level": "info"}

            # ── 해당 연도만 즉시 캐시 저장 (실제 resolve된 ISSN으로 이름 붙임) ──
            cache_manager.save(matched_issns, year_df, cache_dir)
            msg = f"{year}년 캐시 저장 완료: {len(year_df)}건"
            logger.info(msg)
            if "journal" in year_df.columns and year_df["journal"].notna().any():
                journal_name = str(year_df["journal"].dropna().iloc[0])
            rec("collect", status="ok", year=year, n=int(len(year_df)),
                doi_missing_before=int(doi_before), doi_missing_after=int(doi_after))
            yield {"step": "", "done": False, "log": msg, "level": "info"}

            collected_dfs.append(year_df)
            for metrics_part in (metrics_year, metrics_inst):
                for k, v in metrics_part.items():
                    if v is None:
                        # 예: 소속 정보가 하나도 없는 연도는 affiliation_match_rate=None. 이 값을 저장해두면
                        # 다음 연도의 숫자와 더할 때 None + float로 수집 전체가 멈춘다(2026-10-08 Water 2004→2005).
                        all_metrics.setdefault(k, None)
                    elif isinstance(v, (int, float)):
                        all_metrics[k] = (all_metrics.get(k) or 0) + v
                    else:
                        all_metrics[k] = v

            yield {"step": f"✅ {year}년 완료", "done": False}
    else:
        msg = "전체 캐시 히트 — API 호출 없음"
        logger.info(msg)
        yield {"step": "", "done": False, "log": msg, "level": "info"}

    df = pd.concat(collected_dfs, ignore_index=True) if collected_dfs else pd.DataFrame()
    if not df.empty and 'id' in df.columns:
        df = df.drop_duplicates(subset='id', keep='last')

    # total_collected는 이번 실행에서 새로 받은 연도만의 원본 합계가 아니라
    # 캐시 + 신규분을 합친 최종 논문 수(len(df))를 그대로 써야 한다.
    all_metrics["total_collected"] = len(df)
    all_metrics["failed_years"] = failed_years
    all_metrics["error"] = error_msg
    all_metrics["rate_limited"] = rate_limited
    all_metrics["budget_exhausted"] = reset_at is not None
    all_metrics["reset_at"] = reset_at.isoformat() if reset_at else None
    all_metrics.setdefault("doi_enriched", 0)
    all_metrics.setdefault("doi_missing_before", 0)
    all_metrics.setdefault("doi_missing_after", 0)

    msg = f"총 논문 {len(df)}건 (캐시 포함)"
    logger.info(msg)
    yield {"step": "", "done": False, "log": msg, "level": "info"}

    # ── 완료 (시각화는 별도 실행) ─────────────────────────────────
    yield {
        "step": "완료",
        "done": True,
        "metrics": all_metrics,
        "df": df,           # 시각화 버튼 클릭 시 재사용할 DataFrame
        "html_path": "",    # 아직 시각화 미실행
    }


async def run_visualize(df: pd.DataFrame,
                        html_path: str = "network.html") -> AsyncGenerator:
    """
    시각화 전용 파이프라인.
    사용자가 '시각화 생성' 버튼을 클릭했을 때 호출.
    run()이 반환한 df를 그대로 넘겨주면 됨.

    yield 구조:
      - step     : 단계 메시지
      - done     : 완료 여부
      - html_path: (완료 시) 생성된 HTML 파일 경로
    """
    yield {"step": "🌐 네트워크 시각화 생성 중...", "done": False}

    if df is None or df.empty:
        yield {"step": "오류", "done": True,
               "log": "시각화할 데이터가 없습니다.", "level": "error",
               "html_path": ""}
        return

    result_html = visualize.main(df, html_path=html_path, scale=2, base_size=6)
    yield {"step": "✅ 시각화 생성 완료", "done": False}

    yield {
        "step": "완료",
        "done": True,
        "html_path": result_html,
    }


# ---------------------------------------------------------------- 명령줄: 분야 전체 수집
def _log(msg: str) -> None:
    print(f"{datetime.now():%m-%d %H:%M:%S} {msg}", flush=True)


async def _collect_journal(d, journal, year_start: int, year_end: int, keys: list[str], email: str) -> dict:
    final = {}
    async for update in run(issns=journal.issns, year_start=year_start, year_end=year_end, email=email,
                            api_key=keys, cache_dir=d.dataset_dir, domain=d.key):
        if update.get("log"):
            _log(f"  [{update.get('level', 'info')}] {update['log']}")
        if update.get("done"):
            final = update
    return final.get("metrics", {}) if final else {}


def collect_domains(domain_keys: list[str], year_start: int, year_end: int, wait_for_reset: bool = True,
                    max_passes: int = 3) -> None:
    """분야마다 수집 대상 저널 전체를 연도 범위만큼 받는다. 이미 받은 연도는 건너뛴다.
    한 바퀴가 끝나면 실패한 저널이 있을 때만 한 바퀴 더(최대 max_passes) 돈다."""
    from dotenv import load_dotenv
    from config import DOMAINS, OPENALEX_EMAIL, ROOT
    load_dotenv(ROOT / ".env")
    keys = openalex.load_api_keys()
    _log(f"수집 시작: 분야 {', '.join(DOMAINS[k].name for k in domain_keys)} / {year_start}-{year_end} / OpenAlex 키 {len(keys)}개")
    openalex.use_api_keys(keys)

    for n_pass in range(1, max_passes + 1):
        failed = []
        for dkey in domain_keys:
            d = DOMAINS[dkey]
            for i, journal in enumerate(d.journals, 1):
                label = f"[{d.name} {i}/{len(d.journals)}] {journal.name} ({', '.join(journal.issns)})"
                if "미등록" in journal.note:
                    _log(f"{label}: 건너뜀 - {journal.note}")
                    continue
                while True:
                    _log(f"{label} 시작")
                    try:
                        m = asyncio.run(_collect_journal(d, journal, year_start, year_end, keys, OPENALEX_EMAIL))
                    except Exception as e:
                        _log(f"{label}: 오류 - {e}")
                        failed.append(label)
                        break
                    if m.get("budget_exhausted"):
                        # 그사이 .env에 키가 추가됐으면 기다리지 않고 바로 이어 간다(이미 소진된 키는 확인 후 건너뜀).
                        load_dotenv(ROOT / ".env", override=True)
                        new_keys = openalex.load_api_keys()
                        if new_keys != keys:
                            _log(f".env에서 OpenAlex 키 {len(new_keys)}개를 다시 읽어 이어 갑니다(이전 {len(keys)}개)")
                            keys = new_keys
                            openalex.use_api_keys(keys, restart=True)
                            continue
                    if m.get("budget_exhausted") and wait_for_reset and m.get("reset_at"):
                        # 모든 키의 하루 한도가 찼다 -> 초기화 시각(UTC 자정 = 한국 오전 9시)까지 기다렸다가 첫 키부터 다시.
                        reset_at = datetime.fromisoformat(m["reset_at"]) + timedelta(minutes=2)
                        wait_s = max(60.0, (reset_at - datetime.now(timezone.utc)).total_seconds())
                        _log(f"OpenAlex 키 {len(keys)}개 모두 하루 한도 소진 -> {wait_s / 3600:.1f}시간 대기 "
                             f"(재개 예정 {reset_at.astimezone(timezone(timedelta(hours=9))):%m-%d %H:%M} KST)")
                        time.sleep(wait_s)
                        openalex.use_api_keys(keys, restart=True)
                        continue
                    status = ("미등록/ISSN 확인 필요" if m.get("not_found") else
                              f"일부 연도 실패 {m.get('failed_years')}" if m.get("failed_years") else "완료")
                    _log(f"{label}: {status} - 총 {m.get('total_collected', 0):,}편(기존 포함)")
                    if m.get("failed_years") and not m.get("not_found"):
                        failed.append(label)
                    break
        if not failed:
            _log(f"전체 수집 완료 ({n_pass}바퀴)")
            return
        _log(f"{n_pass}바퀴 끝 - 덜 받은 저널 {len(failed)}종: " + " / ".join(failed))
    _log(f"{max_passes}바퀴 후에도 덜 받은 저널이 남아 종료합니다. 같은 명령을 다시 실행하면 빠진 연도만 받습니다.")


if __name__ == "__main__":
    from config import DOMAINS
    parser = argparse.ArgumentParser(description="분야 수집 대상 저널 전체를 OpenAlex에서 받아 dataset/<분야>/에 저장한다.")
    parser.add_argument("--domain", nargs="+", required=True, choices=sorted(DOMAINS), help="config.py의 분야 key (여러 개 가능)")
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--no-wait", action="store_true", help="모든 키 한도 소진 시 기다리지 않고 종료")
    args = parser.parse_args()
    collect_domains(args.domain, args.start, args.end, wait_for_reset=not args.no_wait)
    sys.exit(0)
