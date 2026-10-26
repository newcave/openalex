# pipeline_codes/event_log.py
# 수집/분석 이벤트를 "한 줄 = JSON 하나"로 계속 덧붙여 쌓는 기록(JSON Lines).
#
# 텍스트 로그(logs/journal_app.log*)는 사람이 읽는 용도라 문구가 바뀌면 집계가 깨진다. 모니터링
# 화면은 이 구조화된 기록만 읽는다.
#   - 덧붙이기만 한다(기존 줄을 고치거나 지우지 않음) -> 기록이 계속 보존된다.
#   - 월별 파일(collection_events_YYYY-MM.jsonl)로 나눠 파일 하나가 무한정 커지지 않게 한다.
#   - 서버로 옮길 때는 이 폴더만 가져가면 된다. 위치는 환경변수 OPENALEX_EVENT_DIR로 바꿀 수 있다.
#   - backfill_*.jsonl: 이 기록이 생기기 전의 텍스트 로그를 scripts/backfill_collection_events.py로
#     한 번 변환한 것(source="log").

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
EVENT_DIR = Path(os.environ.get("OPENALEX_EVENT_DIR") or Path(__file__).resolve().parent.parent / "logs" / "events")

# 실패 사유 분류. 모니터링 그래프의 색 구분과 로그 표기에 같이 쓴다.
REASONS = {
    "budget": "일일 한도 소진",
    "rate_limit": "요청 제한(429)",
    "server": "서버 오류(5xx)",
    "network": "네트워크 오류",
    # 제목에 ':' '*' 같은 검색 문법 문자나 번역 제목이 섞여 OpenAlex가 거절한 DOI 검색. 재시도해도 같은 결과.
    "bad_request": "잘못된 검색어(400)",
    "not_found": "저널 없음",
    "other": "기타 오류",
}


def classify(message: str) -> str:
    m = (message or "").lower()
    if "한도 소진" in m or "budget" in m:
        return "budget"
    if "429" in m:
        return "rate_limit"
    if "400 client error" in m or "bad request" in m or "not a valid field" in m or "wildcards" in m:
        return "bad_request"
    if any(code in m for code in ("500", "502", "503", "504", "5xx")):
        return "server"
    if any(k in m for k in ("timeout", "timed out", "connection", "연결")):
        return "network"
    if "찾을 수 없" in m or "not found" in m:
        return "not_found"
    return "other"


def record(event: str, **fields) -> None:
    """event: "collect"(연도 수집 결과) | "lookup"(저널 조회 실패) | "doi"(DOI 역검색 실패 집계) | "analysis"
    공통 필드: ts(KST), event, status("ok"|"empty"|"fail"), domain, source("live")"""
    now = datetime.now(KST)
    row = {"ts": now.isoformat(timespec="seconds"), "event": event, "source": "live", **fields}
    try:
        EVENT_DIR.mkdir(parents=True, exist_ok=True)
        path = EVENT_DIR / f"collection_events_{now:%Y-%m}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError:
        # 기록 실패가 수집 자체를 멈추게 하면 안 된다.
        pass


def read_all() -> list[dict]:
    rows = []
    if not EVENT_DIR.exists():
        return rows
    for path in sorted(EVENT_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # 쓰는 도중 끊긴 줄 등은 건너뛴다
    rows.sort(key=lambda r: r.get("ts", ""))
    return rows
