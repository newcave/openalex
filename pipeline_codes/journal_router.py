# pipeline_codes/journal_router.py
# 수집할 저널(ISSN)을 어느 분야 dataset 폴더에 저장할지 정한다. "데이터 수집" 화면에서 분야를
# "AI 자동 분류"로 골랐을 때 쓴다.
#
#   1) config.py 수집 대상 목록에 있는 ISSN  -> 목록에 적힌 분야(여러 개일 수 있음). AI를 부르지 않는다.
#   2) 목록에 없는 ISSN -> OpenAlex에서 저널 정보(이름·주요 토픽·최근 논문 제목)를 가져와 OpenAI(config.OPENAI_MODEL)가
#      분야를 고른다(0개~여러 개). 판단 결과는 dataset/ai_journal_assignments.json에 저장하고, 같은 저널은
#      다음부터 저장된 판단을 그대로 쓴다(AI를 다시 부르지 않음). 판단이 틀렸으면 이 파일을 직접 고치면 된다.

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from config import DATA_ROOT, DOMAINS, OPENAI_MODEL, ROOT, domains_for_issn

load_dotenv(ROOT / ".env")

ROUTER_MODEL = OPENAI_MODEL
ASSIGNMENTS_PATH = DATA_ROOT / "dataset" / "ai_journal_assignments.json"
OPENALEX_URL = "https://api.openalex.org"
N_SAMPLE_TITLES = 20

SYSTEM_PROMPT = """당신은 한국수자원공사 연구원의 논문 수집 담당자입니다. 새로 수집할 학술지 하나가 주어지면,
아래 연구소(분야) 가운데 이 학술지의 논문을 연구 동향 분석 자료로 쓸 만한 분야를 고르세요.

{catalog}

판단 규칙:
1. 학술지 이름, 주요 토픽, 최근 논문 제목만 근거로 판단하세요.
2. 각 분야의 기존 수집 저널과 성격이 비슷한지를 기준으로 삼으세요.
3. 여러 분야에 해당하면 모두 고르세요. 어느 분야와도 관련이 약하면 빈 목록을 고르세요.
4. 반드시 JSON으로만 답하세요: {{"domains": ["분야 key", ...], "reason": "한국어 1~2문장 근거"}}"""


def _catalog() -> str:
    lines = []
    for d in DOMAINS.values():
        names = ", ".join(dict.fromkeys(j.name for j in d.journals))
        lines.append(f"- key={d.key} | {d.name} ({d.department}) | 기존 수집 저널: {names}")
    return "\n".join(lines)


def _openalex_params(**params) -> dict:
    params["mailto"] = "hjhj@kwater.or.kr"
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    return params


def lookup_source(issns: list[str]) -> dict | None:
    """ISSN들로 OpenAlex 저널(source)을 찾아 이름·토픽·최근 논문 제목을 모은다. 없으면 None."""
    for issn in issns:
        r = requests.get(f"{OPENALEX_URL}/sources", timeout=30, params=_openalex_params(
            filter=f"issn:{issn}", select="id,display_name,issn,works_count,topics,host_organization_name"))
        r.raise_for_status()
        results = r.json().get("results") or []
        if results:
            src = results[0]
            break
    else:
        return None
    source_id = src["id"].rsplit("/", 1)[-1]
    r = requests.get(f"{OPENALEX_URL}/works", timeout=30, params=_openalex_params(
        filter=f"primary_location.source.id:{source_id}", sort="publication_date:desc",
        select="display_name", **{"per-page": N_SAMPLE_TITLES}))
    r.raise_for_status()
    return {
        "source_id": source_id,
        "name": src.get("display_name"),
        "publisher": src.get("host_organization_name"),
        "issns_openalex": src.get("issn") or [],
        "works_count": src.get("works_count"),
        "topics": [t["display_name"] for t in (src.get("topics") or [])[:10]],
        "sample_titles": [w["display_name"] for w in r.json().get("results") or [] if w.get("display_name")],
    }


def classify(source: dict) -> dict:
    """OpenAI로 분야를 고른다. 반환: {"domains": [key, ...], "reason": str}"""
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 비어 있습니다. 프로젝트 루트 .env에 채운 뒤 다시 실행하세요.")
    user = "\n".join([
        f"학술지: {source['name']} (발행: {source.get('publisher') or '-'}, 논문 {source.get('works_count') or 0:,}편)",
        f"주요 토픽: {', '.join(source['topics']) or '-'}",
        "최근 논문 제목:",
        *[f"- {t}" for t in source["sample_titles"]],
    ])
    resp = OpenAI(api_key=key).chat.completions.create(
        model=ROUTER_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT.format(catalog=_catalog())},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    domains = [k for k in data.get("domains") or [] if k in DOMAINS]
    return {"domains": list(dict.fromkeys(domains)), "reason": str(data.get("reason") or "")}


def load_assignments() -> dict:
    try:
        return json.loads(ASSIGNMENTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_assignment(entry: dict) -> None:
    data = load_assignments()
    data[entry["source_id"]] = entry
    ASSIGNMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ASSIGNMENTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def route(issns: list[str]) -> dict:
    """저널 하나(같은 저널의 Print·Electronic ISSN 묶음)의 저장 분야를 정한다.
    반환: {"domains": [key, ...], "by": "목록" | "AI" | "AI(저장된 판단)" | "미등록", "name": str, "reason": str}"""
    listed = list(dict.fromkeys(k for issn in issns for k in domains_for_issn(issn)))
    if listed:
        return {"domains": listed, "by": "목록", "name": ", ".join(issns), "reason": "수집 대상 목록(config.py)에 있음"}

    source = lookup_source(issns)
    if source is None:
        return {"domains": [], "by": "미등록", "name": ", ".join(issns), "reason": "OpenAlex에서 이 ISSN의 저널을 찾지 못함"}

    saved = load_assignments().get(source["source_id"])
    if saved:
        return {"domains": saved["domains"], "by": "AI(저장된 판단)", "name": saved["name"], "reason": saved["reason"]}

    decision = classify(source)
    _save_assignment({
        "source_id": source["source_id"],
        "name": source["name"],
        "issns_input": issns,
        "issns_openalex": source["issns_openalex"],
        "domains": decision["domains"],
        "reason": decision["reason"],
        "model": ROUTER_MODEL,
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return {"domains": decision["domains"], "by": "AI", "name": source["name"], "reason": decision["reason"]}
