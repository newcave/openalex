# webapp/backend/ai_insights.py
# 전체기간 분석·구간별분석·키워드 검색 화면의 "AI 해석(참고용)" 텍스트를 다루는 모듈.
# 모든 함수는 첫 인자로 분야별 서비스(webapp.backend.services.DomainService)를 받는다 - 프롬프트의 분야명
# ("수자원 분야 연구 동향...")과 읽어올 데이터가 분야마다 다르기 때문이다.
#
# 전체기간 분석과 구간별분석은 안정성이 서로 달라서 처리 방식을 분리했다.
#
#   - 전체기간 분석: "분석 진행하기" 마지막 단계에서 generate_full_labels()가 커뮤니티마다 이름+해설을
#     생성해 그 분야 근거문서 폴더의 ai_labels.json에 저장한다. 다시 분석하면 커뮤니티 구성이 바뀔 수
#     있어 매 실행마다 새로 만든다.
#
#   - 구간별분석: 구간별 커뮤니티 구성은 같은 데이터로 다시 돌려도 Louvain 특성상 달라질 수 있다는 게
#     실제로 확인됐다(2000-2005 구간에서 논문 발표 당시 10개 -> 지금 12개로 달라짐). 그래서 파일에
#     저장해두는 방식 자체를 쓰지 않는다. 사용자가 단어를 선택하고 "AI 해석 보기"를 누른 그 순간에만
#     지금 화면에 있는 선택 데이터를 generate_live_insight()로 즉석 호출하고, 결과는 저장하지 않는다
#     (app.py의 콜백에서 직접 호출함 - 이 모듈은 호출 함수만 제공).

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from config import GEMMA_BASE_URL, GEMMA_MODEL, OPENAI_MODEL


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL = OPENAI_MODEL
# provider 값 -> (모델명, 화면 표시 이름). 'AI와 대화하기'는 대화창에서 고른다.
PROVIDERS = {
    "openai": (OPENAI_MODEL, f"OpenAI ({OPENAI_MODEL})"),
    "gemma": (GEMMA_MODEL, "Gemma 4 12B"),
}
LIVE_PROVIDER = "gemma"   # 'AI 해석 보기'(즉석 생성 3종)에 쓰는 모델. OpenAI와 비교해 본 뒤 Gemma로 정했다.
# 수자원 논문 Section III-D(시기별 연구흐름) 문단. 구간별분석 흐름 요약을 생성할 때 "문체 예시"로만 쓴다.
TRAJECTORY_STYLE_EXAMPLE_PATH = Path(__file__).resolve().parent / "trajectory_style_example.json"

SYSTEM_PROMPT = """당신은 {field} 분야 연구 동향 분석 보고서를 돕는 해설가입니다.
주어진 "대표 용어"와 "논문 제목 예시"만 근거로 삼아, 이 용어 묶음이 어떤 연구 흐름을 보여주는지
한국어로 3~4문장의 짧은 해설을 작성하세요.

반드시 지켜야 할 규칙:
1. 제공된 용어/논문 제목에 없는 사실을 새로 지어내지 마세요.
2. 이미 정해진 연구주제 라벨이 주어지면 그 라벨을 바꾸거나 새로 짓지 말고, 그대로 인용하며 설명만 덧붙이세요.
3. 통계적 유의성 검정 없이 뽑힌 "구간별 분석" 데이터라는 설명이 주어지면, "떠오르는 주제입니다"처럼
   단정하지 말고 "이 구간에서 상대적으로 늘어난 용어들의 구조"라는 어조로 서술하세요.
4. 과장되거나 홍보성인 표현 없이 담백하게, 연구자가 참고할 수 있는 수준으로 쓰세요."""


def _field(svc) -> str:
    return svc.d.name


def _client(provider: str = "openai"):
    from openai import OpenAI
    if provider == "gemma":
        return OpenAI(base_url=GEMMA_BASE_URL, api_key="not-needed")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY가 비어 있습니다. 프로젝트 루트의 .env 파일에 "
            "OPENAI_API_KEY=sk-... 형태로 채운 뒤 다시 실행하세요."
        )
    return OpenAI(api_key=key)


def _generate_one(svc, terms: list[str], paper_titles: list[str], label: str | None = None,
                   extra_context: str = "", provider: str = "openai") -> str:
    """해설 한 단락을 생성하는 AI 호출. 전체기간·구간별 화면의 "AI 해석 보기"가 이 함수를 거친다."""
    client = _client(provider)
    parts = []
    if label:
        parts.append(f"연구주제 라벨: {label}")
    parts.append(f"대표 용어: {', '.join(terms)}")
    parts.append("논문 제목 예시:\n" + "\n".join(f"- {t}" for t in paper_titles))
    if extra_context:
        parts.append(f"추가 맥락: {extra_context}")
    user_prompt = "\n".join(parts)

    resp = client.chat.completions.create(
        model=PROVIDERS[provider][0],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.replace("{field}", _field(svc))},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------- 3번 탭: 전체기간 분석
def full_insight(svc, cluster: int) -> dict | None:
    """전체기간 분석 커뮤니티 해설: generate_full_labels()가 저장한 해설({"ko":...}). 없으면 None."""
    for c in svc.load_ai_labels().get("clusters", []):
        if int(c["cluster"]) == int(cluster) and c.get("insight"):
            return {"ko": c["insight"]}
    return None


def load_paper_trajectories() -> dict:
    """논문 Section III-D(시기별 연구흐름 비교)의 내용. overview는 전체 흐름(구간과 무관하게 항상
    동일), by_window는 그 안에서 특정 구간에 해당하는 사실만 추려 재구성한 것."""
    if not TRAJECTORY_STYLE_EXAMPLE_PATH.exists():
        return {"overview": None, "by_window": {}}
    data = json.loads(TRAJECTORY_STYLE_EXAMPLE_PATH.read_text(encoding="utf-8"))
    data.pop("_comment", None)
    return data


FULL_LABEL_SYSTEM_PROMPT = """당신은 {field} 분야 연구 동향 분석 보고서를 돕는 해설가입니다.
주어진 "대표 용어"와 "논문 제목 예시"만 근거로, 이 용어 묶음을 가리키는 짧은 한국어 연구주제명과
3~4문장의 해설을 만드세요. 번호나 기호("C0" 등)는 절대 붙이지 말고, 사람이 읽는 제목처럼 쓰세요.

이 용어들은 {period} 전체 기간에 걸쳐 Hamed-Rao Mann-Kendall 검정과 Benjamini-Hochberg FDR 보정으로
증가 추세가 통계적으로 유의하다고 확인된 용어(Rising Term) 가운데, 논문 제목 안에서 서로 자주 함께
쓰여 하나의 커뮤니티로 묶인 것들입니다. "다른 커뮤니티의 대표 용어"도 함께 주어지니, 연구주제명이
그 커뮤니티들과 겹치지 않고 구별되도록 지으세요.

반드시 지켜야 할 규칙:
1. 제공된 용어/논문 제목에 없는 사실을 새로 지어내지 마세요.
2. 연구주제명은 짧은 명사구로 쓰세요.
3. 과장되거나 홍보성인 표현 없이 담백하게, 연구자가 참고할 수 있는 수준으로 쓰세요.
4. 반드시 JSON으로만 답하세요: {"title": "짧은 연구주제명", "body": "해설 3~4문장"}"""


def generate_full_labels(svc, emit=print) -> dict:
    """분야의 "분석 진행하기" 마지막 단계에서 호출된다. 커뮤니티마다 대표 용어
    15개 + 논문 제목 8개와, 이름이 겹치지 않도록 다른
    커뮤니티의 대표 용어 5개씩을 함께 보내 이름+해설을 만들고 ai_labels.json에 저장한다."""
    client = _client()
    y0, y1 = svc.full_year_range()
    system = FULL_LABEL_SYSTEM_PROMPT.replace("{field}", _field(svc)).replace("{period}", f"{y0}-{y1}")

    clusters = [item["cluster"] for item in svc.full_cluster_legend()]
    top_terms = {cid: svc.full_cluster_terms(cid, top_n=15)["term"].tolist() for cid in clusters}

    out = []
    for cid in clusters:
        papers = svc.full_cluster_papers(cid, None, limit=8)["display_name"].tolist()
        others = [f"- {', '.join(top_terms[o][:5])}" for o in clusters if o != cid]
        user_prompt = (
            f"대표 용어: {', '.join(top_terms[cid])}\n"
            "논문 제목 예시:\n" + "\n".join(f"- {t}" for t in papers) + "\n"
            "다른 커뮤니티의 대표 용어:\n" + ("\n".join(others) if others else "(없음)")
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        title = str(data.get("title", "")).strip() or ", ".join(top_terms[cid][:4])
        out.append({"cluster": int(cid), "label": title, "insight": str(data.get("body", "")).strip(),
                    "top_terms": top_terms[cid]})
        emit(f"  C{cid}: {title}")

    result = {
        "_comment": "분석 진행하기 때마다 OpenAI로 새로 생성되는 참고용 라벨. 전문가 검수 전.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "clusters": out,
    }
    svc.ai_labels_path.parent.mkdir(parents=True, exist_ok=True)
    svc.ai_labels_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------- 3번 탭: 전체기간 분석 (여러 커뮤니티 선택, 저장 없음)
def generate_live_insight_full(svc, terms: list[str], paper_titles: list[str], provider: str = LIVE_PROVIDER) -> str:
    """3번 탭에서 여러 커뮤니티에 걸쳐 단어를 선택하고 "AI 해석 보기"를 눌렀을 때 호출된다.
    선택한 단어는 전체기간 Rising 용어(Hamed-Rao MK + BH FDR로 통계적 검증까지
    끝난 집합)에서 고른 것이므로, generate_live_insight()와 달리 tau>0 캐비어트를 붙이지 않는다."""
    return _generate_one(svc, terms, paper_titles, provider=provider)


# ---------------------------------------------------------------- 4번 탭: 구간별분석 (저장 없음)
def generate_live_insight(svc, terms: list[str], paper_titles: list[str], provider: str = LIVE_PROVIDER) -> str:
    """사용자가 4번 탭에서 단어를 선택하고 "AI 해석 보기"를 누른 바로 그 순간 호출된다.
    구간별 커뮤니티 구성은 재실행마다 달라질 수 있어 결과를 파일에 저장하지 않는다 - 매번 지금
    화면에 보이는 선택을 그대로 다시 보낸다."""
    return _generate_one(
        svc, terms, paper_titles,
        extra_context="이 결과는 통계적 유의성 검정 없이 해당 구간에서 증가 방향(tau>0)만으로 뽑은 용어입니다.",
        provider=provider,
    )


TOPIC_SYSTEM_PROMPT = """당신은 {field} 분야 연구 동향 분석 보고서를 돕는 해설가입니다.
주어진 "대표 용어"와 "논문 제목 예시"만 근거로, 이 용어 묶음을 가리키는 짧은 한국어 연구주제명과
3~4문장의 해설을 만드세요. 번호나 기호("C0" 등)는 절대 붙이지 말고, 사람이 읽는 제목처럼 쓰세요.

반드시 지켜야 할 규칙:
1. 제공된 용어/논문 제목에 없는 사실을 새로 지어내지 마세요.
2. 통계적 유의성 검정 없이 뽑힌 "구간별 분석" 데이터이므로, "떠오르는 주제입니다"처럼 단정하지 말고
   "이 구간에서 상대적으로 늘어난 용어들의 구조"라는 어조로 서술하세요.
3. 과장되거나 홍보성인 표현 없이 담백하게, 연구자가 참고할 수 있는 수준으로 쓰세요.
4. 반드시 JSON으로만 답하세요: {"title": "짧은 연구주제명", "body": "해설 3~4문장"}"""


def generate_live_topic(svc, terms: list[str], paper_titles: list[str]) -> dict:
    """4번 탭에서 커뮤니티 노드를 클릭한 순간 호출된다. 원래는 "C3 · model, river, flood" 처럼
    대표 용어를 기계적으로 이어붙인 라벨을 보여줬는데, 3번 탭처럼 사람이 읽을 수 있는 주제명 +
    해설을 그 자리에서 만들어 보여주기 위한 함수다. 결과는 저장하지 않는다(generate_live_insight와
    같은 이유 - 구간별 커뮤니티 구성이 재분석마다 달라질 수 있음).
    반환값: {"title": str, "body": str}"""
    client = _client()
    user_prompt = (
        f"대표 용어: {', '.join(terms)}\n"
        "논문 제목 예시:\n" + "\n".join(f"- {t}" for t in paper_titles)
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TOPIC_SYSTEM_PROMPT.replace("{field}", _field(svc))},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# 구간 개수/이름은 데이터 연도 범위에 따라 달라지므로({n_windows}, {window_list}, {json_format}),
# 실행할 때 실제 구간으로 채운다. 수자원(2000-2025)이면 예전 고정 문구와 똑같이 채워진다.
TRAJECTORY_SYSTEM_PROMPT = """당신은 {field} 분야 연구 동향 분석을 돕는 해설가입니다.
{n_windows}개 시기({window_list})에 대해 각 시기 안에서 독립적으로
묶인 연구주제 커뮤니티 목록(그 시기 안에서의 순위, 비중%, 대표 용어)이 주어집니다. 이 가운데 여러
시기에 걸쳐 대표 용어가 의미적으로 이어지는 "연구 흐름" 2~4개를 찾아, 시기에 따라 내용과 순위·비중이
어떻게 달라졌는지 간결하게 설명하세요.

아래는 과거 같은 방식의 분석에서 실제로 작성됐던 설명의 "문체 예시"입니다 - 문장 구조와 톤만
참고하고, 내용(구체적으로 어떤 연구 흐름인지, 홍수/수질/기후변화 같은 특정 주제)은 그대로 가져오지
마세요. 반드시 지금 주어지는 데이터에서 실제로 나타나는 흐름만 찾아 새로 작성해야 합니다 - 다음에
새 데이터로 다시 분석하면 내용은 또 달라질 수 있습니다:

---문체 예시 시작---
{example}
---문체 예시 끝---

반드시 지켜야 할 규칙:
1. 주어진 순위/비중 숫자와 대표 용어 외의 사실을 새로 지어내지 마세요.
2. 통계적 유의성 검정 없이 tau>0만으로 뽑힌 구간별 자료이므로, "떠오르는 주제"처럼 단정하지 말고
   "그 구간에서 상대적으로 늘어난 용어의 구조"라는 어조로 쓰세요.
3. overview는 찾아낸 흐름들을 합쳐서 6~10문장 정도로 짧게 쓰세요(예시처럼 길게 쓰지 마세요).
4. by_window는 {n_windows}개 구간 각각에 대해, 그 구간에서 각 흐름이 어떤 모습이었는지 1~2문장으로만 쓰세요.
5. 과장되거나 홍보성인 표현 없이 담백하게 쓰세요.
6. 반드시 JSON으로만 답하세요:
{json_format}"""


def _periods_summary(svc, top_terms_per_cluster: int = 10, top_clusters_per_period: int = 6) -> dict:
    summary = {}
    for window in svc.windows():
        if not (svc.d.windowed_output / window / "tau_positive_unigram_clusters.parquet").exists():
            continue
        terms_df = svc.load_window_terms(window)
        sizes = terms_df.groupby("cluster").size().sort_values(ascending=False)
        total = int(sizes.sum())
        clusters = []
        for rank, (cid, size) in enumerate(sizes.items(), start=1):
            if rank > top_clusters_per_period:
                break
            top_terms = (
                terms_df[terms_df["cluster"].eq(cid)]
                .sort_values("전체기간_평균비중(%)", ascending=False)["term"]
                .head(top_terms_per_cluster).tolist()
            )
            clusters.append({"rank": rank, "share_pct": round(size / total * 100, 1), "top_terms": top_terms})
        summary[window] = clusters
    return summary


def generate_trajectory_overview(svc) -> dict:
    """"분석 진행하기"(4번 탭)가 끝난 직후 run_windowed_analysis 콜백 안에서 호출된다. 전체 구간을
    통째로 비교하는 내용이라 이번 실행 결과로만 의미가 있으므로 파일에 저장하지 않고, 매번
    새로 생성해서 dcc.Store(windowed-trajectory-data)에만 담아둔다.
    반환값: {"overview": str, "by_window": {window: str, ...}}"""
    client = _client()
    periods = _periods_summary(svc)
    windows = [w for w in svc.windows() if w in periods]

    lines = []
    labels = svc.window_labels()
    for window in windows:
        lines.append(f"[{labels[window]}]")
        for c in periods[window]:
            lines.append(f"  순위 {c['rank']}위, 비중 {c['share_pct']}%, 대표 용어: {', '.join(c['top_terms'])}")
    user_prompt = "\n".join(lines)

    example = load_paper_trajectories().get("overview", {}).get("ko", "")
    json_format = (
        '{"overview": "...", "by_window": {'
        + ", ".join(f'"{w}": "..."' for w in windows)
        + "}}"
    )
    system = (
        TRAJECTORY_SYSTEM_PROMPT
        .replace("{field}", _field(svc))
        .replace("{n_windows}", str(len(windows)))
        .replace("{window_list}", ", ".join(w.replace("_", "-") for w in windows))
        .replace("{json_format}", json_format)
        .replace("{example}", example)
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------- 5번 탭: 키워드 검색 (저장 없음)
PAIR_SYSTEM_PROMPT = """당신은 {field} 분야 연구 동향 분석을 돕는 해설가입니다.
사용자가 선택한 단어 조합(보통 2개)에 대해, 주어진 데이터만 근거로 짧은 한국어 인사이트를
작성하세요. 입력은 각 단어의 전체기간({period}) Rising 여부·소속 커뮤니티, 구간별(tau>0)
소속 커뮤니티, 두 단어가 함께 등장한 논문 수와 제목 예시, 전체기간 네트워크에서 두 커뮤니티가
직접 연결돼 있는지(공유 문서수)로 구성됩니다.

판단 기준:
1. 함께 등장한 논문이 있다면: 그 논문들이 주로 어떤 내용인지, 각 단어의 커뮤니티/트렌드 맥락과
   엮어서 어떤 연구 흐름을 보이는지 설명하세요.
2. 함께 등장한 논문이 없다면: 각 단어의 Rising 여부, 소속 커뮤니티, 두 커뮤니티의 네트워크 연결
   여부를 근거로 "아직 다뤄지지 않았지만 유망해 보이는 조합"인지 "서로 무관해 보이는 억지 조합"인지
   판단하고, 그 판단의 근거(트렌드가 둘 다 증가 중인지, 커뮤니티가 연결/인접한지 등)를 함께 제시하세요.

반드시 지켜야 할 규칙:
1. 주어진 데이터에 없는 사실을 새로 지어내지 마세요.
2. 구간별({n_windows}개 구간) 자료는 통계적 유의성 검정 없이 tau>0만으로 뽑힌 것이므로, "떠오르는 주제"처럼
   단정하지 말고 "그 구간에서 상대적으로 늘어난 용어"라는 어조로 쓰세요.
3. 과장되거나 홍보성인 표현 없이, 연구자가 참고할 수 있는 수준으로 4~6문장 정도로 담백하게 쓰세요.
4. "유망하다"/"억지다" 같은 판단을 내릴 때는 반드시 근거를 함께 말하세요 - 근거 없는 단정은 금지."""


def generate_keyword_pair_insight(svc, context: dict, provider: str = LIVE_PROVIDER) -> str:
    """5번 탭에서 2개 이상 단어를 고르고 "AI 해석 보기"를 눌렀을 때 호출된다. 결과는 저장하지
    않는다 - 선택이 바뀔 때마다, 그리고 데이터가 새로 들어와 커뮤니티 구성이 달라질 때마다
    매번 지금 상태 그대로 다시 호출한다."""
    client = _client(provider)
    y0, y1 = svc.full_year_range()
    system = (
        PAIR_SYSTEM_PROMPT
        .replace("{field}", _field(svc))
        .replace("{period}", f"{y0}-{y1}")
        .replace("{n_windows}", str(len(svc.windows())))
    )
    user_prompt = json.dumps(context, ensure_ascii=False, indent=2)
    resp = client.chat.completions.create(
        model=PROVIDERS[provider][0],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()
