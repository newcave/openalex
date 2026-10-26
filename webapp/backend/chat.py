# webapp/backend/chat.py
# 글로벌 트렌드 분석 화면의 "AI와 대화하기" 대화창.
#   - briefing(): 대화창을 열면 첫 메시지로 보여주는 현황 브리핑. AI를 부르지 않고 저장된 분석 결과로 채운다
#     (숫자가 틀릴 일이 없고 즉시 뜬다).
#   - answer(): 이후 질문에 대한 AI 답변(생성되는 대로 조각을 내보낸다 - 스트리밍). 그 분야 분석 결과를
#     [분석 자료]로 넘기고, 자료에 있는 것만 사실로 말하게 한다. 자료 밖 일반 지식은 답변 끝 "[참고]" 문단에만
#     쓰게 해서 섞이지 않게 한다.
#   - 모델은 대화창에서 OpenAI / Gemma 중 고른다(provider). Gemma는 입력+출력 합계 8,192 토큰까지만 처리해서
#     [분석 자료]를 줄인 판(compact)을 보내고, 그래도 넘치면 오래된 대화부터 뺀다(_request).
#   - briefing_all() / answer_all(): '통합' 화면용. 6개 분야를 각각 분석한 결과를 모아 브리핑하고 답한다
#     (분야를 합쳐 다시 분석한 것이 아님 - 논문 수가 많은 분야로 결과가 쏠리지 않게).
# 대화 기록은 화면(dcc.Store)에만 있고 저장하지 않는다.

from __future__ import annotations

import json
import re
import urllib.request

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config import DOMAINS, GEMMA_BASE_URL, GEMMA_CONTEXT
from webapp.backend import ai_insights

REFERENCE_MARK = "[참고]"   # 일반 지식 문단 시작 표시. app.py가 이 표시로 문단을 나눠 다른 색으로 보여준다.
STREAM_END = "\x00"        # 스트리밍 끝 표시. 이 뒤에 숫자 표기를 정리한 최종 답변 전체가 온다(app.py의 /api/chat-stream).
GEMMA_ANSWER_TOKENS = 1500  # Gemma 답변 길이 상한. 입력은 GEMMA_CONTEXT에서 이만큼 뺀 범위에 맞춘다.

SYSTEM_PROMPT = """당신은 한국수자원공사 연구자를 돕는 '{field} 분야 R&D 글로벌 트렌드' 분석 도우미입니다.
아래 [분석 자료]는 {field} 분야 논문을 수집해 실제로 분석한 결과입니다. 사용자의 질문에 한국어로 답하세요.

반드시 지켜야 할 규칙:
1. 분석 결과에 대한 사실(숫자, 트렌드 후보 이름, 단어, 논문 제목, 연도, 추세)은 [분석 자료]에 있는 그대로만 말하세요.
   자료에 없는 숫자·논문·저자·기관·연도를 절대 만들어내지 마세요.
2. [분석 자료]로 답할 수 없는 질문이면 "분석 자료에는 해당 내용이 없습니다"라고 먼저 분명히 말하세요.
3. 이해를 돕는 일반 지식(분석 자료 밖의 배경 설명)을 덧붙일 때는 본문과 섞지 말고, 답변 맨 끝에
   "[참고]"로 시작하는 별도 문단 하나에만 쓰세요. 이 문단은 분석 결과가 아닌 일반 지식입니다.
   확실하지 않은 일반 지식은 쓰지 마세요. 덧붙일 내용이 없으면 [참고] 문단을 쓰지 마세요.
4. 트렌드 후보 이름과 해설은 AI가 붙인 참고용 라벨이므로, 단정적으로 과장하지 말고 담백하게 설명하세요.
5. 답변은 질문 범위에 맞게 충분히 쓰세요. 핵심 답 → 근거(연도별 비중, 추세 지표, 저널, 논문 제목) → 정리 순서로,
   필요하면 소제목과 목록을 쓰세요. 근거 없이 분량만 늘리지는 마세요.
6. 소수는 소수점 첫째 자리까지만 쓰세요(예: 24.8%). 숫자에는 굵게(**) 표시를 하지 마세요.
7. 가능한 한 구체적으로 답하세요. [분석 자료]의 연도별 비중, 추세 지표, 저널, 논문 제목, 초록 내용을 직접 인용해
   근거를 보여주세요(예: "2005년 1.2%에서 2025년 4.8%로", "2024년 Water Research 논문 '...'에서는 ...").
8. 초록은 초록이 있는 일부 대표 논문에서만 제공됩니다(출판사에 따라 초록이 없는 저널이 많음). 초록 내용을 근거로
   말할 때는 "초록이 있는 일부 논문 기준"임을 밝히고, 그것을 전체 경향처럼 일반화하지 마세요.
9. '유망한', '앞으로 중요할' 분야를 물으면 Rising 기준으로 판단하세요. Rising 단어는 연도별 비중에
   Mann-Kendall 추세검정과 BH-FDR 보정(q≤0.05)을 해서 통계적으로 일관되게 증가한 단어이며, 모든 트렌드 후보는
   Rising 단어로 이루어져 있습니다. 후보를 비교할 때는 증가의 일관성(MK tau, p값)과 Rising 단어 수를 주 근거로,
   연평균 문서 비중(현재 규모)을 보조 근거로 쓰세요. Sen 기울기나 최근5년-초기5년 비중 차 같은 증가폭만으로
   유망하다고 판단하지 마세요. 자료에 MK 지표가 없는 후보는 Rising 단어 수와 비중 추이로 판단하고 그 한계를 밝히세요.
   전망은 분석 결과에 근거한 해석임을 밝히세요.

[분석 자료]
{context}"""

_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def _num(i: int) -> str:
    return _CIRCLED[i] if i < len(_CIRCLED) else f"{i + 1}."


def _overview(svc) -> dict:
    data = svc.cache_overview()
    y0, y1 = (data["year_min"], data["year_max"]) if not data.get("empty") else svc.full_year_range()
    return {"year_min": y0, "year_max": y1,
            "n_journals": 0 if data.get("empty") else data["n_journals"],
            "n_papers": 0 if data.get("empty") else data["total_papers"],
            "n_rising": svc.n_full_rising_terms()}


def briefing(svc, cluster: int) -> str:
    """대화창 첫 메시지(마크다운). 트렌드 후보는 번호를 붙여 한 줄에 하나씩."""
    o = _overview(svc)
    name = svc.d.name
    legend = svc.full_cluster_legend()
    trends = "  \n".join(f"{_num(i)} {item['label']}" for i, item in enumerate(legend))
    label = svc.full_cluster_label(cluster)
    insight = (ai_insights.full_insight(svc, cluster) or {}).get("ko")
    if not insight:
        terms = ", ".join(svc.full_cluster_terms(cluster, top_n=10)["term"].tolist())
        insight = f"이 트렌드 후보의 대표 단어는 {terms}입니다."
    return (
        f"**{name}** 분야에서는 {o['year_min']}년~{o['year_max']}년까지 저널 {o['n_journals']}종에서 수집한 "
        f"논문 {o['n_papers']:,}편에 대해 전처리 과정을 거쳐 총 {o['n_rising']:,}개의 Rising 단어가 "
        f"선별되었습니다.\n\n"
        f"{name} 분야 R&D 글로벌 트렌드는 현재 다음 {len(legend)}개가 있습니다.\n\n"
        f"{trends}\n\n"
        f"선택하신 **'{label}'**에 대한 AI 해석(참고용)은 다음과 같습니다.\n\n"
        f"> {insight}\n\n"
        f"궁금하신 점이 있으면 추가 질문해 주세요."
    )


_SHARE_YEARS = (2000, 2005, 2010, 2015, 2020, 2025)
_detail_cache: dict = {}


def _yearly_share(svc, cluster: int):
    """트렌드 후보의 연도별 (문서 수, 문서 비중%) 표. 02_full_period/cluster_trends_fdr_assoc.parquet."""
    path = svc.d.full_output / "cluster_trends_fdr_assoc.parquet"
    df = svc._cached("chat_cluster_trends", lambda: pd.read_parquet(path), (path.stat().st_mtime_ns,))
    return df[df["cluster"].eq(int(cluster))].sort_values("year")


def _share_points(svc, cluster: int) -> str:
    """5년 단위 문서 비중(비교용 요약). 예: 2000 1.2% → 2005 1.5% → ..."""
    ys = _yearly_share(svc, cluster).set_index("year")["document_share_pct"]
    return " → ".join(f"{y} {ys[y]:.1f}%" for y in _SHARE_YEARS if y in ys.index)


def _paper_details(svc, papers: pd.DataFrame) -> pd.DataFrame:
    """수집 원본(dataset)에서 논문들의 초록·피인용 수를 찾는다(분석 결과 파일에는 없음). 파일 이름 끝이
    연도(저널_ISSN_연도.parquet)라서 그 논문들의 연도 파일만 읽는다."""
    out = []
    want = set(papers["id"])
    years = {str(int(y)) for y in papers["year"]}
    for f in svc.d.dataset_dir.glob("*.parquet"):
        if f.stem.rsplit("_", 1)[-1] not in years:
            continue
        t = pq.read_table(f, columns=["id", "abstract", "cited_by_count"])
        sub = t.filter(pc.is_in(t.column("id"), value_set=pa.array(list(want))))
        if sub.num_rows:
            out.append(sub.to_pandas())
    if not out:
        return pd.DataFrame(columns=["id", "abstract", "cited_by_count"])
    return pd.concat(out).drop_duplicates("id")


def _cluster_details(svc, cluster: int) -> dict:
    """선택한 트렌드 후보의 추가 자료: 저널 분포, 최근 대표 논문, 초록이 있는 대표 논문. (분야, 후보)별로 기억해 둔다."""
    key = (svc.d.key, int(cluster), (svc.d.full_review / "all_papers.csv").stat().st_mtime_ns)
    if key in _detail_cache:
        return _detail_cache[key]
    papers = svc.full_cluster_papers(cluster, None, limit=100000)
    journals = papers["journal"].value_counts().head(8)
    y_max = int(papers["year"].max()) if len(papers) else None
    recent = papers[papers["year"] >= (y_max - 2)].head(15) if y_max else papers.head(0)
    # 초록 후보: 핵심 단어를 많이 포함한 논문(최근 우선) 300편 중 초록이 있는 15편
    pool = papers.sort_values(["matched_count", "year"], ascending=[False, False]).head(300)
    det = _paper_details(svc, pd.concat([pool, recent]))
    det = det.set_index("id")
    with_abs = pool[pool["id"].map(lambda i: bool(str(det["abstract"].get(i, "") or "").strip()))].head(15)
    result = {
        "n_papers": len(papers),
        "journals": journals,
        "recent": recent,
        "abstracts": with_abs,
        "details": det,
        "n_pool_with_abstract": int(pool["id"].map(lambda i: bool(str(det["abstract"].get(i, "") or "").strip())).sum()),
        "n_pool": len(pool),
    }
    if len(_detail_cache) >= 4:   # 최근 후보 몇 개만 기억(메모리 절약)
        _detail_cache.pop(next(iter(_detail_cache)))
    _detail_cache[key] = result
    return result


def _context(svc, cluster: int, compact: bool = False) -> str:
    """AI에게 넘기는 [분석 자료]: 분야 개요 + 트렌드 후보 전체 요약 + 선택한 후보의 단어·논문 제목.
    compact=True(Gemma)면 후보 요약의 AI 해설을 빼고 단어·논문·초록 수를 줄인다."""
    o = _overview(svc)
    lines = [
        f"분야: {svc.d.name} ({svc.d.department})",
        f"수집: {o['year_min']}~{o['year_max']}년, 저널 {o['n_journals']}종, 논문 {o['n_papers']:,}편",
        f"수집 대상 저널: {', '.join(dict.fromkeys(j.name for j in svc.d.journals))}",
        f"Rising 단어(제목 단어 중 연도별 비중이 통계적으로 유의하게 증가한 단어): {o['n_rising']:,}개",
        "트렌드 후보 = Rising 단어들이 논문 제목에서 함께 쓰이는 관계로 묶인 커뮤니티",
        "",
        "[트렌드 후보 전체]",
    ]
    summary = svc.load_full_summary().set_index("cluster")
    for i, item in enumerate(svc.full_cluster_legend()):
        cid = item["cluster"]
        row = summary.loc[cid] if cid in summary.index else None
        terms = ", ".join(svc.full_cluster_terms(cid, top_n=5 if compact else 10)["term"].tolist())
        insight = "" if compact else (ai_insights.full_insight(svc, cid) or {}).get("ko", "")
        trend = ""
        if row is not None:
            trend = (f", 연평균 문서비중 {row['mean_document_share_pct']:.1f}%, "
                     f"최근5년-초기5년 비중 차 {row['recent5_minus_first5_pct_point']:.1f}%p")
        lines.append(f"{_num(i)} {item['label']} (단어 {item['term_count']}개{trend})")
        lines.append(f"   대표 단어: {terms}")
        lines.append(f"   5년 단위 문서 비중: {_share_points(svc, cid)}")
        if insight:
            lines.append(f"   AI 해설: {insight}")
    return "\n".join(lines + _selected_lines(svc, cluster, compact=compact))


def _selected_lines(svc, cluster: int, title: str = "사용자가 선택한 트렌드 후보", compact: bool = False) -> list[str]:
    """한 트렌드 후보의 자세한 자료: 연도별 비중, 추세 지표, 저널, 단어, 논문 제목, 초록.
    compact=True(Gemma)면 단어 20개, 논문 제목 10편, 최근 논문 8편, 초록 4편(300자)으로 줄인다."""
    n_terms, n_papers, n_recent, n_abs, abs_len = (20, 10, 8, 4, 300) if compact else (40, 20, 15, 15, 700)
    summary = svc.load_full_summary().set_index("cluster")
    terms_df = svc.full_cluster_terms(cluster, top_n=n_terms)
    papers = svc.full_cluster_papers(cluster, None, limit=n_papers)
    det = dict(_cluster_details(svc, cluster))
    det["recent"], det["abstracts"] = det["recent"].head(n_recent), det["abstracts"].head(n_abs)
    yearly = _yearly_share(svc, cluster)
    row = summary.loc[cluster] if cluster in summary.index else None
    cited = lambda i: det["details"]["cited_by_count"].get(i)  # noqa: E731
    lines = [
        "",
        f"[{title}: {svc.full_cluster_label(cluster)}]",
        f"이 후보의 단어가 제목에 들어간 논문: {det['n_papers']:,}편",
        "연도별 문서 수와 문서 비중(그 해 전체 논문 중 이 후보 단어가 들어간 논문 비율): "
        + ", ".join(f"{int(r['year'])} {int(r['documents'])}편({r['document_share_pct']:.1f}%)" for _, r in yearly.iterrows()),
    ]
    if row is not None:
        lines.append(f"추세 지표: Mann-Kendall tau {row['mk_tau']:.2f}(p={row['mk_p']:.3g}), "
                     f"Sen 기울기 연 {row['sen_slope_pct_point_per_year']:.2f}%p, "
                     f"최근5년-초기5년 비중 차 {row['recent5_minus_first5_pct_point']:.1f}%p")
    lines += [
        "많이 다룬 저널(논문 수): " + ", ".join(f"{j}({n:,})" for j, n in det["journals"].items()),
        "단어(포함 논문 수): " + ", ".join(f"{t}({n})" for t, n in zip(terms_df["term"], terms_df["document_count"])),
        "대표 논문 제목(핵심 단어를 많이 포함한 순):",
        *[f"- ({r['year']}, {r['journal']}) {r['display_name']}" for _, r in papers.iterrows()],
        "최근 3년 대표 논문 제목(피인용 수):",
        *[f"- ({r['year']}, {r['journal']}, 피인용 {cited(r['id']) if cited(r['id']) is not None else '-'}) {r['display_name']}"
          for _, r in det["recent"].iterrows()],
        f"초록(핵심 단어를 많이 포함한 논문 {det['n_pool']}편 중 초록이 있는 것은 {det['n_pool_with_abstract']}편이며, "
        f"그중 {len(det['abstracts'])}편의 초록을 {abs_len}자까지 제공):",
    ]
    for _, r in det["abstracts"].iterrows():
        abstract = str(det["details"]["abstract"].get(r["id"], "")).strip().replace("\n", " ")
        lines.append(f"- ({r['year']}, {r['journal']}) {r['display_name']}\n  초록: {abstract[:abs_len]}")
    return lines


_BOLD_NUMBER = re.compile(r"\*\*([^*\n]*\d[^*\n]*)\*\*")   # 숫자가 들어간 굵은 글씨
_LONG_DECIMAL = re.compile(r"(?<![\w./-])(\d+\.\d{2,})(?![\w/-]|\.\d)")


def tidy_numbers(text: str) -> str:
    """AI 답변의 숫자 표기 정리: 숫자가 들어간 굵은 글씨는 풀고, 소수는 첫째 자리까지 반올림한다."""
    text = _BOLD_NUMBER.sub(r"\1", text)
    return _LONG_DECIMAL.sub(lambda m: f"{float(m.group(1)):.1f}", text)


def _count_tokens(msgs: list[dict]) -> int:
    """Gemma(vLLM) 토크나이저로 센 대화 전체 토큰 수(채팅 템플릿 포함)."""
    req = urllib.request.Request(GEMMA_BASE_URL.removesuffix("/v1") + "/tokenize",
                                 data=json.dumps({"model": ai_insights.PROVIDERS["gemma"][0], "messages": msgs}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["count"]


def _request(provider: str, system: str, messages: list[dict]) -> dict:
    """chat.completions.create 인자. Gemma는 입력+출력 한도에 맞게 오래된 대화(첫 브리핑부터)를 빼고
    답변 길이(max_tokens)를 남은 만큼으로 정한다. 마지막 질문만 남겨도 넘치면 오류."""
    msgs = [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in messages]
    args = {"model": ai_insights.PROVIDERS[provider][0], "messages": msgs}
    if provider != "gemma":
        return args
    budget = GEMMA_CONTEXT - GEMMA_ANSWER_TOKENS
    n = _count_tokens(msgs)
    while n > budget and len(msgs) > 2:
        del msgs[1]
        n = _count_tokens(msgs)
    if GEMMA_CONTEXT - n < 300:
        raise RuntimeError(f"분석 자료와 질문이 Gemma 처리 한도({GEMMA_CONTEXT:,} 토큰)를 넘습니다({n:,} 토큰). "
                           "질문을 줄이거나 OpenAI로 바꿔 주세요.")
    args["max_tokens"] = min(GEMMA_ANSWER_TOKENS, GEMMA_CONTEXT - n)
    return args


def _stream(provider: str, system: str, messages: list[dict]):
    """답변을 생성되는 대로 조각(str)으로 내보낸다. with 블록이라 화면이 중간에 끊기면 모델 쪽 생성도 멈춘다."""
    with ai_insights._client(provider).chat.completions.create(**_request(provider, system, messages), stream=True) as s:
        for chunk in s:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


def answer(svc, cluster: int, messages: list[dict], provider: str = "openai"):
    """messages: [{"role": "assistant"|"user", "content": str}, ...] (첫 메시지는 브리핑). 답변 조각을 내보낸다."""
    context = _context(svc, cluster, compact=provider == "gemma")
    yield from _stream(provider, SYSTEM_PROMPT.format(field=svc.d.name, context=context), messages)


# ---------------------------------------------------------------- 통합 (6개 분야 모아 보기)
ALL = "__all__"   # 대화 저장소(chat-store)의 domain 값으로 '통합'을 나타낸다

SYSTEM_PROMPT_ALL = SYSTEM_PROMPT.replace(
    "'{field} 분야 R&D 글로벌 트렌드' 분석 도우미입니다.\n아래 [분석 자료]는 {field} 분야 논문을 수집해 실제로 분석한 결과입니다.",
    "'물분야 R&D 글로벌 트렌드' 통합 분석 도우미입니다.\n아래 [분석 자료]는 {field}의 논문을 분야마다 따로 수집·분석한 결과를 "
    "모은 것입니다(분야를 합쳐 다시 분석한 것이 아님). 여러 분야를 비교·종합하는 질문에는 어느 분야의 결과인지 밝혀서 답하세요.",
)
_unique_cache: dict = {}


def integrated_overview(services: list) -> dict:
    """통합 화면 숫자. 여러 분야에 같은 저널이 있어 단순 합계는 중복이 생기므로, 저널은 ISSN으로·논문은
    OpenAlex id로 중복을 뺀 값도 함께 계산한다(dataset 파일이 바뀔 때만 다시 계산)."""
    files = sorted(f for s in services for f in s.d.dataset_dir.glob("*.parquet")) if services else []
    sig = tuple((str(f), f.stat().st_mtime_ns) for f in files)
    if _unique_cache.get("sig") != sig:
        ids, issns = set(), set()
        for f in files:
            issns.add(f.stem.rsplit("_", 2)[-2].upper())
            ids.update(pq.read_table(f, columns=["id"]).column(0).to_pylist())
        _unique_cache.update(sig=sig, n_papers=len(ids), n_journals=len(issns))
    analyzed = [s for s in services if s.has_full_results()]
    return {
        "n_domains": len(services),
        "n_analyzed": len(analyzed),
        "n_journals": _unique_cache["n_journals"],
        "n_papers": _unique_cache["n_papers"],
        "n_rising": sum(s.n_full_rising_terms() for s in analyzed),
        "n_trends": sum(len(s.full_cluster_legend()) for s in analyzed),
        "years": (min((_overview(s)["year_min"] for s in analyzed), default=None),
                  max((_overview(s)["year_max"] for s in analyzed), default=None)),
    }


def briefing_all(services: list) -> str:
    o = integrated_overview(services)
    lines = [
        f"전체 {o['n_domains']}개 분야(연구소)에서 {o['years'][0]}년~{o['years'][1]}년까지 저널 {o['n_journals']}종에서 "
        f"수집한 논문 {o['n_papers']:,}편(분야 간 중복 제외)을 분야마다 따로 분석해, 총 {o['n_rising']:,}개의 "
        f"Rising 단어와 {o['n_trends']}개의 트렌드 후보가 도출되었습니다.",
        "",
        "분야별 R&D 글로벌 트렌드는 다음과 같습니다.",
    ]
    for s in services:
        lines.append("")
        if not s.has_full_results():
            lines.append(f"**{s.d.name}** ({s.d.department}) — 아직 분석 결과가 없습니다.")
            continue
        legend = s.full_cluster_legend()
        lines.append(f"**{s.d.name}** ({s.d.department}, 트렌드 후보 {len(legend)}개)  ")
        lines.append("  \n".join(f"{_num(i)} {item['label']}" for i, item in enumerate(legend)))
    lines += ["", "여러 분야에 공통으로 나타나는 트렌드나 분야 간 비교 등 궁금하신 점이 있으면 추가 질문해 주세요."]
    return "\n".join(lines)


def _context_all(services: list, compact: bool = False) -> str:
    """통합 대화의 [분석 자료]. compact=True(Gemma)면 후보별로 이름·문서비중만 넣는다(대표 단어·5년 단위 비중·
    AI 해설을 넣으면 83개 후보에서 Gemma 한도를 넘는다)."""
    o = integrated_overview(services)
    lines = [f"전체: {o['n_domains']}개 분야, 저널 {o['n_journals']}종, 논문 {o['n_papers']:,}편(분야 간 중복 제외), "
             f"Rising 단어 {o['n_rising']:,}개, 트렌드 후보 {o['n_trends']}개",
             "Rising 단어 = 제목 단어 중 연도별 비중이 통계적으로 유의하게 증가한 단어. "
             "트렌드 후보 = Rising 단어들이 논문 제목에서 함께 쓰이는 관계로 묶인 커뮤니티. 분야마다 따로 분석했다."]
    for s in services:
        ov = _overview(s)
        lines += ["", f"### {s.d.name} 분야 ({s.d.department}): {ov['year_min']}~{ov['year_max']}년, 저널 {ov['n_journals']}종, "
                      f"논문 {ov['n_papers']:,}편, Rising 단어 {ov['n_rising']:,}개"]
        if not s.has_full_results():
            lines.append("분석 결과 없음")
            continue
        summary = s.load_full_summary().set_index("cluster")
        for i, item in enumerate(s.full_cluster_legend()):
            cid = item["cluster"]
            row = summary.loc[cid] if cid in summary.index else None
            trend = (f", 연평균 문서비중 {row['mean_document_share_pct']:.1f}%, 최근5년-초기5년 비중 차 "
                     f"{row['recent5_minus_first5_pct_point']:.1f}%p") if row is not None else ""
            lines.append(f"{_num(i)} {item['label']} (단어 {item['term_count']}개{trend})")
            if compact:
                continue
            insight = ((ai_insights.full_insight(s, cid) or {}).get("ko") or "")[:260]
            lines.append(f"   대표 단어: {', '.join(s.full_cluster_terms(cid, top_n=8)['term'].tolist())}")
            lines.append(f"   5년 단위 문서 비중: {_share_points(s, cid)}")
            if insight:
                lines.append(f"   AI 해설: {insight}")
    return "\n".join(lines)


_PICK_PROMPT = """아래는 물분야 트렌드 분석 대화와 트렌드 후보 목록입니다.
사용자의 마지막 질문이 특정 트렌드 후보 하나 또는 두 개를 가리키면(이름을 줄이거나 "그 안에는"처럼 앞 대화를
이어받은 경우 포함) 그 후보의 번호를 쉼표로 구분해 답하세요(최대 2개). 분야 전체나 여러 분야를 넓게 묻는 질문이면
"없음"이라고만 답하세요. 번호나 "없음" 외에는 아무것도 쓰지 마세요.

[트렌드 후보 목록]
{candidates}

[대화]
{dialog}"""


def _pick_clusters(services: list, messages: list[dict]) -> list[tuple]:
    """통합 대화에서 질문이 가리키는 트렌드 후보(최대 2개)를 AI에게 짧게 물어 고른다 -> [(svc, cluster), ...]."""
    options, lines = {}, []
    for s in services:
        if not s.has_full_results():
            continue
        for item in s.full_cluster_legend():
            key = str(len(options) + 1)
            options[key] = (s, item["cluster"])
            lines.append(f"{key}. [{s.d.name}] {item['label']}")
    dialog = "\n".join(f"{'사용자' if m['role'] == 'user' else 'AI'}: {m['content'][:600]}" for m in messages[-5:])
    try:
        resp = ai_insights._client().chat.completions.create(
            model=ai_insights.MODEL,
            messages=[{"role": "user", "content": _PICK_PROMPT.format(candidates="\n".join(lines), dialog=dialog)}],
        )
        picked = re.findall(r"\d+", resp.choices[0].message.content or "")
    except Exception:   # 고르기에 실패해도 답변은 기본 자료로 계속한다
        return []
    return [options[k] for k in dict.fromkeys(picked) if k in options][:2]


def answer_all(services: list, messages: list[dict], provider: str = "openai"):
    """통합 대화 답변 조각을 내보낸다. Gemma는 한도 때문에 줄인 자료만 보내고, 질문이 가리키는 후보의
    자세한 자료(_pick_clusters)는 붙이지 않는다."""
    context = _context_all(services, compact=provider == "gemma")
    if provider != "gemma":
        for s, cid in _pick_clusters(services, messages):
            context += "\n" + "\n".join(_selected_lines(s, cid, title=f"질문이 가리키는 트렌드 후보({s.d.name} 분야)"))
    system = SYSTEM_PROMPT_ALL.format(field=f"전체 {len(services)}개 분야(연구소)", context=context)
    yield from _stream(provider, system, messages)
