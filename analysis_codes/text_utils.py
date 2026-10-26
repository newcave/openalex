# analysis_codes/text_utils.py  (도구: n_02에서 사용)
# 제목(display_name) 텍스트를 유니그램/바이그램으로 쪼개는 공통 함수.
# 아직 용어 정규화 사전을 적용하기 전 단계 — 여기서 나온 원시 빈도표를 보고
# 정규화 사전(같은 뜻 다른 표기 통합)을 사람이 직접 작성하는 데 쓴다.

import re

# 표준 영어 불용어. 도메인 용어(study/model/using 등)는 일부러 빼지 않았다 —
# 빈도표에 그대로 노출시켜서 정규화 사전 작성 시 사람이 직접 판단하게 하기 위함.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "in", "on",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "this", "that", "these", "those", "it", "its",
    "as", "not", "no", "nor", "so", "than", "too", "very", "s", "t", "just",
    "we", "our", "their", "his", "her", "which", "who", "whom", "what", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "there", "here", "over",
    "under", "again", "further", "once", "new", "based", "using", "via", "vs",
}

_TOKEN_RX = re.compile(r"[a-zA-Z][a-zA-Z\-]*[a-zA-Z]|[a-zA-Z]")

# 제목에 실제로 섞여 들어오는 HTML 유사 태그(<scp>...</scp>, CO<sub>2</sub> 등).
# 태그 구문 자체만 지우고 내부 텍스트는 남긴다 — <scp>Danube River Basin</scp> ->
# "Danube River Basin", CO<sub>2</sub> -> "CO2". (2026-09-07 sub/scp 전수 점검에서
# 확인: reports/rising_cluster_review/markup_audit.md)
_TAG_RX = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")

# 표준 하이픈(U+002D) 외의 하이픈류 문자. 이 문자가 쓰이면 현재 토크나이저가
# 복합어를 분리해버린다(sub‐basin -> sub, basin). 뜻은 같은 하이픈이므로 표준
# 하이픈으로 정규화해 sub-basin처럼 하나의 복합어 토큰으로 유지한다.
# U+2010 hyphen, U+2011 non-breaking hyphen, U+2012 figure dash,
# U+2013 en dash, U+2212 minus sign.
_HYPHEN_RX = re.compile("[‐‑‒–−]")

# 제목 맨 앞에 붙는 저널 특유의 접두어. 실제 연구 내용이 아니라 글 형식이 만든
# 표기라 hess/opinions가 매 문서 동시출현해 별도 군집으로 뭉치는 부작용이 있었다
# (rising_cluster_review 군집 5). 접두어 구문만 제거하고 "opinions" 단어 자체를
# 전역 삭제하지는 않는다 — 본문에 정당하게 등장하는 opinions는 그대로 남는다.
_TITLE_PREFIX_RX = re.compile(r"^\s*hess\s+opinions\s*:\s*", re.IGNORECASE)


def clean_title(title: str) -> str:
    """토큰화 전에 제목을 정제한다: 태그 구문 제거(내용 보존), 하이픈 정규화,
    HESS Opinions 접두어 제거. 순서가 중요하다 — 태그를 먼저 지워야 접두어
    정규식이 <scp>HESS Opinions</scp>: 같은 변형에도 안정적으로 걸린다."""
    if not isinstance(title, str):
        return ""
    t = _TAG_RX.sub("", title)
    t = _HYPHEN_RX.sub("-", t)
    t = _TITLE_PREFIX_RX.sub("", t)
    return t


def raw_tokens(title: str) -> list[str]:
    """제목을 소문자 알파벳 토큰으로 쪼갠다. 하이픈은 살려둔다
    (bio-reactor / bioreactor / waste water 같은 표기 차이를 빈도표에서
    구분해서 보여줘야 정규화가 필요한지 눈으로 판단할 수 있다).
    토큰화 전에 clean_title로 태그/하이픈 표기/제목 접두어를 먼저 정리한다."""
    if not isinstance(title, str):
        return []
    cleaned = clean_title(title)
    return [t.strip("-") for t in _TOKEN_RX.findall(cleaned.lower()) if t.strip("-")]


def unigrams(title: str) -> set[str]:
    """불용어·1글자 토큰을 제외한 유니그램 집합 (문서당 중복 제거)."""
    return {t for t in raw_tokens(title) if t not in STOPWORDS and len(t) > 1}


def bigrams(title: str) -> set[str]:
    """연속된 두 토큰의 구(phrase). 둘 다 실질어(불용어 아님)일 때만 남긴다.
    한쪽이라도 불용어면 제외 — 안 그러면 "analysis of"/"effect of"/"model for" 같은
    일반 문장 패턴이 상위권을 차지해서 진짜 기술 용어(river basin, climate change 등)
    신호가 흐려진다."""
    tokens = raw_tokens(title)
    result = set()
    for a, b in zip(tokens, tokens[1:]):
        if a in STOPWORDS or b in STOPWORDS:
            continue
        if len(a) <= 1 or len(b) <= 1:
            continue
        result.add(f"{a} {b}")
    return result
