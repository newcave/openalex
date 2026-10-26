# analysis_codes/normalization.py  (도구: n_03에서 사용)
# 용어 정규화 사전(variant -> canonical) 관리 + 정규화가 안 된 후보 그룹 탐색.
#
# 사전 자체는 "사람이 판단해서 채워나가는 것"이 원칙이라 CSV 하나로만 관리한다
# (언제든 손으로도 열어서 고칠 수 있게). 이 모듈이 자동으로 하는 일은 딱 하나,
# "표기가 비슷한 용어들을 후보로 묶어서 보여주는 것"까지다 — 그 후보가 진짜
# 같은 뜻인지, 어떤 표기를 대표형으로 쓸지는 절대 자동으로 정하지 않는다
# (잘못 합치면 서로 다른 개념이 하나로 뭉개지는 게 더 위험하기 때문).

from pathlib import Path

import pandas as pd

from config import DEFAULT

DICT_PATH = DEFAULT.dict_path  # 인자 없이 쓰면 기본 분야(config.DEFAULT_DOMAIN) 사전

# 후보 그룹을 찾을 때만 쓰는 대략적인 규칙. 접미사를 최소한으로만 둔 이유는
# 공격적으로 자르면(-tion, -ment 등) 서로 다른 개념을 잘못 묶을 위험이 커지기 때문
# (예: treatment의 -ment를 자르면 treat와 같은 그룹이 되어버림).
#
# -ing/-ed는 일부러 뺐다. model(명사)/modeling(동명사·과정)/modeled(과거분사·능동-수동)처럼
# 문법적 역할이 달라지면 실제로 다른 뉘앙스를 가리키는 경우가 많다는 게 실제 데이터로
# 확인됐다(flood/flooding/flooded, change/changing/changed 등). 단순 단복수(-s/-es)보다
# 훨씬 위험한 패턴이라 자동 후보에서는 제외하고, 필요하면 사람이 "직접 추가"로 판단해서 넣는다.
_SUFFIXES = ["es", "s"]


def normalization_key(term: str) -> str:
    """하이픈/공백 제거 + 가벼운 접미사 제거로 '후보 묶음 키'를 만든다.
    이 키가 같다고 반드시 같은 뜻은 아니다 — 사람이 후보를 눈으로 보고 정한다."""
    key = term.replace("-", "").replace(" ", "")
    for suf in _SUFFIXES:
        if key.endswith(suf) and len(key) - len(suf) >= 3:
            return key[: -len(suf)]
    return key


def load_dict(path: Path | None = None) -> dict[str, str]:
    path = Path(path) if path else DICT_PATH
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return dict(zip(df["variant"], df["canonical"]))


def save_mapping(variants: list[str], canonical: str) -> None:
    """variants(표기 여러 개) -> canonical(대표형) 매핑을 사전에 추가/갱신한다."""
    DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DICT_PATH.exists():
        existing = pd.read_csv(DICT_PATH)
    else:
        existing = pd.DataFrame(columns=["variant", "canonical"])

    existing = existing[~existing["variant"].isin(variants)]
    new_rows = pd.DataFrame({"variant": variants, "canonical": canonical})
    result = pd.concat([existing, new_rows], ignore_index=True).sort_values("variant")
    result.to_csv(DICT_PATH, index=False)


def apply_dict(terms: pd.Series, mapping: dict[str, str]) -> pd.Series:
    if not mapping:
        return terms
    return terms.map(lambda t: mapping.get(t, t))


def _all_candidate_clusters(
    freq: pd.Series, mapping: dict[str, str], min_group_freq: int
) -> list[dict]:
    """find_candidate_clusters/candidate_progress가 공유하는 내부 로직.
    max_groups로 자르기 전 '전체' 후보 목록을 반환한다 (진행 상황 계산용)."""
    unresolved = freq[~freq.index.isin(mapping.keys())]

    groups: dict[str, list[tuple[str, int]]] = {}
    for term, count in unresolved.items():
        key = normalization_key(term)
        groups.setdefault(key, []).append((term, int(count)))

    clusters = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        total = sum(c for _, c in members)
        if total < min_group_freq:
            continue
        members.sort(key=lambda x: -x[1])
        clusters.append({"key": key, "members": members, "total": total})

    clusters.sort(key=lambda c: -c["total"])
    return clusters


def find_candidate_clusters(
    freq: pd.Series, mapping: dict[str, str], min_group_freq: int = 20, max_groups: int = 30
) -> list[dict]:
    """
    freq: term(index) -> count(value). 순서는 상관없음.
    이미 사전에 등록된 표기(variant)는 처리 완료로 보고 후보에서 제외한다.
    반환: [{"key": str, "members": [(term, count), ...], "total": int}, ...] (total 내림차순, 상위 max_groups개만)
    """
    return _all_candidate_clusters(freq, mapping, min_group_freq)[:max_groups]


def candidate_progress(freq: pd.Series, mapping: dict[str, str], min_group_freq: int = 20) -> dict:
    """
    정규화 작업 진행 상황 요약.
    남은 후보 그룹 수는 계속 커 보일 수 있지만(전체 어휘가 원래 많으니까), 빈도 기준
    커버리지(coverage)를 보면 실제로는 이미 자주 등장하는 용어부터 처리되고 있다는 게
    드러난다 — 빈도순으로 정렬해서 큰 것부터 보여주기 때문에 나머지는 대개 희귀어다.
    """
    all_clusters = _all_candidate_clusters(freq, mapping, min_group_freq)
    total_occurrences = int(freq.sum())
    resolved_occurrences = int(freq[freq.index.isin(mapping.keys())].sum())
    coverage = resolved_occurrences / total_occurrences if total_occurrences else 0.0
    return {
        "remaining_clusters": len(all_clusters),
        "resolved_occurrences": resolved_occurrences,
        "total_occurrences": total_occurrences,
        "coverage": coverage,
    }
