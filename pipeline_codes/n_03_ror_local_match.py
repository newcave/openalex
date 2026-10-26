# pipeline_codes/n_03_ror_local_match.py
"""
OpenAlex가 매칭하지 못한 affiliation 문자열을 ROR 데이터 덤프로 매칭한다.

네트워크 호출 없음. 사전 준비:
  1) https://doi.org/10.5281/zenodo.6347574 에서 최신 릴리스 다운로드 (CC0)
  2) zip 안의 v2 스키마 JSON 파일 경로를 DUMP_PATH 로 지정
     (예: v2.x-2026-xx-xx-ror-data.json, 약 12만 레코드)

주의: 이 모듈의 매칭 결과는 OpenAlex 매칭보다 신뢰도가 낮다.
반드시 ror_source 컬럼으로 출처를 구분해 두고,
빈도 상위 결과는 수작업으로 검수한 뒤 확정할 것.
"""
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

from pipeline_codes.utils import logger, collect_metrics

try:
    from rapidfuzz import process as _rf_process
    from rapidfuzz import fuzz as _rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


# ---------------------------------------------------------------- 정규화

_ABBREV = [
    (r"\buniv\b", "university"), (r"\bunivs\b", "university"),
    (r"\binst\b", "institute"), (r"\bnatl\b", "national"),
    (r"\bnat\b", "national"), (r"\bdept\b", "department"),
    (r"\blab\b", "laboratory"), (r"\blabs\b", "laboratory"),
    (r"\bres\b", "research"), (r"\bctr\b", "center"),
    (r"\bcentre\b", "center"), (r"\btech\b", "technology"),
    (r"\bsci\b", "science"), (r"\beng\b", "engineering"),
    (r"\bagr\b", "agriculture"), (r"\bacad\b", "academy"),
    (r"\bsurv\b", "survey"), (r"\bmin\b", "ministry"),
    (r"\bhosp\b", "hospital"), (r"\bcoll\b", "college"),
]

# 주소·학과 등 기관 식별에 방해가 되는 조각
_NOISE_RX = re.compile(
    r"\b(p\.?o\.? box|zip|postal code|e-?mail|corresponding author|"
    r"present address|street|avenue|road|blvd)\b", re.I
)


def normalize(s: str) -> str:
    """대소문자·발음기호·구두점·약어를 정규화한 매칭 키를 만든다."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pat, rep in _ABBREV:
        s = re.sub(pat, rep, s)
    s = re.sub(r"^(the|of|at)\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def segment(aff: str) -> list[str]:
    """
    affiliation 문자열을 매칭 후보 조각으로 분해.
    긴 조각을 먼저 시도하도록 길이 내림차순 정렬해 반환한다.
    """
    if not isinstance(aff, str) or not aff.strip():
        return []
    aff = _NOISE_RX.sub(" ", aff)
    parts = [p.strip() for p in re.split(r"[,;()\[\]/]|\band\b", aff)]
    parts = [p for p in parts if len(p) >= 4]
    cand = list(dict.fromkeys(parts + [aff.strip()]))
    return sorted(cand, key=len, reverse=True)


# ---------------------------------------------------------------- 인덱스

class RorIndex:
    """ROR 덤프를 메모리에 올려 이름 -> ROR ID 색인을 구축한다."""

    def __init__(self, dump_path: str | Path, include_inactive: bool = False):
        self.by_name: dict[str, set[str]] = defaultdict(set)
        self.by_acronym: dict[str, set[str]] = defaultdict(set)
        self.meta: dict[str, dict] = {}
        self._load(Path(dump_path), include_inactive)
        self._keys = list(self.by_name.keys())  # fuzzy 후보군
        logger.info(
            f"ROR 인덱스 구축 완료: 레코드 {len(self.meta)}건, "
            f"이름키 {len(self.by_name)}종, 약어 {len(self.by_acronym)}종"
        )

    def _load(self, path: Path, include_inactive: bool):
        records = json.loads(path.read_text(encoding="utf-8"))
        for rec in records:
            status = (rec.get("status") or "active").lower()
            if not include_inactive and status != "active":
                continue

            rid = (rec.get("id") or "").strip()
            if not rid:
                continue

            names = rec.get("names") or []
            display = next(
                (n.get("value") for n in names
                 if "ror_display" in (n.get("types") or [])),
                None,
            ) or (names[0].get("value") if names else "")

            locs = rec.get("locations") or []
            geo = (locs[0].get("geonames_details") or {}) if locs else {}

            self.meta[rid] = {
                "ror": rid,
                "display_name": display,
                "country_code": geo.get("country_code") or "",
                "types": rec.get("types") or [],
                "status": status,
            }

            for n in names:
                val = n.get("value")
                key = normalize(val)
                if not key:
                    continue
                types = n.get("types") or []
                # 약어는 중의성이 매우 높으므로(IWA, KIST 등) 별도 색인에 둔다
                if "acronym" in types:
                    self.by_acronym[key].add(rid)
                elif len(key) >= 6:
                    self.by_name[key].add(rid)

    # ------------------------------------------------------------ 매칭

    def _disambiguate(self, rids: set[str], country_hint: str | None) -> str | None:
        if len(rids) == 1:
            return next(iter(rids))
        if country_hint:
            narrowed = {r for r in rids
                        if self.meta[r]["country_code"] == country_hint}
            if len(narrowed) == 1:
                return next(iter(narrowed))
        return None  # 모호하면 포기 (오매칭보다 결측이 낫다)

    def match(self, aff: str,
              country_hint: str | None = None,
              fuzzy_cutoff: int = 92) -> dict | None:
        """
        반환: {ror, display_name, method, score} 또는 None
        method: 'exact' | 'acronym' | 'fuzzy'
        """
        cands = segment(aff)
        if not cands:
            return None

        # 1) 정확 매칭 (긴 조각 우선)
        for seg in cands:
            key = normalize(seg)
            if len(key) < 6:
                continue
            hit = self._disambiguate(self.by_name.get(key, set()), country_hint)
            if hit:
                return {"ror": hit,
                        "display_name": self.meta[hit]["display_name"],
                        "method": "exact", "score": 100}

        # 2) 약어 매칭 (국가 힌트가 있을 때만 허용 — 중의성 방지)
        if country_hint:
            for seg in cands:
                key = normalize(seg)
                if not (2 <= len(key) <= 12):
                    continue
                hit = self._disambiguate(self.by_acronym.get(key, set()), country_hint)
                if hit:
                    return {"ror": hit,
                            "display_name": self.meta[hit]["display_name"],
                            "method": "acronym", "score": 90}

        # 3) 유사도 매칭 (임계값을 높게 유지할 것)
        if _HAS_RAPIDFUZZ:
            for seg in cands[:3]:
                key = normalize(seg)
                if len(key) < 8:
                    continue
                best = _rf_process.extractOne(
                    key, self._keys,
                    scorer=_rf_fuzz.token_sort_ratio,
                    score_cutoff=fuzzy_cutoff,
                )
                if best:
                    hit = self._disambiguate(self.by_name[best[0]], country_hint)
                    if hit:
                        return {"ror": hit,
                                "display_name": self.meta[hit]["display_name"],
                                "method": "fuzzy", "score": int(best[1])}
        return None


# ---------------------------------------------------------------- main

def main(df: pd.DataFrame,
         dump_path: str,
         review_out: str | None = "ror_local_match_review.csv",
         fuzzy_cutoff: int = 92) -> tuple[pd.DataFrame, dict]:
    """
    n_02 가 만든 unmatched_affiliations 를 로컬 ROR 덤프로 보강한다.
    고유 문자열 단위로 1회만 매칭하므로 중복 연산이 없다.
    """
    if "unmatched_affiliations" not in df.columns:
        logger.warning("unmatched_affiliations 컬럼 없음 - n_02 를 먼저 실행하십시오")
        return df, {}

    uniq: set[str] = set()
    for lst in df["unmatched_affiliations"]:
        if isinstance(lst, list):
            uniq.update(lst)

    if not uniq:
        logger.info("미매칭 affiliation 없음 - 스킵")
        return df, {}

    logger.info(f"로컬 ROR 매칭 시작: 고유 문자열 {len(uniq)}종")
    index = RorIndex(dump_path)

    resolved: dict[str, dict] = {}
    for s in uniq:
        r = index.match(s, fuzzy_cutoff=fuzzy_cutoff)
        if r:
            resolved[s] = r

    # 검수용 리뷰 파일 — fuzzy/acronym 결과는 반드시 눈으로 확인할 것
    if review_out:
        pd.DataFrame([
            {"raw_affiliation_string": k, **v} for k, v in resolved.items()
        ]).sort_values(["method", "score"]).to_csv(
            review_out, index=False, encoding="utf-8-sig")
        logger.info(f"매칭 결과 검수 파일 -> {review_out}")

    # df 반영 (출처를 구분해 기록)
    def _augment(row):
        rors = list(row["ror"] or [])
        added = []
        for s in (row["unmatched_affiliations"] or []):
            hit = resolved.get(s)
            if hit and hit["ror"] not in rors:
                rors.append(hit["ror"])
                added.append(hit["ror"])
        return pd.Series({
            "ror": sorted(set(rors)),
            "ror_local_added": added,
        })

    aug = df.apply(_augment, axis=1)
    df["ror"] = aug["ror"]
    df["ror_local_added"] = aug["ror_local_added"]
    df["ror_source"] = df.apply(
        lambda r: "openalex+local" if r["ror_local_added"]
        else (r.get("ror_source") or ("local" if r["ror"] else None)),
        axis=1,
    )

    by_method: dict[str, int] = defaultdict(int)
    for v in resolved.values():
        by_method[v["method"]] += 1

    metrics = collect_metrics(
        unmatched_unique=len(uniq),
        local_resolved=len(resolved),
        local_resolve_rate=round(len(resolved) / len(uniq), 4),
        **{f"method_{k}": v for k, v in by_method.items()},
    )
    logger.info(f"로컬 ROR 매칭 완료: {len(resolved)}/{len(uniq)}종 해결 {dict(by_method)}")
    return df, metrics
