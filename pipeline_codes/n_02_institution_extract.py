# pipeline_codes/n_02_institution_extract.py
"""
OpenAlex authorships -> 기관 정보 추출.

핵심: 네트워크 호출이 전혀 없다.
OpenAlex는 raw affiliation 문자열을 Institution 엔티티에 이미 매칭하여
ROR ID와 정규화된 display_name, country_code, type을 함께 제공한다.
따라서 이 단계는 순수 파싱이며, ROR API를 호출할 이유가 없다.

기존 n_02_ror_extract.py + n_03_ror_enrich.py 를 대체한다.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

from pipeline_codes.utils import logger, collect_metrics


# ---------------------------------------------------------------- helpers

def _as_list(cell) -> list:
    """authorships 셀(list, numpy.ndarray, 또는 JSON 문자열)을 list로 정규화.

    parquet에서 읽은 list 컬럼은 pandas가 numpy.ndarray로 반환하므로
    list만 검사하면 항상 빈 결과가 나온다.
    """
    if isinstance(cell, list):
        return cell
    if isinstance(cell, np.ndarray):
        return cell.tolist()
    if isinstance(cell, str) and cell.strip():
        try:
            parsed = json.loads(cell)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _norm_ror(v) -> str:
    """ROR 값을 https://ror.org/<id> 형태로 정규화. 잘못된 값은 빈 문자열."""
    if not isinstance(v, str):
        return ""
    v = v.strip().rstrip("/")
    if not v:
        return ""
    rid = v.rsplit("/", 1)[-1].strip().lower()
    # ROR ID는 0으로 시작하는 9자 crockford base32 (마지막 2자는 체크섬)
    if len(rid) != 9 or not rid.startswith("0"):
        return ""
    return f"https://ror.org/{rid}"


# ---------------------------------------------------------------- parsing

def parse_authorships(cell) -> dict:
    """
    authorships 하나를 파싱해 아래 4가지를 반환한다.

      institutions            : [{ror, display_name, country_code, type}, ...] 중복 제거
      author_institutions     : 저자-기관 연결 (공저 네트워크 분석용)
      unmatched_affiliations  : OpenAlex가 기관에 매칭하지 못한 원문 문자열
      n_affiliations          : 전체 affiliation 인스턴스 수 (커버리지 계산용)
    """
    auths = _as_list(cell)

    inst_map: dict[str, dict] = {}
    author_links: list[dict] = []
    unmatched: list[str] = []
    n_aff = 0

    for a in auths:
        if not isinstance(a, dict):
            continue

        rors_here = []
        for inst in _as_list(a.get("institutions")):
            if not isinstance(inst, dict):
                continue
            ror = _norm_ror(inst.get("ror"))
            if not ror:
                continue
            rors_here.append(ror)
            if ror not in inst_map:
                inst_map[ror] = {
                    "ror": ror,
                    # OpenAlex가 이미 정규화해 둔 기관명. ROR API 조회 불필요.
                    "display_name": inst.get("display_name") or "",
                    "country_code": inst.get("country_code") or "",
                    "type": inst.get("type") or "",
                }

        author = a.get("author") or {}
        author_links.append({
            "author_id": author.get("id") or "",
            "author_name": author.get("display_name") or "",
            "position": a.get("author_position") or "",
            "is_corresponding": bool(a.get("is_corresponding")),
            "rors": sorted(set(rors_here)),
        })

        # affiliation 단위로 매칭 실패분을 정확히 식별한다.
        # (행 단위로 "ROR 0개"를 보는 것보다 훨씬 정밀한 타겟팅)
        affs = _as_list(a.get("affiliations"))
        for aff in affs:
            if not isinstance(aff, dict):
                continue
            raw = (aff.get("raw_affiliation_string") or "").strip()
            if not raw:
                continue
            n_aff += 1
            if not _as_list(aff.get("institution_ids")):
                unmatched.append(raw)

        # affiliations 블록이 없는 구형 레코드 보완
        if not affs:
            for raw in _as_list(a.get("raw_affiliation_strings")):
                raw = (raw or "").strip()
                if not raw or "orcid" in raw.lower():
                    continue
                n_aff += 1
                if not rors_here:
                    unmatched.append(raw)

    return {
        "institutions": sorted(inst_map.values(), key=lambda d: d["ror"]),
        "author_institutions": author_links,
        "unmatched_affiliations": sorted(set(unmatched)),
        "n_affiliations": n_aff,
    }


# ---------------------------------------------------------------- main

def main(df: pd.DataFrame,
         unmatched_out: str | None = "unmatched_affiliations.csv"
         ) -> tuple[pd.DataFrame, dict]:
    """
    df에 아래 컬럼을 추가한다.

      institutions            : list[dict]  (ror/display_name/country_code/type)
      ror                     : list[str]   (기존 파이프라인 호환용)
      org_names               : list[str]   (기존 파이프라인 호환용)
      author_institutions     : list[dict]
      unmatched_affiliations  : list[str]
      ror_source              : 'openalex'  (출처 추적용. 이후 단계에서 갱신)

    unmatched_out 경로에는 미매칭 문자열의 빈도표를 저장한다.
    다음 단계(로컬 ROR 덤프 매칭)의 입력이자, 수작업 검수 우선순위 목록이 된다.
    """
    if "authorships" not in df.columns:
        logger.warning("authorships 컬럼 없음")
        for col in ("institutions", "ror", "org_names",
                    "author_institutions", "unmatched_affiliations"):
            df[col] = [[] for _ in range(len(df))]
        df["ror_source"] = None
        return df, {}

    parsed = df["authorships"].apply(parse_authorships)

    df["institutions"] = parsed.apply(lambda d: d["institutions"])
    df["ror"] = parsed.apply(lambda d: [i["ror"] for i in d["institutions"]])
    df["org_names"] = parsed.apply(
        lambda d: [i["display_name"] for i in d["institutions"] if i["display_name"]] or None
    )
    df["author_institutions"] = parsed.apply(lambda d: d["author_institutions"])
    df["unmatched_affiliations"] = parsed.apply(lambda d: d["unmatched_affiliations"])
    df["ror_source"] = df["ror"].apply(lambda x: "openalex" if x else None)

    # --- 미매칭 문자열 빈도표 -------------------------------------------
    counter: Counter = Counter()
    for lst in df["unmatched_affiliations"]:
        counter.update(lst)

    if unmatched_out and counter:
        pd.DataFrame(
            sorted(counter.items(), key=lambda kv: -kv[1]),
            columns=["raw_affiliation_string", "freq"],
        ).to_csv(unmatched_out, index=False, encoding="utf-8-sig")
        logger.info(f"미매칭 affiliation {len(counter)}종 -> {unmatched_out}")

    # --- 메트릭 -----------------------------------------------------------
    total_aff = int(parsed.apply(lambda d: d["n_affiliations"]).sum())
    unmatched_inst = int(sum(counter.values()))
    rows_with_ror = int(df["ror"].apply(bool).sum())

    metrics = collect_metrics(
        rows_total=len(df),
        rows_with_ror=rows_with_ror,
        rows_without_ror=len(df) - rows_with_ror,
        affiliation_instances=total_aff,
        affiliation_unmatched=unmatched_inst,
        affiliation_match_rate=(
            round(1 - unmatched_inst / total_aff, 4) if total_aff else None
        ),
        unmatched_unique_strings=len(counter),
    )

    logger.info(
        f"기관 파싱 완료: {rows_with_ror}/{len(df)}건에 ROR 존재 | "
        f"affiliation 매칭률 {metrics.get('affiliation_match_rate')} | "
        f"미매칭 고유 문자열 {len(counter)}종"
    )
    return df, metrics


# ---------------------------------------------------------------- 연도별 커버리지

def coverage_by_year(df: pd.DataFrame, year_col: str = "publication_year") -> pd.DataFrame:
    """
    연도별 기관정보 보유율. 트렌드 분석 전 반드시 확인할 것.
    2000년대 초반은 원본 메타데이터에 affiliation 자체가 없는 경우가 많아,
    이를 보정하지 않으면 '기관 수 증가' 같은 인공적 추세가 만들어진다.
    """
    if year_col not in df.columns:
        raise KeyError(f"{year_col} 컬럼 없음")

    g = df.groupby(year_col).agg(
        n_works=("ror", "size"),
        n_with_ror=("ror", lambda s: int(s.apply(lambda x: len(x) > 0).sum())),
    )
    g["ror_coverage"] = (g["n_with_ror"] / g["n_works"]).round(4)
    return g.reset_index()
