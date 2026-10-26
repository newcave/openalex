# webapp/backend/monitor.py
# "수집현황 모니터링" 화면용 집계. 원본은 pipeline_codes/event_log.py의 구조화 기록(logs/events/*.jsonl)뿐이다.
#
# 화면이 15초마다 갱신되므로 매번 기록 전체를 다시 읽지 않는다.
#   - signature(): 기록 파일들의 (이름, 크기, 수정시각). 이게 같으면 내용도 같다.
#   - 기록이 바뀌었을 때만 전체를 한 번 읽어 ①일자별 집계(rollup) ②콘솔 줄 목록을 만들어 두고,
#     분야/기간/실패만 필터와 페이지 이동은 그 결과를 잘라 쓴다.

from __future__ import annotations

import unicodedata
from datetime import datetime, timedelta

import pandas as pd

from pipeline_codes import event_log
from config import DOMAINS, LEGACY_DOMAIN_NAMES

LOG_PAGE = 300  # 로그 창 한 번에 보여주는 줄 수


def domain_name(key: str | None) -> str:
    if key in DOMAINS:
        return DOMAINS[key].name
    return LEGACY_DOMAIN_NAMES.get(key, key or "-")


def signature() -> tuple:
    if not event_log.EVENT_DIR.exists():
        return ()
    out = []
    for p in sorted(event_log.EVENT_DIR.glob("*.jsonl")):
        st = p.stat()
        out.append((p.name, st.st_size, st.st_mtime_ns))
    return tuple(out)


# ---------------------------------------------------------------- 기록이 바뀔 때만 다시 계산하는 캐시
_cache: dict = {"sig": None}


def _store() -> dict:
    sig = signature()
    if _cache["sig"] != sig:
        df = _load(sig)
        _cache.clear()
        _cache.update(sig=sig, df=df, rollup=_rollup(df), last_ok=_last_ok(df), lines={})
    return _cache


def _load(_sig) -> pd.DataFrame:
    rows = [r for r in event_log.read_all() if r.get("event") != "backfill_meta"]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("status", "domain", "reason", "journal", "message", "kind", "issns", "reasons", "year", "n", "years"):
        if col not in df.columns:
            df[col] = None
    df["ts"] = pd.to_datetime(df["ts"].str[:19])
    df["date"] = df["ts"].dt.normalize()
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    return df.sort_values("ts").reset_index(drop=True)


def _rollup(df: pd.DataFrame) -> pd.DataFrame:
    """일자별 집계: (date, domain, metric, reason, value)
    metric = papers(내려받은 논문) | years(수집된 저널×연도) | fail(실패 요청: 연도·저널조회 1건, DOI는 논문 수만큼)"""
    cols = ["date", "domain", "metric", "reason", "value"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    parts = []
    ok = df[df["event"].eq("collect") & df["status"].eq("ok")]
    if not ok.empty:
        g = ok.groupby(["date", "domain"], dropna=False)["n"].agg(["sum", "size"]).reset_index()
        parts.append(g.assign(metric="papers", reason=None, value=g["sum"])[cols])
        parts.append(g.assign(metric="years", reason=None, value=g["size"])[cols])
    rows = []
    for r in df[df["status"].eq("fail")].itertuples():
        if r.event == "doi" and isinstance(r.reasons, dict):
            rows += [(r.date, r.domain, "fail", k, int(v)) for k, v in r.reasons.items()]
        elif r.event in ("collect", "lookup"):
            rows.append((r.date, r.domain, "fail", r.reason or "other", 1))
    if rows:
        f = pd.DataFrame(rows, columns=cols)
        parts.append(f.groupby(["date", "domain", "metric", "reason"], dropna=False, as_index=False)["value"].sum())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def _last_ok(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    ok = df[df["event"].eq("collect") & df["status"].eq("ok")]
    return ok.groupby("domain")["ts"].max().to_dict()


def _since(days: int | None):
    return pd.Timestamp(datetime.now() - timedelta(days=days)).normalize() if days else None


# ---------------------------------------------------------------- 화면에서 부르는 함수
def chart_data(domain: str | None, days: int | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(날짜x분야 논문 수, 날짜x사유 실패 수, 요약) - 미리 만들어 둔 일자별 집계에서 잘라낸다."""
    st = _store()
    r = st["rollup"]
    if domain:
        r = r[r["domain"].eq(domain)]
    if (since := _since(days)) is not None:
        r = r[r["date"] >= since]
    papers = r[r["metric"].eq("papers")].rename(columns={"value": "n"})[["date", "domain", "n"]]
    fails = (r[r["metric"].eq("fail")].groupby(["date", "reason"], as_index=False)["value"].sum()
             .rename(columns={"value": "count"}))
    last = [v for k, v in st["last_ok"].items() if not domain or k == domain]
    summary = {
        "papers": int(papers["n"].sum()),
        "years": int(r.loc[r["metric"].eq("years"), "value"].sum()),
        "fails": int(fails["count"].sum()),
        "last_ok": max(last) if last else None,
    }
    return papers, fails, summary


def log_page(domain: str | None, days: int | None, fails_only: bool, offset: int = 0) -> tuple[list[tuple[str, str]], int]:
    """(현재 페이지 줄 [(level, text)], 전체 줄 수). 최신순, LOG_PAGE줄씩."""
    st = _store()
    key = (domain, days, fails_only)
    if key not in st["lines"]:
        df = st["df"]
        if not df.empty:
            if domain:
                df = df[df["domain"].eq(domain)]
            if (since := _since(days)) is not None:
                df = df[df["date"] >= since]
        st["lines"][key] = [(r["level"], _format(r)) for r in _log_rows(df, fails_only)]
    lines = st["lines"][key]
    return lines[offset:offset + LOG_PAGE], len(lines)


# ---------------------------------------------------------------- 콘솔 줄 만들기
def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _cell(s: str, width: int) -> str:
    """한글은 2칸 폭으로 계산해 고정폭 칸을 맞춘다(넘치면 … 처리)."""
    s = s or ""
    if _width(s) > width:
        out, w = "", 0
        for c in s:
            cw = _width(c)
            if w + cw > width - 1:
                break
            out, w = out + c, w + cw
        s = out + "…"
    return s + " " * (width - _width(s))


def _format(r: dict) -> str:
    return "  ".join([
        r["ts"].strftime("%Y-%m-%d %H:%M:%S"),
        _cell(domain_name(r["domain"]), 8),
        _cell(r["tag"], 8),
        _cell(r["target"], 44),
        _cell(r["year"], 9),
        r["text"],
    ])


def _issn_text(issns) -> str:
    return ", ".join(issns) if isinstance(issns, list) and issns else ""


def _s(v) -> str:
    return v if isinstance(v, str) else ""


def _log_rows(df: pd.DataFrame, fails_only: bool) -> list[dict]:
    """콘솔 표시용 행(최신순). DOI 검색 실패는 줄이 너무 많아(논문 1건당 1줄) 같은 분 같은 저널끼리 묶는다."""
    if df.empty:
        return []
    data = df[df["status"].eq("fail")] if fails_only else df
    rows = []
    doi = data[data["event"].eq("doi")].copy()
    if not doi.empty:
        doi["minute"] = doi["ts"].dt.floor("min")
        for (minute, dom, jour), g in doi.groupby(["minute", "domain", doi["journal"].fillna("")], dropna=False):
            reasons: dict = {}
            for rs in g["reasons"]:
                for k, v in (rs or {}).items():
                    reasons[k] = reasons.get(k, 0) + int(v)
            years = sorted({int(y) for y in g["year"].dropna()})
            rows.append({
                "ts": g["ts"].max(), "domain": dom, "level": "fail", "tag": "DOI",
                "target": jour or _issn_text(g["issns"].iloc[0]),
                "year": f"{years[0]}–{years[-1]}" if len(years) > 1 else (str(years[0]) if years else ""),
                "text": f"검색 실패 {int(g['n'].sum())}건 (" + ", ".join(
                    f"{event_log.REASONS.get(k, k)} {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])) + ")",
            })
    for r in data[~data["event"].eq("doi")].itertuples():
        target = _s(r.journal)
        issn = _issn_text(r.issns)
        if issn:
            target = f"{target} ({issn})" if target else f"ISSN {issn}"
        if r.event == "collect" and r.status == "ok":
            row = {"level": "ok", "tag": "OK", "text": f"{r.n:,}건"}
        elif r.event == "collect" and r.status == "empty":
            row = {"level": "dim", "tag": "EMPTY", "text": "0건"}
        elif r.event == "analysis":
            kind = "전체기간 분석" if r.kind == "full" else "구간별분석"
            if r.status == "ok":
                extra = getattr(r, "n_rising", None)
                clusters = getattr(r, "clusters", None)
                detail = " · ".join(x for x in (
                    f"Rising {int(extra)}" if pd.notna(extra) else "",
                    f"커뮤니티 {int(clusters)}" if pd.notna(clusters) else "") if x)
                row = {"level": "info", "tag": "ANALYSIS", "text": f"{kind} 완료" + (f" ({detail})" if detail else "")}
            else:
                row = {"level": "fail", "tag": "ANALYSIS", "text": f"{kind} 실패 — {r.message}"}
            target = ""
        else:  # fail
            reason = event_log.REASONS.get(r.reason, r.reason or "기타")
            what = "저널 조회" if r.event == "lookup" else "수집"
            years = r.years if isinstance(r.years, list) else []
            span = f" · 미수집 {years[0]}–{years[-1]}" if len(years) > 1 else ""
            row = {"level": "fail", "tag": "FAIL", "text": f"{what} 실패 — {reason}{span}"}
        rows.append({"ts": r.ts, "domain": r.domain, "target": target,
                     "year": str(int(r.year)) if pd.notna(r.year) else "", **row})
    rows.sort(key=lambda x: x["ts"], reverse=True)
    return rows
