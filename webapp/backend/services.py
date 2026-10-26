# webapp/backend/services.py
# app.py가 쓰는 데이터 접근 계층: results/<분야>/의 산출물(parquet/csv/json)을 읽어 화면용으로 가공하고,
# "수집 시작"/"분석 진행하기" 버튼이 눌리면 pipeline_codes/pipeline.py, analysis_codes/pipeline.py를 호출한다.
# 여기서는 새 분석 로직을 만들지 않는다.
#
# 분야(수자원/상하수도 ...)마다 DomainService 인스턴스가 하나씩 있고, 각자 config.py에
# 정의된 자기 폴더만 읽고 쓴다. 앱에서는 svc.get(domain_key)로 꺼내 쓴다.

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_codes import pipeline as collection_pipeline          # noqa: E402
from analysis_codes import pipeline as analysis_pipeline              # noqa: E402
from analysis_codes import n_06_windowed_analysis as windowed_analysis   # noqa: E402
from config import DATA_ROOT, DOMAINS, DEFAULT_DOMAIN, OPENALEX_EMAIL, Domain  # noqa: E402
from pipeline_codes import n_01_openalex as openalex                  # noqa: E402

# .env를 여기서 먼저 읽는다. app.py가 이 모듈을 가장 먼저 import하므로, 여기서 안 읽으면
# .env의 OPENALEX_API_KEY가 수집에 쓰이지 않는다(ai_insights.py가 .env를 읽는 건 그보다 뒤).
load_dotenv(ROOT / ".env")

EMAIL = OPENALEX_EMAIL
# .env의 OpenAlex 키 목록(OPENALEX_API_KEY, OPENALEX_API_KEY_2, ...). 앞 키 한도가 차면 다음 키로 넘어간다.
API_KEYS = openalex.load_api_keys()
API_KEY = API_KEYS[0] if API_KEYS else ""

# 클러스터/구간 색상 (Okabe-Ito 색맹안전 팔레트, 부족하면 순환)
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
           "#D55E00", "#CC79A7", "#999999", "#000000"]

STATE_DIR = ROOT / "webapp" / ".state"   # 분야별 마지막 수집 기록(collect_job_<분야>.json)

_CACHE_FILENAME_RX = re.compile(r"^(?P<journal>.+)_(?P<issn>\d{4}-[\dXx]{3}[\dXx])_(?P<year>\d{4})\.parquet$")


def color_for(idx: int) -> str:
    return PALETTE[int(idx) % len(PALETTE)]


def _node_size(value: float, max_value: float, min_px: float = 52.0, max_px: float = 118.0) -> float:
    """면적(반지름^2)이 값에 비례하도록 제곱근 스케일링 — 선형 스케일은 작은 값의 원이
    실제 비율보다 훨씬 작아 보이는 착시를 만든다. 최소 크기를 확보해 작은 군집도 라벨이 들어갈 공간을 준다."""
    if max_value <= 0:
        return min_px
    ratio = max(0.0, min(1.0, value / max_value))
    return float(min_px + (max_px - min_px) * (ratio ** 0.5))


def _keep_strong_edges(pairs: list[tuple[int, int, float]], keep_ratio: float = 0.55) -> list[tuple[int, int, float]]:
    """군집 간 엣지가 사실상 전부-대-전부로 이어져 거미줄처럼 보이는 것을 막기 위해,
    가중치 상위 keep_ratio만 남긴다(최소 1개는 남김)."""
    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    n_keep = max(1, int(round(len(pairs) * keep_ratio)))
    return pairs[:n_keep]


def _files_sig(*paths: Path) -> tuple:
    """캐시 무효화용 서명: 파일별 수정시각(없으면 None).
    "분석 진행하기"/"수집 시작"은 백그라운드 프로세스에서 돌기 때문에, 그쪽에서 캐시를 비워도
    대시보드 본 프로세스의 캐시는 그대로 남는다. 그래서 키 대신 파일 수정시각으로 새 결과를 감지한다."""
    out = []
    for p in paths:
        try:
            out.append(p.stat().st_mtime_ns)
        except OSError:
            out.append(None)
    return tuple(out)


def _dir_sig(directory: Path, pattern: str = "*.parquet") -> tuple:
    files = list(directory.glob(pattern)) if directory.exists() else []
    if not files:
        return (0, None)
    return (len(files), max(f.stat().st_mtime_ns for f in files))


class DomainService:
    def __init__(self, domain: Domain):
        self.d = domain
        self._cache: dict = {}

    # ------------------------------------------------------------ 공용 캐시
    def _cached(self, key, loader, sig=None):
        hit = self._cache.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        value = loader()
        self._cache[key] = (sig, value)
        return value

    def invalidate_cache(self, prefix: str | None = None):
        if prefix is None:
            self._cache.clear()
            return
        for k in [k for k in self._cache if str(k).startswith(prefix)]:
            del self._cache[k]

    # ------------------------------------------------------------ 결과 존재 여부
    def has_full_results(self) -> bool:
        return (self.d.full_review / "terms.csv").exists() and \
            (self.d.full_output / "cluster_trends_summary_fdr_assoc.csv").exists()

    # ------------------------------------------------------------ 구간 목록 (연도 범위에 따라 동적으로)
    def windows(self) -> list[str]:
        """구간별분석(4번 탭)이 실제로 쓰는 구간 id 목록("2000_2005" 형태).

        하드코딩 대신 windows_summary.csv(직전 "분석 진행하기"가 compute_windows()로 실제 연도
        범위를 보고 계산해 남긴 결과)를 그대로 읽는다. 아직 한 번도 실행 전이면(파일 없음)
        기본 범위(2000-2025)를 같은 함수에 넣어 나온 구간으로 대체한다."""
        path = self.d.windowed_output / "windows_summary.csv"

        def _load():
            if path.exists():
                df = pd.read_csv(path)
                if not df.empty:
                    return sorted(df["window"].tolist(), key=lambda w: int(w.split("_")[0]))
            return [f"{s}_{e}" for s, e in windowed_analysis.compute_windows(2000, 2025)]
        return self._cached("window_ids", _load, _files_sig(path))

    def window_labels(self) -> dict[str, str]:
        return {w: w.replace("_", "–", 1) for w in self.windows()}

    def full_year_range(self) -> tuple[int, int]:
        """전체기간 분석이 실제로 다룬 연도 범위(연도별 추이 파일 기준). 실행 전이면 구간 목록 기준."""
        path = self.d.full_output / "cluster_trends_fdr_assoc.parquet"

        def _load():
            if path.exists():
                years = pd.read_parquet(path, columns=["year"])["year"]
                if not years.empty:
                    return int(years.min()), int(years.max())
            ws = self.windows()
            return int(ws[0].split("_")[0]), int(ws[-1].split("_")[1])
        return self._cached("full_year_range", _load, _files_sig(path, self.d.windowed_output / "windows_summary.csv"))

    # ------------------------------------------------------------ 원본 롱포맷(코퍼스+용어) 로더 — 논문 검색용
    def get_long_and_corpus(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(long: id/year/term 정규화 완료, corpus: id/year/journal/display_name) — 필요할 때 1회만 로드."""
        src = self.d.shared_source

        def _load():
            corpus = pd.read_parquet(
                src / "filtered_corpus.parquet",
                columns=["id", "year", "language", "journal", "display_name"],
            )
            corpus = corpus[corpus["language"].eq("en")].drop_duplicates("id")
            long = pd.read_parquet(src / "unigram_long.parquet", columns=["id", "year", "term"])
            mapping = pd.read_csv(src / "term_normalization_dict.csv")
            term_map = dict(zip(mapping["variant"], mapping["canonical"]))
            long = long[long["id"].isin(corpus["id"])].copy()
            long["term"] = long["term"].map(lambda t: term_map.get(t, t))
            long = long.drop_duplicates(["id", "term"])
            return long, corpus
        sig = _files_sig(src / "filtered_corpus.parquet", src / "unigram_long.parquet", src / "term_normalization_dict.csv")
        return self._cached("long_corpus", _load, sig)

    def papers_for_terms(self, terms: list[str], year_start: int | None = None, year_end: int | None = None,
                          limit: int = 300) -> pd.DataFrame:
        """주어진 용어 중 하나라도 제목에 포함된 논문 메타데이터. (연도범위는 선택)"""
        empty = pd.DataFrame(columns=["id", "year", "journal", "display_name", "matched_terms", "url"])
        if not terms:
            return empty
        long, corpus = self.get_long_and_corpus()
        term_set = set(terms)
        hits = long[long["term"].isin(term_set)]
        if year_start is not None:
            hits = hits[hits["year"].between(year_start, year_end)]
        if hits.empty:
            return empty
        grouped = hits.groupby("id")["term"].agg(lambda s: sorted(set(s))).reset_index(name="matched_terms")
        grouped["matched_count"] = grouped["matched_terms"].str.len()
        out = grouped.merge(corpus[["id", "year", "journal", "display_name"]], on="id", how="left")
        out["matched_terms"] = out["matched_terms"].map(lambda s: " | ".join(s))
        out["url"] = out["id"]
        out = out.sort_values(["matched_count", "year"], ascending=[False, False]).head(limit)
        return out.reset_index(drop=True)

    def co_occurrence_papers(self, terms: list[str], limit: int = 50) -> pd.DataFrame:
        """papers_for_terms는 용어 중 하나라도(OR) 들어간 논문을 찾는다. 이 함수는 주어진 용어가
        전부(AND) 같은 제목에 등장한 논문만 찾는다 - 5번 탭 키워드 조합 인사이트에서 "같이 쓰인 적이
        있는지"를 판단하는 데 쓴다."""
        empty = pd.DataFrame(columns=["id", "year", "journal", "display_name", "matched_terms", "url"])
        if not terms or len(terms) < 2:
            return empty
        long, corpus = self.get_long_and_corpus()
        term_set = set(terms)
        hits = long[long["term"].isin(term_set)]
        if hits.empty:
            return empty
        grouped = hits.groupby("id")["term"].agg(set).reset_index(name="matched_terms")
        grouped = grouped[grouped["matched_terms"].map(lambda s: term_set.issubset(s))]
        if grouped.empty:
            return empty
        out = grouped.merge(corpus[["id", "year", "journal", "display_name"]], on="id", how="left")
        out["matched_terms"] = out["matched_terms"].map(lambda s: " | ".join(sorted(s)))
        out["url"] = out["id"]
        out = out.sort_values("year", ascending=False).head(limit)
        return out.reset_index(drop=True)

    def full_cluster_edge_weight(self, cluster_a: int, cluster_b: int) -> float:
        """전체분석 네트워크에서 두 커뮤니티가 직접 연결돼 있다면 그 가중치(공유 문서수 기반),
        연결이 없으면 0.0. 같은 커뮤니티면 호출하는 쪽에서 따로 처리한다."""
        edges = self.load_full_edges()
        inter = edges[edges["cluster_a"].ne(edges["cluster_b"])]
        match = inter[
            ((inter["cluster_a"] == cluster_a) & (inter["cluster_b"] == cluster_b))
            | ((inter["cluster_a"] == cluster_b) & (inter["cluster_b"] == cluster_a))
        ]
        if match.empty:
            return 0.0
        return float(match["documents"].sum())

    # ------------------------------------------------------------ 전체분석 데이터 접근
    def load_full_terms(self) -> pd.DataFrame:
        path = self.d.full_review / "terms.csv"
        return self._cached("full_terms", lambda: pd.read_csv(path), _files_sig(path))

    def load_full_edges(self) -> pd.DataFrame:
        path = self.d.full_review / "edges.csv"
        return self._cached("full_edges", lambda: pd.read_csv(path), _files_sig(path))

    def n_full_rising_terms(self) -> int:
        return len(self.load_full_terms()) if self.has_full_results() else 0

    @property
    def ai_labels_path(self) -> Path:
        return self.d.full_review / "ai_labels.json"

    def load_ai_labels(self) -> dict:
        """"분석 진행하기" 마지막 단계에서 OpenAI로 생성해 저장한 커뮤니티 라벨+해설."""
        path = self.ai_labels_path

        def _load():
            if not path.exists():
                return {"clusters": []}
            return json.loads(path.read_text(encoding="utf-8"))
        return self._cached("full_ai_labels", _load, _files_sig(path))

    def full_cluster_label(self, cluster: int) -> str:
        for c in self.load_ai_labels().get("clusters", []):
            if int(c["cluster"]) == int(cluster):
                return c.get("label", f"군집 {cluster}")
        # 라벨 생성 전/실패 시: 대표 용어를 이어붙여 대신 보여준다.
        terms = self.full_cluster_terms(cluster, top_n=4)["term"].tolist()
        return ", ".join(terms) if terms else f"군집 {cluster}"

    def load_full_summary(self) -> pd.DataFrame:
        path = self.d.full_output / "cluster_trends_summary_fdr_assoc.csv"

        def _load():
            df = pd.read_csv(path)
            df["label"] = df["cluster"].map(self.full_cluster_label)
            return df
        sig = _files_sig(path, self.ai_labels_path, self.d.full_review / "terms.csv")
        return self._cached("full_summary", _load, sig)

    def full_cluster_graph_elements(self) -> list[dict]:
        """클러스터(군집) 수준 cytoscape 노드/엣지. 실제 용어 전체가 아니라 군집만 보여준다."""
        summary = self.load_full_summary()
        edges = self.load_full_edges()
        inter = edges[edges["cluster_a"].ne(edges["cluster_b"])]
        inter_weight = (
            inter.rename(columns={"cluster_a": "src", "cluster_b": "dst"})[["src", "dst", "documents"]]
            .groupby(["src", "dst"])["documents"].sum().reset_index()
        )
        elements = []
        max_terms = summary["term_count"].max()
        for _, row in summary.iterrows():
            cid = int(row["cluster"])
            elements.append({
                "data": {
                    "id": f"c{cid}", "label": row["label"],
                    "cluster": cid, "term_count": int(row["term_count"]),
                    "mean_share": round(float(row["mean_document_share_pct"]), 2),
                    "color": color_for(cid), "size": _node_size(row["term_count"], max_terms),
                }
            })
        seen = set()
        pairs = []
        for _, row in inter_weight.iterrows():
            a, b = int(row["src"]), int(row["dst"])
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((key[0], key[1], float(row["documents"])))
        kept = _keep_strong_edges(pairs)
        max_w = max((w for _, _, w in kept), default=1.0) or 1.0
        for a, b, w in kept:
            elements.append({"data": {"source": f"c{a}", "target": f"c{b}", "weight": w / max_w}})
        return elements

    def full_cluster_legend(self) -> list[dict]:
        """그래프 아래에 보여줄 범례: 색상 + 번호 + 짧은 라벨 + 용어수."""
        summary = self.load_full_summary()
        return [
            {"cluster": int(r["cluster"]), "color": color_for(int(r["cluster"])),
             "label": r["label"], "term_count": int(r["term_count"])}
            for _, r in summary.sort_values("cluster").iterrows()
        ]

    def full_cluster_terms(self, cluster: int, top_n: int = 60) -> pd.DataFrame:
        terms = self.load_full_terms()
        sub = terms[terms["cluster"].eq(int(cluster))].sort_values("document_count", ascending=False)
        return sub.head(top_n).reset_index(drop=True)

    def full_cluster_papers(self, cluster: int, terms: list[str] | None, limit: int = 200) -> pd.DataFrame:
        path = self.d.full_review / "all_papers.csv"
        df = self._cached("full_all_papers", lambda: pd.read_csv(path), _files_sig(path))
        sub = df[df["cluster"].eq(int(cluster))]
        if terms:
            term_set = set(terms)
            sub = sub[sub["matched_terms"].map(lambda s: bool(set(str(s).split(" | ")) & term_set))]
        return sub.sort_values(["matched_count", "year"], ascending=[False, False]).head(limit).reset_index(drop=True)

    # ------------------------------------------------------------ 구간별분석 데이터 접근
    def load_window_summary(self) -> pd.DataFrame:
        path = self.d.windowed_output / "windows_summary.csv"

        def _load():
            df = pd.read_csv(path)
            df["label"] = df["window"].map(self.window_labels())
            return df
        return self._cached("window_summary", _load, _files_sig(path))

    def load_window_terms(self, window: str) -> pd.DataFrame:
        path = self.d.windowed_output / window / "tau_positive_unigram_clusters.parquet"
        return self._cached(f"window_terms_{window}", lambda: pd.read_parquet(path), _files_sig(path))

    def _window_top_terms(self, window: str, top_n: int = 3) -> dict:
        terms_df = self.load_window_terms(window)
        return {
            cid: sub.sort_values("전체기간_평균비중(%)", ascending=False)["term"].head(top_n).tolist()
            for cid, sub in terms_df.groupby("cluster")
        }

    def window_cluster_graph_elements(self, window: str) -> list[dict]:
        """구간별 클러스터 수준 그래프. 전체분석과 동일하게 노드 라벨은 짧게, 엣지는 상위만 남긴다."""
        terms_df = self.load_window_terms(window)
        long, _ = self.get_long_and_corpus()
        start, end = (int(x) for x in window.split("_"))
        long_w = long[long["year"].between(start, end)]
        graph = windowed_analysis.build_graph_from_terms(long_w, terms_df["term"].tolist())
        term_to_cluster = dict(zip(terms_df["term"], terms_df["cluster"]))

        sizes = terms_df.groupby("cluster").size()
        shares = terms_df.groupby("cluster")["전체기간_평균비중(%)"].mean()
        top4 = self._window_top_terms(window, top_n=4)
        max_size = sizes.max()
        elements = []
        for cid in sorted(sizes.index):
            label_terms = ", ".join(top4.get(cid, []))
            elements.append({
                "data": {
                    "id": f"c{cid}", "label": label_terms,
                    "cluster": int(cid), "term_count": int(sizes[cid]),
                    "mean_share": round(float(shares.get(cid, 0.0)), 3),
                    "color": color_for(cid), "size": _node_size(sizes[cid], max_size),
                }
            })
        cross_weight: dict[tuple[int, int], float] = {}
        for u, v, data in graph.edges(data=True):
            cu, cv = term_to_cluster.get(u), term_to_cluster.get(v)
            if cu is None or cv is None or cu == cv:
                continue
            key = tuple(sorted((int(cu), int(cv))))
            cross_weight[key] = cross_weight.get(key, 0.0) + data.get("raw_cooccurrence", data.get("weight", 0.0))
        pairs = [(a, b, w) for (a, b), w in cross_weight.items()]
        kept = _keep_strong_edges(pairs)
        max_w = max((w for _, _, w in kept), default=1.0) or 1.0
        for a, b, w in kept:
            elements.append({"data": {"source": f"c{a}", "target": f"c{b}", "weight": w / max_w}})
        return elements

    def window_cluster_legend(self, window: str) -> list[dict]:
        terms_df = self.load_window_terms(window)
        sizes = terms_df.groupby("cluster").size()
        top_terms = self._window_top_terms(window, top_n=6)
        return [
            {"cluster": int(cid), "color": color_for(int(cid)),
             "label": ", ".join(top_terms.get(cid, [])), "term_count": int(sizes[cid])}
            for cid in sorted(sizes.index)
        ]

    def window_cluster_terms(self, window: str, cluster: int, top_n: int = 60) -> pd.DataFrame:
        df = self.load_window_terms(window)
        sub = df[df["cluster"].eq(int(cluster))].sort_values("전체기간_평균비중(%)", ascending=False)
        return sub.head(top_n).reset_index(drop=True)

    def window_cluster_papers(self, window: str, terms: list[str], limit: int = 200) -> pd.DataFrame:
        start, end = (int(x) for x in window.split("_"))
        return self.papers_for_terms(terms, year_start=start, year_end=end, limit=limit)

    # ------------------------------------------------------------ 키워드 검색 (전체분석 + 구간 통합)
    def _existing_window_term_paths(self) -> list[Path]:
        return [self.d.windowed_output / w / "tau_positive_unigram_clusters.parquet" for w in self.windows()]

    def all_known_terms(self) -> list[str]:
        """자동완성용 전체 용어 사전 — 전체분석 Rising 용어 + 구간별 tau>0 용어 합집합.
        아직 분석 전인 분야는 있는 결과만 모은다(없으면 빈 목록)."""
        full_path = self.d.full_review / "terms.csv"
        window_paths = self._existing_window_term_paths()

        def _load():
            vocab = set(self.load_full_terms()["term"]) if full_path.exists() else set()
            for w, p in zip(self.windows(), window_paths):
                if p.exists():
                    vocab |= set(self.load_window_terms(w)["term"])
            return sorted(vocab)
        return self._cached("all_known_terms", _load, _files_sig(full_path, *window_paths))

    def keyword_search(self, keyword: str) -> dict:
        keyword = (keyword or "").strip().lower()
        result: dict = {"keyword": keyword, "full": None, "windows": []}
        if not keyword:
            return result

        if self.has_full_results():
            full_terms = self.load_full_terms()
            hit = full_terms[full_terms["term"].str.lower().eq(keyword)]
            if not hit.empty:
                cid = int(hit.iloc[0]["cluster"])
                result["full"] = {
                    "cluster": cid,
                    "label": self.full_cluster_label(cid),
                    "document_count": int(hit.iloc[0]["document_count"]),
                }

        labels = self.window_labels()
        for w in self.windows():
            if not (self.d.windowed_output / w / "tau_positive_unigram_clusters.parquet").exists():
                continue
            terms_df = self.load_window_terms(w)
            hit_w = terms_df[terms_df["term"].str.lower().eq(keyword)]
            if hit_w.empty:
                continue
            cid = int(hit_w.iloc[0]["cluster"])
            cluster_terms = terms_df[terms_df["cluster"].eq(cid)].sort_values(
                "전체기간_평균비중(%)", ascending=False
            )["term"].head(10).tolist()
            result["windows"].append({
                "window": w, "label": labels[w], "cluster": cid,
                "top_terms": cluster_terms,
            })
        return result

    # ------------------------------------------------------------ 데이터 수집 탭
    def dataset_dir_label(self) -> str:
        try:
            return self.d.dataset_dir.relative_to(DATA_ROOT).as_posix() + "/"
        except ValueError:
            return str(self.d.dataset_dir)

    def cache_overview(self) -> dict:
        """분야별 dataset 폴더를 실제로 스캔해서 '지금 이미 모아져 있는 논문' 현황을 만든다.
        파일 전체를 읽지 않고 parquet 메타데이터(행 수)만 읽어서 빠르게 처리한다."""
        dataset_dir = self.d.dataset_dir

        def _load():
            rows = []
            for f in sorted(dataset_dir.glob("*.parquet")) if dataset_dir.exists() else []:
                m = _CACHE_FILENAME_RX.match(f.name)
                if not m:
                    continue
                journal = m.group("journal").replace("_", " ")
                year = int(m.group("year"))
                try:
                    n = pq.ParquetFile(f).metadata.num_rows
                except Exception:
                    n = 0
                rows.append({"journal": journal, "issn": m.group("issn"), "year": year, "count": n})
            df = pd.DataFrame(rows, columns=["journal", "issn", "year", "count"])
            if df.empty:
                return {"empty": True}

            per_journal = (
                df.groupby(["journal", "issn"])
                .agg(total_papers=("count", "sum"), year_min=("year", "min"), year_max=("year", "max"),
                     years_covered=("year", "nunique"))
                .reset_index()
                .sort_values("total_papers", ascending=False)
            )
            year_min, year_max = int(df["year"].min()), int(df["year"].max())
            pivot = df.pivot_table(index="journal", columns="year", values="count", aggfunc="sum", fill_value=0)
            pivot = pivot.reindex(columns=range(year_min, year_max + 1), fill_value=0)
            pivot = pivot.reindex(per_journal["journal"])  # 논문 수 많은 저널이 위로

            return {
                "empty": False,
                "total_papers": int(df["count"].sum()),
                "n_journals": df["journal"].nunique(),
                "year_min": year_min,
                "year_max": year_max,
                "n_files": len(df),
                "per_journal": per_journal,
                "pivot": pivot,
            }
        return self._cached("cache_overview", _load, _dir_sig(dataset_dir))

    def journal_reference_list(self) -> list[dict]:
        """지금까지 이 분야 dataset 폴더에 실제로 논문이 쌓여있는 저널들의 이름·ISSN을 저널명 가나다순으로.
        ISSN은 그 저널의 논문이 실제로 수집됐을 때 쓰인 대표 ISSN 하나(파일명에 남은 것)다."""
        data = self.cache_overview()
        if data.get("empty"):
            return []
        per_journal = data["per_journal"][["journal", "issn"]].sort_values("journal")
        return per_journal.to_dict("records")

    def abstract_overview(self) -> dict:
        """수집 원본 중 초록이 있는 논문 수. 초록이 없으면 빈 문자열로 저장돼 있어 abstract 칸을 실제로 읽는다
        (전체 분야 약 2초, dataset 폴더가 바뀔 때만 다시 계산)."""
        dataset_dir = self.d.dataset_dir

        def _load():
            total = with_abstract = 0
            for f in dataset_dir.glob("*.parquet") if dataset_dir.exists() else []:
                try:
                    col = pq.read_table(f, columns=["abstract"]).column(0)
                except Exception:
                    continue
                total += len(col)
                filled = pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(pc.fill_null(col, ""))), 0)
                with_abstract += pc.sum(filled).as_py() or 0
            return {"total": total, "with_abstract": with_abstract}
        return self._cached("abstract_overview", _load, _dir_sig(dataset_dir))

    def dataset_size_mb(self) -> float:
        files = self.d.dataset_dir.glob("*.parquet") if self.d.dataset_dir.exists() else []
        return sum(f.stat().st_size for f in files) / 1024 / 1024

    def run_collection_sync(self, issns: list[str], year_start: int, year_end: int, emit) -> dict:
        """pipeline.run()(비동기 제너레이터)을 동기 함수 안에서 돌리며 단계마다 emit(msg)를 호출."""
        async def _go():
            final = {}
            async for update in collection_pipeline.run(
                issns=issns, year_start=year_start, year_end=year_end, email=EMAIL, api_key=API_KEYS,
                cache_dir=self.d.dataset_dir, domain=self.d.key,
            ):
                if update.get("log"):
                    emit(f"[{update.get('level', 'info')}] {update['log']}")
                if update.get("step"):
                    emit(update["step"])
                if update.get("done"):
                    final = update
            return final
        result = asyncio.run(_go())
        self.invalidate_cache("cache_overview")
        return result

    # ------------------------------------------------------------ 분석 실행("분석 진행하기")
    # 실제 분석 순서는 analysis_codes/pipeline.py에 있다. 여기서는 웹앱 전용 일(AI 라벨, 화면 캐시)만 붙인다.
    def run_full_pipeline(self, emit) -> dict:
        saved = analysis_pipeline.load_saved_full(self.d)
        if saved:
            # 수집 데이터가 지난 분석 이후 그대로면 다시 계산하지 않는다(화면은 이 경우 버튼을 숨기므로 거의 안 옴).
            emit("수집 데이터가 지난 분석 이후 바뀌지 않아 저장된 최신 결과를 보여줍니다.")
            return saved

        def generate_labels(emit_):
            import webapp.backend.ai_insights as ai_insights
            # 이전 실행의 라벨이 새 커뮤니티 번호에 잘못 붙지 않게 먼저 지운다.
            self.ai_labels_path.unlink(missing_ok=True)
            ai_insights.generate_full_labels(self, emit_)

        result = analysis_pipeline.run_full(self.d, emit, label_fn=generate_labels)
        self.invalidate_cache("full_")
        return result

    def full_status(self) -> tuple[str, dict | None]:
        """글로벌 트렌드 분석 상태와 (최신이면) 결과 요약.
        ready: 저장된 결과가 지금 수집 데이터 기준 최신 / stale: 결과는 있지만 그 뒤 새 논문이 수집됨
        new: 수집 데이터는 있지만 분석한 적 없음 / no_data: 수집된 논문 없음"""
        saved = analysis_pipeline.load_saved_full(self.d)
        if saved:
            return "ready", saved
        if self.cache_overview().get("empty"):
            return "no_data", None
        return ("stale" if self.has_full_results() else "new"), None

    def run_windowed_pipeline(self, emit) -> dict:
        result = analysis_pipeline.run_windowed(self.d, emit)
        self.invalidate_cache("window_")
        return result


# ---------------------------------------------------------------- 수집 기록 ("다시 시작하기"용)
def collect_job_path(job_key: str) -> Path:
    """job_key: 분야 key, 또는 AI 자동 분류 수집이면 "auto"."""
    return STATE_DIR / f"collect_job_{job_key}.json"


def load_collect_job(job_key: str) -> dict | None:
    """마지막 수집 요청과 덜 끝난 저널(pending). 브라우저가 아니라 서버 파일에 두므로 새로고침·화면
    전환·다음 날 접속에도 남는다. 형식: {"year_start", "year_end", "groups", "pending", "updated_at", "reset_at"}"""
    try:
        return json.loads(collect_job_path(job_key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_collect_job(job_key: str, job: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    collect_job_path(job_key).write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 분야별 인스턴스
_services: dict[str, DomainService] = {}


def get(domain_key: str | None) -> DomainService:
    key = domain_key if domain_key in DOMAINS else DEFAULT_DOMAIN
    if key not in _services:
        _services[key] = DomainService(DOMAINS[key])
    return _services[key]
