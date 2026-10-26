# streamlit_app.py — Streamlit Community Cloud용 읽기 전용 결과 뷰어
# 수집·분석·AI 라벨 생성은 EC2의 Dash 앱(webapp/app.py)에서 한다. 여기서는 이미 만들어진
# results/<분야>/ 산출물만 webapp/backend/services.py의 DomainService로 읽어 보여준다(새 분석 로직 없음).
#
# 데이터 위치:
#   - 로컬: .env의 OPENALEX_DATA_ROOT (예: ../openalex-data)
#   - Streamlit Cloud: Secrets의 GITHUB_TOKEN(openalex-data 읽기 권한)으로 비공개 리포에서 결과 폴더만
#     받아온다. 수집 원본 dataset/(parquet 수백 개)은 받지 않는다.
#
# 실행: streamlit run streamlit_app.py

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
DATA_REPO_DEFAULT = "newcave/openalex-data"
# 뷰어가 읽는 폴더만 받는다(sparse checkout). 수자원(논문) 결과는 archive 아래에 있다.
DATA_SPARSE_DIRS = ["results", "archive/수자원_논문/results"]

st.set_page_config(page_title="OpenAlex 연구주제 뷰어", page_icon="💧", layout="wide")


def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)
    except Exception:   # secrets.toml이 아예 없으면 예외가 난다(로컬 실행)
        return None


@st.cache_resource(show_spinner="openalex-data에서 분석 결과를 받는 중...")
def _clone_data_repo(repo: str, token: str) -> str:
    dest = ROOT / ".data" / repo.split("/")[-1]
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://x-access-token:{token}@github.com/{repo}.git"
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(dest)],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *DATA_SPARSE_DIRS],
                       check=True, capture_output=True)
        # 토큰이 .git/config에 남지 않게 원격 주소를 토큰 없는 주소로 되돌린다.
        subprocess.run(["git", "-C", str(dest), "remote", "set-url", "origin", f"https://github.com/{repo}.git"],
                       check=True, capture_output=True)
    return str(dest)


def _resolve_data_root() -> None:
    """config.py가 import될 때 DATA_ROOT를 정하므로, 그보다 먼저 OPENALEX_DATA_ROOT를 맞춰둔다."""
    configured = os.environ.get("OPENALEX_DATA_ROOT") or dotenv_values(ROOT / ".env").get("OPENALEX_DATA_ROOT")
    if configured and (ROOT / configured).resolve().exists():
        return
    token = _secret("GITHUB_TOKEN")
    if token:
        os.environ["OPENALEX_DATA_ROOT"] = _clone_data_repo(_secret("DATA_REPO") or DATA_REPO_DEFAULT, token)


_resolve_data_root()

import config                                      # noqa: E402
from config import DATA_ROOT, Domain               # noqa: E402
from webapp.backend import services                # noqa: E402


class ViewerService(services.DomainService):
    @property
    def ai_labels_path(self) -> Path:
        # 수자원(논문) 버전은 커뮤니티 이름이 interpretations.json에 있다(ai_labels.json과 같은 형식).
        path = self.d.full_review / "ai_labels.json"
        legacy = self.d.full_review / "interpretations.json"
        return legacy if not path.exists() and legacy.exists() else path


def _archive_domain() -> Domain:
    """archive/수자원_논문: 연구소별 분야 체계 이전의 '수자원 분야' 논문 버전 (archive/README.md 참고)."""
    base = DATA_ROOT / "archive" / "수자원_논문"
    results = base / "results"
    corpus = results / "01_corpus"
    return Domain(
        key="water_paper", name="수자원(논문)", department="논문 버전(2026-10-20까지 웹앱의 수자원 분야)",
        journals=(), dataset_dir=base / "dataset",
        shared_source=corpus,
        corpus_path=corpus / "filtered_corpus.parquet",
        dict_path=corpus / "term_normalization_dict.csv",
        full_output=results / "02_full_period",
        full_review=results / "03_full_period_review",
        windowed_output=results / "04_windowed",
    )


@st.cache_resource
def _services() -> dict[str, ViewerService]:
    """결과가 있는 분야만. DomainService가 파일별 캐시를 갖고 있으니 프로세스당 하나씩 둔다."""
    domains = [*config.DOMAINS.values(), _archive_domain()]
    out = {}
    for d in domains:
        svc = ViewerService(d)
        if svc.has_full_results():
            out[d.name] = svc
    return out


# ------------------------------------------------------------ 그림
def _network_figure(elements: list[dict]) -> go.Figure:
    """DomainService가 cytoscape용으로 만든 군집 노드/엣지를 plotly 그림으로 그린다."""
    nodes = [e["data"] for e in elements if "source" not in e["data"]]
    edges = [e["data"] for e in elements if "source" in e["data"]]
    g = nx.Graph()
    g.add_nodes_from(n["id"] for n in nodes)
    g.add_weighted_edges_from((e["source"], e["target"], e["weight"]) for e in edges)
    pos = nx.spring_layout(g, seed=0, weight="weight", k=1.6)

    fig = go.Figure()
    for e in edges:
        (x0, y0), (x1, y1) = pos[e["source"]], pos[e["target"]]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line={"width": 1 + 6 * e["weight"], "color": "rgba(140,140,140,0.45)"}))
    fig.add_trace(go.Scatter(
        x=[pos[n["id"]][0] for n in nodes], y=[pos[n["id"]][1] for n in nodes],
        mode="markers+text", showlegend=False, cliponaxis=False,
        text=[f"{n['cluster']}. {n['label']}" for n in nodes], textposition="top center",
        marker={"size": [n["size"] / 2 for n in nodes], "color": [n["color"] for n in nodes],
                "line": {"width": 1, "color": "white"}},
        customdata=[[n["term_count"], n["mean_share"]] for n in nodes],
        hovertemplate="%{text}<br>용어 %{customdata[0]}개 · 평균 문서비중 %{customdata[1]}%<extra></extra>",
    ))
    # 노드 라벨이 그림 가장자리에서 잘리지 않게 좌우·위쪽 여백을 둔다.
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      xaxis={"visible": False, "range": [min(xs) - 0.6, max(xs) + 0.6]},
                      yaxis={"visible": False, "range": [min(ys) - 0.2, max(ys) + 0.3]})
    return fig


def _papers_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("해당하는 논문이 없습니다.")
        return
    st.dataframe(
        df[["year", "journal", "display_name", "matched_terms", "url"]],
        column_config={
            "year": st.column_config.NumberColumn("연도", format="%d"),
            "journal": "저널", "display_name": "제목", "matched_terms": "일치 용어",
            "url": st.column_config.LinkColumn("OpenAlex", display_text="열기"),
        },
        hide_index=True, use_container_width=True,
    )


# ------------------------------------------------------------ 탭
def _full_tab(svc: ViewerService) -> None:
    summary = svc.load_full_summary()
    labels = dict(zip(summary["cluster"], summary["label"]))
    y0, y1 = svc.full_year_range()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rising 용어", f"{svc.n_full_rising_terms():,}개")
    c2.metric("연구주제 후보(커뮤니티)", f"{len(summary)}개")
    c3.metric("분석 기간", f"{y0}–{y1}")

    st.subheader("커뮤니티 네트워크")
    st.caption("원 크기 = 용어 수, 선 굵기 = 커뮤니티 사이 공유 논문 수(상위 연결만 표시)")
    st.plotly_chart(_network_figure(svc.full_cluster_graph_elements()), use_container_width=True)

    st.subheader("커뮤니티 요약")
    st.dataframe(
        summary[["cluster", "label", "term_count", "mean_document_share_pct", "mk_tau",
                 "recent5_minus_first5_pct_point", "top_terms"]],
        column_config={
            "cluster": "번호", "label": "이름", "term_count": "용어 수",
            "mean_document_share_pct": st.column_config.NumberColumn("평균 문서비중(%)", format="%.1f"),
            "mk_tau": st.column_config.NumberColumn("MK tau", format="%.2f"),
            "recent5_minus_first5_pct_point": st.column_config.NumberColumn("최근5년-초기5년(%p)", format="%.1f"),
            "top_terms": "대표 용어",
        },
        hide_index=True, use_container_width=True,
    )

    trends_path = svc.d.full_output / "cluster_trends_fdr_assoc.parquet"
    if trends_path.exists():
        st.subheader("연도별 문서 비중")
        trends = pd.read_parquet(trends_path)
        trends["커뮤니티"] = trends["cluster"].map(lambda c: f"{c}. {labels.get(c, c)}")
        fig = px.line(trends.sort_values("year"), x="year", y="document_share_pct", color="커뮤니티",
                      color_discrete_sequence=services.PALETTE,
                      labels={"year": "연도", "document_share_pct": "문서 비중(%)"})
        fig.update_layout(height=420, margin={"l": 10, "r": 10, "t": 10, "b": 10})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("커뮤니티 자세히 보기")
    cluster = st.selectbox("커뮤니티", sorted(labels), format_func=lambda c: f"{c}. {labels[c]}",
                           key=f"full_cluster_{svc.d.key}")
    left, right = st.columns([1, 2])
    with left:
        terms = svc.full_cluster_terms(cluster)
        st.dataframe(terms[["term", "document_count"]],
                     column_config={"term": "용어", "document_count": "논문 수"},
                     hide_index=True, use_container_width=True, height=420)
    with right:
        picked = st.multiselect("용어로 논문 좁히기", terms["term"].tolist(), key=f"full_terms_{svc.d.key}")
        _papers_table(svc.full_cluster_papers(cluster, picked or None))


def _window_tab(svc: ViewerService) -> None:
    if not (svc.d.windowed_output / "windows_summary.csv").exists():
        st.info("이 분야는 아직 구간별 분석 결과가 없습니다.")
        return
    labels = svc.window_labels()
    window = st.radio("구간", svc.windows(), format_func=labels.get, horizontal=True, key=f"window_{svc.d.key}")
    row = svc.load_window_summary().set_index("window").loc[window]
    c1, c2, c3 = st.columns(3)
    c1.metric("구간 논문 수", f"{int(row['documents_in_window']):,}편")
    c2.metric("tau>0 용어", f"{int(row['tau_positive_terms']):,}개")
    c3.metric("커뮤니티", f"{int(row['clusters'])}개")

    st.plotly_chart(_network_figure(svc.window_cluster_graph_elements(window)), use_container_width=True)

    legend = svc.window_cluster_legend(window)
    cluster = st.selectbox("커뮤니티", [c["cluster"] for c in legend],
                           format_func=lambda c: next(f"{c}. {x['label']} ({x['term_count']}개)"
                                                      for x in legend if x["cluster"] == c),
                           key=f"window_cluster_{svc.d.key}_{window}")
    left, right = st.columns([1, 2])
    with left:
        terms = svc.window_cluster_terms(window, cluster)
        st.dataframe(terms[["term", "전체기간_평균비중(%)"]],
                     column_config={"term": "용어",
                                    "전체기간_평균비중(%)": st.column_config.NumberColumn("평균비중(%)", format="%.3f")},
                     hide_index=True, use_container_width=True, height=420)
    with right:
        picked = st.multiselect("용어로 논문 찾기 (비우면 대표 용어 10개)", terms["term"].tolist(),
                                key=f"window_terms_{svc.d.key}_{window}")
        _papers_table(svc.window_cluster_papers(window, picked or terms["term"].head(10).tolist()))


def _keyword_tab(svc: ViewerService) -> None:
    vocab = svc.all_known_terms()
    keyword = st.selectbox("키워드 (전체기간 Rising 용어 + 구간별 tau>0 용어)", vocab, index=None,
                           placeholder="용어를 입력해 고르세요", key=f"keyword_{svc.d.key}")
    if not keyword:
        return
    result = svc.keyword_search(keyword)
    if result["full"]:
        f = result["full"]
        st.success(f"전체기간: {f['cluster']}. {f['label']} 커뮤니티 · 논문 {f['document_count']:,}편")
    else:
        st.write("전체기간 분석의 Rising 용어는 아닙니다.")
    if result["windows"]:
        st.dataframe(
            pd.DataFrame([{"구간": w["label"], "커뮤니티": w["cluster"], "같은 커뮤니티 대표 용어": ", ".join(w["top_terms"])}
                          for w in result["windows"]]),
            hide_index=True, use_container_width=True,
        )
    papers = svc.papers_for_terms([keyword], limit=1000)
    if not papers.empty:
        st.subheader("연도별 논문 수")
        st.bar_chart(papers.groupby("year").size().rename("논문 수"))
        st.subheader("논문")
        _papers_table(papers.head(200))


# ------------------------------------------------------------ 화면
def main() -> None:
    svcs = _services()
    st.sidebar.title("OpenAlex 연구주제 뷰어")
    if not svcs:
        st.title("OpenAlex 연구주제 뷰어")
        st.warning(f"분석 결과를 찾지 못했습니다. 데이터 위치: `{DATA_ROOT}`  \n"
                   "로컬은 .env의 OPENALEX_DATA_ROOT, Streamlit Cloud는 Secrets의 GITHUB_TOKEN을 확인하세요.")
        return
    name = st.sidebar.radio("분야", list(svcs))
    svc = svcs[name]
    st.sidebar.caption(svc.d.department)
    st.sidebar.caption("읽기 전용 뷰어입니다. 수집·분석은 EC2 웹앱에서 합니다.")

    st.title(svc.d.title)
    full, window, keyword = st.tabs(["전체기간 분석", "구간별 분석", "키워드 검색"])
    with full:
        _full_tab(svc)
    with window:
        _window_tab(svc)
    with keyword:
        _keyword_tab(svc)


main()
