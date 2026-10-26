# webapp/app.py
# 분야(연구소)별 시기별 주요 연구주제 탐색 대시보드 (Dash).
# 사이드바 구성
#   - 데이터: 데이터셋 현황 / 데이터 수집 / 수집현황 모니터링  (분야 전체 공통 화면)
#   - 분야별 분석: 분야를 고르면 그 분야 데이터로 탭(글로벌 트렌드 분석 / 키워드 검색)이 다시 그려진다.
#     구간별분석 탭은 화면에서만 숨겨 두었다(코드·콜백은 그대로, build_domain_content의 tab_style).
#
# 이 파일은 화면(레이아웃·콜백)만 담당한다.
#   - 분야 정의(폴더 경로·라벨 방식 등): config.py
#   - 분야별 데이터 읽기 / 수집·분석 실행: webapp/backend/services.py  (svc.get(domain))
#   - AI 해석 텍스트: webapp/backend/ai_insights.py,  모니터링 탭 집계: webapp/backend/monitor.py
# 분야별 분석 콜백은 State("domain-store", "data")로 지금 보고 있는 분야를 받아서 그 분야 서비스만 쓴다.
#
# 실행(프로젝트 루트에서): python webapp/app.py

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent
ROOT = WEBAPP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diskcache
import flask
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, DiskcacheManager, Input, Output, State, ctx, dash_table, dcc, html, no_update
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

import webapp.backend.services as svc
import webapp.backend.ai_insights as ai_insights
import webapp.backend.monitor as monitor
from webapp.backend import chat
from pipeline_codes import event_log, journal_router
from config import DOMAINS, DEFAULT_DOMAIN, LEGACY_DOMAIN_NAMES

# ---------------------------------------------------------------- 백그라운드 콜백 매니저
_diskcache = diskcache.Cache(str(WEBAPP_DIR / ".cache_diskcache"))
background_manager = DiskcacheManager(_diskcache)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    background_callback_manager=background_manager,
    suppress_callback_exceptions=True,
    title="Rising 연구주제 탐색",
)
server = app.server

# 분석 도중 분야를 바꾸면 화면이 다른 분야로 다시 그려지므로, 그 작업 결과가 엉뚱한 분야 화면에
# 꽂히지 않게 분석을 중단시킨다. 수집은 공통 화면이라 분야를 바꿔도 중단되지 않는다.
CANCEL_ON_DOMAIN_SWITCH = [Input("domain-store", "data")]

CYTO_STYLESHEET = [
    {"selector": "node", "style": {
        "label": "data(label)", "background-color": "data(color)",
        "background-opacity": 0.94,
        "width": "data(size)", "height": "data(size)",
        "font-family": "Inter, -apple-system, sans-serif",
        "font-size": "12px", "font-weight": 700, "line-height": 1.35,
        "text-wrap": "wrap", "text-max-width": "150px",
        "text-valign": "bottom", "text-halign": "center", "text-margin-y": 8,
        "color": "#1a1d23",
        "text-outline-width": 3, "text-outline-color": "#f4f6f9", "text-outline-opacity": 1,
        "border-width": 3, "border-color": "#ffffff",
        "shadow-blur": 14, "shadow-color": "rgba(20,24,33,0.22)", "shadow-opacity": 1, "shadow-offset-x": 0, "shadow-offset-y": 3,
        "z-index": 10,
    }},
    {"selector": "edge", "style": {
        "width": "mapData(weight, 0, 1, 1, 7)", "line-color": "#c7cde0",
        "curve-style": "bezier", "opacity": 0.5,
    }},
    {"selector": "node:selected", "style": {
        "border-width": 5, "border-color": "#0067ac",
    }},
]

CYTO_LAYOUT = {"name": "circle", "padding": 60, "radius": 190, "animate": False}


def cyto_layout(nonce=None):
    """CYTO_LAYOUT에 nonce를 섞어 새 dict를 만든다.

    dash-cytoscape는 elements만 바뀌었을 때 레이아웃을 자동으로 다시 계산해주지 않는 경우가 있고,
    layout prop을 다시 보내도 내용이 이전과 완전히 같으면 프론트엔드가 변경으로 인식하지 않는다.
    그래서 값이 실제로 달라지는 nonce 필드를 끼워 넣어 레이아웃 재계산을 강제로 트리거한다."""
    layout = dict(CYTO_LAYOUT)
    if nonce is not None:
        layout["_nonce"] = nonce
    return layout


# 네트워크 그림은 마우스로 확대·이동할 수 있고, '원래대로'를 누르면 배치를 다시 계산해 처음 화면(전체가 보이게)으로
# 돌아간다. 확대·이동하다 그림을 놓쳐도 되돌릴 수 있게 하려는 것.
for _graph in ("full", "windowed"):
    app.callback(
        Output(f"{_graph}-cytoscape", "layout", allow_duplicate=True),
        Input(f"{_graph}-cyto-reset", "n_clicks"),
        prevent_initial_call=True,
    )(lambda n_clicks: cyto_layout(f"reset-{n_clicks}"))


# ================================================================== 공용 UI 헬퍼
def panel(title, children, hint=None, extra_header=None):
    header = [html.Div(title, className="panel-title")]
    if extra_header is not None:
        header = [html.Div([html.Span(title)], className="panel-title"), extra_header]
    body = []
    if hint:
        body.append(html.Div(hint, className="panel-hint"))
    body.extend(children if isinstance(children, list) else [children])
    return html.Div(
        [html.Div(header, className="d-flex justify-content-between align-items-center")] + body,
        className="panel",
    )


def paper_table(df: pd.DataFrame, table_id: str):
    if df.empty:
        return html.Div("표시할 논문이 없습니다. 커뮤니티나 단어를 먼저 선택하세요.",
                         className="panel-hint", style={"padding": "8px 0"})
    view = df.copy()
    view["display_name"] = view.apply(lambda r: f"[{r['display_name']}]({r['url']})", axis=1)
    cols = [c for c in ["year", "journal", "display_name", "matched_terms"] if c in view.columns]
    return dash_table.DataTable(
        id=table_id,
        columns=[
            {"name": "연도", "id": "year"},
            {"name": "저널", "id": "journal"},
            {"name": "제목", "id": "display_name", "presentation": "markdown"},
            {"name": "매칭된 단어", "id": "matched_terms"},
        ] if set(["year", "journal", "display_name", "matched_terms"]).issubset(view.columns) else
        [{"name": c, "id": c} for c in cols],
        data=view[cols].to_dict("records"),
        page_size=10,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "10px 12px", "fontFamily": "Inter, sans-serif",
                    "maxWidth": "420px", "overflow": "hidden", "textOverflow": "ellipsis"},
        style_data={"border": "none"},
        style_header={"border": "none"},
        markdown_options={"link_target": "_blank"},
    )


def term_table(df: pd.DataFrame, table_id: str, count_col: str, count_label: str,
                selectable: bool = True, initial_selected: list[int] | None = None):
    if df.empty:
        return html.Div("이 커뮤니티에 단어가 없습니다.", className="panel-hint")
    view = df[["term", count_col]].rename(columns={count_col: count_label})
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": "용어", "id": "term"}, {"name": count_label, "id": count_label}],
        data=view.to_dict("records"),
        row_selectable="multi" if selectable else False,
        selected_rows=initial_selected or [],
        page_size=15,
        style_as_list_view=True,
        style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
        style_cell={"textAlign": "left", "padding": "9px 12px", "fontFamily": "Inter, sans-serif"},
        style_data={"border": "none"},
        style_header={"border": "none"},
        sort_action="native",
    )


def metadata_table(df: pd.DataFrame, table_id: str = "collect-metadata-table", limit: int = 500):
    if df is None or df.empty:
        return html.Div("수집된 논문이 없습니다.", className="panel-hint")
    view = df.copy()
    view["ror_matched"] = view["ror"].map(lambda x: "\u2713" if isinstance(x, (list,)) and len(x) > 0 else "")
    view = view.sort_values(["year"], ascending=False)
    truncated = len(view) > limit
    view = view.head(limit)
    table = dash_table.DataTable(
        id=table_id,
        columns=[
            {"name": "연도", "id": "year"},
            {"name": "저널", "id": "journal"},
            {"name": "제목", "id": "display_name"},
            {"name": "DOI", "id": "doi"},
            {"name": "피인용수", "id": "cited_by_count"},
            {"name": "기관정보", "id": "ror_matched"},
        ],
        data=view[["year", "journal", "display_name", "doi", "cited_by_count", "ror_matched"]].to_dict("records"),
        page_size=10,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "10px 12px", "fontFamily": "Inter, sans-serif",
                    "maxWidth": "380px", "overflow": "hidden", "textOverflow": "ellipsis"},
        style_data={"border": "none"},
        style_header={"border": "none"},
        sort_action="native",
        filter_action="native",
    )
    note = f"총 {len(df):,}건 중 최근 {limit}건만 표시합니다." if truncated else f"총 {len(df):,}건."
    return html.Div([html.Div(note, className="panel-hint"), table])


def stat_tile(value: str, label: str) -> html.Div:
    return html.Div([
        html.Div(value, style={"fontSize": "24px", "fontWeight": 800, "color": "var(--ink)"}),
        html.Div(label, style={"fontSize": "11.5px", "color": "var(--ink-soft)", "marginTop": "2px"}),
    ], style={"background": "#fff", "border": "1px solid var(--border)", "borderRadius": "12px",
              "padding": "14px 18px", "minWidth": "140px", "flex": "1"})


def cluster_info_card(title: str, term_count: int, share_label: str, share_value, note: str | None = None) -> html.Div:
    children = [
        html.B(title),
        html.Div(f"용어 {term_count}개 · {share_label} {share_value}%", style={"fontSize": "12px", "marginTop": "2px"}),
    ]
    if note:
        children.append(html.Div(note, className="panel-hint", style={"marginBottom": "0", "marginTop": "2px"}))
    return html.Div(children, className="info-card")


def ai_insight_block(insight: dict | str | None, chat_button: bool = False) -> html.Div:
    """공식 라벨(전문가 검수)과 확실히 구분되도록 별도 스타일로 보여주는 "AI 해석(참고용)" 박스.
    insight는 {"ko": ..., "en": ...}(논문 원문, 둘 다 있을 수 있음), {"ko": ...}(새로 생성된 경우),
    평범한 문자열(4번 탭 즉석 생성), 또는 None(아직 없음) 중 하나다.
    chat_button=True면 제목 옆에 'AI와 대화하기' 버튼을 붙인다(글로벌 트렌드 분석 커뮤니티 상세)."""
    if not insight and not chat_button:
        return html.Div()
    if isinstance(insight, str):
        insight = {"ko": insight}
    insight = insight or {}
    title = html.Div("AI 해석 (참고용)", className="ai-insight-title")
    if chat_button:
        title = html.Div([title, dbc.Button("AI와 대화하기", id="chat-open-btn", size="sm", className="chat-open-btn")],
                         className="ai-insight-head")
    children = [title]
    if insight.get("ko"):
        children.append(html.Div(insight["ko"], className="ai-insight-body"))
    if insight.get("en"):
        children.append(html.Div(insight["en"], className="ai-insight-body ai-insight-en"))
    return html.Div(children, className="ai-insight-box")


def legend_block(items: list[dict]) -> html.Div:
    rows = []
    for it in items:
        rows.append(html.Div([
            html.Span(style={
                "display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "4px",
                "backgroundColor": it["color"], "marginRight": "8px", "flexShrink": "0",
            }),
            html.Span(it["label"], style={"color": "var(--ink-soft)", "flex": "1"}),
            html.Span(f"{it['term_count']}개", style={"color": "var(--ink-soft)", "fontSize": "11.5px", "flexShrink": "0"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "4px", "padding": "5px 0",
                  "borderBottom": "1px solid #f0f1f5", "fontSize": "12.5px"}))
    return html.Div(rows, className="mt-2")


def _n_years(s: svc.DomainService) -> int:
    y0, y1 = s.full_year_range()
    return y1 - y0 + 1


# ================================================================== 공통 화면 1: 데이터셋 현황
def journal_status_rows(s: svc.DomainService) -> list[dict]:
    """분야 수집 대상 저널마다 실제 dataset 폴더에 수집됐는지. 목록에 없는데 폴더에 있는 저널(AI 배정 등)도 붙인다."""
    data = s.cache_overview()
    collected = {} if data.get("empty") else {r["issn"].upper(): r for r in data["per_journal"].to_dict("records")}
    rows, used = [], set()

    def _row(journal, issn, hit, state, note=""):
        return {"journal": journal, "issn": issn, "state": state,
                "papers": f"{hit['total_papers']:,}" if hit else "",
                "years": f"{hit['year_min']}–{hit['year_max']} ({hit['years_covered']}개 연도)" if hit else "",
                "note": note}

    for j in s.d.journals:
        hit = next((collected[x.upper()] for x in j.issns if x.upper() in collected), None)
        if hit:
            used.add(hit["issn"].upper())
        state = "✓ 수집됨" if hit else ("✗ OpenAlex 미등록" if "미등록" in j.note else "미수집")
        rows.append(_row(j.name, ", ".join(j.issns), hit, state, j.note))
    for issn, hit in collected.items():
        if issn not in used:
            rows.append(_row(hit["journal"], hit["issn"], hit, "✓ 목록 외 (AI 배정 등)"))
    return rows


def _abstract_text(s: svc.DomainService) -> str:
    """초록 있는 논문 수 (전체 대비 %). 수집된 논문이 없으면 빈칸."""
    a = s.abstract_overview()
    if not a["total"]:
        return ""
    return f"{a['with_abstract']:,}건 ({a['with_abstract'] / a['total'] * 100:.1f}%)"


def dataset_summary_table() -> dash_table.DataTable:
    rows = []
    for key, d in DOMAINS.items():
        s = svc.get(key)
        data = s.cache_overview()
        status = journal_status_rows(s)
        rows.append({
            "domain": d.name, "department": d.department,
            "journals": f"{sum(r['state'].startswith('✓') and '목록 외' not in r['state'] for r in status)} / {len(d.journals)}",
            "papers": "" if data.get("empty") else f"{data['total_papers']:,}",
            "abstract": _abstract_text(s),
            "years": "" if data.get("empty") else f"{data['year_min']}–{data['year_max']}",
            "size": f"{s.dataset_size_mb():,.0f} MB",
            "analysis": "글로벌 트렌드" if s.has_full_results() else "-",
        })
    return dash_table.DataTable(
        columns=[
            {"name": "분야", "id": "domain"}, {"name": "연구소", "id": "department"},
            {"name": "수집된 대상 저널", "id": "journals"},
            {"name": "논문 수", "id": "papers"}, {"name": "초록 현황", "id": "abstract"},
            {"name": "연도 범위", "id": "years"},
            {"name": "용량", "id": "size"}, {"name": "분석 결과", "id": "analysis"},
        ],
        data=rows, style_as_list_view=True,
        style_cell={"textAlign": "left", "padding": "9px 12px", "fontFamily": "Inter, sans-serif", "fontSize": "13px"},
        style_data={"border": "none"}, style_header={"border": "none", "fontWeight": 700},
    )


def dataset_heatmap_block(s: svc.DomainService):
    """분야 하나의 저널×연도 수집 논문 수 히트맵."""
    data = s.cache_overview()
    if data.get("empty"):
        return html.Div(f"{s.dataset_dir_label()} 폴더에 아직 수집된 논문이 없습니다. "
                        "'데이터 수집' 화면에서 이 분야를 골라 수집하세요.", className="panel-hint")
    pivot = data["pivot"]
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=list(pivot.index),
        colorscale=[[0, "#f4f6f9"], [0.001, "#cfe6f5"], [1, "#0067ac"]],
        zmin=0, xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}년: %{z:,}건<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(
        height=min(120 + 26 * len(pivot.index), 640),
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#1a1d23"),
        xaxis=dict(side="top", tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def dataset_detail_block(s: svc.DomainService) -> html.Div:
    """분야 하나의 수집 현황: 요약 숫자(초록 현황 포함) + 대상 저널별 수집 상태."""
    data = s.cache_overview()
    status = dash_table.DataTable(
        columns=[{"name": "저널", "id": "journal"}, {"name": "ISSN", "id": "issn"}, {"name": "상태", "id": "state"},
                 {"name": "논문 수", "id": "papers"}, {"name": "수집 연도", "id": "years"}, {"name": "비고", "id": "note"}],
        data=journal_status_rows(s), style_as_list_view=True, sort_action="native",
        style_cell={"textAlign": "left", "padding": "8px 12px", "fontFamily": "Inter, sans-serif", "fontSize": "12.5px",
                    "whiteSpace": "normal", "height": "auto"},
        style_data={"border": "none"}, style_header={"border": "none", "fontWeight": 700},
        style_data_conditional=[{"if": {"filter_query": '{state} = "미수집"'}, "color": "var(--ink-soft)"},
                                {"if": {"filter_query": '{state} contains "미등록"'}, "color": "#b3261e"}],
    )
    if data.get("empty"):
        return status
    tiles = html.Div([
        stat_tile(f"{data['total_papers']:,}건", "전체 수집 논문"),
        stat_tile(_abstract_text(s), "초록 현황 (초록 있는 논문)"),
        stat_tile(f"{data['n_journals']}개", "저널 수"),
        stat_tile(f"{data['year_min']}–{data['year_max']}", "커버 연도 범위"),
        stat_tile(f"{data['n_files']}개", "수집 파일(저널×연도)"),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"})
    return html.Div([tiles, status])


def build_dataset_page():
    return html.Div([
        html.Div([
            html.Div(html.Img(src=app.get_asset_url("kwater-logo.png"), alt="K-water"), className="app-logo"),
            html.Div([
                html.H3("물분야 글로벌 저널 트렌드 분석"),
                html.Div("분야(연구소)별 수집 대상 저널과 실제로 수집된 논문 현황입니다. dataset/<분야>/ 폴더를 직접 "
                         "읽어 계산하므로 수집이 끝나면 '새로고침'으로 최신 상태를 볼 수 있습니다.", className="subtitle"),
            ]),
        ], className="app-header"),
        html.Div(dbc.Button("새로고침", id="ds-refresh", className="btn-secondary", size="sm"), className="mb-3"),
        panel("분야별 요약", hint="여러 분야에 겹치는 저널은 분야마다 따로 수집·저장됩니다. "
                                  "초록 현황 = 초록이 있는 논문 수 (전체 수집 논문 대비 %).",
              children=[dcc.Loading(html.Div(id="ds-summary"))]),
        panel("저널×연도 수집 현황",
              hint="색이 진할수록 논문이 많습니다. 여기서 고른 분야가 아래 '분야별 상세'에도 나옵니다.",
              extra_header=dcc.Dropdown(id="ds-domain", clearable=False, value=DEFAULT_DOMAIN, style={"width": "260px"},
                                        options=[{"label": f"{d.name} ({d.department})", "value": k}
                                                 for k, d in DOMAINS.items()]),
              children=[dcc.Loading(html.Div(id="ds-heatmap"))]),
        panel("분야별 상세", children=[dcc.Loading(html.Div(id="ds-detail"))]),
    ], className="section-gap")


@app.callback(
    Output("ds-summary", "children"),
    Output("ds-heatmap", "children"),
    Output("ds-detail", "children"),
    Input("page-store", "data"),
    Input("ds-refresh", "n_clicks"),
    Input("ds-domain", "value"),
)
def refresh_dataset_page(page, n_refresh, domain):
    """화면을 열 때·새로고침·분야 변경 때마다 dataset 폴더를 다시 스캔한다(바뀐 폴더만 실제로 다시 읽음)."""
    if page != "dataset":
        return no_update, no_update, no_update
    s = svc.get(domain)
    summary = no_update if ctx.triggered_id == "ds-domain" else dataset_summary_table()
    return summary, dataset_heatmap_block(s), dataset_detail_block(s)


# ================================================================== 공통 화면 2: 데이터 수집
AUTO = "auto"   # "AI 자동 분류" 선택값 = 수집 기록 파일 key


def _domain_issn_lines(d) -> str:
    """분야 수집 대상 저널을 입력칸 형식으로(한 줄 = 저널 하나, 같은 저널의 Print·Online은 콤마)."""
    return "\n".join(", ".join(j.issns) for j in d.journals)


def build_collect_page():
    form = dbc.Row([
        dbc.Col(panel(
            "수집 설정",
            extra_header=dbc.Button("저널 목록 보기", id="journal-ref-btn", size="sm", className="btn-secondary"),
            children=[
                dbc.Label("저장할 분야", style={"fontSize": "12px"}),
                dcc.Dropdown(
                    id="collect-domain", clearable=False, value=DEFAULT_DOMAIN,
                    options=[{"label": f"{d.name} ({d.department})", "value": k} for k, d in DOMAINS.items()]
                    + [{"label": "AI 자동 분류 (목록에 없는 새 저널)", "value": AUTO}],
                ),
                html.Div(id="collect-domain-hint", className="panel-hint", style={"marginTop": "6px"}),
                dcc.Textarea(
                    id="collect-issn-textarea", className="form-control", style={"height": "200px", "fontSize": "13px"},
                    placeholder=("한 줄에 저널 하나씩 입력합니다. 같은 저널의 Print·Online ISSN은 콤마로 구분합니다.\n"
                                 "예: 1234-5678, 8765-4321"),
                ),
                dbc.Row([
                    dbc.Col([dbc.Label("시작 연도", style={"fontSize": "12px"}),
                             dbc.Input(id="collect-year-start", type="number", value=2000)], width=6),
                    dbc.Col([dbc.Label("끝 연도", style={"fontSize": "12px"}),
                             dbc.Input(id="collect-year-end", type="number", value=2025)], width=6),
                ], className="mt-2 g-2"),
                html.Div([
                    dbc.Button("수집 시작", id="collect-start-btn", className="btn-primary"),
                    dbc.Button("다시 시작하기", id="collect-resume-btn", className="btn-secondary",
                               style={"marginLeft": "8px", "display": "none"}),
                    dbc.Button("중지", id="collect-stop-btn", className="btn-secondary", disabled=True,
                               style={"marginLeft": "8px"}),
                ], className="mt-3"),
                html.Div(id="collect-resume-info", className="mt-2"),
            ],
        ), width=5),
        dbc.Col(panel(
            "진행 상황",
            children=[
                dcc.Loading(html.Pre(id="collect-log", className="log-box", style={"height": "520px"})),
                html.Div(id="collect-result", className="mt-2"),
            ],
        ), width=7),
    ], className="section-gap")

    return html.Div([
        html.Div([
            html.H3("데이터 수집"),
            html.Div("OpenAlex에서 저널별 논문을 받아 선택한 분야의 dataset 폴더에 저장합니다. 이미 받은 연도는 "
                     "건너뛰므로 중단돼도 다시 누르면 이어서 받습니다. 수집 중 다른 화면으로 이동해도 수집은 계속됩니다.",
                     className="subtitle"),
        ], className="app-header"),
        form,
        panel(
            "수집된 논문 메타데이터",
            hint="위에서 수집한 ISSN 그룹의 논문 목록입니다.",
            children=[dcc.Loading(html.Div(id="collect-metadata-wrap"))],
        ),
        dbc.Offcanvas(
            html.Div(id="journal-ref-content"),
            id="journal-ref-offcanvas",
            placement="end",
            is_open=False,
            scrollable=True,
            style={"width": "480px"},
        ),
    ], className="section-gap")


@app.callback(
    Output("collect-issn-textarea", "value"),
    Output("collect-domain-hint", "children"),
    Output("collect-resume-btn", "style", allow_duplicate=True),
    Output("collect-resume-info", "children", allow_duplicate=True),
    Input("collect-domain", "value"),
    prevent_initial_call="initial_duplicate",
)
def on_collect_domain(mode):
    """분야를 고르면 그 분야 수집 대상 저널 전체를 입력칸에 채운다(필요 없는 줄은 지우고 수집)."""
    if mode == AUTO:
        hint = ("입력한 저널마다 저장할 분야를 정합니다. 수집 대상 목록에 있는 ISSN은 목록의 분야로, 목록에 없는 "
                f"ISSN은 OpenAlex 저널 정보(토픽·최근 논문 제목)를 보고 AI({journal_router.ROUTER_MODEL})가 "
                "분야를 고릅니다(여러 분야 가능). AI 판단은 dataset/ai_journal_assignments.json에 저장되어 "
                "같은 저널은 다시 판단하지 않습니다.")
        return "", hint, *_resume_state(AUTO)
    s = svc.get(mode)
    n_missing = sum("미등록" in j.note for j in s.d.journals)
    hint = f"저장 위치: {s.dataset_dir_label()} · 수집 대상 저널 {len(s.d.journals)}종"
    if n_missing:
        hint += f" (그중 {n_missing}종은 OpenAlex 미등록이라 수집되지 않음)"
    return _domain_issn_lines(s.d), hint, *_resume_state(mode)


def reference_journal_list(s: svc.DomainService) -> html.Div:
    """분야에 정해둔 수집 대상 저널 목록. 저널마다 ISSN 입력칸에 그대로 붙여넣을 수 있는 한 줄과,
    지금 이 분야 dataset 폴더에 수집돼 있는지(연도 범위·논문 수)를 함께 보여준다."""
    items = [
        dbc.Button(f"전체 ISSN 다시 채우기 ({len(s.d.journals)}종)", id="journal-ref-select-all-btn",
                   className="btn-primary", size="sm", style={"marginBottom": "10px"}),
        html.Div(
            "버튼을 누르면 아래 저널 전체의 ISSN이 입력칸에 한 줄에 저널 하나씩 채워집니다. "
            "일부만 수집하려면 회색 칸의 ISSN 한 줄을 복사해 붙여넣으세요.",
            className="panel-hint"),
    ]
    status = {r["issn"]: r for r in journal_status_rows(s)}
    for j in s.d.journals:
        labels = ([f"Print {j.print_issn}"] if j.print_issn else []) + ([f"Electronic {j.e_issn}"] if j.e_issn else [])
        r = status.get(", ".join(j.issns), {})
        collected = r.get("state", "").startswith("✓")
        children = [
            html.Div(j.name, style={"fontWeight": 600, "fontSize": "13.5px"}),
            html.Div(" · ".join(labels), style={"color": "var(--ink-soft)", "fontSize": "12px", "marginTop": "2px"}),
            html.Code(", ".join(j.issns), style={"fontSize": "12px", "background": "#f4f6f9", "padding": "2px 8px",
                                                 "borderRadius": "6px", "marginTop": "4px", "width": "fit-content",
                                                 "userSelect": "all"}),
            html.Div(f"✓ 수집됨: {r['years']}, {r['papers']}건" if collected else r.get("state", "미수집"),
                     style={"color": "#1a7f37" if collected else "var(--ink-soft)", "fontSize": "12px", "marginTop": "2px"}),
        ]
        if j.note:
            children.append(html.Div(j.note, style={"color": "#8a5a00", "fontSize": "12px", "marginTop": "2px"}))
        items.append(html.Div(children, className="cache-row"))
    return html.Div(items)


def ai_assignment_list() -> html.Div:
    data = sorted(journal_router.load_assignments().values(), key=lambda a: a.get("decided_at", ""), reverse=True)
    if not data:
        return html.Div("아직 AI가 분야를 판단한 저널이 없습니다.", className="panel-hint")
    rows = []
    for a in data:
        names = ", ".join(DOMAINS[k].name for k in a.get("domains", []) if k in DOMAINS) or "해당 분야 없음"
        rows.append(html.Div([
            html.Div(a["name"], style={"fontWeight": 600, "fontSize": "13.5px"}),
            html.Div(f"{', '.join(a.get('issns_input', []))} → {names}", style={"fontSize": "12.5px", "marginTop": "2px"}),
            html.Div(f"{a.get('reason', '')} ({a.get('model', '')}, {a.get('decided_at', '')[:10]})",
                     style={"color": "var(--ink-soft)", "fontSize": "12px", "marginTop": "2px"}),
        ], className="cache-row"))
    return html.Div([html.Div("판단이 틀렸으면 dataset/ai_journal_assignments.json에서 domains를 고치세요.",
                              className="panel-hint")] + rows)


@app.callback(
    Output("journal-ref-offcanvas", "is_open"),
    Output("journal-ref-offcanvas", "title"),
    Output("journal-ref-content", "children"),
    Input("journal-ref-btn", "n_clicks"),
    State("journal-ref-offcanvas", "is_open"),
    State("collect-domain", "value"),
    prevent_initial_call=True,
)
def toggle_journal_reference(n_clicks, is_open, mode):
    if mode == AUTO:
        return not is_open, "AI가 분야를 판단한 저널", ai_assignment_list()
    s = svc.get(mode)
    return not is_open, f"{s.d.name} 분야 수집 대상 저널 ({s.d.department})", reference_journal_list(s)


@app.callback(
    Output("collect-issn-textarea", "value", allow_duplicate=True),
    Output("journal-ref-offcanvas", "is_open", allow_duplicate=True),
    Input("journal-ref-select-all-btn", "n_clicks"),
    State("collect-domain", "value"),
    prevent_initial_call=True,
)
def select_all_reference_issns(n_clicks, mode):
    if not n_clicks or mode == AUTO:
        return no_update, no_update
    return _domain_issn_lines(DOMAINS[mode]), False


@app.callback(
    Output("collect-result", "children", allow_duplicate=True),
    Output("collect-resume-btn", "style", allow_duplicate=True),
    Output("collect-resume-info", "children", allow_duplicate=True),
    Input("collect-stop-btn", "n_clicks"),
    State("collect-domain", "value"),
    prevent_initial_call=True,
)
def on_collect_stop(n_clicks, mode):
    """중지 후 안내. 수집 기록은 저널 하나 끝날 때마다 저장되므로, 받던 저널부터 '다시 시작하기'로 이어 받을 수 있다."""
    if not n_clicks:
        return no_update, no_update, no_update
    msg = html.Div("수집을 중지했습니다. 이미 받은 연도는 저장되어 있습니다. '다시 시작하기'를 누르면 받던 저널부터 "
                   "빠진 연도만 이어서 받습니다.", className="panel-warn")
    return msg, *_resume_state(mode)


def _resume_state(job_key: str) -> tuple[dict, html.Div]:
    """'다시 시작하기' 버튼 표시 여부(style)와 그 아래 안내문. 지난 수집 기록(서버 파일)에 덜 끝난
    저널이 남아 있을 때만 버튼을 보여준다."""
    hidden = {"marginLeft": "8px", "display": "none"}
    job = svc.load_collect_job(job_key)
    pending = (job or {}).get("pending") or []
    if not pending:
        if job:
            return hidden, html.Div(f"지난 수집({job['year_start']}–{job['year_end']})은 모두 끝났습니다.",
                                    className="panel-hint", style={"marginBottom": "0"})
        return hidden, html.Div()
    text = (f"지난 수집에서 미완료 {len(pending)}종이 남아 있습니다 ({job['year_start']}–{job['year_end']}). "
            "'다시 시작하기'를 누르면 입력칸과 상관없이 이 저널들만 중단된 연도부터 이어서 받습니다.")
    reset_iso = job.get("reset_at")
    if reset_iso and datetime.fromisoformat(reset_iso) > datetime.now(timezone.utc):
        text += f" (OpenAlex 한도 초기화: {_format_kst(reset_iso)})"
    return {"marginLeft": "8px"}, html.Div(text, className="panel-hint", style={"marginBottom": "0"})


def _collect_into(s: svc.DomainService, issns: list[str], year_start: int, year_end: int, emit) -> tuple:
    """저널 하나를 분야 하나에 수집. 반환: (kind, 결과 문구, 수집된 df, metrics)
    kind: ok / partial / fail / notfound (rate limit 여부는 metrics["rate_limited"])"""
    label = ", ".join(issns)
    final = s.run_collection_sync(issns, year_start, year_end, emit)
    metrics = final.get("metrics", {}) if final else {}
    df = final.get("df") if final else None
    total = metrics.get("total_collected")
    total_txt = f"{total:,}건" if isinstance(total, int) else "알 수 없음"
    failed = metrics.get("failed_years") or []
    if metrics.get("not_found"):
        # ISSN 자체가 틀린 경우라 다시 시도해도 같으므로 '다시 시작하기' 대상에서 뺀다.
        return "notfound", f"✗ {label}: OpenAlex에서 이 ISSN의 저널을 찾지 못함 (ISSN 확인 필요)", df, metrics
    if failed and not total:
        return "fail", f"✗ {label}: 수집 실패 — {metrics.get('error')}", df, metrics
    if failed:
        return "partial", f"⚠ {label}: 총 {total_txt} (기존 수집분 포함) — {_year_ranges(failed)}년 못 받음", df, metrics
    return "ok", f"✓ {label}: 총 {total_txt} (기존 수집분 포함)", df, metrics


@app.callback(
    Output("collect-result", "children"),
    Output("collect-metadata-wrap", "children"),
    Output("collect-resume-btn", "style"),
    Output("collect-resume-info", "children"),
    Input("collect-start-btn", "n_clicks"),
    Input("collect-resume-btn", "n_clicks"),
    State("collect-issn-textarea", "value"),
    State("collect-year-start", "value"),
    State("collect-year-end", "value"),
    State("collect-domain", "value"),
    background=True,
    manager=background_manager,
    progress=Output("collect-log", "children"),
    running=[(Output("collect-start-btn", "disabled"), True, False),
             (Output("collect-resume-btn", "disabled"), True, False),
             (Output("collect-domain", "disabled"), True, False),
             (Output("collect-stop-btn", "disabled"), False, True)],
    # '중지'를 누르면 수집 작업 프로세스를 멈춘다. 이미 저장된 연도는 남고, 받던 연도는 저장되지 않는다
    # (연도 단위로 끝까지 받은 것만 저장하므로 반쯤 받은 파일은 생기지 않음).
    cancel=[Input("collect-stop-btn", "n_clicks")],
    prevent_initial_call=True,
)
def start_collection(set_progress, n_start, n_resume, issn_text, year_start, year_end, mode):
    """'수집 시작'은 입력칸의 ISSN·연도로, '다시 시작하기'는 지난 수집 기록에 남은 미완료 저널만
    지난 연도 범위로 다시 돌린다. 어느 쪽이든 이미 수집된 연도는 건너뛰므로 중단된 연도부터
    이어진다. 끝나면 덜 끝난 저널(⚠/✗/⏸)을 서버 파일에 기록해 다음 '다시 시작하기'에 쓴다.
    mode가 AUTO면 저널마다 journal_router.route()로 저장할 분야를 정해 그 분야(들)에 받는다."""
    resume = ctx.triggered_id == "collect-resume-btn"
    log_lines: list[str] = []

    def emit(msg: str):
        log_lines.append(msg)
        set_progress("\n".join(log_lines[-500:]))

    if resume:
        job = svc.load_collect_job(mode) or {}
        groups = job.get("pending") or []
        if not groups:
            emit("이어서 받을 미완료 저널이 없습니다.")
            return (html.Div("이어서 받을 미완료 저널이 없습니다.", className="panel-hint"), no_update,
                    *_resume_state(mode))
        year_start, year_end = job["year_start"], job["year_end"]
        emit(f"[다시 시작하기] 지난 수집의 미완료 {len(groups)}종을 {year_start}–{year_end} 범위로 이어서 받습니다.")
    else:
        groups = [[x.strip() for x in line.split(",") if x.strip()]
                  for line in (issn_text or "").splitlines() if line.strip()]
        if not groups:
            emit("ISSN이 입력되지 않았습니다.")
            return html.Div("ISSN을 입력하세요.", className="panel-warn"), no_update, no_update, no_update
        if year_start is None or year_end is None or int(year_start) > int(year_end):
            emit("연도 범위가 올바르지 않습니다.")
            return html.Div("시작/끝 연도를 확인하세요.", className="panel-warn"), no_update, no_update, no_update
        year_start, year_end = int(year_start), int(year_end)
        job = {"groups": groups, "year_start": year_start, "year_end": year_end}

    if mode == AUTO:
        emit("[AI 자동 분류] 저널마다 저장할 분야를 정한 뒤 수집합니다.")
    else:
        emit(f"[{DOMAINS[mode].name}] 저장 위치: {svc.get(mode).dataset_dir_label()}")
    results = []   # (kind, 문구, issns)
    collected_dfs = []
    stopped_by_limit = False
    reset_at = None

    def save_progress(done_count: int):
        # 저널 하나 끝날 때마다 기록: 중간에 오류/화면 이동으로 멈춰도 아직 안 한 저널 + 덜 끝난 저널이 남는다.
        unfinished = [g for kind, _, g in results if kind in ("partial", "fail", "skip")] + groups[done_count:]
        pending = [list(g) for g in dict.fromkeys(tuple(g) for g in unfinished)]
        job.update(pending=pending, updated_at=datetime.now(timezone.utc).isoformat(), reset_at=reset_at)
        svc.save_collect_job(mode, job)
        return pending

    for idx, issns in enumerate(groups):
        label = ", ".join(issns)
        if stopped_by_limit:
            results.append(("skip", f"⏸ {label}: 건너뜀 (앞에서 OpenAlex 요청 한도 초과로 수집 중단)", issns))
            continue

        targets = [mode]
        if mode == AUTO:
            try:
                r = journal_router.route(issns)
            except Exception as e:
                emit(f"[error] {label} 분야 판단 실패: {e}")
                results.append(("fail", f"✗ {label}: 분야 판단 실패 — {e}", issns))
                save_progress(idx + 1)
                continue
            names = ", ".join(DOMAINS[k].name for k in r["domains"]) or "해당 분야 없음"
            emit(f"[분야 판단 · {r['by']}] {r['name']} -> {names} ({r['reason']})")
            targets = r["domains"]
            if not targets:
                kind = "notfound" if r["by"] == "미등록" else "nodomain"
                results.append((kind, f"✗ {label} ({r['name']}): {r['reason'] if kind == 'notfound' else '맞는 분야가 없어 수집하지 않음'}", issns))
                save_progress(idx + 1)
                continue

        for key in targets:
            s = svc.get(key)
            prefix = f"[{s.d.name}] " if mode == AUTO else ""
            emit(f"=== {prefix}{label} 수집 시작 ===")
            try:
                kind, text, df, metrics = _collect_into(s, issns, year_start, year_end, emit)
            except Exception as e:
                # 예상 못 한 오류는 이 저널만 실패로 두고 다음으로 넘어간다(이미 저장된 연도는 남아 있음).
                emit(f"[error] {label} 수집 중 오류: {e}")
                results.append(("fail", f"✗ {prefix}{label}: 수집 중 오류 — {e}", issns))
                continue
            if df is not None and not df.empty:
                collected_dfs.append(df)
            results.append((kind, prefix + text, issns))
            emit(f"=== {prefix}{label} 완료 ===\n")
            if metrics.get("rate_limited"):
                stopped_by_limit = True
                reset_at = metrics.get("reset_at") or reset_at
                emit("OpenAlex 요청 한도 초과 — 남은 저널은 건너뜁니다.")
                break
        save_progress(idx + 1)

    pending = save_progress(len(groups))

    emit("전체 수집 완료." if not stopped_by_limit else "요청 한도 초과로 수집을 중단했습니다.")
    combined = pd.concat(collected_dfs, ignore_index=True) if collected_dfs else pd.DataFrame()
    colors = {"ok": "var(--ink)", "partial": "#8a5a00", "fail": "#b3261e", "notfound": "#b3261e",
              "skip": "var(--ink-soft)", "nodomain": "var(--ink-soft)"}
    lines = [html.Div(text, style={"color": colors[kind]}) for kind, text, _ in results]
    if pending and not stopped_by_limit:
        lines.append(html.Div(
            "'다시 시작하기'를 누르면 위에서 덜 끝난 저널만 빠진 연도부터 이어서 받습니다.",
            className="panel-hint", style={"marginTop": "8px", "marginBottom": "0"}))
    blocks = []
    if stopped_by_limit:
        blocks.append(_rate_limit_banner(reset_at))
    blocks.append(html.Div(lines, className="info-card"))
    return (html.Div(blocks), metadata_table(combined), *_resume_state(mode))


_KST = timezone(timedelta(hours=9))


def _format_kst(reset_at_iso: str) -> str:
    """UTC ISO 시각 -> "10월 9일 오전 9:00 (약 14시간 5분 후)" """
    reset_kst = datetime.fromisoformat(reset_at_iso).astimezone(_KST)
    left = max(reset_kst - datetime.now(_KST), timedelta(0))
    hours, minutes = int(left.total_seconds() // 3600), int(left.total_seconds() % 3600 // 60)
    ampm = "오전" if reset_kst.hour < 12 else "오후"
    hour12 = reset_kst.hour % 12 or 12
    return (f"{reset_kst.month}월 {reset_kst.day}일 {ampm} {hour12}:{reset_kst.minute:02d}"
            f" (약 {hours}시간 {minutes}분 후)")


def _rate_limit_banner(reset_at_iso: str | None) -> html.Div:
    """OpenAlex 한도 초과로 수집을 멈췄을 때 결과 위에 띄우는 경고. 하루 한도 소진이 확인됐으면
    응답 헤더의 초기화 시각을 한국시간으로, 아니면(일시적 429) 잠시 뒤 다시 하라고 안내한다."""
    tail = " '다시 시작하기'를 누르면 덜 끝난 저널만 중단된 연도부터 이어서 받습니다."
    if reset_at_iso:
        head = "⚠️ 오늘 OpenAlex 요청 한도를 모두 사용해 수집을 멈췄습니다."
        body = f"한도는 {_format_kst(reset_at_iso)}에 초기화됩니다. 그 이후" + tail
    else:
        head = "⚠️ OpenAlex가 요청을 계속 거절(429)해 수집을 멈췄습니다."
        body = "일시적인 제한일 수 있으니 잠시 뒤" + tail
    return html.Div([html.B(head), html.Div(body, style={"marginTop": "4px"})],
                    className="panel-warn", style={"fontSize": "13px"})


def _year_ranges(years: list[int]) -> str:
    """[2021, 2022, 2023, 2025] -> "2021–2023, 2025" """
    years = sorted(set(int(y) for y in years))
    parts, start = [], None
    for i, y in enumerate(years):
        if start is None:
            start = y
        if i == len(years) - 1 or years[i + 1] != y + 1:
            parts.append(str(start) if start == y else f"{start}–{y}")
            start = None
    return ", ".join(parts)


# ================================================================== 탭 3: 전체분석
FULL_NOTICE = {
    "stale": "지난 분석 이후 새 논문이 수집되었습니다. '분석하기'를 누르면 새 데이터로 다시 분석해 결과를 보여줍니다(몇 분 걸립니다).",
    "new": "아직 이 분야의 글로벌 트렌드 분석 결과가 없습니다. '분석하기'를 누르면 분석해 결과를 보여줍니다(몇 분 걸립니다).",
    "no_data": "이 분야에 수집된 논문이 없습니다. 먼저 '데이터 수집' 화면에서 수집하세요.",
}


def _full_status_text(summary: dict) -> str:
    return (f"Rising {summary['n_rising']:,}개 \u2192 트렌드 후보 {summary['clusters']}개 "
            f"(modularity={float(summary['modularity']):.3f})")


def build_tab_full(s: svc.DomainService):
    """분야 화면을 열면 저장된 최신 결과를 바로 보여준다. 지난 분석 이후 새 논문이 수집됐으면(또는 분석한 적이
    없으면) 결과 대신 안내문과 '분석하기' 버튼만 보여주고, 누르면 실제로 분석해서 결과를 띄운다."""
    state, summary = s.full_status()
    ready = state == "ready"
    intro = [html.Div(
        "수집된 논문 제목에서 Rising 용어를 추출하고 함께 쓰이는 관계로 트렌드 후보(커뮤니티)를 도출한 결과입니다. "
        "연구주제 이름은 전문가 검수 전 AI가 붙인 참고용 라벨입니다.",
        className="panel-hint", style={"marginBottom": "8px"},
    )]
    return html.Div(intro + [
        html.Div(FULL_NOTICE.get(state, ""), id="full-notice",
                 className="panel-warn" if not ready else "", style={"marginBottom": "10px"} if not ready else {}),
        html.Div([
            dbc.Button("분석하기", id="full-run-btn", className="btn-primary",
                       style={"display": "none"} if ready or state == "no_data" else {"marginRight": "12px"}),
            html.Span(_full_status_text(summary) if ready else "", id="full-run-status", className="status-chip"),
        ], className="mb-3"),
        html.Div(dcc.Loading(html.Pre(id="full-log", className="log-box", style={"height": "90px"})),
                 id="full-log-wrap", style={"display": "none"}),
        dbc.Row([
            dbc.Col(panel(
                "트렌드 후보",
                hint="원 크기 = 용어 수. 노드를 클릭하면 오른쪽에 해당 커뮤니티의 단어 목록이 나타납니다.",
                children=[
                    html.Div([cyto.Cytoscape(
                        id="full-cytoscape", layout=CYTO_LAYOUT,
                        style={"width": "100%", "height": "620px"},
                        stylesheet=CYTO_STYLESHEET, elements=s.full_cluster_graph_elements() if ready else [],
                    ), dbc.Button("원래대로", id="full-cyto-reset", size="sm", className="cyto-reset-btn",
                                  title="확대·이동한 그림을 처음 배치로 되돌립니다")], className="cyto-wrap"),
                    html.Div(legend_block(s.full_cluster_legend()) if ready else None, id="full-legend"),
                ],
            ), width=6),
            dbc.Col(panel(
                "커뮤니티 상세",
                hint="노드를 클릭하면 그 커뮤니티의 단어가 나옵니다. 체크한 단어는 다른 커뮤니티를 눌러도 유지됩니다.",
                children=[
                    html.Div(id="full-cluster-info", className="mb-2"),
                    html.Div(id="full-term-table-wrap"),
                ],
            ), width=6),
        ], className="g-3"),
        panel(
            "선택한 단어 (여러 커뮤니티 합산)",
            children=[
                html.Div(id="full-selection-summary", className="mb-2"),
                html.Div([
                    dbc.Button("전체 선택 초기화", id="full-clear-selection-btn", size="sm", className="btn-secondary me-2"),
                    dbc.Button("AI 해석 보기", id="full-ai-insight-btn", size="sm", className="btn-secondary"),
                ]),
                dcc.Loading(html.Div(id="full-ai-insight-wrap", className="mt-2")),
            ],
        ),
        panel("선택한 단어가 들어간 논문", children=[html.Div(id="full-papers-wrap")]),
        dcc.Store(id="full-selected-cluster"),
        dcc.Store(id="full-selected-words", data=[]),
    ], className="section-gap")


@app.callback(
    Output("full-run-status", "children"),
    Output("full-cytoscape", "elements"),
    Output("full-cytoscape", "layout"),
    Output("full-legend", "children"),
    Output("full-selected-words", "data", allow_duplicate=True),
    Output("full-run-btn", "style"),
    Output("full-notice", "children"),
    Output("full-notice", "className"),
    Input("full-run-btn", "n_clicks"),
    State("domain-store", "data"),
    background=True,
    manager=background_manager,
    progress=Output("full-log", "children"),
    running=[(Output("full-run-btn", "disabled"), True, False),
             (Output("full-log-wrap", "style"), {}, {})],
    cancel=CANCEL_ON_DOMAIN_SWITCH,
    prevent_initial_call=True,
)
def run_full_analysis(set_progress, n_clicks, domain):
    s = svc.get(domain)
    log_lines: list[str] = []

    def emit(msg: str):
        log_lines.append(msg)
        set_progress("\n".join(log_lines[-300:]))

    try:
        result = s.run_full_pipeline(emit)
    except Exception as e:
        emit(f"[오류] {e}")
        return f"분석 실패: {e}", no_update, no_update, no_update, no_update, no_update, no_update, no_update
    elements = s.full_cluster_graph_elements()
    layout = cyto_layout(n_clicks)
    legend = legend_block(s.full_cluster_legend())
    return _full_status_text(result), elements, layout, legend, [], {"display": "none"}, "", ""


@app.callback(
    Output("full-cluster-info", "children"),
    Output("full-term-table-wrap", "children"),
    Output("full-selected-cluster", "data"),
    Input("full-cytoscape", "tapNodeData"),
    State("full-selected-words", "data"),
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def on_full_node_click(node_data, selected_words, domain):
    if not node_data:
        return no_update, no_update, no_update
    s = svc.get(domain)
    cid = node_data["cluster"]
    label = s.full_cluster_label(cid)
    note = "연구주제 이름: AI 생성(참고용, 전문가 검수 전)"
    info = html.Div([
        cluster_info_card(label, node_data["term_count"], f"{_n_years(s)}년 평균 문서비중", node_data["mean_share"],
                          note=note),
        ai_insight_block(ai_insights.full_insight(s, cid), chat_button=True),
    ])
    terms_df = s.full_cluster_terms(cid)
    already = set(selected_words or [])
    initial_selected = [i for i, t in enumerate(terms_df["term"]) if t in already]
    table = term_table(terms_df, "full-term-table", "document_count", "포함 논문 수",
                        selectable=True, initial_selected=initial_selected)
    return info, table, cid


@app.callback(
    Output("full-selected-words", "data", allow_duplicate=True),
    Input("full-term-table", "selected_rows"),
    State("full-term-table", "data"),
    State("full-selected-words", "data"),
    prevent_initial_call=True,
)
def merge_full_selection(selected_rows, table_data, selected_words):
    """구간별분석 탭과 같은 방식: 지금 보이는 커뮤니티 표의 체크 상태를 다른 커뮤니티에서
    이미 골라둔 단어와 합친다."""
    if table_data is None:
        return no_update
    current_table_terms = {row["term"] for row in table_data}
    checked_now = {table_data[i]["term"] for i in (selected_rows or [])}
    kept_from_others = [t for t in (selected_words or []) if t not in current_table_terms]
    return kept_from_others + sorted(checked_now)


@app.callback(
    Output("full-selected-words", "data", allow_duplicate=True),
    Input("full-clear-selection-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_full_selection(n_clicks):
    return []


@app.callback(
    Output("full-selection-summary", "children"),
    Output("full-papers-wrap", "children"),
    Output("full-ai-insight-wrap", "children", allow_duplicate=True),
    Input("full-selected-words", "data"),
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def on_full_selection_change(selected_words, domain):
    if not selected_words:
        empty = html.Div("아직 선택된 단어가 없습니다. 왼쪽 그래프에서 커뮤니티를 클릭하고 단어를 체크하세요.",
                          className="panel-hint")
        return empty, empty, html.Div()
    s = svc.get(domain)
    terms_df = s.load_full_terms()
    term_to_cluster = dict(zip(terms_df["term"], terms_df["cluster"]))
    chips = html.Div([
        html.Span(f"{t} (C{term_to_cluster.get(t, '?')})", className="status-chip",
                   style={"marginRight": "6px", "marginBottom": "6px"})
        for t in selected_words
    ])
    papers = s.papers_for_terms(selected_words)
    # 선택이 바뀌면 이전 선택에 대한 AI 해석은 더 이상 맞지 않으므로 비운다(다시 눌러야 보임).
    return chips, paper_table(papers, "full-papers-table"), html.Div()


@app.callback(
    Output("full-ai-insight-wrap", "children"),
    Input("full-ai-insight-btn", "n_clicks"),
    State("full-selected-words", "data"),
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def on_full_ai_insight_click(n_clicks, selected_words, domain):
    if not selected_words:
        return html.Div("먼저 왼쪽에서 단어를 선택하세요.", className="panel-warn")
    s = svc.get(domain)
    papers = s.papers_for_terms(selected_words, limit=8)["display_name"].tolist()
    try:
        text = ai_insights.generate_live_insight_full(s, selected_words, papers)
    except RuntimeError as e:
        return html.Div(str(e), className="panel-warn")
    except Exception as e:
        return html.Div(f"AI 해석 생성 중 오류가 발생했습니다: {e}", className="panel-warn")
    return ai_insight_block(text)


# ================================================================== 탭 4: 구간별분석
def build_tab_windowed(s: svc.DomainService):
    windows = s.windows()
    labels = s.window_labels()
    return html.Div([
        html.Div(
            "5~6년 단위로 나눠서 주요 연구주제를 탐색합니다. (데이터의 연도 범위가 넓어지면 "
            "구간 수는 자동으로 조정됩니다 - 항상 5개 구간으로 고정되는 것은 아닙니다.)",
            className="panel-hint", style={"marginBottom": "8px"},
        ),
        html.Div([
            dbc.Button("분석 진행하기", id="windowed-run-btn", className="btn-primary"),
        ], className="mb-3"),
        dcc.Loading(html.Pre(id="windowed-log", className="log-box", style={"height": "90px"})),
        html.Div([
            html.Label("구간 선택", style={"fontSize": "11px", "color": "var(--ink-soft)", "display": "block"}),
            dcc.Dropdown(
                id="window-select",
                options=[{"label": labels[w], "value": w} for w in windows],
                value=windows[0], clearable=False, style={"width": "220px"},
            ),
        ], className="mb-3 mt-2"),
        dbc.Row([
            dbc.Col(panel(
                "구간 내 용어 구조 네트워크",
                children=[
                    html.Div("\u26a0\ufe0f 통계적 유의성 검정 없이 '이 구간에서 증가 방향(tau>0)'만으로 뽑은 용어입니다 "
                             "\u2014 \"부상 주제\"가 아니라 \"그 구간에서 상대적으로 늘어난 용어의 구조\"로 해석하세요.",
                             className="panel-warn"),
                    html.Div([cyto.Cytoscape(
                        id="windowed-cytoscape", layout=CYTO_LAYOUT,
                        style={"width": "100%", "height": "560px"},
                        stylesheet=CYTO_STYLESHEET, elements=[],
                    ), dbc.Button("원래대로", id="windowed-cyto-reset", size="sm", className="cyto-reset-btn",
                                  title="확대·이동한 그림을 처음 배치로 되돌립니다")], className="cyto-wrap"),
                    html.Div(id="windowed-legend"),
                ],
            ), width=6),
            dbc.Col([
                panel(
                    "구간별 연구흐름 (AI 해석)",
                    hint=f"\"분석 진행하기\"를 누르면 {len(windows)}개 구간 전체를 한 번에 비교해서 생성됩니다.",
                    children=[
                        html.Div(id="windowed-trajectory-overview", className="mb-2"),
                        html.Div(id="windowed-trajectory-current"),
                    ],
                ),
                panel(
                    "커뮤니티 상세",
                    hint="노드를 클릭하면 그 커뮤니티의 단어가 나옵니다. 체크한 단어는 다른 커뮤니티를 눌러도 유지됩니다.",
                    children=[
                        html.Div(id="windowed-cluster-info", className="mb-2"),
                        html.Div(id="windowed-term-table-wrap"),
                    ],
                ),
            ], width=6),
        ], className="g-3"),
        panel(
            "선택한 단어 (여러 커뮤니티 합산)",
            children=[
                html.Div(id="windowed-selection-summary", className="mb-2"),
                html.Div([
                    dbc.Button("전체 선택 초기화", id="windowed-clear-selection-btn", size="sm", className="btn-secondary me-2"),
                    dbc.Button("AI 해석 보기", id="windowed-ai-insight-btn", size="sm", className="btn-secondary"),
                ]),
                dcc.Loading(html.Div(id="windowed-ai-insight-wrap", className="mt-2")),
            ],
        ),
        panel("선택한 단어가 들어간 논문 (해당 구간)", children=[html.Div(id="windowed-papers-wrap")]),
        dcc.Store(id="windowed-selected-cluster"),
        dcc.Store(id="windowed-selected-words", data=[]),
        dcc.Store(id="windowed-has-run", data=False),
        dcc.Store(id="windowed-trajectory-data"),
    ], className="section-gap")


@app.callback(
    Output("windowed-trajectory-overview", "children"),
    Output("windowed-trajectory-current", "children"),
    Input("windowed-trajectory-data", "data"),
    Input("window-select", "value"),
)
def render_trajectory_blocks(data, window):
    """"분석 진행하기" 전에는 windowed-trajectory-data가 비어 있으므로 아무것도 보여주지 않는다."""
    if not data:
        return html.Div(), html.Div()
    overview = ai_insight_block({"ko": data.get("overview", "")})
    current = ai_insight_block({"ko": data.get("by_window", {}).get(window, "")})
    return overview, current


@app.callback(
    Output("windowed-cytoscape", "elements"),
    Output("windowed-cytoscape", "layout"),
    Output("windowed-legend", "children"),
    Output("windowed-selected-words", "data", allow_duplicate=True),
    Input("window-select", "value"),
    State("windowed-has-run", "data"),
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def load_window_graph(window, has_run, domain):
    """구간을 바꿔도, 이번 세션에서 "분석 진행하기"를 한 번이라도 눌렀을 때만 보여준다.
    아직 실행 전이면 빈 화면을 유지해 실행하지 않은 결과가 먼저 보이지 않게 한다."""
    if not has_run:
        return [], cyto_layout(window), html.Div(), []
    s = svc.get(domain)
    try:
        elements = s.window_cluster_graph_elements(window)
        legend = legend_block(s.window_cluster_legend(window))
    except FileNotFoundError:
        elements, legend = [], html.Div()
    return elements, cyto_layout(window), legend, []


@app.callback(
    Output("windowed-cytoscape", "elements", allow_duplicate=True),
    Output("windowed-cytoscape", "layout", allow_duplicate=True),
    Output("windowed-legend", "children", allow_duplicate=True),
    Output("windowed-selected-words", "data", allow_duplicate=True),
    Output("windowed-has-run", "data"),
    Output("windowed-trajectory-data", "data"),
    Output("window-select", "options"),
    Output("window-select", "value"),
    Input("windowed-run-btn", "n_clicks"),
    State("window-select", "value"),
    State("domain-store", "data"),
    background=True,
    manager=background_manager,
    progress=Output("windowed-log", "children"),
    running=[(Output("windowed-run-btn", "disabled"), True, False)],
    cancel=CANCEL_ON_DOMAIN_SWITCH,
    prevent_initial_call=True,
)
def run_windowed_analysis(set_progress, n_clicks, window, domain):
    """그래프 갱신을 파이프라인 실행과 분리된 별도 콜백으로 두면, 버튼 클릭 시 두 콜백이
    동시에 발동해 파이프라인이 끝나기도 전에(디스크의 예전 결과로) 그래프가 먼저 떠 버린다.
    그래서 전체분석 탭처럼 파이프라인 실행과 그래프 갱신을 하나의 백그라운드 콜백으로 합쳐서,
    run_windowed_pipeline()이 완전히 끝난 뒤에만 그래프를 그리도록 한다.

    windowed-log는 여기서는 progress 채널로만 갱신한다(전체분석 탭과 동일한 방식) - 같은
    Output을 이 콜백의 "최종 반환값"에도 넣으면 진행 중 progress 스트리밍이 반영되지 않고
    완료될 때까지 로딩 스피너만 보이는 문제가 있었다.

    구간별 연구흐름(windowed-trajectory-data)도 이 안에서 같이 생성한다 - 전체 구간을 통째로
    비교하는 내용이라 "분석 진행하기"를 누른 이 실행의 데이터로만 의미가 있고, 새 데이터로
    다시 분석할 때마다 자동으로 다시 생성돼야 하기 때문이다(파일로 저장하지 않음).

    구간 목록(window-select의 options/value)도 여기서 다시 계산해서 갱신한다 - 데이터의 연도
    범위가 달라지면 run_windowed_pipeline()이 실제 범위를 보고 구간을 다시 나누므로(5~6년 단위),
    드롭다운도 재실행 즉시 그 결과를 반영해야 한다. 기존에 선택했던 구간이 새 목록에도 그대로
    있으면 유지하고, 없으면(구간 경계가 달라진 경우) 첫 구간으로 둔다."""
    s = svc.get(domain)
    log_lines: list[str] = []

    def emit(msg: str):
        log_lines.append(msg)
        set_progress("\n".join(log_lines[-300:]))

    try:
        s.run_windowed_pipeline(emit)
    except Exception as e:
        emit(f"[오류] {e}")
        return (no_update,) * 8

    new_windows = s.windows()
    new_options = [{"label": s.window_labels()[w], "value": w} for w in new_windows]
    selected_window = window if window in new_windows else new_windows[0]

    elements = s.window_cluster_graph_elements(selected_window)
    legend = legend_block(s.window_cluster_legend(selected_window))

    trajectory_data = None
    try:
        emit("구간별 연구흐름 비교 생성 중...")
        trajectory_data = ai_insights.generate_trajectory_overview(s)
        emit("구간별 연구흐름 생성 완료")
    except Exception as e:
        emit(f"구간별 연구흐름 생성 실패(건너뜀): {e}")

    return (elements, cyto_layout(n_clicks), legend, [], True, trajectory_data,
            new_options, selected_window)


@app.callback(
    Output("windowed-cluster-info", "children"),
    Output("windowed-term-table-wrap", "children"),
    Output("windowed-selected-cluster", "data"),
    Input("windowed-cytoscape", "tapNodeData"),
    State("window-select", "value"),
    State("windowed-selected-words", "data"),
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def on_window_node_click(node_data, window, selected_words, domain):
    if not node_data:
        return no_update, no_update, no_update
    s = svc.get(domain)
    cid = node_data["cluster"]
    terms_df = s.window_cluster_terms(window, cid)
    top_terms = terms_df["term"].head(15).tolist()

    insight_box = html.Div()
    try:
        papers = s.window_cluster_papers(window, top_terms, limit=8)["display_name"].tolist()
        topic = ai_insights.generate_live_topic(s, top_terms, papers)
        title = topic["title"]
        insight_box = ai_insight_block({"ko": topic["body"]})
    except Exception:
        # 키가 없거나 호출에 실패해도 화면이 비지 않도록, 대표 용어를 그대로 이어붙인 제목으로 대체한다.
        title = ", ".join(terms_df["term"].head(4).tolist())

    info = html.Div([
        cluster_info_card(title, node_data["term_count"], "구간 평균 문서비중", node_data["mean_share"]),
        insight_box,
    ])
    already = set(selected_words or [])
    initial_selected = [i for i, t in enumerate(terms_df["term"]) if t in already]
    table = term_table(terms_df, "windowed-term-table", "전체기간_평균비중(%)", "구간 내 평균비중(%)",
                        selectable=True, initial_selected=initial_selected)
    return info, table, cid


@app.callback(
    Output("windowed-selected-words", "data", allow_duplicate=True),
    Input("windowed-term-table", "selected_rows"),
    State("windowed-term-table", "data"),
    State("windowed-selected-words", "data"),
    prevent_initial_call=True,
)
def merge_window_selection(selected_rows, table_data, selected_words):
    """현재 보이는 커뮤니티 표의 체크 상태를, 다른 커뮤니티에서 이미 골라둔 단어와 합친다.
    현재 표에 있는 단어들만 이번 체크 상태로 갱신하고, 다른 커뮤니티의 기존 선택은 그대로 둔다."""
    if table_data is None:
        return no_update
    current_table_terms = {row["term"] for row in table_data}
    checked_now = {table_data[i]["term"] for i in (selected_rows or [])}
    kept_from_others = [t for t in (selected_words or []) if t not in current_table_terms]
    return kept_from_others + sorted(checked_now)


@app.callback(
    Output("windowed-selected-words", "data", allow_duplicate=True),
    Input("windowed-clear-selection-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_window_selection(n_clicks):
    return []


@app.callback(
    Output("windowed-selection-summary", "children"),
    Output("windowed-papers-wrap", "children"),
    Output("windowed-ai-insight-wrap", "children", allow_duplicate=True),
    Input("windowed-selected-words", "data"),
    State("window-select", "value"),
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def on_window_selection_change(selected_words, window, domain):
    if not selected_words:
        empty = html.Div("아직 선택된 단어가 없습니다. 왼쪽 그래프에서 커뮤니티를 클릭하고 단어를 체크하세요.",
                          className="panel-hint")
        return empty, empty, html.Div()
    s = svc.get(domain)
    terms_df = s.load_window_terms(window)
    term_to_cluster = dict(zip(terms_df["term"], terms_df["cluster"]))
    chips = html.Div([
        html.Span(f"{t} (C{term_to_cluster.get(t, '?')})", className="status-chip",
                   style={"marginRight": "6px", "marginBottom": "6px"})
        for t in selected_words
    ])
    papers = s.window_cluster_papers(window, selected_words)
    # 선택이 바뀌면 이전 선택에 대한 AI 해석은 더 이상 맞지 않으므로 비운다(다시 눌러야 보임).
    return chips, paper_table(papers, "windowed-papers-table"), html.Div()


@app.callback(
    Output("windowed-ai-insight-wrap", "children"),
    Input("windowed-ai-insight-btn", "n_clicks"),
    State("windowed-selected-words", "data"),
    State("window-select", "value"),
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def on_window_ai_insight_click(n_clicks, selected_words, window, domain):
    """선택된 단어로 그 순간 AI(Gemma)를 호출한다(저장 안 함) - 구간별 커뮤니티는 재분석 때마다
    구성이 달라질 수 있어서, 미리 만들어두지 않고 매번 지금 선택을 그대로 보낸다."""
    if not selected_words:
        return html.Div("먼저 왼쪽에서 단어를 선택하세요.", className="panel-warn")
    s = svc.get(domain)
    papers = s.window_cluster_papers(window, selected_words, limit=8)["display_name"].tolist()
    try:
        text = ai_insights.generate_live_insight(s, selected_words, papers)
    except RuntimeError as e:
        return html.Div(str(e), className="panel-warn")
    except Exception as e:
        return html.Div(f"AI 해석 생성 중 오류가 발생했습니다: {e}", className="panel-warn")
    return ai_insight_block(text)


# ================================================================== 탭 5: 키워드 검색
def _keyword_hint(s: svc.DomainService) -> str:
    n = s.n_full_rising_terms()
    if n:
        return (f"입력하면 실제 Rising 용어 사전(글로벌 트렌드 분석 {n:,}개 + 구간별 용어)에서 자동완성됩니다. "
                "여러 개 선택할 수 있습니다.")
    return ("아직 이 분야의 글로벌 트렌드 분석 결과가 없습니다. '1. 글로벌 트렌드 분석'에서 "
            "'분석 진행하기'를 먼저 실행하면 그 결과 용어로 자동완성됩니다.")


def build_tab_keyword(s: svc.DomainService):
    return html.Div([
        panel(
            "검색",
            children=[
                html.Div(_keyword_hint(s), id="keyword-hint", className="panel-hint"),
                dcc.Dropdown(
                    id="keyword-input",
                    options=[{"label": t, "value": t} for t in s.all_known_terms()],
                    multi=True, placeholder="예: microplastics, lstm, drought ...",
                    style={"maxWidth": "640px"},
                ),
            ],
        ),
        html.Div(id="keyword-result"),
        html.Div([
            dbc.Button("AI 해석 보기", id="keyword-ai-insight-btn", size="sm", className="btn-secondary"),
            html.Div("단어를 2개 이상 선택하면, 두 단어가 같이 쓰인 연구가 어떤 흐름인지(또는 아직 "
                     "없다면 앞으로 유망한 조합인지 억지 조합인지) AI가 그 자리에서 해석해줍니다.",
                     className="panel-hint", style={"marginTop": "6px"}),
            dcc.Loading(html.Div(id="keyword-ai-insight-wrap", className="mt-2")),
        ], className="mt-3"),
        html.Div(id="keyword-papers-wrap"),
    ], className="section-gap")


@app.callback(
    Output("keyword-input", "options"),
    Output("keyword-hint", "children"),
    Input("main-tabs", "active_tab"),
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def refresh_keyword_options(active_tab, domain):
    """다른 탭에서 '분석 진행하기'를 돌리면 용어 사전이 바뀌므로, 키워드 탭에 들어올 때마다 다시 읽는다."""
    if active_tab != "tab-keyword":
        return no_update, no_update
    s = svc.get(domain)
    return [{"label": t, "value": t} for t in s.all_known_terms()], _keyword_hint(s)


@app.callback(
    Output("keyword-result", "children"),
    Output("keyword-papers-wrap", "children"),
    Input("keyword-input", "value"),
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def do_keyword_search(keywords, domain):
    if not keywords:
        return html.Div("검색어를 하나 이상 선택하세요.", className="panel-warn"), ""
    s = svc.get(domain)
    n_rising = s.n_full_rising_terms()

    cards = []
    for keyword in keywords:
        result = s.keyword_search(keyword)
        row = [html.B(keyword, style={"fontSize": "13px"})]
        if result["full"]:
            f = result["full"]
            row.append(html.Div(
                f"글로벌 트렌드 분석 \u2192 {f['label']}  ({f['document_count']}건)",
                style={"fontSize": "12.5px", "marginTop": "3px"},
            ))
        else:
            row.append(html.Div(f"글로벌 트렌드 분석 \u2192 Rising {n_rising:,}개 안에는 없음", className="panel-hint"))
        # 구간별분석 결과 줄은 화면에서 숨김(구간별분석 탭과 함께). keyword_search()는 result["windows"]를 계속 계산한다.
        cards.append(html.Div(row, className="info-card"))

    papers = s.papers_for_terms([k.lower() for k in keywords])
    papers_block = panel(f"선택한 단어({', '.join(keywords)})가 들어간 논문",
                          children=[paper_table(papers, "keyword-papers-table")])
    return html.Div(cards, className="mt-3"), papers_block


def _keyword_pair_context(s: svc.DomainService, keywords: list[str]) -> dict:
    y0, y1 = s.full_year_range()
    per_term = {}
    full_clusters = {}
    for kw in keywords:
        result = s.keyword_search(kw)
        entry = {f"rising_in_full_period_{y0}_{y1}": result["full"] is not None}
        if result["full"]:
            entry["full_period_community"] = result["full"]["label"]
            full_clusters[kw] = result["full"]["cluster"]
        entry["positive_trend_in_periods"] = [
            {"period": w["label"], "community_top_terms": w["top_terms"][:8]}
            for w in result["windows"]
        ]
        per_term[kw] = entry

    co_df = s.co_occurrence_papers([k.lower() for k in keywords], limit=8)
    context = {
        "selected_terms": keywords,
        "per_term": per_term,
        "co_occurrence_paper_count": int(len(co_df)),
        "co_occurrence_sample_titles": co_df["display_name"].tolist() if not co_df.empty else [],
    }
    if len(keywords) == 2 and len(full_clusters) == 2:
        ca, cb = list(full_clusters.values())
        if ca == cb:
            context["full_period_network_relation"] = "두 단어가 같은 커뮤니티에 속함"
        else:
            w = s.full_cluster_edge_weight(ca, cb)
            context["full_period_network_relation"] = (
                f"서로 다른 커뮤니티, 네트워크상 직접 연결(공유 문서 가중치={w})" if w
                else "서로 다른 커뮤니티, 네트워크상 직접 연결 없음"
            )
    return context


@app.callback(
    Output("keyword-ai-insight-wrap", "children"),
    Input("keyword-ai-insight-btn", "n_clicks"),
    State("keyword-input", "value"),
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def on_keyword_ai_insight_click(n_clicks, keywords, domain):
    if not keywords or len(keywords) < 2:
        return html.Div("먼저 단어를 2개 이상 선택하세요.", className="panel-warn")
    s = svc.get(domain)
    context = _keyword_pair_context(s, keywords)
    try:
        text = ai_insights.generate_keyword_pair_insight(s, context)
    except RuntimeError as e:
        return html.Div(str(e), className="panel-warn")
    except Exception as e:
        return html.Div(f"AI 해석 생성 중 오류가 발생했습니다: {e}", className="panel-warn")
    return ai_insight_block(text)


# ================================================================== 분야별 화면 (헤더 + 3개 탭)
def build_domain_content(domain: str):
    s = svc.get(domain)
    d = s.d
    header = html.Div([
        html.Div(html.Img(src=app.get_asset_url("kwater-logo.png"), alt="K-water"), className="app-logo"),
        html.Div([
            html.H3(d.title),
            html.Div(d.description, className="subtitle"),
            html.Div(
                "전체 파이프라인: 데이터 수집 및 전처리(OpenAlex) \u2192 키워드 단어(Rising Term) 추출 "
                "\u2192 연구주제 후보(Community) 도출(장기 / 단기)",
                className="subtitle-pipeline",
            ),
        ]),
    ], className="app-header")
    tabs = dbc.Tabs([
        dbc.Tab(build_tab_full(s), label="1. 글로벌 트렌드 분석", tab_id="tab-full"),
        # 구간별분석: 화면에서만 숨김(탭 버튼 display:none). 내용·콜백은 그대로 살아 있어 다시 보이려면 tab_style만 지우면 된다.
        dbc.Tab(build_tab_windowed(s), label="구간별분석", tab_id="tab-windowed", tab_style={"display": "none"}),
        dbc.Tab(build_tab_keyword(s), label="2. 키워드 검색", tab_id="tab-keyword"),
    ], id="main-tabs", active_tab="tab-full")
    return [header, tabs]


# 세 탭은 분야 화면이 그려질 때 한 번에 마운트되고, 이후에는 dbc.Tabs가 CSS로 보이기/숨기기만
# 한다(선택 안 된 탭도 DOM에는 계속 남아있음). 그래서 "분석 진행하기"로 그려낸 전체분석/
# 구간별분석 그래프는 다른 탭에 갔다가 돌아와도 다시 계산할 필요 없이 그대로 남아있다.
# 전체분석 cytoscape는 처음엔 elements=[]로 비어 있는 채 마운트되므로(숨겨진 탭이라 폭 0이어도
# 그릴 게 없어 문제되지 않음), 실제로 그래프가 채워지는 시점은 항상 그 탭을 보고 있을 때뿐이라
# 예전에 겪었던 "숨겨진 컨테이너 기준으로 배치가 굳어버리는" 문제도 재발하지 않는다.
# 분야를 바꾸면 화면 전체(탭·선택 상태 포함)가 그 분야 것으로 새로 그려진다.


@app.callback(
    Output("domain-content", "children"),
    Output("rendered-domain", "data"),
    Input("domain-store", "data"),
    Input("page-store", "data"),
    State("rendered-domain", "data"),
)
def render_domain(domain, page, rendered):
    """분야 화면은 그 분야로 실제로 들어갈 때 그린다. 숨겨진 상태에서 그리면 트렌드 후보 그래프가 폭 0 기준으로
    배치돼 원이 한곳에 뭉친다. 같은 분야로 돌아오면 다시 그리지 않는다(선택 상태 유지)."""
    if page != "domain" or domain == rendered:
        return no_update, no_update
    return build_domain_content(domain), domain


# ================================================================== 사이드바 + 앱 레이아웃
def _nav_id(key: str) -> str:
    return f"nav-{key}"


# 분야와 무관한 공통 화면: (page-store 값, 사이드바 이름, 화면 컨테이너 id)
COMMON_PAGES = [
    ("dataset", "데이터셋 현황", "dataset-content"),
    ("collect", "데이터 수집", "collect-content"),
    ("monitor", "수집현황 모니터링", "monitor-content"),
    ("integrated", "통합", "integrated-content"),   # 사이드바에서는 '분야별 분석' 맨 위에 보인다
]
DATA_PAGES = ("dataset", "collect", "monitor")
DEFAULT_PAGE = "integrated"   # 앱을 열면 처음 보이는 화면

sidebar = html.Div([
    html.Div("데이터", className="sidebar-title"),
    dbc.Nav([dbc.NavLink(label, id=f"nav-page-{page}", n_clicks=0, active=(page == DEFAULT_PAGE))
             for page, label, _ in COMMON_PAGES if page in DATA_PAGES],
            vertical=True, pills=True, className="sidebar-nav"),
    html.Div("분야별 분석", className="sidebar-title", style={"marginTop": "22px"}),
    dbc.Nav(
        [dbc.NavLink("통합 (전체 분야)", id="nav-page-integrated", n_clicks=0, active=(DEFAULT_PAGE == "integrated"))]
        + [dbc.NavLink(d.nav_label, id=_nav_id(key), n_clicks=0) for key, d in DOMAINS.items()],
        vertical=True, pills=True, className="sidebar-nav",
    ),
    html.Div("분석 중 다른 분야를 누르면 분석이 중단됩니다. 수집은 화면을 옮겨도 계속됩니다.",
             className="sidebar-note"),
], className="app-sidebar")


@app.callback(
    Output("domain-store", "data"),
    Output("page-store", "data"),
    *[Output(f"nav-page-{page}", "active") for page, _, _ in COMMON_PAGES],
    *[Output(_nav_id(key), "active") for key in DOMAINS],
    *[Input(f"nav-page-{page}", "n_clicks") for page, _, _ in COMMON_PAGES],
    *[Input(_nav_id(key), "n_clicks") for key in DOMAINS],
    Input({"type": "goto-domain", "key": ALL}, "n_clicks"),   # 통합 화면의 '○○ 분야 상세보기'
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def switch_page(*args):
    """공통 화면을 누르면 분야 화면은 숨기기만 하고(domain-store를 안 바꾸므로 진행 중인 분석이 취소되지
    않음), 분야를 누르면 그 분야 화면으로 간다(다른 분야면 다시 그림)."""
    current = args[-1]
    clicked = ctx.triggered_id or ""
    if isinstance(clicked, dict):
        # 통합 화면 카드가 새로 그려질 때도 호출되므로 실제로 누른 경우만 이동한다.
        if not ctx.triggered or not ctx.triggered[0]["value"]:
            return (no_update,) * (2 + len(COMMON_PAGES) + len(DOMAINS))
        clicked = _nav_id(clicked["key"])
    common_off = [False] * len(COMMON_PAGES)
    for i, (page, _, _) in enumerate(COMMON_PAGES):
        if clicked == f"nav-page-{page}":
            actives = list(common_off)
            actives[i] = True
            return no_update, page, *actives, *[False] * len(DOMAINS)
    key = clicked[len("nav-"):] if clicked.startswith("nav-") else current
    if key not in DOMAINS:
        key = current
    # 같은 분야를 다시 누르면 화면을 다시 그리지 않는다(진행 중 작업이 괜히 중단되지 않게).
    return (no_update if key == current else key), "domain", *common_off, *[k == key for k in DOMAINS]


@app.callback(
    Output("domain-content", "style"),
    *[Output(container, "style") for _, _, container in COMMON_PAGES],
    Output("monitor-content", "children"),
    Input("page-store", "data"),
    State("monitor-content", "children"),
)
def show_page(page, monitor_children):
    """화면은 모두 한 번 만들어 두고 보이기/숨기기만 한다(수집 진행 로그 등이 화면을 옮겨도 남게)."""
    hidden = {"display": "none"}
    styles = [{} if page == p else hidden for p, _, _ in COMMON_PAGES]
    monitor = (monitor_children or build_monitor_page()) if page == "monitor" else no_update
    return ({} if page == "domain" else hidden), *styles, monitor


# ================================================================== 수집현황 모니터링
DOMAIN_COLORS = ["#0072B2", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#D55E00", "#F0E442", "#999999"]
MONITOR_COLORS = {  # 분야(막대)·실패 사유(막대) 색. Okabe-Ito 계열(색맹 안전).
    **{key: DOMAIN_COLORS[i % len(DOMAIN_COLORS)] for i, key in enumerate([*DOMAINS, *LEGACY_DOMAIN_NAMES])},
    "budget": "#D55E00", "rate_limit": "#E69F00", "server": "#CC79A7",
    "network": "#56B4E9", "bad_request": "#A6761D", "not_found": "#999999", "other": "#000000",
}
LOG_TAG_COLORS = {"ok": "#7ee787", "fail": "#ff7b72", "info": "#79c0ff", "dim": "#6e7681"}


def build_monitor_page():
    return html.Div([
        html.Div([html.H3("수집현황 모니터링")], className="app-header"),
        html.Div([
            dcc.Dropdown(id="mon-domain", clearable=False, value="all", style={"width": "200px"},
                         options=[{"label": "전체 분야", "value": "all"}]
                         + [{"label": d.nav_label, "value": k} for k, d in DOMAINS.items()]),
            dbc.RadioItems(id="mon-days", inline=True, value=30, className="mon-radio",
                           options=[{"label": "7일", "value": 7}, {"label": "30일", "value": 30},
                                    {"label": "전체", "value": 0}]),
            dbc.Button("새로고침", id="mon-refresh", className="btn-secondary", size="sm"),
            html.Span(id="mon-updated", className="panel-hint", style={"marginBottom": "0"}),
        ], className="mon-toolbar"),
        html.Div(id="mon-tiles", className="mon-tiles"),
        panel("일자별 수집", children=[dcc.Graph(id="mon-chart", config={"displayModeBar": False})]),
        panel("로그", extra_header=html.Div([
                  dbc.Checklist(id="mon-fails-only", options=[{"label": "실패만", "value": "fail"}], value=[],
                                switch=True),
                  html.Span(id="mon-log-range", className="panel-hint", style={"marginBottom": "0"}),
                  dbc.Button("이전", id="mon-log-older", className="btn-secondary", size="sm"),
                  dbc.Button("최근", id="mon-log-newer", className="btn-secondary", size="sm"),
              ], className="mon-log-tools"),
              children=[html.Div(id="mon-log", className="log-box mon-log")]),
        dcc.Interval(id="mon-interval", interval=15_000),  # 수집 중 자동 갱신(15초). 한도 조회는 하지 않음.
        dcc.Store(id="mon-sig"),            # 마지막으로 그린 기록 파일 상태 -> 그대로면 자동 갱신 생략
        dcc.Store(id="mon-offset", data=0),  # 로그 창 페이지 위치(최신에서 몇 줄 뒤부터)
    ], className="section-gap")


def _monitor_chart(papers: pd.DataFrame, fails: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.62, 0.38],
                        subplot_titles=("내려받은 논문 수", "실패 요청 수 (사유별)"))
    for key in [*DOMAINS, *LEGACY_DOMAIN_NAMES]:
        sub = papers[papers["domain"].eq(key)]
        if sub.empty:
            continue
        name = monitor.domain_name(key)
        fig.add_bar(x=sub["date"], y=sub["n"], name=name, marker_color=MONITOR_COLORS[key],
                    hovertemplate="%{x|%Y-%m-%d}<br>" + name + " %{y:,}건<extra></extra>",
                    row=1, col=1)
    for reason in event_log.REASONS:
        sub = fails[fails["reason"].eq(reason)] if not fails.empty else fails
        if sub.empty:
            continue
        label = event_log.REASONS[reason]
        fig.add_bar(x=sub["date"], y=sub["count"], name=label, marker_color=MONITOR_COLORS[reason],
                    hovertemplate="%{x|%Y-%m-%d}<br>" + label + " %{y:,}건<extra></extra>", row=2, col=1)
    fig.update_layout(
        barmode="stack", height=460, margin=dict(l=8, r=8, t=28, b=8), bargap=0.25,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#1a1d23"),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
    )
    fig.update_xaxes(type="date", tickformat="%m-%d", showgrid=False)
    fig.update_yaxes(gridcolor="#eceff4", zeroline=False, tickformat=",")
    fig.update_annotations(font_size=12, x=0, xanchor="left")
    return fig


def _monitor_log(lines: list[tuple[str, str]]):
    """줄마다 문자열 하나(칸은 monitor._format이 고정폭으로 맞춤). 색만 줄 단위로 준다."""
    if not lines:
        return html.Div("기록 없음", style={"color": LOG_TAG_COLORS["dim"]})
    return [html.Div(text, style=None if level == "ok" else {"color": LOG_TAG_COLORS[level]})
            for level, text in lines]


@app.callback(
    Output("mon-tiles", "children"),
    Output("mon-chart", "figure"),
    Output("mon-log", "children"),
    Output("mon-log-range", "children"),
    Output("mon-updated", "children"),
    Output("mon-sig", "data"),
    Output("mon-offset", "data"),
    Input("mon-refresh", "n_clicks"),
    Input("mon-interval", "n_intervals"),
    Input("mon-domain", "value"),
    Input("mon-days", "value"),
    Input("mon-fails-only", "value"),
    Input("mon-log-older", "n_clicks"),
    Input("mon-log-newer", "n_clicks"),
    State("page-store", "data"),
    State("mon-tiles", "children"),
    State("mon-sig", "data"),
    State("mon-offset", "data"),
)
def refresh_monitor(n_refresh, n_interval, domain, days, fails_only, n_older, n_newer,
                    page, current_tiles, last_sig, offset):
    trig = ctx.triggered_id
    sig = [list(x) for x in monitor.signature()]
    if trig == "mon-interval" and (page != "monitor" or sig == last_sig):
        # 화면이 숨겨져 있거나 기록 파일이 그대로면 아무것도 다시 그리지 않는다.
        return (no_update,) * 7
    dom = None if domain == "all" else domain
    days = days or None
    fails_only = bool(fails_only)
    offset = offset or 0
    if trig == "mon-log-older":
        offset += monitor.LOG_PAGE
    elif trig == "mon-log-newer":
        offset = max(0, offset - monitor.LOG_PAGE)
    elif trig != "mon-interval":
        offset = 0  # 필터·새로고침이면 최신부터

    lines, total = monitor.log_page(dom, days, fails_only, offset)
    if total and offset >= total:
        offset = (total - 1) // monitor.LOG_PAGE * monitor.LOG_PAGE
        lines, total = monitor.log_page(dom, days, fails_only, offset)
    log_range = f"{offset + 1:,}–{offset + len(lines):,} / {total:,}줄" if total else ""

    if trig in ("mon-log-older", "mon-log-newer"):
        # 페이지 이동은 로그 창만 바꾼다.
        return no_update, no_update, _monitor_log(lines), log_range, no_update, no_update, offset

    papers, fails, s = monitor.chart_data(dom, days)
    # OpenAlex 남은 한도는 요청 1회(1크레딧)가 들어서 처음 열 때와 '새로고침' 때만 조회한다.
    if trig in (None, "mon-refresh") or not (isinstance(current_tiles, list) and len(current_tiles) == 4):
        budget_tile = stat_tile(_budget_text(), "OpenAlex 남은 한도 (오늘)")
    else:
        budget_tile = current_tiles[3]
    tiles = [
        stat_tile(f"{s['papers']:,}건", "내려받은 논문"),
        stat_tile(f"{s['years']:,}개", "수집된 저널×연도"),
        stat_tile(f"{s['fails']:,}건", "실패 요청"),
        budget_tile,
    ]
    updated = f"갱신 {datetime.now():%H:%M:%S}"
    if s["last_ok"] is not None:
        updated += f" · 마지막 수집 {s['last_ok']:%Y-%m-%d %H:%M}"
    return tiles, _monitor_chart(papers, fails), _monitor_log(lines), log_range, updated, sig, offset


def _budget_text() -> str:
    from pipeline_codes import n_01_openalex as openalex
    if not svc.API_KEY:
        return "키 없음"
    openalex.use_api_keys(svc.API_KEYS)  # 지금 쓰는 키(앞 키가 소진됐으면 다음 키)의 한도를 보여준다
    b = openalex.check_budget()
    if b["remaining"] is None:
        return "확인 불가"
    return f"{int(b['remaining']):,} / 10,000" if not b["exhausted"] else "소진"


# ================================================================== 통합 (전체 분야)
# 6개 분야를 각각 분석한 글로벌 트렌드를 한 화면에 모아 보여주고, 그 전체를 근거로 AI와 대화한다.
# 분야를 합쳐 다시 분석하지 않는다(논문 수가 많은 분야로 결과가 쏠리는 것을 피하려고).
def build_integrated_page():
    return html.Div([
        html.Div([
            html.Div(html.Img(src=app.get_asset_url("kwater-logo.png"), alt="K-water"), className="app-logo"),
            html.Div([
                html.H3("물분야 글로벌 저널 트렌드 분석"),
                html.Div("6개 분야(연구소)의 글로벌 트렌드 분석 결과를 한곳에 모았습니다. 분야마다 따로 분석한 결과이며, "
                         "분야를 합쳐 다시 분석한 것은 아닙니다. 'AI와 대화하기'에서 분야 간 공통 트렌드나 비교를 물어볼 수 "
                         "있습니다.", className="subtitle"),
            ]),
        ], className="app-header"),
        html.Div(dbc.Button("AI와 대화하기", id="chat-open-all-btn", className="chat-open-btn chat-open-btn-lg"),
                 className="mb-3"),
        dcc.Loading(html.Div(id="integrated-body")),
    ], className="section-gap")


def _integrated_body():
    services = [svc.get(k) for k in DOMAINS]
    o = chat.integrated_overview(services)
    tiles = html.Div([
        stat_tile(f"{o['n_analyzed']} / {o['n_domains']}개", "분석된 분야"),
        stat_tile(f"{o['n_journals']}종", "저널 (분야 간 중복 제외)"),
        stat_tile(f"{o['n_papers']:,}편", "논문 (분야 간 중복 제외)"),
        stat_tile(f"{o['n_rising']:,}개", "Rising 단어 (분야별 합계)"),
        stat_tile(f"{o['n_trends']}개", "트렌드 후보 (분야별 합계)"),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"})
    cards = []
    for s in services:
        if s.has_full_results():
            legend = s.full_cluster_legend()
            ov = s.cache_overview()
            meta = f"논문 {ov['total_papers']:,}편 \u00b7 Rising {s.n_full_rising_terms():,}개 \u00b7 트렌드 후보 {len(legend)}개"
            body = html.Div([html.Div([html.Span(chat._num(i), className="trend-num"), html.Span(item["label"])],
                                      className="trend-item") for i, item in enumerate(legend)])
        else:
            meta, body = "", html.Div("아직 분석 결과가 없습니다.", className="panel-hint")
        cards.append(dbc.Col(html.Div([
            html.Div([html.Span(f"{s.d.name} 분야", className="trend-card-title"),
                      html.Span(s.d.department, className="trend-card-dept")]),
            html.Div(meta, className="trend-card-meta"),
            body,
            dbc.Button(f"{s.d.name} 분야 상세보기 \u2192", id={"type": "goto-domain", "key": s.d.key},
                       size="sm", className="btn-secondary trend-card-btn"),
        ], className="panel trend-card"), md=6, xl=4))
    return html.Div([tiles, dbc.Row(cards, className="g-3")])


@app.callback(
    Output("integrated-body", "children"),
    Input("page-store", "data"),
)
def refresh_integrated_page(page):
    """통합 화면을 열 때마다 분야별 최신 분석 결과로 다시 그린다."""
    if page != "integrated":
        return no_update
    return _integrated_body()


@app.callback(
    Output("chat-store", "data", allow_duplicate=True),
    Output("chat-panel", "style", allow_duplicate=True),
    Output("chat-subtitle", "children", allow_duplicate=True),
    Output("chat-echo", "children", allow_duplicate=True),
    Input("chat-open-all-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_chat_all(n_clicks):
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    services = [svc.get(k) for k in DOMAINS]
    data = {"domain": chat.ALL, "cluster": None,
            "messages": [{"role": "assistant", "content": chat.briefing_all(services)}]}
    return data, {}, f"통합 \u00b7 전체 {len(services)}개 분야", []


# ================================================================== AI와 대화하기 (오른쪽 대화창)
# 글로벌 트렌드 분석 > 커뮤니티 상세 > 'AI와 대화하기'를 누르면 화면 오른쪽에 대화창이 열린다(왼쪽은 그대로).
# 첫 메시지는 저장된 분석 결과로 만든 브리핑(chat.briefing), 이후 질문은 AI가 분석 자료에 근거해 답한다(chat.answer).
# 답변 모델은 대화창 위에서 OpenAI / Gemma 중 고르고(chat-model, 기본 Gemma), 답변은 생성되는 대로 이어 붙여 보여준다(스트리밍).
# 대화 기록은 chat-store에만 있고 저장하지 않는다. 분야를 바꾸거나 다른 화면으로 가면 닫힌다.
def build_chat_panel():
    return html.Div([
        html.Div([
            html.Div([html.Div("AI와 대화하기", className="chat-title"),
                      html.Div(id="chat-subtitle", className="chat-subtitle"),
                      dbc.RadioItems(id="chat-model", value="gemma", inline=True, persistence=True,
                                     options=[{"label": name, "value": key}
                                              for key, (_, name) in ai_insights.PROVIDERS.items()],
                                     className="chat-model")]),
            dbc.Button("\u2715", id="chat-close-btn", className="chat-close-btn", title="대화창 닫기"),
        ], className="chat-header"),
        html.Div([html.Div(id="chat-history", className="chat-history"),
                  html.Div(id="chat-echo", className="chat-history")],
                 id="chat-messages", className="chat-messages"),
        html.Div([
            dbc.Input(id="chat-input", placeholder="궁금한 점을 입력하고 Enter", autoComplete="off", debounce=False),
            dbc.Button("보내기", id="chat-send-btn", className="btn-primary"),
        ], className="chat-input-row"),
        html.Div("분석 결과에 근거해 답합니다. 분석 자료 밖의 일반 지식은 회색 [참고] 상자로 따로 표시됩니다.",
                 className="chat-footnote"),
        dcc.Store(id="chat-store"),     # 대화 기록(브리핑 + 주고받은 질문·답변)
        html.Div(id="chat-scroll-dummy", style={"display": "none"}),
    ], id="chat-panel", className="chat-panel", style={"display": "none"})


def _chat_bubble(m: dict):
    if m["role"] == "user":
        return html.Div([html.Div(m["content"], className="chat-bubble chat-user"),
                         html.Div(className="chat-avatar chat-avatar-user", title="나")], className="chat-row chat-row-user")
    avatar = html.Div(className="chat-avatar chat-avatar-ai", title="AI")
    main, _, ref = m["content"].partition(chat.REFERENCE_MARK)
    parts = [dcc.Markdown(main.strip(), className="chat-md")]
    if m.get("model"):   # 어느 모델이 쓴 답변인지(OpenAI·Gemma 비교용)
        parts.insert(0, html.Div(ai_insights.PROVIDERS[m["model"]][1], className="chat-model-tag"))
    if ref.strip():
        parts.append(html.Div([html.Div("[참고] 분석 결과가 아닌 일반 지식", className="chat-ref-title"),
                               dcc.Markdown(ref.strip(), className="chat-md")], className="chat-ref"))
    return html.Div([avatar, html.Div(parts, className="chat-bubble chat-ai")], className="chat-row")


@app.callback(
    Output("chat-store", "data", allow_duplicate=True),
    Output("chat-panel", "style", allow_duplicate=True),
    Output("chat-subtitle", "children"),
    Output("chat-echo", "children", allow_duplicate=True),
    Input("chat-open-btn", "n_clicks"),
    State("full-selected-cluster", "data"),
    State("domain-store", "data"),
    prevent_initial_call=True,
)
def open_chat(n_clicks, cluster, domain):
    """새로 열 때마다 그 트렌드 후보에 대한 브리핑으로 대화를 새로 시작한다."""
    if not n_clicks or cluster is None:
        return no_update, no_update, no_update, no_update
    s = svc.get(domain)
    cid = int(cluster)
    data = {"domain": domain, "cluster": cid,
            "messages": [{"role": "assistant", "content": chat.briefing(s, cid)}]}
    return data, {}, f"{s.d.name} 분야 \u00b7 {s.full_cluster_label(cid)}", []


@app.callback(
    Output("chat-panel", "style"),
    Input("chat-close-btn", "n_clicks"),
    Input("domain-store", "data"),
    Input("page-store", "data"),
    prevent_initial_call=True,
)
def close_chat(n_clicks, domain, page):
    return {"display": "none"}


# 질문을 보내면 브라우저가 질문 말풍선을 바로 그리고 /api/chat-stream에 답변을 요청해, 받는 조각을 AI 말풍선에
# 이어 붙인다. 서버 콜백으로 하면 Dash가 답변이 끝날 때까지 화면 갱신을 미뤄서 한 번에 뜨기 때문이다.
#   Enter/보내기 -> [브라우저] chat-echo에 질문 말풍선 + 빈 AI 말풍선(chat-streaming), 입력칸 비움
#               -> [서버] chat_stream: 답변 조각들 ... STREAM_END + 숫자 표기를 정리한 최종 답변
#               -> [브라우저] chat-store에 질문·답변 추가, chat-echo 비움 -> render_chat(마크다운으로 다시 그림)
@server.route("/api/chat-stream", methods=["POST"])
def chat_stream():
    req = flask.request.get_json()
    provider = req.get("model") if req.get("model") in ai_insights.PROVIDERS else "gemma"

    def generate():
        parts = []
        try:
            if req["domain"] == chat.ALL:
                pieces = chat.answer_all([svc.get(k) for k in DOMAINS], req["messages"], provider)
            else:
                pieces = chat.answer(svc.get(req["domain"]), req["cluster"], req["messages"], provider)
            for piece in pieces:
                parts.append(piece)
                yield piece
            final = chat.tidy_numbers("".join(parts).strip())
        except Exception as e:
            final = f"답변을 만들지 못했습니다: {e}"
        yield chat.STREAM_END + final

    # X-Accel-Buffering: nginx가 응답을 모았다가 한 번에 보내지 않게 한다.
    return flask.Response(generate(), mimetype="text/plain; charset=utf-8",
                          headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


app.clientside_callback(
    """function(n_clicks, n_submit, text, store, echo, model) {
        var nu = window.dash_clientside.no_update;
        text = (text || '').trim();
        if (!text || !store || (echo && echo.length)) { return [nu, nu]; }
        function div(cls, children, title, id) {
            var props = {className: cls, children: children};
            if (title) { props.title = title; }
            if (id) { props.id = id; }
            return {type: 'Div', namespace: 'dash_html_components', props: props};
        }
        function scroll() {
            var el = document.getElementById('chat-messages');
            if (el) { el.scrollTop = el.scrollHeight; }
        }
        var echoEls = [
            div('chat-row chat-row-user', [div('chat-bubble chat-user', text), div('chat-avatar chat-avatar-user', null, '나')]),
            div('chat-row', [div('chat-avatar chat-avatar-ai', null, 'AI'),
                             div('chat-bubble chat-ai chat-pending', '답변을 작성하고 있습니다\u2026', null, 'chat-streaming')])
        ];
        setTimeout(scroll, 50);
        var history = store.messages.concat([{role: 'user', content: text}]);
        var send = history.map(function(m) { return {role: m.role, content: m.content}; });
        function finish(answer) {
            window.dash_clientside.set_props('chat-store', {data: Object.assign({}, store,
                {messages: history.concat([{role: 'assistant', content: answer, model: model}])})});
            window.dash_clientside.set_props('chat-echo', {children: []});
        }
        fetch('""" + app.config.requests_pathname_prefix + """api/chat-stream', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({domain: store.domain, cluster: store.cluster, messages: send, model: model})
        }).then(function(resp) {
            if (!resp.ok) { throw new Error('HTTP ' + resp.status); }
            var reader = resp.body.getReader(), decoder = new TextDecoder(), buf = '';
            function read() {
                return reader.read().then(function(r) {
                    if (!r.done) { buf += decoder.decode(r.value, {stream: true}); }
                    var end = buf.indexOf('\u0000');
                    if (end >= 0) { finish(buf.slice(end + 1)); return; }
                    if (r.done) { finish(buf || '답변을 만들지 못했습니다: 응답이 비어 있습니다.'); return; }
                    var el = document.getElementById('chat-streaming');
                    if (el && buf) {
                        var atBottom = (function(m) { return !m || m.scrollHeight - m.scrollTop - m.clientHeight < 40; })(
                            document.getElementById('chat-messages'));
                        el.textContent = buf;
                        el.className = 'chat-bubble chat-ai chat-streaming';
                        if (atBottom) { scroll(); }
                    }
                    return read();
                });
            }
            return read();
        }).catch(function(e) { finish('답변을 만들지 못했습니다: ' + e.message); });
        return ['', echoEls];
    }""",
    Output("chat-input", "value"),
    Output("chat-echo", "children"),
    Input("chat-send-btn", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-store", "data"),
    State("chat-echo", "children"),
    State("chat-model", "value"),
    prevent_initial_call=True,
)


@app.callback(
    Output("chat-history", "children"),
    Input("chat-store", "data"),
)
def render_chat(data):
    return [_chat_bubble(m) for m in (data or {}).get("messages", [])]


# 답변이 붙으면 대화창을 맨 아래로 내린다(질문을 보낼 때는 위 clientside 콜백이 내린다).
app.clientside_callback(
    """function(children) {
        setTimeout(function() {
            var el = document.getElementById('chat-messages');
            if (el) { el.scrollTop = el.scrollHeight; }
        }, 50);
        return '';
    }""",
    Output("chat-scroll-dummy", "children"),
    Input("chat-history", "children"),
)


app.layout = html.Div([
    dcc.Store(id="domain-store", data=DEFAULT_DOMAIN),
    dcc.Store(id="page-store", data=DEFAULT_PAGE),
    dcc.Store(id="rendered-domain"),   # domain-content에 지금 그려져 있는 분야
    sidebar,
    html.Div(
        dbc.Container([
            html.Div(build_dataset_page(), id="dataset-content",
                     style=None if DEFAULT_PAGE == "dataset" else {"display": "none"}),
            html.Div(build_collect_page(), id="collect-content", style={"display": "none"}),
            html.Div(id="monitor-content", style={"display": "none"}),
            html.Div(build_integrated_page(), id="integrated-content",
                     style=None if DEFAULT_PAGE == "integrated" else {"display": "none"}),
            html.Div(id="domain-content", style={"display": "none"}),
        ], fluid=True, style={"maxWidth": "1360px"}),
        className="app-main",
    ),
    build_chat_panel(),
], className="app-shell")


if __name__ == "__main__":
    # 외부에 열려 있는 서버라 debug=False(개발용 디버그 화면·자동 재시작 끔). 코드를 고친 뒤에는 앱을 껐다 켜야 반영된다.
    app.run(debug=False, host="0.0.0.0", port=8050)
