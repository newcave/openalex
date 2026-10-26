# pipeline_codes/n_04_visualize.py
import ast
import itertools
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

from pipeline_codes.utils import logger


CENTRALITY_CONFIGS = {
    "degree": {
        "label": "연결 중심성 (Degree)",
        "description": "직접 연결된 기관 수 기준",
        "tooltip": "연결 중심성 (Degree Centrality)\n\n해당 기관이 직접 공동 연구한 다른 기관의 수를 나타냅니다.\n값이 높을수록 많은 기관과 직접 협업한 '허브' 기관입니다.\n\n활용: 네트워크에서 가장 활발하게 협업하는 기관을 파악할 때 사용합니다.\n노드 색상: 빨강(높음) → 파랑(낮음)",
        "color_high": "#ff6b6b",
        "color_low":  "#4a90d9",
    },
    "eigenvector": {
        "label": "고유벡터 중심성 (Eigenvector)",
        "description": "영향력 있는 기관과의 연결 기준",
        "tooltip": "고유벡터 중심성 (Eigenvector Centrality)\n\n단순히 연결 수만 보는 게 아니라, 연결된 기관의 '영향력'까지 고려합니다.\n영향력 있는 기관과 많이 연결될수록 점수가 높아집니다.\n\n활용: 핵심 연구 그룹의 중심에 있는 기관을 파악할 때 유용합니다.\n노드 색상: 보라(높음) → 파랑(낮음)",
        "color_high": "#a29bfe",
        "color_low":  "#4a90d9",
    },
    "betweenness": {
        "label": "매개 중심성 (Betweenness)",
        "description": "기관 간 정보 흐름의 교량 역할 기준",
        "tooltip": "매개 중심성 (Betweenness Centrality)\n\n두 기관 사이의 최단 경로에 얼마나 자주 등장하는지를 측정합니다.\n값이 높을수록 서로 다른 연구 그룹을 연결하는 '다리' 역할을 합니다.\n\n활용: 제거되면 네트워크가 단절될 수 있는 핵심 연결 기관을 찾을 때 유용합니다.\n노드 색상: 주황(높음) → 파랑(낮음)",
        "color_high": "#ffa94d",
        "color_low":  "#4a90d9",
    },
    "closeness": {
        "label": "근접 중심성 (Closeness)",
        "description": "전체 네트워크 내 접근 용이성 기준",
        "tooltip": "근접 중심성 (Closeness Centrality)\n\n해당 기관에서 네트워크 내 다른 모든 기관까지의 평균 거리를 역수로 나타냅니다.\n값이 높을수록 다른 기관들에 빠르게 접근할 수 있는 '중심부' 기관입니다.\n\n활용: 정보나 기술이 빠르게 확산될 수 있는 위치의 기관을 파악할 때 유용합니다.\n노드 색상: 초록(높음) → 파랑(낮음)",
        "color_high": "#51cf66",
        "color_low":  "#4a90d9",
    },
}


def _parse_org_names(cell) -> list[str]:
    if isinstance(cell, str):
        try:
            lst = ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(cell, list):
        lst = cell
    else:
        return []
    return [x for x in lst if isinstance(x, str) and x.strip()]


def build_full_graph(df: pd.DataFrame) -> nx.Graph:
    node_weight: defaultdict[str, int] = defaultdict(int)
    edge_weight: defaultdict[tuple, int] = defaultdict(int)

    for insts in df['org_names'].apply(_parse_org_names):
        unique_insts = set(insts)
        for inst in unique_insts:
            node_weight[inst] += 1
        for a, b in itertools.combinations(sorted(unique_insts), 2):
            edge_weight[(a, b)] += 1

    G = nx.Graph()
    for inst, w in node_weight.items():
        G.add_node(inst, weight=w)
    for (a, b), w in edge_weight.items():
        G.add_edge(a, b, weight=w)

    logger.info(f"전체 그래프: 노드 {G.number_of_nodes():,}개, 엣지 {G.number_of_edges():,}개")
    return G


def compute_centralities(G: nx.Graph) -> dict[str, dict[str, float]]:
    logger.info("중심성 지표 계산 중...")
    centralities = {}
    centralities["degree"] = nx.degree_centrality(G)

    try:
        largest_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        eig = nx.eigenvector_centrality_numpy(largest_cc, weight='weight')
        centralities["eigenvector"] = {n: eig.get(n, 0.0) for n in G.nodes()}
    except Exception as e:
        logger.warning(f"Eigenvector 계산 실패, degree로 대체: {e}")
        centralities["eigenvector"] = centralities["degree"]

    centralities["betweenness"] = nx.betweenness_centrality(G, weight='weight', normalized=True)
    centralities["closeness"] = nx.closeness_centrality(G, distance='weight')
    logger.info("중심성 지표 계산 완료")
    return centralities


def filter_graph(G: nx.Graph, centrality_scores: dict[str, float],
                 top_n: int, min_edge_weight: int) -> nx.Graph:
    top_nodes = set(
        sorted(centrality_scores, key=centrality_scores.get, reverse=True)[:top_n]
    )
    sub = G.subgraph(top_nodes).copy()
    weak_edges = [(u, v) for u, v, d in sub.edges(data=True)
                  if d.get('weight', 1) < min_edge_weight]
    sub.remove_edges_from(weak_edges)
    isolates = list(nx.isolates(sub))
    sub.remove_nodes_from(isolates)
    logger.info(f"필터 후: 노드 {sub.number_of_nodes()}개, 엣지 {sub.number_of_edges()}개")
    return sub


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _lerp_color(c1, c2, t):
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    r = int(r1 * t + r2 * (1 - t))
    g = int(g1 * t + g2 * (1 - t))
    b = int(b1 * t + b2 * (1 - t))
    return f"#{r:02x}{g:02x}{b:02x}"


def _make_pyvis_html(sub: nx.Graph, centrality_scores: dict[str, float],
                     cfg: dict, scale: int = 5, base_size: int = 10) -> str:
    net = Network(
        height="750px", width="100%",
        notebook=False, bgcolor="#1e1e1e", font_color="white",
        directed=False,
    )
    net.show_buttons(filter_=["physics"])

    scores = list(centrality_scores.values())
    min_s = min(scores) if scores else 0
    max_s = max(scores) if scores else 1

    for node, data in sub.nodes(data=True):
        w = data.get('weight', 1)
        score = centrality_scores.get(node, 0)
        size = base_size + scale * w
        ratio = (score - min_s) / (max_s - min_s) if max_s != min_s else 0.5
        color = _lerp_color(cfg["color_high"], cfg["color_low"], 1 - ratio)
        net.add_node(
            node,
            label=node,
            title=f"빈도: {w}회 | {cfg['label']}: {score:.4f}",
            value=size,
            color={
                "background": color,
                "border": color,
                "highlight": {"background": "#ffffff", "border": color},
                "hover": {"background": "#ffffff", "border": color},
            },
        )

    for u, v, data in sub.edges(data=True):
        w = data.get('weight', 1)
        net.add_edge(
            u, v,
            value=w,
            title=f"공동출현: {w}회",
            color={"color": "rgba(150,170,200,0.3)", "highlight": "rgba(255,255,255,0.8)"},
        )

    return net.generate_html()


def _build_tabbed_html(html_parts: dict[str, str], stats: dict) -> str:
    tab_buttons = ""
    tab_contents = ""
    tooltip_divs = ""

    for i, (key, cfg) in enumerate(CENTRALITY_CONFIGS.items()):
        active = "active" if i == 0 else ""
        display = "block" if i == 0 else "none"
        s = stats.get(key, {})
        tooltip_divs += f'<div class="tip-box" id="tip-{key}">{cfg["tooltip"]}</div>\n'
        tab_buttons += f"""
        <div class="tab-wrap">
          <button id="btn-{key}" class="tab-btn {active}" onclick="showTab('{key}')">{cfg['label']}</button>
          <span class="qmark" onmouseenter="showTip('{key}',event)" onmouseleave="hideTip()">?</span>
        </div>"""
        tab_contents += f"""
        <div id="tab-{key}" class="tab-pane" style="display:{display}">
          <div class="tab-meta">
            <span>{cfg['description']}</span>
            <span>노드 <b>{s.get('nodes',0)}</b>개 &nbsp;·&nbsp; 엣지 <b>{s.get('edges',0)}</b>개</span>
          </div>
          <div class="net-wrap">{html_parts[key]}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>기관 협업 네트워크</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', 'Noto Sans KR', sans-serif; background: #1e1e1e; color: #fff; }}
.tab-bar {{
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  padding: 12px 16px; background: #252526; border-bottom: 1px solid #3c3c3c;
}}
.tab-wrap {{ display: flex; align-items: center; gap: 5px; }}
.tab-btn {{
  padding: 6px 16px; border: 1px solid #555; border-radius: 4px;
  cursor: pointer; background: #3c3c3c; color: #ccc;
  font-size: 13px; font-family: inherit; transition: all 0.15s;
}}
.tab-btn:hover {{ background: #505050; color: #fff; }}
.tab-btn.active {{ background: #0078d4; color: #fff; border-color: #0078d4; }}
.qmark {{
  width: 18px; height: 18px; border-radius: 50%;
  background: #3c3c3c; border: 1px solid #666;
  color: #aaa; font-size: 11px; font-weight: bold;
  display: inline-flex; align-items: center; justify-content: center;
  cursor: help; user-select: none; flex-shrink: 0;
}}
.qmark:hover {{ background: #0078d422; border-color: #0078d4; color: #4fc3f7; }}
.tip-box {{
  position: fixed; display: none;
  background: #2d2d2d; border: 1px solid #555; border-radius: 8px;
  color: #ddd; font-size: 13px; line-height: 1.7;
  padding: 14px 18px; width: 360px; max-width: 90vw;
  box-shadow: 0 8px 24px rgba(0,0,0,0.6);
  z-index: 9999; pointer-events: none;
  white-space: pre-wrap; word-break: keep-all;
}}
.tab-meta {{
  display: flex; justify-content: center; gap: 24px;
  padding: 6px 16px; background: #252526; border-bottom: 1px solid #3c3c3c;
  font-size: 12px; color: #aaa;
}}
.tab-meta b {{ color: #fff; }}
.net-wrap iframe {{ border: none; display: block; width: 100%; height: 750px; }}
</style>
</head>
<body>
{tooltip_divs}
<div class="tab-bar">{tab_buttons}</div>
{tab_contents}
<script>
function showTab(key) {{
  document.querySelectorAll('.tab-pane').forEach(el => el.style.display='none');
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-'+key).style.display='block';
  document.getElementById('btn-'+key).classList.add('active');
}}
var _tip=null;
function showTip(key,e) {{
  if(_tip) _tip.style.display='none';
  var box=document.getElementById('tip-'+key);
  box.style.display='block'; _tip=box;
  var x=e.clientX+14, y=e.clientY+14;
  if(x+380>window.innerWidth) x=e.clientX-380;
  if(y+300>window.innerHeight) y=e.clientY-300;
  box.style.left=x+'px'; box.style.top=y+'px';
}}
function hideTip() {{ if(_tip){{ _tip.style.display='none'; _tip=null; }} }}
</script>
</body>
</html>"""


def main(df: pd.DataFrame, html_path: str,
         top_n: int = 50, min_edge_weight: int = 2,
         scale: int = 2, base_size: int = 6) -> str:
    if 'org_names' not in df.columns:
        raise ValueError("org_names 컬럼이 없습니다.")

    G = build_full_graph(df)
    if G.number_of_nodes() == 0:
        raise ValueError("기관 데이터가 없습니다.")

    centralities = compute_centralities(G)

    html_parts = {}
    stats = {}
    for key, cfg in CENTRALITY_CONFIGS.items():
        sub = filter_graph(G, centralities[key], top_n=top_n, min_edge_weight=min_edge_weight)
        sub_scores = {n: centralities[key].get(n, 0) for n in sub.nodes()}
        html_parts[key] = _make_pyvis_html(sub, sub_scores, cfg, scale=scale, base_size=base_size)
        stats[key] = {"nodes": sub.number_of_nodes(), "edges": sub.number_of_edges()}
        logger.info(f"{key}: 노드 {stats[key]['nodes']}개, 엣지 {stats[key]['edges']}개")

    final_html = _build_tabbed_html(html_parts, stats)
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    Path(html_path).write_text(final_html, encoding="utf-8")
    logger.info(f"시각화 저장 완료: {html_path}")
    return html_path