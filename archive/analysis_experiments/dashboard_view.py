# analysis/dashboard_view.py
# scripts/dashboard.py의 "트렌드 분석" 페이지에서 호출하는 렌더링 함수.
# 무거운 계산(토큰화)은 term_frequency.py가 미리 끝내둔 롱포맷 테이블을 읽어서
# groupby만 하므로 화면 조작(연도/저널/n-gram 변경)이 빠르다.
#
# 이 페이지는 세 서브탭으로 구성된다 (이 순서 그대로 st.tabs에 넘긴다):
#   - 용어 정규화 : 정규화 사전을 실제로 채워나가는 작업 화면 (다른 탭 결과에 다 영향을 주므로 먼저)
#   - 빈도 분석   : 정규화 사전이 적용된 빈도표 + 워드클라우드 (특정 기간 스냅샷)
#   - 신호 태그   : 용어별 추세/속도/버스트 등을 태그로 보여주는 화면 (구 '추이 보기'의 5구간+임계값
#                   방식을 대체 — analysis/trend_tags.py, 근거는 analysis/lifecycle.py 주석 참고)
# 앞으로 weak signal 등 단계가 늘어날 때마다 서브탭이 추가될 예정.

import altair as alt
import colorsys
import json
import math
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from wordcloud import WordCloud

from analysis import build_dataset, cluster_alignment, cluster_trends, normalization, trend_tags
from analysis.term_frequency import UNIGRAM_PATH, BIGRAM_PATH

# 게이트2(최소빈도) 값은 analysis/trend_tags.py에 있다 (export_trend_tags.py와 같은 값을
# 쓰기 위해 그쪽으로 옮김). 여기선 그대로 가져다 쓴다.
MIN_TOTAL_COUNT = trend_tags.MIN_TOTAL_COUNT

YEAR_PRESETS = {
    "2000-04": (2000, 2004),
    "2005-09": (2005, 2009),
    "2010-14": (2010, 2014),
    "2015-19": (2015, 2019),
    "2020-25": (2020, 2025),
}


@st.cache_data(show_spinner=False)
def _load_long(ngram_type: str) -> pd.DataFrame:
    path = UNIGRAM_PATH if ngram_type == "유니그램" else BIGRAM_PATH
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_doc_counts() -> pd.DataFrame:
    """상대빈도(%)·최소빈도 게이트의 분모로 쓸 논문 수.

    build_dataset.load_analyzable_corpus()를 쓴다 — filtered_corpus.parquet 전체가 아니라
    "지금 토큰화 가능한"(현재는 영문만, text_utils.py 참고) 서브셋으로 좁힌 것. 안 그러면
    한글 논문(용어 기여 0)이 분모에만 들어가 비율이 미세하게 과소계산된다.
    """
    return build_dataset.load_analyzable_corpus(columns=["id", "year", "journal"])


def _make_wordcloud(freq: dict):
    """PIL 이미지로 직접 반환한다 (matplotlib 미사용).
    이 환경에서는 matplotlib과 wordcloud를 같은 프로세스에서 같이 쓰면
    폰트 렌더링 시점에 freetype 버전 충돌로 프로세스가 죽는 문제가 있어,
    wordcloud가 자체 제공하는 to_image()로 우회한다."""
    wc = WordCloud(width=1400, height=700, background_color="white", colormap="viridis")
    wc.generate_from_frequencies(freq)
    return wc.to_image()


def render() -> None:
    if not UNIGRAM_PATH.exists() or not BIGRAM_PATH.exists():
        st.warning(
            "빈도표 데이터가 없습니다. 먼저 아래 두 스크립트를 순서대로 실행해주세요:\n\n"
            "1. `python analysis/build_dataset.py`\n"
            "2. `python analysis/term_frequency.py`"
        )
        return

    subtab_norm, subtab_freq, subtab_tags, subtab_clusters = st.tabs(
        ["🔧 용어 정규화", "📊 빈도 분석", "🧭 신호 태그", "📈 군집 추세"]
    )

    with subtab_norm:
        _render_normalization_tab()

    with subtab_freq:
        _render_frequency_tab()

    with subtab_tags:
        _render_signal_tags_tab()

    with subtab_clusters:
        _render_cluster_trends_tab()



@st.cache_data(show_spinner=False)
def _load_cluster_trends(version: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if version == "BH-FDR 보정 후":
        long_path = cluster_trends.ROOT / "analysis" / "output_fdr" / "cluster_trends_fdr.parquet"
        summary_path = cluster_trends.ROOT / "analysis" / "output_fdr" / "cluster_trends_summary_fdr.csv"
    else:
        long_path = cluster_trends.OUTPUT_LONG_PATH
        summary_path = cluster_trends.OUTPUT_SUMMARY_PATH
    if not long_path.exists() or not summary_path.exists():
        return None
    series = pd.read_parquet(long_path)
    summary = pd.read_csv(summary_path)
    return series, summary


def _render_cluster_trends_single_legacy() -> None:
    version = st.radio(
        "분석 버전",
        ["기존 p-value 기준", "BH-FDR 보정 후"],
        horizontal=True,
        key="cluster_trends_version",
        help="기존 결과와 BH-FDR(q≤0.05)로 다시 선별한 Rising 용어 결과를 비교합니다.",
    )
    loaded = _load_cluster_trends(version)
    if loaded is None:
        command = "python analysis\\run_fdr_pipeline.py" if version == "BH-FDR 보정 후" else "python analysis\\cluster_trends.py"
        st.warning(f"선택한 분석 버전의 군집 추세 데이터가 없습니다. 먼저 다음 명령을 실행해주세요:\n\n`{command}`")
        return

    series, summary = loaded
    if "label" not in summary.columns:
        summary = summary.copy()
        summary["label"] = summary["cluster"].map(lambda cluster: f"FDR 군집 {int(cluster)}")
    st.caption(
        f"{version} 결과입니다. 각 군집의 Rising 용어가 하나 이상 제목에 등장한 논문 수를 해당 연도의 전체 영어 논문 수로 나눈 비중입니다. "
        "한 논문이 여러 군집의 용어를 포함하면 군집별로 중복 집계됩니다."
    )
    if version == "BH-FDR 보정 후":
        st.info(
            "FDR 결과의 잠정 주제명은 FDR로 다시 선별된 군집의 상위 용어를 바탕으로 별도로 제안한 라벨입니다. "
            "따라서 기존 p-value 기준 결과의 주제명과 다를 수 있으며, 두 결과의 군집 구성과 해석을 비교해야 합니다."
        )
    st.info(
        "이 분석은 전체 연구주제의 성장률이 아니라, 이미 Rising으로 판정된 용어들이 형성한 군집의 연도별 분포를 보여줍니다. "
        "따라서 군집의 주제 구조와 시간적 변화는 확인할 수 있지만, 수자원 분야 전체에서 해당 주제가 가장 빠르게 성장했다고 단정하는 지표는 아닙니다."
    )

    display_summary = summary.rename(columns={
        "cluster": "군집",
        "label": "잠정 주제명",
        "term_count": "용어 수",
        "mean_document_share_pct": "평균 논문 비중(%)",
        "mk_tau": "MK tau",
        "mk_p": "MK p",
        "sen_slope_pct_point_per_year": "Sen 기울기(%p/년)",
        "recent5_minus_first5_pct_point": "최근5년-초기5년(%p)",
        "top_terms": "평균비중 상위 용어",
    })
    summary_columns = [
        "군집", "잠정 주제명", "용어 수", "평균 논문 비중(%)", "MK tau", "MK p",
        "Sen 기울기(%p/년)", "최근5년-초기5년(%p)", "평균비중 상위 용어",
    ]
    for column in summary_columns:
        if column not in display_summary.columns:
            display_summary[column] = ""
    display_summary = display_summary[summary_columns]
    st.subheader("군집별 추세 요약")
    st.dataframe(display_summary, width="stretch", hide_index=True)

    selected_cluster = st.selectbox(
        "시계열을 볼 군집",
        summary["cluster"].tolist(),
        format_func=lambda cluster: f"군집 {cluster}: {summary.loc[summary['cluster'].eq(cluster), 'label'].iloc[0]}",
        key="cluster_trend_selected",
    )
    selected = series[series["cluster"].eq(selected_cluster)].sort_values("year")
    line = alt.Chart(selected).mark_line(point=True).encode(
        x=alt.X("year:O", title="연도"),
        y=alt.Y("document_share_pct:Q", title="군집 포함 논문 비중(%)"),
        tooltip=[
            alt.Tooltip("year:O", title="연도"),
            alt.Tooltip("documents:Q", title="군집 포함 논문 수"),
            alt.Tooltip("total_documents:Q", title="전체 영어 논문 수"),
            alt.Tooltip("document_share_pct:Q", title="비중(%)", format=".3f"),
        ],
    ).properties(height=360)
    st.altair_chart(line, width="stretch")
    st.download_button(
        "⬇️ 군집 추세 시계열 CSV",
        data=series.to_csv(index=False).encode("utf-8-sig"),
        file_name="cluster_trends.csv",
        mime="text/csv",
        key="cluster_trends_download",
    )


def _summary_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.rename(columns={
        "cluster": "군집",
        "label": "잠정 주제명",
        "term_count": "용어 수",
        "mean_document_share_pct": "평균 논문 비중(%)",
        "mk_tau": "MK tau",
        "mk_p": "MK p",
        "sen_slope_pct_point_per_year": "Sen 기울기(%p/년)",
        "recent5_minus_first5_pct_point": "최근5년-초기5년(%p)",
        "top_terms": "평균비중 상위 용어",
    })
    columns = [
        "군집", "잠정 주제명", "용어 수", "평균 논문 비중(%)", "MK tau", "MK p",
        "Sen 기울기(%p/년)", "최근5년-초기5년(%p)", "평균비중 상위 용어",
    ]
    for column in columns:
        if column not in display.columns:
            display[column] = ""
    return display[columns]


def _cluster_share_chart(series: pd.DataFrame, cluster: int, title: str) -> alt.Chart:
    selected = series[series["cluster"].eq(cluster)].sort_values("year")
    return alt.Chart(selected).mark_line(point=True).encode(
        x=alt.X("year:O", title="연도"),
        y=alt.Y("document_share_pct:Q", title="군집 포함 논문 비중(%)"),
        tooltip=[
            alt.Tooltip("year:O", title="연도"),
            alt.Tooltip("documents:Q", title="군집 포함 논문 수"),
            alt.Tooltip("document_share_pct:Q", title="비중(%)", format=".3f"),
        ],
    ).properties(title=title, height=300)


def _render_cluster_trends_tab() -> None:
    raw_loaded = _load_cluster_trends("기존 p-value 기준")
    fdr_loaded = _load_cluster_trends("BH-FDR 보정 후")
    if raw_loaded is None or fdr_loaded is None:
        st.warning(
            "기존 p-value 및 BH-FDR 군집 추세 결과가 모두 필요합니다. "
            "`python analysis\\cluster_trends.py`와 `python analysis\\run_fdr_pipeline.py`를 실행해주세요."
        )
        return

    raw_series, raw_summary = raw_loaded
    fdr_series, fdr_summary = fdr_loaded
    st.caption("왼쪽은 기존 p-value 기준, 오른쪽은 BH-FDR 보정 후 기준입니다. FDR 적용 후에는 Rising 용어와 Louvain 군집을 다시 계산했습니다.")
    st.info(
        "두 표의 군집 번호는 서로 같은 군집이라는 뜻이 아닙니다. 정확한 군집 대응과 용어 겹침은 '검증 비교' 탭에서 확인하세요."
    )
    st.caption(
        "네트워크: 군집별 상위 용어(문서 빈도 기준)만 노드로 표시합니다. 좌표는 실제 동시출현 그래프에 "
        "스프링 레이아웃을 한 번 계산해서 고정한 것이라, 같은 군집끼리 엣지가 촘촘해 자연스럽게 뭉쳐 보이고 "
        "군집 간 약한 연결은 옅은 선으로만 이어집니다. 특정 군집 하나를 고르면 그 군집 내부만 확대해서 볼 수 있습니다."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("기존 p-value 기준")
        st.dataframe(_summary_for_display(raw_summary), width="stretch", hide_index=True)
        raw_network_cluster = st.selectbox("Louvain 군집", ["전체 군집"] + raw_summary["cluster"].tolist(), key="raw_louvain_cluster")
        raw_top_n = st.slider("표시할 상위 용어 수(전체 군집 선택 시 군집별)", 10, 40, 15, step=5, key="raw_louvain_top_n")
        raw_nodes, raw_edges = _load_louvain_network("기존 p-value 기준", raw_network_cluster, raw_top_n)
        _render_louvain_network(raw_nodes, raw_edges)
    with right:
        st.subheader("BH-FDR 보정 후")
        st.dataframe(_summary_for_display(fdr_summary), width="stretch", hide_index=True)
        fdr_network_cluster = st.selectbox("Louvain 군집", ["전체 군집"] + fdr_summary["cluster"].tolist(), key="fdr_louvain_cluster")
        fdr_top_n = st.slider("표시할 상위 용어 수(전체 군집 선택 시 군집별)", 10, 40, 15, step=5, key="fdr_louvain_top_n")
        fdr_nodes, fdr_edges = _load_louvain_network("BH-FDR 보정 후", fdr_network_cluster, fdr_top_n)
        _render_louvain_network(fdr_nodes, fdr_edges)

    common_clusters = sorted(set(raw_summary["cluster"]) & set(fdr_summary["cluster"]))
    st.divider()
    _render_fdr_comparison(raw_series, fdr_series)


def _render_fdr_comparison(raw_series: pd.DataFrame, fdr_series: pd.DataFrame) -> None:
    st.subheader("FDR 전후 군집 대응 및 시계열 비교")
    st.caption("FDR 전후 군집을 공통 용어가 최대가 되도록 대응시킨 뒤, 대응된 군집의 두 시계열을 하나의 그래프에서 비교합니다.")
    raw_partition_path = cluster_trends.ROOT / "analysis" / "output_임계15" / "rising_unigram_clusters.parquet"
    fdr_partition_path = cluster_trends.ROOT / "analysis" / "output_fdr" / "rising_unigram_clusters_fdr.parquet"
    alignment = cluster_alignment.compute(raw_partition_path, fdr_partition_path)
    raw_summary_path = cluster_trends.ROOT / "analysis" / "output_임계15" / "cluster_trends_summary.csv"
    fdr_summary_path = cluster_trends.ROOT / "analysis" / "output_fdr" / "cluster_trends_summary_fdr.csv"
    raw_labels = pd.read_csv(raw_summary_path).set_index("cluster")["label"].to_dict()
    fdr_labels = pd.read_csv(fdr_summary_path).set_index("cluster")["label"].to_dict()
    alignment.insert(1, "raw_label", alignment["raw_cluster"].map(raw_labels))
    alignment.insert(3, "fdr_label", alignment["fdr_cluster"].map(fdr_labels))
    alignment_display = alignment.rename(columns={
        "raw_cluster": "기존 군집", "raw_label": "기존 주제명", "fdr_cluster": "FDR 군집", "fdr_label": "FDR 주제명",
        "raw_terms": "기존 용어 수", "fdr_terms": "FDR 용어 수", "shared_terms": "공통 용어 수", "jaccard": "Jaccard",
        "raw_retained_pct": "기존 용어 유지율(%)", "fdr_supported_pct": "FDR 용어 중 공통 비율(%)",
    })
    st.dataframe(alignment_display, width="stretch", hide_index=True)
    pairs = list(alignment.itertuples(index=False))
    pair_index = st.selectbox(
        "비교할 대응 군집쌍", range(len(pairs)),
        format_func=lambda index: f"기존 {pairs[index].raw_cluster} ↔ FDR {pairs[index].fdr_cluster} (Jaccard {pairs[index].jaccard:.3f})",
        key="fdr_alignment_pair_main",
    )
    pair = pairs[pair_index]
    raw_plot = raw_series[raw_series["cluster"].eq(pair.raw_cluster)][["year", "document_share_pct"]].copy()
    raw_plot["분석 버전"] = "기존 p-value 기준"
    fdr_plot = fdr_series[fdr_series["cluster"].eq(pair.fdr_cluster)][["year", "document_share_pct"]].copy()
    fdr_plot["분석 버전"] = "BH-FDR 보정 후"
    plot = pd.concat([raw_plot, fdr_plot], ignore_index=True)
    chart = alt.Chart(plot).mark_line(point=True).encode(
        x=alt.X("year:O", title="연도"), y=alt.Y("document_share_pct:Q", title="군집 포함 논문 비중(%)"),
        color=alt.Color("분석 버전:N", title="분석 버전"),
        tooltip=["year:O", "분석 버전:N", alt.Tooltip("document_share_pct:Q", title="비중(%)", format=".3f")],
    ).properties(title="대응 군집의 연도별 비중 비교", height=400)
    st.altair_chart(chart, width="stretch")


def _render_validation_tab() -> None:
    st.subheader("최소빈도 기준 민감도")
    st.caption(
        "10·15·20건 비교는 15건이 통계적으로 최적이라는 것을 증명하는 분석이 아닙니다. "
        "최소빈도 선택을 바꿨을 때 분석 대상 용어와 Louvain 구조가 얼마나 달라지는지 확인하는 안정성 점검입니다."
    )
    sensitivity_path = cluster_trends.ROOT / "reports" / "min_count_sensitivity.csv"
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path).rename(columns={
            "min_doc_count": "최소빈도(건)",
            "candidate_terms": "분석대상 용어 수",
            "rising_overlap_with_15_jaccard": "Rising 용어집 Jaccard(15 기준)",
            "rising_terms": "Rising 용어 수",
            "edges": "네트워크 연결 수",
            "isolated_terms": "고립 용어 수",
            "clusters": "군집 수",
            "largest_cluster": "최대 군집 크기",
            "smallest_cluster": "최소 군집 크기",
            "modularity": "Modularity",
            "cluster_sizes": "군집 크기(내림차순)",
        })
        st.dataframe(sensitivity, width="stretch", hide_index=True)
        st.info(
            "이 표는 15건을 선택하는 근거 중 하나일 뿐입니다. 현재 결과에서는 10건에서 작은 고립 군집이 생기고, "
            "15건과 20건에서는 5개 군집 구조가 유지되었습니다. 따라서 15건은 희귀 용어를 줄이면서 전문용어를 남기는 "
            "실무적 절충값으로 설명할 수 있지만, 유일한 최적값이라고 주장할 수는 없습니다."
        )
    else:
        st.warning("민감도 분석 결과가 없습니다. `python analysis\\sensitivity_min_count.py`를 먼저 실행해주세요.")

    st.divider()
    st.subheader("FDR 전후 군집 대응")
    st.caption(
        "FDR 적용 전후에 네트워크와 Louvain을 각각 다시 수행한 뒤, 군집 번호가 아니라 공통 용어 수가 최대가 되도록 군집을 대응시킨 비교입니다. "
        "이는 FDR 군집 자체를 해석하는 표가 아니라, FDR 적용으로 군집 구성이 얼마나 달라졌는지 확인하는 검증표입니다."
    )
    raw_partition_path = cluster_trends.ROOT / "analysis" / "output_임계15" / "rising_unigram_clusters.parquet"
    fdr_partition_path = cluster_trends.ROOT / "analysis" / "output_fdr" / "rising_unigram_clusters_fdr.parquet"
    if not raw_partition_path.exists() or not fdr_partition_path.exists():
        st.warning("FDR 전후 군집 결과가 없습니다. FDR 파이프라인을 먼저 실행해주세요.")
        return
    alignment = cluster_alignment.compute(raw_partition_path, fdr_partition_path)
    raw_summary_path = cluster_trends.ROOT / "analysis" / "output_임계15" / "cluster_trends_summary.csv"
    fdr_summary_path = cluster_trends.ROOT / "analysis" / "output_fdr" / "cluster_trends_summary_fdr.csv"
    raw_labels = pd.read_csv(raw_summary_path).set_index("cluster")["label"].to_dict()
    fdr_labels = pd.read_csv(fdr_summary_path).set_index("cluster")["label"].to_dict()
    alignment.insert(1, "raw_label", alignment["raw_cluster"].map(raw_labels))
    alignment.insert(3, "fdr_label", alignment["fdr_cluster"].map(fdr_labels))
    alignment_display = alignment.rename(columns={
        "raw_cluster": "기존 군집", "raw_label": "기존 주제명",
        "fdr_cluster": "FDR 군집", "fdr_label": "FDR 주제명",
        "raw_terms": "기존 용어 수", "fdr_terms": "FDR 용어 수",
        "shared_terms": "공통 용어 수", "jaccard": "Jaccard",
        "raw_retained_pct": "기존 용어 유지율(%)",
        "fdr_supported_pct": "FDR 용어 중 공통 비율(%)",
    })
    st.dataframe(alignment_display, width="stretch", hide_index=True)

    raw_loaded = _load_cluster_trends("기존 p-value 기준")
    fdr_loaded = _load_cluster_trends("BH-FDR 보정 후")
    if raw_loaded is not None and fdr_loaded is not None:
        raw_series, _ = raw_loaded
        fdr_series, _ = fdr_loaded
        pair_options = list(alignment.itertuples(index=False))
        pair_index = st.selectbox(
            "비교할 대응 군집쌍",
            range(len(pair_options)),
            format_func=lambda index: (
                f"기존 {pair_options[index].raw_cluster} ↔ FDR {pair_options[index].fdr_cluster} "
                f"(Jaccard {pair_options[index].jaccard:.3f})"
            ),
            key="fdr_alignment_pair",
        )
        pair = pair_options[pair_index]
        raw_plot = raw_series[raw_series["cluster"].eq(pair.raw_cluster)][["year", "document_share_pct"]].copy()
        raw_plot["버전"] = "기존 p-value"
        fdr_plot = fdr_series[fdr_series["cluster"].eq(pair.fdr_cluster)][["year", "document_share_pct"]].copy()
        fdr_plot["버전"] = "BH-FDR"
        comparison_plot = pd.concat([raw_plot, fdr_plot], ignore_index=True)
        st.subheader("대응 군집의 연도별 비중 비교")
        st.caption("군집 번호가 아니라 위에서 대응시킨 군집쌍을 비교합니다. 두 선을 겹쳐 보면 FDR 적용 후 시간 변화가 어떻게 달라졌는지 확인할 수 있습니다.")
        chart = alt.Chart(comparison_plot).mark_line(point=True).encode(
            x=alt.X("year:O", title="연도"),
            y=alt.Y("document_share_pct:Q", title="군집 포함 논문 비중(%)"),
            color=alt.Color("버전:N", title="분석 버전"),
            tooltip=["year:O", "버전:N", alt.Tooltip("document_share_pct:Q", title="비중(%)", format=".3f")],
        ).properties(height=360)
        st.altair_chart(chart, width="stretch")

    st.divider()
    st.subheader("Louvain 동시출현 네트워크")
    st.caption(
        "선택한 분석 버전과 군집의 상위 용어를 노드로 표시합니다. 노드 크기는 문서 등장 빈도, 선의 굵기는 동시출현 횟수를 뜻합니다. "
        "전체 용어를 한 화면에 표시하면 읽기 어려워 문서 빈도 상위 용어를 중심으로 그립니다."
    )
    network_version = st.radio(
        "네트워크 분석 버전", ["기존 p-value 기준", "BH-FDR 보정 후"], horizontal=True, key="louvain_network_version"
    )
    network_loaded = _load_cluster_trends(network_version)
    if network_loaded is not None:
        _, network_summary = network_loaded
        network_cluster = st.selectbox(
            "네트워크를 볼 군집", network_summary["cluster"].tolist(), key="louvain_network_cluster"
        )
        network_top_n = st.slider("표시할 상위 용어 수", 15, 80, 35, step=5, key="louvain_network_top_n")
        nodes, edges = _load_louvain_network(network_version, int(network_cluster), network_top_n)
        _render_louvain_network(nodes, edges)


@st.cache_data(show_spinner="동시출현 네트워크 계산 중...")
def _load_louvain_network(version: str, cluster: int | str, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if version == "BH-FDR 보정 후":
        partition_path = cluster_trends.ROOT / "analysis" / "output_fdr" / "rising_unigram_clusters_fdr.parquet"
    else:
        partition_path = cluster_trends.ROOT / "analysis" / "output_임계15" / "rising_unigram_clusters.parquet"
    partition = pd.read_parquet(partition_path)
    term_column = "용어" if "용어" in partition.columns else partition.columns[1]
    all_clusters = cluster == "전체 군집"
    if all_clusters:
        candidate_partition = partition
    else:
        candidate_partition = partition.loc[partition["cluster"].eq(int(cluster))]
    terms = set(candidate_partition[term_column])

    source = cluster_trends.ROOT / "analysis" / "output_임계15"
    corpus = pd.read_parquet(source / "filtered_corpus.parquet", columns=["id", "language"])
    english_ids = set(corpus.loc[corpus["language"].eq("en"), "id"])
    long = pd.read_parquet(source / "unigram_long.parquet", columns=["id", "term"])
    mapping = pd.read_csv(source / "term_normalization_dict.csv")
    mapping = dict(zip(mapping["variant"], mapping["canonical"]))
    long["term"] = long["term"].map(lambda term: mapping.get(term, term))
    long = long[long["id"].isin(english_ids) & long["term"].isin(terms)].drop_duplicates(["id", "term"])
    term_counts = long.groupby("term")["id"].nunique().sort_values(ascending=False)
    if all_clusters:
        selected_terms = []
        for _, group in candidate_partition.groupby("cluster"):
            group_terms = group[term_column].tolist()
            selected_terms.extend(term_counts.reindex(group_terms).dropna().sort_values(ascending=False).head(top_n).index.tolist())
        top_terms = selected_terms
    else:
        top_terms = term_counts.head(top_n).index.tolist()
    long = long[long["term"].isin(top_terms)]
    doc_codes = long["id"].astype("category")
    term_codes = long["term"].astype("category")
    matrix = sparse.csr_matrix(
        (np.ones(len(long)), (doc_codes.cat.codes, term_codes.cat.codes)),
        shape=(len(doc_codes.cat.categories), len(term_codes.cat.categories)),
    )
    co = (matrix.T @ matrix).tocoo()
    terms_index = term_codes.cat.categories
    term_to_cluster = partition.set_index(term_column)["cluster"].to_dict()
    # 군집 간 연결도 그대로 남긴다 — before/2606_statics_all/network_community.html과 같은 느낌을
    # 내려면 인위적으로 엣지를 지우는 대신, _render_louvain_network에서 실제 그래프에 스프링
    # 레이아웃을 얼려서 같은 군집끼리 저절로 뭉쳐 보이게 한다.
    edges = []
    for row, col, weight in zip(co.row, co.col, co.data):
        if row < col and weight >= 2:
            edges.append({"source": terms_index[row], "target": terms_index[col], "weight": int(weight)})
    nodes = pd.DataFrame({
        "term": top_terms,
        "documents": [int(term_counts[t]) for t in top_terms],
        "cluster": [int(term_to_cluster[t]) for t in top_terms],
    })
    return nodes, pd.DataFrame(edges, columns=["source", "target", "weight"])


def _cluster_palette(cluster_ids: list) -> dict:
    """군집 수만큼 색상을 자동 생성한다 — before/2606_statics_all/network_community.html은
    matplotlib colormap(tab20/gist_rainbow)으로 이 일을 하는데, 여기선 같은 효과를 표준
    라이브러리 colorsys로 낸다 (matplotlib을 새로 끌어올 필요 없이 색상환을 고르게 나눔)."""
    n = max(len(cluster_ids), 1)
    colors = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.65, 0.92)
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return dict(zip(cluster_ids, colors))


def _hybrid_cluster_layout(nodes: pd.DataFrame, edges: pd.DataFrame, k: float = 1.2,
                            safety_factor: float = 1.5) -> dict:
    """군집별로 스프링 레이아웃을 따로 계산한 뒤, 군집끼리 겹치지 않을 만큼만 떨어뜨려 놓은
    앵커 위치로 옮겨 붙인다.

    처음엔 before/2606_statics_all/network_community.html처럼 전체 그래프(군집 간 엣지 포함)에
    스프링 레이아웃을 통째로 돌리는 방식을 그대로 시도했었다. 그런데 실제로 재보니(스크래치패드
    검증 스크립트) 여기 Rising 용어 네트워크는 상위 15개 기준 군집 간 엣지가 전체의 약 80%를
    차지해서(고빈도 용어일수록 여러 주제에 걸쳐 등장하기 때문 — 기관 네트워크와 다른 지점),
    전체를 한 번에 스프링 레이아웃 돌리면 군집이 절반 정도밖에 안 갈라졌다(같은 군집 노드가
    자기 군집 중심에 제일 가까운 비율이 56~63%에 그침).
    그래서 군집 내부는 실제 동시출현 강도로 스프링 레이아웃을 돌려 유기적인 모양을 살리고,
    군집 자체는 서로 겹치지 않을 최소 거리만큼 앵커를 벌려서 확실히 분리한다(검증 결과 top_n
    10~40 전 구간에서 자기 군집 최근접 비율 100%). 군집 간 엣지는 지우지 않고 옅은 선으로
    남겨서 어떤 군집끼리 실제로 이어지는지는 그대로 보여준다."""
    cluster_of = nodes.set_index("term")["cluster"].to_dict()
    cluster_ids = sorted(nodes["cluster"].unique())
    n_clusters = len(cluster_ids)

    graph = nx.Graph()
    graph.add_nodes_from(nodes["term"])
    for row in edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=float(row.weight))

    local_radius_by_cluster = {c: 0.35 + 0.05 * (nodes["cluster"] == c).sum() for c in cluster_ids}
    max_local_radius = max(local_radius_by_cluster.values())
    anchor_radius = safety_factor * max_local_radius / math.sin(math.pi / n_clusters) if n_clusters > 1 else 0.0

    pos = {}
    for index, cluster_id in enumerate(cluster_ids):
        members = nodes.loc[nodes["cluster"].eq(cluster_id), "term"].tolist()
        local_pos = nx.spring_layout(graph.subgraph(members), weight="weight", k=k, seed=42, iterations=200)
        xs = np.array([p[0] for p in local_pos.values()])
        ys = np.array([p[1] for p in local_pos.values()])
        span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)
        local_radius = local_radius_by_cluster[cluster_id]
        cx = anchor_radius * math.cos(2 * math.pi * index / n_clusters)
        cy = anchor_radius * math.sin(2 * math.pi * index / n_clusters)
        for term, (x, y) in local_pos.items():
            pos[term] = (cx + (x - xs.mean()) / span * local_radius, cy + (y - ys.mean()) / span * local_radius)
    return pos


def _render_louvain_network(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    if nodes.empty:
        st.info("선택한 군집에 표시할 용어가 없습니다.")
        return
    if edges.empty:
        st.info("선택한 상위 용어 사이에 동시출현 2건 이상인 연결이 없습니다.")
        st.dataframe(nodes, width="stretch", hide_index=True)
        return

    pos = _hybrid_cluster_layout(nodes, edges)
    scale = 900

    network = Network(
        height="680px", width="100%", bgcolor="#171827", font_color="#F4F5FA",
        notebook=False, cdn_resources="in_line",
    )
    network.toggle_physics(False)

    max_documents = max(nodes["documents"].max(), 1)
    cluster_ids = sorted(nodes["cluster"].unique())
    color_by_cluster = _cluster_palette(cluster_ids)

    for row in nodes.itertuples(index=False):
        x, y = pos[row.term]
        size = 12 + 28 * (row.documents / max_documents) ** 0.5
        network.add_node(
            row.term,
            label=row.term,
            title=f"{row.term}<br>군집 {row.cluster}<br>문서 {row.documents:,}건",
            value=float(row.documents),
            size=float(size),
            group=f"cluster_{row.cluster}",
            color=color_by_cluster[row.cluster],
            x=float(x * scale),
            y=float(y * scale),
            physics=False,
            fixed=True,
        )
    max_weight = max(edges["weight"].max(), 1)
    for row in edges.itertuples(index=False):
        network.add_edge(
            row.source,
            row.target,
            value=float(row.weight),
            width=min(row.weight / max_weight * 6, 6),
            color="rgba(180,180,180,0.35)",
            title=f"동시출현 {row.weight:,}건",
        )
    network.options = json.loads("""
    {
        "interaction": {"hover": true, "zoomView": true, "dragView": true},
        "physics": {"enabled": false},
        "nodes": {"borderWidth": 1,
            "font": {"color": "#F4F5FA", "size": 16, "face": "Arial", "strokeWidth": 3, "strokeColor": "#171827"}},
        "edges": {"smooth": false, "selectionWidth": 2}
    }
    """)
    legend = " · ".join(
        f"<span style='color:{color_by_cluster[c]}'>● 군집 {c}</span>" for c in cluster_ids
    )
    st.markdown(f"군집별 색상: {legend}", unsafe_allow_html=True)
    components.html(network.generate_html(notebook=False), height=650, scrolling=False)


def _render_frequency_tab() -> None:
    doc_counts = _load_doc_counts()
    journals = ["(전체)"] + sorted(doc_counts["journal"].unique())
    year_min, year_max = int(doc_counts["year"].min()), int(doc_counts["year"].max())
    all_years = list(range(year_min, year_max + 1))

    if "trend_years" not in st.session_state:
        st.session_state["trend_years"] = all_years

    st.caption("연도 선택 (기본값: 전체). 아래 버튼으로 5년 구간을 한 번에 고를 수도 있습니다.")
    preset_cols = st.columns(len(YEAR_PRESETS) + 1)
    for col, (label, (p0, p1)) in zip(preset_cols[:-1], YEAR_PRESETS.items()):
        if col.button(label, key=f"trend_preset_{label}"):
            st.session_state["trend_years"] = [y for y in all_years if p0 <= y <= p1]
    if preset_cols[-1].button("전체", key="trend_preset_all"):
        st.session_state["trend_years"] = all_years

    selected_years = st.pills("연도", all_years, selection_mode="multi", key="trend_years")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        ngram_type = st.radio("용어 단위", ["유니그램", "바이그램"], horizontal=True, key="trend_ngram_type")
    with col2:
        sel_journal = st.selectbox("저널", journals, key="trend_journal")
    with col3:
        top_n = st.slider("표시 개수", 10, 100, 30, step=10, key="trend_top_n")

    if not selected_years:
        st.info("연도를 1개 이상 선택해주세요.")
        return

    long_df = _load_long(ngram_type)
    long_df = long_df[long_df["year"].isin(selected_years)]
    doc_slice = doc_counts[doc_counts["year"].isin(selected_years)]
    if sel_journal != "(전체)":
        long_df = long_df[long_df["journal"] == sel_journal]
        doc_slice = doc_slice[doc_slice["journal"] == sel_journal]

    total_docs = doc_slice["id"].nunique()
    if total_docs == 0 or long_df.empty:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    # 정규화 사전 적용: "용어 정규화" 탭에서 사전에 등록한 표기들이 여기서 하나로 합쳐진다.
    mapping = normalization.load_dict()
    long_df = long_df.assign(term=normalization.apply_dict(long_df["term"], mapping))

    # 문서 단위 빈도(document frequency): 한 논문 제목 안에서 같은 용어가 중복 등장해도 1건으로 센다.
    freq = long_df.groupby("term")["id"].nunique().sort_values(ascending=False)
    top = freq.head(top_n)

    st.caption(
        f"대상 논문 {total_docs:,}건 · 고유 용어 {len(freq):,}개"
        + (f" · 정규화 사전 {len(mapping):,}개 표기 적용됨" if mapping else " · 정규화 사전 아직 비어있음")
    )

    col_table, col_cloud = st.columns([2, 3])
    with col_table:
        table = top.reset_index()
        table.columns = ["용어", "건수"]
        table["비중"] = (table["건수"] / total_docs * 100).round(1).astype(str) + "%"
        st.dataframe(table, width="stretch", height=420)
    with col_cloud:
        img = _make_wordcloud(top.to_dict())
        st.image(img, width="stretch")


@st.cache_data(show_spinner="태그 계산 중 (용어당 통계검정 1회, 수 초 소요)...")
def _compute_tags_cached(term_year_counts: pd.DataFrame, term_year_share: pd.DataFrame,
                          candidate_terms: list) -> pd.DataFrame:
    return trend_tags.compute_tags(term_year_counts, term_year_share, candidate_terms)


def _term_trend_band_chart(share_series: pd.Series, recent_years: int = trend_tags.ZSCORE_RECENT_YEARS) -> alt.Chart:
    """실제 값(선+점) 위에 Sen_slope 추세선(빨간 점선)과 '과거 평균±표준편차' 밴드를 한
    차트에 겹친다. 원래는 두 차트(추세선 오버레이/밴드)로 나뉘어 있었는데, 결국 같은 실제값
    시계열 위에 오버레이만 다른 거라 하나로 합쳤다. 최근 N년 점만 주황색으로 강조."""
    trend_series = trend_tags.trend_line_for_term(share_series)
    history = share_series.iloc[:-recent_years]
    hist_mean, hist_std = history.mean(), history.std(ddof=0)
    band_df = pd.DataFrame({
        "연도": share_series.index,
        "하단": max(hist_mean - hist_std, 0.0),
        "상단": hist_mean + hist_std,
    })
    band = alt.Chart(band_df).mark_area(opacity=0.18, color="#72B7B2").encode(
        x=alt.X("연도:O"), y=alt.Y("하단:Q", title="비중(%)"), y2="상단:Q"
    )

    recent_cut = share_series.index[-recent_years]
    line_df = pd.DataFrame({
        "연도": share_series.index,
        "값": share_series.values,
        "구간": ["최근" if y >= recent_cut else "과거" for y in share_series.index],
    })
    line = alt.Chart(line_df).mark_line(color="#4C78A8").encode(x=alt.X("연도:O"), y=alt.Y("값:Q"))
    points = alt.Chart(line_df).mark_point(filled=True, size=90).encode(
        x=alt.X("연도:O"), y=alt.Y("값:Q"),
        color=alt.Color("구간:N", scale=alt.Scale(domain=["과거", "최근"], range=["#4C78A8", "#F58518"])),
    )

    trend_df = pd.DataFrame({"연도": trend_series.index, "값": trend_series.values})
    trend = alt.Chart(trend_df).mark_line(strokeDash=[6, 4], color="#E45756").encode(
        x=alt.X("연도:O"), y=alt.Y("값:Q")
    )

    return (band + line + points + trend).properties(height=320)


def _term_burst_chart(share_series: pd.Series) -> alt.Chart:
    """용어 하나의 연도별 버스트 강도를 막대로 — 히트맵과 같은 색스케일을 써서 두 그래프가
    같은 색 언어를 쓰게 한다. 양수(빨강 계열)=급등 가속, 음수(파랑 계열)=급감 가속
    (scheme="redblue"는 기본이 저값=빨강/고값=파랑이라 관례와 반대라 reverse=True로 뒤집음)."""
    burst_series = trend_tags.macd_burstiness_series(share_series)
    burst_df = pd.DataFrame({"연도": burst_series.index, "버스트강도": burst_series.values})
    zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#B0B0B0").encode(y="y:Q")
    bars = alt.Chart(burst_df).mark_bar().encode(
        x=alt.X("연도:O"),
        y=alt.Y("버스트강도:Q"),
        color=alt.Color("버스트강도:Q", scale=alt.Scale(scheme="redblue", domainMid=0, reverse=True), legend=None),
        tooltip=["연도", "버스트강도"],
    )
    return (zero_rule + bars).properties(height=280)


def _burst_heatmap(term_year_share: pd.DataFrame, terms: list) -> alt.Chart:
    rows = []
    for term in terms:
        burst_series = trend_tags.macd_burstiness_series(term_year_share.loc[term])
        for year, val in burst_series.items():
            rows.append({"용어": term, "연도": year, "버스트강도": val})
    heat_df = pd.DataFrame(rows)
    return alt.Chart(heat_df).mark_rect().encode(
        x=alt.X("연도:O"),
        y=alt.Y("용어:N", sort=terms),
        color=alt.Color("버스트강도:Q", scale=alt.Scale(scheme="redblue", domainMid=0, reverse=True)),
        tooltip=["용어", "연도", "버스트강도"],
    ).properties(height=max(280, 22 * len(terms)))


def _render_signal_tags_tab() -> None:
    st.caption(
        "용어별로 신규진입 여부·추세·속도·최근 급등·버스트 이력을 태그(컬럼)로 보여줍니다. "
        "각 컬럼 뜻은 표 아래 '컬럼 설명'에 있습니다.\n\n"
        f"※ 최소빈도 필터(현재 26년 통틀어 {MIN_TOTAL_COUNT}건 이상)를 통과한 용어만 분석 대상으로 삼습니다. "
    )

    doc_counts_all = _load_doc_counts()
    journals = ["(전체)"] + sorted(doc_counts_all["journal"].unique())

    col1, col2 = st.columns([1, 1])
    with col1:
        ngram_type = st.radio("용어 단위", ["유니그램", "바이그램"], horizontal=True, key="tags_ngram_type")
    with col2:
        sel_journal = st.selectbox("저널", journals, key="tags_journal")

    mapping = normalization.load_dict()
    long_df_all = _load_long(ngram_type)
    long_df_all = long_df_all.assign(term=normalization.apply_dict(long_df_all["term"], mapping))

    long_df = long_df_all
    doc_counts = doc_counts_all
    if sel_journal != "(전체)":
        long_df = long_df[long_df["journal"] == sel_journal]
        doc_counts = doc_counts[doc_counts["journal"] == sel_journal]

    if doc_counts.empty or long_df.empty:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        return

    year_doc_totals = doc_counts.groupby("year")["id"].nunique().sort_index()
    term_year_counts = long_df.groupby(["term", "year"])["id"].nunique().unstack(fill_value=0)
    term_year_counts = term_year_counts.reindex(columns=year_doc_totals.index, fill_value=0)
    term_year_share = term_year_counts.div(year_doc_totals, axis=1) * 100

    overall_total = term_year_counts.sum(axis=1)
    candidate_terms = list(overall_total[overall_total >= MIN_TOTAL_COUNT].sort_values(ascending=False).index)

    if not candidate_terms:
        st.info("선택한 조건에서 비교할 만큼 충분한 빈도의 용어가 없습니다.")
        return

    # "전체 저널"은 export_trend_tags.py가 미리 계산해둔 파일이 있으면 그걸 읽어서 매번
    # 재계산(용어당 통계검정 1회씩, 수 초)하는 걸 건너뛴다. 특정 저널 하나만 골랐을 때는
    # 조합이 저널 수(12개)만큼 늘어나서 그것까진 미리 안 만들어뒀으니 그때그때 계산한다.
    precomputed = trend_tags.load_precomputed(ngram_type) if sel_journal == "(전체)" else None
    if precomputed is not None:
        tags_df = precomputed
        if len(tags_df) != len(candidate_terms):
            st.warning(
                f"미리 계산된 파일({len(tags_df):,}개)과 현재 데이터 기준 용어 수({len(candidate_terms):,}개)가 "
                "다릅니다 — 데이터나 정규화 사전이 바뀐 뒤 `python -m analysis.export_trend_tags`를 "
                "다시 안 돌린 것 같습니다. 최신화 전까지는 이 파일 내용을 보여줍니다."
            )
    else:
        tags_df = _compute_tags_cached(term_year_counts, term_year_share, candidate_terms)

    # z-score의 '최근 N년' 기준은 통계 검정이 아니라 단순 평균·표준편차라 계산이 가벼우므로,
    # MK/Sen_slope처럼 캐시에 박아두지 않고 슬라이더로 매번 다시 계산한다 (compute_tags의
    # 무거운 부분은 그대로 캐시/precomputed 값을 씀).
    recent_years = st.slider(
        "'최근 N년' 기준 (zscore·밴드 그래프에 씀)", 1, 10, trend_tags.ZSCORE_RECENT_YEARS, key="tags_recent_years"
    )
    zscore_col = f"최근{recent_years}년_zscore"
    fresh_zscore = trend_tags.compute_zscore_column(term_year_share, tags_df["용어"].tolist(), recent_years)
    tags_df = tags_df.drop(columns=[c for c in tags_df.columns if c.endswith("년_zscore")])
    tags_df[zscore_col] = tags_df["용어"].map(fresh_zscore).round(2)

    st.divider()
    st.subheader("📋 태그 테이블")
    st.markdown(
        "- **신규등장(2015~)** — 'O'면 2015년 이후 처음 등장, 'X'면 그 전부터 있던 용어\n"
        "- **추세확인(MK)** — 26년 동안 해당 용어가 통계적으로 늘었는지/줄었는지를 Mann-Kendall 검정으로 확인\n"
        "  - `MK_tau`: 26년치 조합으로 증가했으면 +1, 감소했으면 -1 → 합계하여 최종 tau값 계산\n"
        "  - `MK_p`: tau로 본 추세가 통계적으로 믿을만한가 판단하는 지표로, 0.05 이하일 때 유의미\n"
        "  - 라벨링\n"
        "    - MK_p<0.05 이고 MK_tau>0 이면, `Rising`\n"
        "    - MK_p<0.05 이고 MK_tau<0 이면, `Declining`\n"
        "    - 그 외(MK_p≥0.05) `Not Significant` — '변화가 없다고 증명됨'이 아니라 "
        "'유의미한 추세라고 확신할 통계적 근거가 부족하다'는 뜻 (가설검정 일반 원칙: 유의하지 않다 ≠ 변화가 없다는 증명)\n"
        "    - MK_p 자체가 계산 안 됨(드묾, 자기상관 보정 공식이 수치적으로 깨지는 경우) → `판정불가` — "
        "이땐 Sen_slope(빨간 점선)는 별개 계산이라 정상적으로 나옴, 다만 그 방향이 통계적으로 유의미한지만 확인 불가\n"
        "- **Sen_slope** — 얼마나 빠르게 변하였는가? 1년에 비중이 몇 %p씩 변하는지 보는 지표. 숫자가 클수록 "
        "빠르게 변하는 용어라고 할 수 있으며, 추세선 오버레이 그래프의 빨간 점선이 이 값에 해당\n"
        f"- **Z-Score** (표의 `{zscore_col}`) — 최근 N년에 튀는 애가 있는지 (슬라이더로 N 조절 가능)\n"
        "  - 양수: 최근 갑자기 늘었다\n"
        "  - 음수: 최근 갑자기 줄었다\n"
        "  - 급등/급감에 대해 확인 가능하며, 최근 급등 확인(밴드) 그래프 참고\n"
        "- **버스트** — 빵 떴다가 사라진 애가 있는가? 버스트 히트맵(여러 용어 비교) / 개별 용어 그래프의 "
        "버스트 강도(단일 용어) 참고\n"
        "- **평균비중(%)** — 원래 얼마나 큰데? 추세와 무관하게 크기에 대한 정보 확인 가능. 논문의 전체 몇 %였는지(평균). "
        "앞선 지표들이 '변화'만 보기에, 크고 중요한 단어(water 등)와 원래 희귀한 단어의 구분이 어려워서 넣은 지표임\n\n"
        "(방법론 출처: Mann-Kendall/Sen's slope/z-score는 Marrone 2020, 버스트 강도는 Tattershall et al. 2020 — "
        "Scientometrics 122호. 다만 'MK_p≥0.05→Not Significant' 판정 규칙 자체는 두 논문에 그대로 나온 게 아니라 "
        "이 둘을 조합해 제가 설계한 것입니다 — 자세한 근거는 대화로 설명드립니다.)"
    )

    fc1, fc2 = st.columns([1.3, 1])
    with fc1:
        trend_filter = st.multiselect(
            "추세 필터", trend_tags.TREND_LABELS, default=trend_tags.TREND_LABELS, key="tags_trend_filter"
        )
    with fc2:
        top_n = st.slider("표시 개수(N)", 10, 200, 50, step=10, key="tags_top_n")

    sc1, sc2 = st.columns([2, 1])
    with sc1:
        search = st.text_input("용어 검색(부분일치)", key="tags_search")
    with sc2:
        new_only = st.checkbox("신규진입 용어만", key="tags_new_only")

    filtered = tags_df[tags_df["추세(MK)"].isin(trend_filter)]
    if search:
        filtered = filtered[filtered["용어"].str.contains(search, case=False, na=False)]
    if new_only:
        filtered = filtered[filtered["신규진입(2015~)"]]

    # 위 N개를 "무엇을 기준으로" 뽑을지 결정하는 정렬 — 기본값은 평균비중(가장 크게/흔하게
    # 쓰이는 용어부터). 화면에 뜬 표 자체는 Streamlit 기본기능(컬럼 헤더 클릭)으로 사용자가
    # 원하는 컬럼 기준으로 다시 정렬해서 볼 수 있다 — 그래서 여기엔 별도 정렬 UI를 안 둔다.
    filtered = filtered.sort_values("전체기간_평균비중(%)", ascending=False)
    view_df = filtered.head(top_n).reset_index(drop=True)

    st.caption(
        f"조건에 맞는 {len(filtered):,}개 중 평균비중이 큰 순으로 {len(view_df):,}개 표시 "
        f"(최소빈도 필터 통과 전체 {len(tags_df):,}개 중) · 표의 컬럼 제목을 클릭하면 그 컬럼 기준으로 다시 정렬됩니다"
    )
    # bool 그대로 두면 st.dataframe이 체크박스로 그려서 뭘 뜻하는지 헷갈리므로 표시용으로만 문자열로 바꾼다
    # (view_df 자체는 아래 그래프들에서 계속 boolean/원본 타입 그대로 써야 하니 복사본에만 적용).
    display_df = view_df.copy()
    display_df["신규진입(2015~)"] = display_df["신규진입(2015~)"].map({True: "O", False: "X"})
    st.dataframe(display_df, width="stretch", height=420, hide_index=True)
    st.download_button(
        "⬇️ CSV로 다운로드",
        data=tags_df.to_csv(index=False).encode("utf-8-sig"),  # BOM 있어야 엑셀에서 한글이 안 깨짐
        file_name=f"trend_tags_{ngram_type}_{sel_journal}.csv",
        mime="text/csv",
        key="tags_download_csv",
        help=f"위 표는 상위 {len(view_df):,}개만 보여주지만, 이 버튼은 최소빈도 필터를 통과한 전체 "
             f"{len(tags_df):,}개를 받습니다 (지금 조건: {ngram_type}·{sel_journal}).",
    )

    if view_df.empty:
        return

    st.divider()
    st.subheader("📈 개별 용어 그래프")
    term_options = list(view_df["용어"])
    picked_term = st.selectbox("용어 선택 (위 표에 표시된 용어 중에서)", term_options, key="tags_term_single")

    # 고른 용어의 태그값을 전부 한 줄로 — 아래 그래프 두 개가 정확히 이 중 어떤 숫자를
    # 그림으로 옮긴 건지 바로 옆에서 대조할 수 있게.
    picked_row = view_df[view_df["용어"] == picked_term].iloc[0]
    rank_by_share = int((view_df["전체기간_평균비중(%)"] > picked_row["전체기간_평균비중(%)"]).sum()) + 1
    st.caption(
        f"**{picked_term}** — 추세 `{picked_row['추세(MK)']}` (MK_tau {picked_row['MK_tau']}, "
        f"MK_p {picked_row['MK_p']}) · Sen_slope {picked_row['Sen_slope(%p_연)']}%p/년 · "
        f"{zscore_col} {picked_row[zscore_col]} · "
        f"최대버스트강도 {picked_row['최대버스트강도']} ({picked_row['버스트연도']}년) · "
        f"평균비중 {picked_row['전체기간_평균비중(%)']}% (표시된 {len(view_df)}개 중 {rank_by_share}위)"
    )
    st.caption(
        "아래 그래프들의 y축 '비중(%)'은 그 해 발행된 논문 중 이 용어가 제목에 들어간 논문의 비율입니다 "
        "(표의 평균비중(%)은 이 값을 26년치 평균낸 것). 버스트 강도 그래프의 y축은 별도 계산식(아래 "
        "'어떻게 그려졌나요?' 참고)으로 나온 지표라 단위가 없습니다."
    )

    gc1, gc2 = st.columns([1.3, 1])
    with gc1:
        st.caption(
            "**추세·급등 확인** — 실제 값(파란 선) 위에 Sen_slope로 그은 추세선(빨간 점선)과 "
            "'과거 평균±표준편차' 밴드(청록)를 함께 겹칩니다. 최근 N년만 주황 점으로 강조."
        )
        with st.expander("어떻게 그려졌나요?"):
            st.markdown(
                "**추세선(빨간 점선)**: 태그 테이블의 **Sen_slope** 값을 그대로 그린 직선입니다 (매년 두 점씩 "
                "모든 쌍의 기울기를 구해서 그 중앙값을 쓰는 방식이라 이상치에 잘 안 흔들림). 이 점선이 파란 "
                "실선(실제 값)을 잘 따라가면 `추세(MK)` 판정이 눈으로 봐도 말이 된다는 뜻입니다.\n\n"
                f"**밴드(청록)**: 전체 연도를 **최근 {recent_years}년**(위 슬라이더로 조절)과 **그 이전(과거)**"
                "으로 나눠, 과거 구간만으로 평균±표준편차 범위를 깝니다 — \"과거 기준으로 이 정도는 흔한 "
                f"범위\"라는 뜻. **최근 {recent_years}년**만 주황 점으로 찍어서, 밴드를 벗어났는지 바로 봅니다.\n\n"
                f"태그 테이블의 **{zscore_col}**가 이 밴드 비교를 숫자 하나로 요약한 것입니다 — "
                f"`(최근 {recent_years}년 평균 − 과거 평균) ÷ 과거 표준편차`. zscore가 +1이면 \"최근 평균이 "
                "밴드 윗선에 딱 걸쳐있다\", +2면 \"밴드 폭만큼 두 배 벗어났다\"는 뜻입니다."
            )
        st.altair_chart(
            _term_trend_band_chart(term_year_share.loc[picked_term], recent_years=recent_years), width="stretch"
        )
    with gc2:
        st.caption("**버스트 강도** — 연도별 막대. 빨강 계열(양수)=급등 가속, 파랑 계열(음수)=급감 가속.")
        with st.expander("어떻게 그려졌나요?"):
            st.markdown(
                "주식시장 기술적분석 기법인 MACD를 응용합니다 — 단기(6년)·장기(12년) 이동평균의 "
                "차이(MACD선)를 다시 3년으로 평활화한 값(시그널선)과 비교해서, 그 차이(히스토그램)를 "
                "**그 용어의 역대 최고 비중의 제곱근**으로 나눠 정규화한 것이 이 막대 값입니다 "
                "(Tattershall et al. 2020). 이렇게 정규화하는 이유: water처럼 원래 비중이 큰 용어의 "
                "사소한 변동이, 희귀어의 진짜 버스트보다 부풀려 보이는 걸 막기 위해서입니다.\n\n"
                "MK 검정은 '한쪽 방향으로 쭉'인 경우만 잡고 이런 급등-급락은 상쇄되어 놓칠 수 있는데, "
                "이 그래프가 그걸 보완합니다. 태그 테이블의 **최대버스트강도/버스트연도**가 이 막대들 "
                "중 가장 큰(양수 기준) 값과 그 연도입니다."
            )
        st.altair_chart(_term_burst_chart(term_year_share.loc[picked_term]), width="stretch")

    st.divider()
    st.subheader("🔥 버스트 히트맵 (표시 상위 20개, 여러 용어 동시 비교용)")
    st.caption("행=용어, 열=연도, 색=MACD 기반 버스트 강도(양수=급등 가속, 음수=급감 가속). Marrone(2020) Fig.1/2와 같은 방식.")
    heat_terms = term_options[:20]
    st.altair_chart(_burst_heatmap(term_year_share, heat_terms), width="stretch")


def _render_normalization_tab() -> None:
    st.caption(
        "워드클라우드/빈도표에서 같은 개념이 여러 표기로 흩어져 보이는 걸 통합하는 곳입니다. "
        "아래 후보는 하이픈/공백 유무, 단복수, -ing형 등 **표기가 비슷한 유니그램을 자동으로 묶어서** "
        "보여준 것일 뿐이고, 정말 같은 뜻인지·어떤 표기를 대표형(canonical)으로 쓸지는 사람이 "
        "직접 정해서 저장해야 사전에 반영됩니다. 저장하면 '빈도 분석' 탭 결과에 바로 반영됩니다."
    )

    mapping = normalization.load_dict()

    st.subheader(f"📖 현재 정규화 사전 ({len(mapping)}개 표기 등록됨)")
    if mapping:
        dict_df = pd.DataFrame(sorted(mapping.items()), columns=["표기 (variant)", "대표형 (canonical)"])
        st.dataframe(dict_df, width="stretch", height=250)
    else:
        st.info("아직 등록된 정규화 규칙이 없습니다. 아래 후보들을 보면서 하나씩 채워보세요.")

    st.divider()
    st.subheader("✏️ 직접 추가")
    st.caption(
        "아래 자동 후보는 하이픈/공백/단복수/-ing형처럼 표기가 비슷한 것만 잡아냅니다. "
        "MBR→membrane bioreactor 같은 약어나, 후보에 안 뜬 표기는 여기서 직접 입력해서 등록하세요."
    )
    manual_variants_raw = st.text_input(
        "표기(variant) — 쉼표로 구분해 여러 개 입력 가능",
        key="norm_manual_variants",
        placeholder="예: mbr, membrane bio-reactor, membrane bioreactor",
    )
    manual_canonical = st.text_input("대표형(canonical)", key="norm_manual_canonical")
    if st.button("직접 추가한 표기 저장", key="norm_manual_save"):
        manual_variants = [v.strip() for v in manual_variants_raw.split(",") if v.strip()]
        manual_canonical_clean = manual_canonical.strip()
        if not manual_variants:
            st.warning("표기를 1개 이상 입력해주세요.")
        elif not manual_canonical_clean:
            st.warning("대표형을 입력해주세요.")
        else:
            normalization.save_mapping(manual_variants, manual_canonical_clean)
            st.cache_data.clear()
            st.success(f"{manual_variants} → '{manual_canonical_clean}' 로 저장했습니다.")
            st.rerun()

    st.divider()
    st.subheader("🔍 정규화 후보 (전체 코퍼스 기준, 영향 큰 순)")
    st.caption("특정 연도/저널로 필터링하지 않고 전체 데이터 기준으로 계산합니다 — 사전은 항상 전체에 적용되니까요.")

    uni_long = _load_long("유니그램")
    full_freq = uni_long.groupby("term")["id"].nunique()

    progress = normalization.candidate_progress(full_freq, mapping)
    p1, p2 = st.columns(2)
    p1.metric(
        "빈도 기준 커버리지",
        f"{progress['coverage']:.1%}",
        help=(
            "전체 유니그램 등장 횟수 중, 정규화 사전에 등록된 표기가 차지하는 비중입니다. "
            "후보는 항상 빈도 큰 순으로 보여주기 때문에, 남은 후보 개수가 많아 보여도 "
            "이미 영향력 큰 용어부터 처리되고 있는 경우가 많습니다. 100%를 채울 필요는 없고, "
            "커버리지가 어느 정도 올라오면(예: 40~60%) 나머지는 희귀어라 효과가 작으니 "
            "여기서 멈추고 다음 단계로 넘어가도 괜찮습니다."
        ),
    )
    p2.metric("남은 후보 그룹 수", f"{progress['remaining_clusters']:,}개 (한 번에 30개씩 표시)")

    clusters = normalization.find_candidate_clusters(full_freq, mapping)

    if not clusters:
        st.success("현재 기준으로 더 이상 눈에 띄는 후보가 없습니다.")
        return

    for cluster in clusters:
        members = cluster["members"]
        variants = [t for t, _ in members]
        member_desc = " / ".join(f"{t} ({c:,}건)" for t, c in members)

        with st.expander(f"{member_desc}  —  합계 {cluster['total']:,}건"):
            guess = members[0][0]  # 가장 빈도 높은 표기를 기본 대표형으로 제안
            canonical = st.text_input(
                "대표형(canonical)으로 통합", value=guess, key=f"norm_canonical_{cluster['key']}"
            )
            if st.button("이 대표형으로 저장", key=f"norm_save_{cluster['key']}"):
                normalization.save_mapping(variants, canonical)
                st.cache_data.clear()
                st.success(f"{variants} → '{canonical}' 로 저장했습니다.")
                st.rerun()
