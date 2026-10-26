# analysis_codes/pipeline.py
# 분석 실행: n_01 ~ n_06을 순서대로 호출한다. 웹앱의 "분석 진행하기" 버튼도 이 함수들을 부른다.
#
#   run_full()      전체기간 분석  [n_01 -> n_02] -> n_03 -> n_04 -> n_05 -> (AI 커뮤니티 이름)
#   run_windowed()  구간별 분석    [n_01 -> n_02] -> n_03 로드 -> n_06
#   [n_01 -> n_02] 전처리는 수집 데이터가 지난 전처리 이후 바뀌었을 때만 다시 한다.
#
# 분야별 입력/출력 경로는 config.py의 Domain에서 가져온다.
#
# 실행(프로젝트 루트에서):
#   python -m analysis_codes.pipeline --domain sewerage            전체기간 + 구간별
#   python -m analysis_codes.pipeline --domain sewerage --only full
# 커뮤니티 AI 이름은 OpenAI 호출이 웹앱 데이터 접근 계층을 쓰므로 웹앱에서 실행할 때만 생성된다
# (명령줄로 돌리면 이름 대신 대표 용어가 표시된다).

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_codes import n_01_build_corpus as build_corpus             # noqa: E402
from analysis_codes import n_02_term_frequency as term_frequency         # noqa: E402
from analysis_codes import n_03_rising_terms as rising_terms             # noqa: E402
from analysis_codes import n_04_cluster_network as cluster_network       # noqa: E402
from analysis_codes import n_05_review_docs as review_docs               # noqa: E402
from analysis_codes import n_06_windowed_analysis as windowed_analysis   # noqa: E402
from config import DATA_ROOT, DOMAINS, SEED_NORMALIZATION_DICT, Domain   # noqa: E402
from pipeline_codes import event_log                                     # noqa: E402


def _rel(path: Path) -> str:
    try:
        return path.relative_to(DATA_ROOT).as_posix() + "/"
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- 전처리 (n_01, n_02)
def _dataset_stamp(d: Domain) -> list:
    files = sorted(d.dataset_dir.glob("*.parquet")) if d.dataset_dir.exists() else []
    return [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files]


def ensure_corpus(d: Domain, emit) -> None:
    """dataset 폴더 -> filtered_corpus -> unigram/bigram 롱테이블 (n_01, n_02). 수집 데이터(파일 이름·크기·
    수정시각)가 지난 전처리 이후 바뀌지 않았으면 다시 하지 않는다."""
    stamp_path = d.shared_source / "_cache_stamp.json"
    stamp = _dataset_stamp(d)
    if not stamp:
        raise RuntimeError(
            f"{_rel(d.dataset_dir)}에 수집된 논문이 없습니다. 먼저 '데이터 수집' 화면에서 수집하세요."
        )
    up_to_date = (
        d.corpus_path.exists() and (d.shared_source / "unigram_long.parquet").exists()
        and stamp_path.exists()
        and json.loads(stamp_path.read_text(encoding="utf-8")) == stamp
    )
    if up_to_date:
        emit("수집 데이터가 지난 전처리 이후 바뀌지 않아 기존 전처리 결과를 그대로 씁니다.")
    else:
        filtered = build_corpus.main(cache_dir=d.dataset_dir, output_path=d.corpus_path)
        emit(f"코퍼스 필터링 완료 — {len(filtered):,}건 (type/comment·reply/language 필터)")
        term_frequency.main(corpus_path=d.corpus_path, output_dir=d.shared_source)
        emit("제목 유니그램/바이그램 추출 완료")
        stamp_path.write_text(json.dumps(stamp, ensure_ascii=False), encoding="utf-8")
    if not d.dict_path.exists():
        d.dict_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SEED_NORMALIZATION_DICT, d.dict_path)
        emit("용어 정규화 사전이 없어 시작본(단·복수·철자 표기 통일)을 복사했습니다.")


# ---------------------------------------------------------------- 분석 실행 기록(모니터링용)
def _recorded(d: Domain, kind: str, fn, emit) -> dict:
    try:
        result = fn(emit)
    except Exception as e:
        event_log.record("analysis", domain=d.key, kind=kind, status="fail",
                         reason=event_log.classify(str(e)), message=str(e))
        raise
    summary = {k: v for k, v in result.items() if isinstance(v, (int, float, str))}
    event_log.record("analysis", domain=d.key, kind=kind, status="ok", **summary)
    return result


# ---------------------------------------------------------------- 전체기간 분석
def run_full(d: Domain, emit=print, label_fn=None) -> dict:
    """label_fn(emit): 마지막 단계로 커뮤니티 이름을 생성하는 함수(웹앱이 넘김). 없으면 이 단계를 건너뛴다."""
    return _recorded(d, "full", lambda e: _run_full(d, e, label_fn), emit)


def _run_full(d: Domain, emit, label_fn) -> dict:
    with_labels = label_fn is not None
    total = 6 + int(with_labels)
    n = 0

    def step(msg):
        nonlocal n
        n += 1
        emit(f"{n}/{total} {msg}")

    step("수집 데이터 전처리 중 (필터링 + 제목 토큰화)...")
    ensure_corpus(d, emit)

    step("데이터 로드 중 (수집·전처리된 코퍼스 + 용어 테이블)...")
    corpus, long, counts, year_totals = rising_terms.load_inputs(
        source=d.shared_source, corpus_path=d.corpus_path, dict_path=d.dict_path)
    min_count = rising_terms.MIN_TOTAL_COUNT
    emit(f"로드 완료 — 분석대상 문서 {len(corpus):,}건, "
         f"후보 용어({len(year_totals)}년 {min_count}건 이상) {int((counts.sum(axis=1) >= min_count).sum()):,}개")

    step("Rising 용어 추출 중 (Hamed-Rao Mann-Kendall + Benjamini-Hochberg FDR)...")
    tags = rising_terms.build_fdr_tags(counts, year_totals)
    n_rising = int((tags["추세(MK-FDR)"] == "Rising").sum())
    emit(f"Rising 용어 {n_rising}개 확정")
    if n_rising < 2:
        raise RuntimeError(
            f"Rising 용어가 {n_rising}개뿐이라 동시출현 네트워크를 만들 수 없습니다. "
            "수집 연도 범위나 저널을 늘린 뒤 다시 시도하세요."
        )

    step("동시출현 네트워크 구축 중 (association strength 정규화)...")
    partition, graph_summary = cluster_network.build_clusters_association(long, tags)
    emit(f"엣지 {graph_summary['edges']:,}개 구축 완료")

    step("Louvain 커뮤니티 탐지 중...")
    emit(f"커뮤니티 {graph_summary['clusters']}개 탐지 (modularity={graph_summary['modularity']:.3f})")

    step("연도별 추이 계산 및 결과 저장 중...")
    series, summary = rising_terms.build_cluster_trends(corpus, long, partition)

    out = d.full_output
    out.mkdir(parents=True, exist_ok=True)
    tags.to_parquet(out / "trend_tags_unigram_fdr.parquet", index=False)
    tags.to_csv(out / "trend_tags_unigram_fdr.csv", index=False, encoding="utf-8-sig")
    out_partition = partition.rename(columns={"용어": "term"})
    cols = ["cluster", "term"] + [c for c in out_partition.columns if c not in ("cluster", "term")]
    out_partition = out_partition[cols]
    out_partition.to_parquet(out / "rising_unigram_clusters.parquet", index=False)
    series.to_parquet(out / "cluster_trends_fdr_assoc.parquet", index=False)
    summary.to_csv(out / "cluster_trends_summary_fdr_assoc.csv", index=False, encoding="utf-8-sig")

    emit("근거문서(군집별 용어/엣지/논문 목록) 갱신 중...")
    review_docs.main(partition_source=out, shared_source=d.shared_source, out=d.full_review)

    if with_labels:
        step("커뮤니티 이름 생성 중 (OpenAI)...")
        try:
            label_fn(emit)
        except Exception as e:
            emit(f"커뮤니티 이름 생성 실패 — 대표 용어로 대신 표시합니다: {e}")

    result = {"n_rising": n_rising, **graph_summary}
    _save_full_stamp(d, result)
    emit("완료")
    return result


# ---------------------------------------------------------------- 저장된 전체기간 결과 재사용
# 전체기간 분석이 끝나면 그때의 dataset 파일 목록과 결과 요약을 02_full_period/_analysis_stamp.json에 남긴다.
# 다음에 "분석 진행하기"를 눌렀을 때 dataset이 그대로면 다시 계산하지 않고 저장된 결과를 바로 쓴다.
def _full_stamp_path(d: Domain) -> Path:
    return d.full_output / "_analysis_stamp.json"


def _save_full_stamp(d: Domain, result: dict) -> None:
    summary = {k: v for k, v in result.items() if isinstance(v, (int, float, str))}
    _full_stamp_path(d).write_text(json.dumps({"dataset_stamp": _dataset_stamp(d), "summary": summary},
                                              ensure_ascii=False), encoding="utf-8")


def load_saved_full(d: Domain) -> dict | None:
    """dataset이 지난 전체기간 분석 이후 바뀌지 않았으면 그때의 결과 요약, 아니면 None."""
    try:
        saved = json.loads(_full_stamp_path(d).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if saved.get("dataset_stamp") != _dataset_stamp(d):
        return None
    if not (d.full_review / "terms.csv").exists():
        return None
    return saved.get("summary")


# ---------------------------------------------------------------- 구간별 분석
def run_windowed(d: Domain, emit=print) -> dict:
    return _recorded(d, "windowed", lambda e: _run_windowed(d, e), emit)


def _run_windowed(d: Domain, emit) -> dict:
    total = 3
    n = 0

    def step(msg):
        nonlocal n
        n += 1
        return f"{n}/{total} {msg}"

    emit(step("수집 데이터 전처리 중 (필터링 + 제목 토큰화)..."))
    ensure_corpus(d, emit)

    emit(step("데이터 로드 중..."))
    corpus, long, counts, year_totals = windowed_analysis.load_inputs(
        source=d.shared_source, corpus_path=d.corpus_path, dict_path=d.dict_path)
    out = d.windowed_output
    out.mkdir(parents=True, exist_ok=True)

    year_min, year_max = int(year_totals.index.min()), int(year_totals.index.max())
    win_list = windowed_analysis.compute_windows(year_min, year_max)
    emit(f"데이터 연도 범위 {year_min}-{year_max}({year_max - year_min + 1}년) -> {len(win_list)}개 구간(5~6년 단위)으로 분할")

    window_step = step("")
    summary_rows, all_topterms, done_windows = [], [], []
    for i, (start, end) in enumerate(win_list, 1):
        emit(f"{window_step}구간 {start}-{end} 분석 중 ({i}/{len(win_list)})...")
        try:
            run_summary, topterm_rows = windowed_analysis.run_window(
                start, end, corpus, long, counts, year_totals, output_dir=out)
        except Exception as e:
            # 수집량이 적은 분야는 특정 구간에 용어가 모자라 실패할 수 있다 - 그 구간만 건너뛴다.
            emit(f"  -> {start}-{end}: 분석 불가로 건너뜀 ({e})")
            continue
        summary_rows.append(run_summary)
        all_topterms.extend(topterm_rows)
        done_windows.append((start, end))
        emit(
            f"  -> {start}-{end}: tau>0 {run_summary['tau_positive_terms']}개, "
            f"커뮤니티 {run_summary['clusters']}개, modularity={run_summary['modularity']:.3f}"
        )

    if not summary_rows:
        raise RuntimeError("모든 구간이 분석 불가였습니다. 데이터를 더 수집한 뒤 다시 시도하세요.")

    summary_df = pd.DataFrame([
        {k: (v if not isinstance(v, list) else "|".join(map(str, v))) for k, v in row.items()}
        for row in summary_rows
    ])
    summary_df.to_csv(out / "windows_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_topterms).to_csv(out / "windows_cluster_topterms.csv", index=False, encoding="utf-8-sig")

    emit("완료")
    return {"windows": len(summary_rows), "window_ids": [f"{s}_{e}" for s, e in done_windows]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="분야 하나의 분석(n_01~n_06)을 실행한다.")
    parser.add_argument("--domain", required=True, choices=sorted(DOMAINS), help="config.py의 분야 키")
    parser.add_argument("--only", choices=["full", "windowed"], help="한쪽만 실행 (생략하면 둘 다)")
    args = parser.parse_args()
    domain = DOMAINS[args.domain]
    if args.only in (None, "full"):
        run_full(domain)
    if args.only in (None, "windowed"):
        run_windowed(domain)
