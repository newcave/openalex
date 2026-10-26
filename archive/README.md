# archive (보관용)

## 수자원_논문/
2026-10-20까지 웹앱의 "수자원 분야"였던 논문 버전(저널 12종, 2000~2025). 논문에 실린 확정 결과이며,
연구소별 분야 체계(config.py)로 바뀌면서 웹앱에서 빠졌다. 새 "수자원환경" 분야와는 저널 구성이 다르다.

| 경로 | 내용 |
|---|---|
| `dataset/` | 수집 원본 304개 파일 |
| `results/01_corpus ~ 04_windowed/` | 논문에 쓴 분석 결과. `03_full_period_review/`에 논문 커뮤니티 이름(interpretations.json), 논문 해설(paper_interpretations.json), 분석 과정 설명(00_분석_과정_설명.md) 포함 |
| `collect_job_water.json` | 마지막 수집 기록 |

이 중 두 파일은 지금 코드도 쓰므로 코드 옆에 복사해 두었다:
`results/01_corpus/term_normalization_dict.csv` -> `analysis_codes/term_normalization_seed.csv`,
`results/03_full_period_review/paper_trajectories.json` -> `webapp/backend/trajectory_style_example.json`.

## 논문 실험 스크립트

논문 작성 과정에서 쓴 실험·검증 스크립트와 그 산출물입니다. **웹앱은 이 폴더를 전혀 사용하지 않습니다.**
기록 보존 목적이며, 그대로는 실행되지 않습니다(예전 폴더 구조 `analysis/`, `analysis/output_임계15/`,
`reports/` 기준 경로와 `from analysis import ...`가 남아 있음). 다시 돌려야 한다면 경로와 import를
현재 구조(`analysis_codes/`, `archive/수자원_논문/results/`)에 맞게 고쳐야 합니다.

## analysis_experiments/
| 파일 | 용도 (논문 근거) |
|---|---|
| `explore_cooccurrence_threshold.py`, `explore_louvain_resolution.py` | 동시출현 임계값, Louvain resolution 탐색 |
| `cooccurrence_cluster.py`, `cluster_trends.py`, `cluster_alignment.py` | 초기(FDR 보정 전) 군집화 방식 |
| `sensitivity_min_count.py` | 최소 빈도(10/15/20) 민감도 |
| `weight_scheme_ablation.py` | 엣지 가중치 3종 비교 (§5.5.1) |
| `seed_stability.py`, `seed_stability_tau_windowed.py` | Louvain 시드 안정성 (§5.9) |
| `subcluster_stability.py` | 군집 하위구조 진단 (§5.10) |
| `multisignal_feasibility.py` | 다중신호 타당성 (§3.5.1) |
| `bigram_fdr_supplement.py` | 바이그램 보충표 |
| `run_fdr_assoc_pipeline_windowed.py` | 구간별 MK+FDR 시도 (구간 대부분에서 Rising 0개 -> tau>0 방식으로 대체) |
| `visualize_fdr_assoc_network.py` | 논문용 네트워크 그림 (`results/수자원/03_full_period_review/network.html`) |
| `audit_title_markup.py` | 제목 내 HTML 태그 점검 |
| `dashboard_view.py`, `export_trend_tags.py`, `export_term_charts*.py` | 예전 Streamlit 대시보드와 그래프 일괄 저장 |

## results_수자원_실험산출물/
위 스크립트들과 예전 버전 파이프라인이 만든 파일입니다 (폴더명은 `results/수자원/`의 어느 단계와 관련 있는지를 뜻함).
`01_corpus/analysis_output_사본/`은 예전 `analysis/output/` 폴더에 있던 사본 중 내용이 조금 달라 지우지 않고 남긴 파일입니다.
