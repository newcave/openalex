# 연구소별 연구트렌드 대시보드

OpenAlex에서 연구소(분야)별 수집 대상 저널의 논문을 모으고, 논문 제목에서 부상하는 용어(Rising Term)를
찾아 연구주제 커뮤니티로 묶어 보여주는 Dash 웹앱입니다.

분야 = 연구소: AI · 수자원환경 · 물인프라안전 · 상하수도 · 물에너지 · 경영 (`config.py`에서 정의)

## 실행

```bash
cd /home/openalex/src
.venv/bin/pip install -r requirements.txt   # 처음 한 번
.venv/bin/python webapp/app.py              # http://<서버>:8050
```

- 반드시 이 폴더(프로젝트 루트)에서 실행합니다. 코드가 `dataset/`, `results/`를 데이터 루트(`OPENALEX_DATA_ROOT`, 없으면 프로젝트 루트) 기준 경로로 찾습니다.
- 수집 데이터와 분석 산출물은 비공개 리포 [newcave/openalex-data](https://github.com/newcave/openalex-data)에 있습니다. 이 리포 옆에 클론하고 `.env`에 `OPENALEX_DATA_ROOT=../openalex-data`를 적습니다.
- `webapp/app.py` 마지막 줄의 `app.run(...)`에서 host/port/debug를 설정합니다. 외부 접속은 `host="0.0.0.0"`, 운영 시에는 `debug=False`를 씁니다.

### 서버 부팅 시 자동 실행 (systemd)

AWS EC2 인스턴스가 부팅될 때 웹앱도 자동으로 시작하고, 인스턴스가 종료될 때 웹앱도 함께 종료하려면 서버에서 한 번 등록합니다.

```bash
sudo cp /home/openalex/src/deploy/openalex-webapp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now openalex-webapp
```

상태와 로그 확인:

```bash
sudo systemctl status openalex-webapp
sudo journalctl -u openalex-webapp -f
```

`enable`은 다음 서버 부팅에도 자동 시작하게 합니다. EC2 인스턴스가 정지되면 systemd가 앱 프로세스도 종료합니다. 이 설정은 EC2 인스턴스의 부팅·종료에 연결되며, 별도의 웹사이트 프로세스가 재시작되는 경우까지 앱을 따라 재시작시키지는 않습니다. 앱은 포트 `8050`을 사용하므로, 외부에서 직접 접속할 경우 AWS 보안 그룹 및 도메인/리버스 프록시 설정도 별도로 확인해야 합니다.

### 환경변수 (`.env`, 저장소에 포함하지 않음)
| 이름 | 용도 |
|---|---|
| `OPENAI_API_KEY` | 커뮤니티 이름(AI 라벨)·해설 생성, 새 저널의 분야 자동 분류. 모델은 `config.py`의 `OPENAI_MODEL` |
| `OPENALEX_API_KEY` | OpenAlex 논문 수집 (없어도 수집은 되지만 하루 요청 한도가 낮음) |
| `OPENALEX_DATA_ROOT` | `dataset/`, `results/`, `archive/` 데이터가 있는 폴더 (openalex-data 클론 위치, 상대경로는 프로젝트 루트 기준). 없으면 프로젝트 루트 |

루트에 `.env` 파일로 두거나 서버 환경변수로 설정합니다. 값은 담당자에게 따로 받습니다.

## 화면 구성

| 사이드바 | 내용 |
|---|---|
| 데이터 › 데이터셋 현황 | 분야별 요약(수집된 대상 저널 수, 논문 수, 용량, 분석 결과 유무) + 분야별 저널 수집 상태·연도 히트맵 |
| 데이터 › 데이터 수집 | 분야를 고르면 그 분야 대상 저널 ISSN이 채워지고, 수집하면 `dataset/<분야>/`에 저장. "AI 자동 분류"를 고르면 저널마다 분야를 정해 저장 |
| 데이터 › 수집현황 모니터링 | 일자별 수집량, 실패 요청, 로그 |
| 분야별 분석 › <분야> | 전체기간 분석 / 구간별분석 / 키워드 검색 |

### 데이터 수집 규칙
- 여러 분야에 같은 저널이 있으면 **분야마다 따로** 받습니다(예: Water Research는 AI·수자원환경·상하수도 3곳).
- 이미 받은 연도 파일은 건너뛰므로, 중단돼도 다시 누르면 이어서 받습니다.
- **AI 자동 분류**: 수집 대상 목록(`config.py`)에 있는 ISSN은 목록의 분야로 보냅니다. 목록에 없는 ISSN은 OpenAlex 저널 정보(이름·토픽·최근 논문 제목 20개)를 보고 AI(`config.py`의 `OPENAI_MODEL`)가 분야를 고릅니다(여러 개 또는 없음). 판단 결과는 `dataset/ai_journal_assignments.json`에 쌓이고, 같은 저널은 다시 묻지 않습니다. 판단이 틀렸으면 이 파일의 `domains`를 직접 고치면 됩니다.

## 폴더 구조

```
src/
├── README.md
├── requirements.txt
├── .env                     API 키 (공유 금지)
├── config.py                분야 설정: 분야별 수집 대상 저널, dataset/results 경로 (모든 코드가 공통으로 사용)
│
├── dataset/                 [데이터] 수집 원본. 파일 1개 = 저널 1개의 1년치 (저널명_ISSN_연도.parquet)
│   ├── AI/  수자원환경/  물인프라안전/  상하수도/  물에너지/  경영/
│   └── ai_journal_assignments.json   AI가 분야를 판단한 저널 기록 (AI 자동 분류를 쓰면 생김)
│
├── pipeline_codes/          [코드] 수집: OpenAlex -> dataset/<분야>/        (실행: pipeline.py)
├── analysis_codes/          [코드] 분석: dataset/<분야>/ -> results/<분야>/  (실행: pipeline.py)
├── webapp/                  [코드] 대시보드                                  (실행: app.py)
│
├── results/                 [데이터] 분석 산출물. 분야마다 같은 구조 ("분석 진행하기"를 누르면 생김)
│   └── <분야>/
│       ├── 01_corpus/
│       ├── 02_full_period/
│       ├── 03_full_period_review/
│       └── 04_windowed/
│
├── logs/                    실행 로그(journal_app.log), 수집/분석 이벤트 기록(events/*.jsonl, 모니터링 화면 원본)
└── archive/                 보관용: 수자원 논문 버전 데이터·결과, 논문 실험 스크립트 (웹앱과 무관, archive/README.md 참고)
```

## 전체 흐름

```
[수집] pipeline_codes/pipeline.py   (웹앱 "데이터 수집" 화면이 호출)
   (AI 자동 분류일 때) journal_router -> 저장할 분야 결정
   n_01_openalex -> n_02_institution_extract -> cache_manager(저장)
   결과: dataset/<분야>/*.parquet

[분석] analysis_codes/pipeline.py   (웹앱 "분석 진행하기" 버튼이 호출)
   n_01_build_corpus       코퍼스 필터링 (논문 유형/언어)             -> results/<분야>/01_corpus/
   n_02_term_frequency     제목 -> 유니그램/바이그램 용어 테이블        -> results/<분야>/01_corpus/
   n_03_rising_terms       MK 추세검정 + FDR 보정으로 Rising 용어 선별  (함수만 제공)
   n_04_cluster_network    동시출현 네트워크 + Louvain 커뮤니티 도출     -> results/<분야>/02_full_period/
   n_05_review_docs        커뮤니티별 근거문서(용어/엣지/논문)          -> results/<분야>/03_full_period_review/
   (웹앱에서 실행 시)       AI 커뮤니티 이름 생성                       -> results/<분야>/03_full_period_review/ai_labels.json
   n_06_windowed_analysis  5년 구간별 분석 (전체기간과 별도)           -> results/<분야>/04_windowed/
   * n_01~n_02 전처리는 dataset이 지난 전처리 이후 바뀌었을 때만 다시 한다.

[화면] webapp/app.py
```

## 코드 파일 설명

### pipeline_codes/ (수집)
| 파일 | 역할 |
|---|---|
| `pipeline.py` | 수집 실행. 연도별로 [논문 수집 -> 기관 정보 파싱 -> 저장] 반복 |
| `journal_router.py` | 저널(ISSN)을 저장할 분야 결정 (목록 조회 또는 AI 판단) |
| `n_01_openalex.py` | OpenAlex API 호출. OpenAlex 저널 정보 오류 보정표(`DOI_PREFIX_SOURCES` 등)도 여기 있음 |
| `n_02_institution_extract.py` | 저자 소속기관(ROR) 파싱 |
| `n_03_ror_local_match.py` | (선택, 수동 실행) ROR 덤프로 기관 매칭 보강 |
| `n_04_visualize.py` | 기관 협업 네트워크 시각화 |
| `cache_manager.py` | dataset 폴더 저장/조회 (이미 받은 연도는 다시 받지 않음) |
| `event_log.py` | 수집/분석 이벤트를 `logs/events/`에 기록 |
| `utils.py` | 로거 등 공통 도구 |

### analysis_codes/ (분석)
| 파일 | 역할 |
|---|---|
| `pipeline.py` | 분석 실행. n_01~n_06을 순서대로 호출 (전체기간 `run_full`, 구간별 `run_windowed`) |
| `n_01_build_corpus.py` ~ `n_06_windowed_analysis.py` | 위 "전체 흐름"의 분석 단계 |
| `text_utils.py` | 제목 토큰화, 불용어 (n_02에서 사용) |
| `normalization.py` | 용어 정규화 사전 적용 (n_03에서 사용) |
| `trend_tags.py` | Mann-Kendall 추세 지표 계산 (n_03에서 사용) |
| `term_normalization_seed.csv` | 새 분야의 정규화 사전 시작본 (수자원 논문 때 만든 단·복수·철자 통일 사전) |

웹앱 없이 실행: `python -m analysis_codes.pipeline --domain sewerage` (`--only full` 또는 `--only windowed`로 한쪽만 실행 가능.
분야 key는 `config.py`의 `ai / water_env / infra / sewerage / energy / management`)

### webapp/ (대시보드)
| 파일 | 역할 |
|---|---|
| `app.py` | **유일한 실행 파일.** 화면 레이아웃과 콜백 |
| `assets/` | 로고, CSS (Dash 규칙상 app.py 옆에 있어야 함) |
| `backend/services.py` | app.py가 쓰는 데이터 읽기. 수집/분석 버튼은 pipeline_codes, analysis_codes의 pipeline.py 호출 |
| `backend/ai_insights.py` | OpenAI로 커뮤니티 이름/해설 생성 |
| `backend/trajectory_style_example.json` | 구간별 흐름 요약 생성 시 문체 예시로 쓰는 수자원 논문 문단 |
| `backend/monitor.py` | 수집현황 모니터링 화면 집계 |
| `.state/`, `.cache_diskcache/` | 실행 중 자동 생성 (분야별 마지막 수집 기록, 백그라운드 작업 큐) |

## results/<분야>/ 파일 설명

| 폴더 | 파일 | 내용 |
|---|---|---|
| `01_corpus` | `filtered_corpus.parquet` | 분석 대상 논문 (필터링 후) |
| | `unigram_long.parquet`, `bigram_long.parquet` | 논문별 제목 용어 (1행 = 논문 1편 x 용어 1개) |
| | `term_normalization_dict.csv` | 용어 정규화 사전 (models -> model 등) |
| | `_cache_stamp.json` | 마지막 전처리 때의 dataset 파일 목록 (바뀌었는지 판단용) |
| `02_full_period` | `trend_tags_unigram_fdr.*` | 용어별 추세 판정 (Rising/Declining 등) |
| | `rising_unigram_clusters.parquet` | Rising 용어 -> 커뮤니티 번호 |
| | `cluster_trends_fdr_assoc.parquet`, `cluster_trends_summary_fdr_assoc.csv` | 커뮤니티별 연도 추이와 요약 |
| `03_full_period_review` | `terms.csv`, `edges.csv`, `all_papers.csv` | 웹앱 전체기간 화면이 읽는 근거 데이터 |
| | `ai_labels.json` | AI가 생성한 커뮤니티 이름·해설 |
| | `evidence.md`, `manifest.json` | 사람이 읽는 근거 문서, 입력 파일 해시 |
| `04_windowed` | `<시작>_<끝>/` | 구간별 용어 추세와 커뮤니티 |
| | `windows_summary.csv` | 구간별 요약 (웹앱이 구간 목록을 여기서 읽음) |

## 분야 추가·수정
- **저널 추가/삭제**: `config.py`에서 그 분야의 `J("저널명", "Print ISSN", "Electronic ISSN")` 줄을 고칩니다.
- **새 분야**: `config.py`에서 `_domain("key", "분야명", "연구소명", [...])`을 하나 더 만들어 `DOMAINS`에 넣으면 `dataset/분야명`, `results/분야명`이 같은 구조로 쓰이고 사이드바에도 나타납니다.
- **ISSN 주의**: OpenAlex에서 ISSN이 다른 저널에 잘못 붙어 있는 경우가 있습니다(예: 원본 목록의 Journal of Contaminant Hydrology 전자 ISSN 1879-2553은 OpenAlex에서 다른 물리학 저널). 새 ISSN은 https://api.openalex.org/sources?filter=issn:XXXX-XXXX 로 저널명을 확인한 뒤 넣으세요.
