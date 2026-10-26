# config.py — 분야 설정 (프로젝트 전체 공통)
# 분야 하나 = 여기의 Domain 하나 = 연구소 하나. 대시보드 왼쪽 사이드바의 분야 목록도 이것이다.
# 수집(pipeline_codes), 분석(analysis_codes), 웹앱(webapp)이 모두 이 파일의 경로를 쓴다.
#
# 분야마다 수집 원본 / 분석 산출물 폴더를 완전히 분리한다. 여러 분야에 같은 저널이 있으면 분야마다
# 따로 받는다(같은 폴더를 같이 쓰면 n_01_build_corpus.py가 *.parquet를 전부 읽어 다른 분야 논문이 섞인다).
#
#   dataset/<분야>/                        수집 원본 (데이터 수집 화면이 저장)
#   results/<분야>/01_corpus/              n_01~n_02: 필터링 코퍼스, 제목 용어 테이블, 정규화 사전
#   results/<분야>/02_full_period/         n_03~n_04: 전체기간 Rising 용어, 커뮤니티, 연도별 추이
#   results/<분야>/03_full_period_review/  n_05: 커뮤니티별 근거문서(terms/edges/all_papers), AI 라벨
#   results/<분야>/04_windowed/            n_06: 5년 구간별 분석
#
# dataset/, results/는 DATA_ROOT 아래에 있다. 데이터는 비공개 리포(newcave/openalex-data)에 따로 두고
# .env의 OPENALEX_DATA_ROOT로 가리킨다(예: ../openalex-data, 상대경로는 이 폴더 기준). 없으면 이 폴더.
#
# 새 분야 추가: 아래 _domain(...)으로 Domain 하나를 더 만들어 DOMAINS에 넣으면 된다.
# 저널 목록 출처: 2026-10-20 부서별 수집 대상 저널 목록(사용자 제공). ISSN은 같은 날 OpenAlex에서 확인했다.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
DATA_ROOT = (ROOT / (os.environ.get("OPENALEX_DATA_ROOT")
                     or dotenv_values(ROOT / ".env").get("OPENALEX_DATA_ROOT")
                     or ".")).resolve()


@dataclass(frozen=True)
class Journal:
    name: str
    print_issn: str | None
    e_issn: str | None
    note: str = ""            # 수집 화면에 함께 보여줄 주의사항 (예: OpenAlex 미등록)

    @property
    def issns(self) -> list[str]:
        return [x for x in (self.print_issn, self.e_issn) if x]


@dataclass(frozen=True)
class Domain:
    key: str                  # 내부 식별자 (수집 기록 파일명, 이벤트 로그에 남음 - 바꾸지 말 것)
    name: str                 # 화면·AI 프롬프트에 쓰는 분야명 ("상하수도" -> "상하수도 분야 연구 동향...")
    department: str           # 이 분야를 쓰는 연구소
    journals: tuple           # 수집 대상 저널 (Journal, ...)
    dataset_dir: Path         # 수집 원본 parquet 폴더
    shared_source: Path       # unigram_long / filtered_corpus / 정규화사전 (전체·구간별 분석 공통 입력)
    corpus_path: Path         # 용어빈도 분모로 쓰는 filtered_corpus.parquet
    dict_path: Path           # 용어 정규화 사전
    full_output: Path         # 전체기간 분석 산출물
    full_review: Path         # 전체기간 분석 근거문서(terms/edges/all_papers) + AI 라벨
    windowed_output: Path     # 구간별분석 산출물

    @property
    def nav_label(self) -> str:
        return f"{self.name} 분야"

    @property
    def title(self) -> str:
        return f"{self.name} 분야 시기별 주요 연구주제 탐색"

    @property
    def description(self) -> str:
        return (
            f"{self.department} 수집 대상 저널 {len(self.journals)}종의 논문 데이터를 활용합니다. "
            "수집된 '논문 제목'으로 주요 연구주제를 탐색하기 위해 키워드 단어(Rising Term)를 선별하고, "
            "키워드를 통해 주요 연구주제 후보(Community)를 도출합니다. 연구주제 이름은 분석할 때마다 "
            "AI(OpenAI)가 생성한 참고용 라벨입니다."
        )


def _domain(key: str, name: str, department: str, journals: list[Journal]) -> Domain:
    """분야명(name)이 그대로 dataset/<name>, results/<name> 폴더 이름이 된다."""
    results = DATA_ROOT / "results" / name
    corpus = results / "01_corpus"
    return Domain(
        key=key, name=name, department=department, journals=tuple(journals),
        dataset_dir=DATA_ROOT / "dataset" / name,
        shared_source=corpus,
        corpus_path=corpus / "filtered_corpus.parquet",
        dict_path=corpus / "term_normalization_dict.csv",
        full_output=results / "02_full_period",
        full_review=results / "03_full_period_review",
        windowed_output=results / "04_windowed",
    )


J = Journal
_NOT_IN_OPENALEX = "OpenAlex 미등록(2026-10-20 확인) - 수집되지 않음"
# 한국수자원학회논문집: 신 ISSN 2799-8746/2799-8754는 OpenAlex 미등록이라 구 ISSN으로 받는다.
# OpenAlex에 있는 건 2021년까지다(2022년 ISSN 변경 이후 논문 없음).
_KWRA = J("한국수자원학회논문집", "1226-6280", "2287-6138", "OpenAlex에는 2021년까지만 있음(2022년 ISSN 변경)")
_KSWW = J("상하수도학회지", "1225-7672", "2287-822X")
_WATER_RESEARCH = J("Water Research", "0043-1354", "1879-2448")
_J_HYDROLOGY = J("Journal of Hydrology", "0022-1694", "1879-2707")
_WATER = J("Water", None, "2073-4441")
_KSCE = J("대한토목학회논문집", "1015-6348", "2799-9629")
_KSEE = J("대한환경공학회지", "1225-5025", "2383-7810")
_KSWE = J("한국물환경학회지", "2289-0971", "2289-098X",
          "OpenAlex 저널 정보 오류로 DOI 접두어로 수집(n_01_openalex.py DOI_PREFIX_SOURCES), 2014~2021년만 있음")
_ADV_WATER = J("Advances in Water Resources", "0309-1708", "1872-9657")
_JWRPM = J("Journal of Water Resources Planning and Management", "0733-9496", "1943-5452")
_NRE = J("신재생에너지", "1738-3935", "2713-9999")
_RENEWABLE = J("Renewable Energy", "0960-1481", "1879-0682")

AI = _domain("ai", "AI", "AI연구소", [
    _WATER_RESEARCH,
    J("Journal of Hydraulic Engineering", "0733-9429", "1943-7900"),
    _J_HYDROLOGY,
    _WATER,
    J("Environmental Modelling & Software", "1364-8152", "1873-6726"),
    # arXiv(cs.LG/eess.IV/eess.SP/stat.ML/stat.CO, 2331-8422)는 OpenAlex에서 ISSN 하나가 arXiv 전체라
    # 카테고리 구분 없이 연간 수십만 건이 들어와 제외했다(2026-10-20 결정). ESNT는 목록에 ISSN이 없어 제외.
    _KWRA, _KSCE, _KSWW, _KSEE,
    J("한국실용인공지능학술지", None, "3091-5902", _NOT_IN_OPENALEX),
    J("한국전자통신학회 논문지", "1975-8170", "2288-2189"),
])

WATER_ENV = _domain("water_env", "수자원환경", "수자원환경연구소", [
    J("Atmospheric Research", "0169-8095", "1873-2895"),
    J("Advances in Atmospheric Sciences", "0256-1530", "1861-9533"),
    J("Climate Dynamics", "0930-7575", "1432-0894"),
    J("International Journal of Climatology", "0899-8418", "1097-0088"),
    J("Journal of Applied Meteorology and Climatology", "1558-8424", "1558-8432"),
    J("Journal of Climate", "0894-8755", "1520-0442"),
    J("Water Resources Research", "0043-1397", "1944-7973"),
    _J_HYDROLOGY,
    _ADV_WATER,
    J("Water Resources Management", "0920-4741", "1573-1650"),
    _JWRPM,
    J("Hydrology and Earth System Sciences", "1027-5606", "1607-7938"),
    J("Hydrological Processes", "0885-6087", "1099-1085"),
    J("Journal of Hydrologic Engineering", "1084-0699", "1943-5584"),
    _WATER_RESEARCH,
    # 원본 목록의 Electronic ISSN 1879-2553은 OpenAlex에서 Journal of Physics and Chemistry of Solids의
    # ISSN이다(그대로 쓰면 물리학 논문 2만여 편이 섞임). OpenAlex 기준 올바른 값 1873-6009로 바꿨다.
    J("Journal of Contaminant Hydrology", "0169-7722", "1873-6009"),
    J("Groundwater", "0017-467X", "1745-6584"),
    J("Ecohydrology", "1936-0584", "1936-0592"),
    J("Hydrogeology Journal", "1431-2174", "1435-0157"),
    J("Ecological Informatics", "1574-9541", "1878-0512"),
    J("Ecological Modelling", "0304-3800", "1872-7026"),
    _KSWE,
    J("한국환경영향평가학회지", "1225-7184", None),
    J("한국수생태학회지", "2288-1115", "2288-1123"),
])

INFRA = _domain("infra", "물인프라안전", "물인프라안전연구소", [
    J("Hydropower & Dams", "1352-2523", None, _NOT_IN_OPENALEX),
    _ADV_WATER,
    J("Journal of the Korean Geotechnical Society", "1229-2427", "2288-646X"),
    J("한국지진공학회 논문집", "1226-525X", "2234-1099"),
    J("한국구조물진단유지관리공학회 논문집", "2234-6937", "2287-6979"),
    J("한국터널지하공간학회 논문집", "2233-8292", "2287-4747"),
    J("지하수토양환경", "1598-6438", "2287-8831"),
    J("한국콘크리트학회 논문집", "1229-5515", "2234-2842"),
    J("지구물리와 물리탐사", "1229-1064", "2384-051X"),
])

# key "sewerage"는 기존 상하수도(test) 분야를 그대로 잇는다 - 수집 기록·분석 결과가 이 key로 남아 있다.
# 원본 목록의 중복(Water-Research = Water Research, Water 2회, 상하수도학회지 2회)은 하나로 합쳤다.
SEWERAGE = _domain("sewerage", "상하수도", "상하수도연구소", [
    _WATER_RESEARCH,
    J("Water Research X", None, "2589-9147"),
    _JWRPM,
    _J_HYDROLOGY,
    _WATER,
    J("Sustainability", None, "2071-1050"),
    _KWRA, _KSCE, _KSWW,
    J("Water Science and Technology", "0273-1223", "1996-9732"),
    J("Desalination", "0011-9164", "1873-4464"),
    J("Desalination and Water Treatment", "1944-3994", "1944-3986"),
    J("Water Reuse", "2709-6092", "2709-6106"),
    J("Journal AWWA", "0003-150X", "1551-8833"),
    J("Membrane and Water Treatment", "2005-8624", "2092-7037"),
    _KSWE,
    J("한국수처리학회지", "1225-7192", "2289-0076"),
    _KSEE,
    J("Environmental Engineering Research", "1226-1025", "2005-968X"),
])

# 원본 목록의 Renewable Energy 2회는 하나로 합쳤다.
ENERGY = _domain("energy", "물에너지", "물에너지연구소", [
    J("Applied Energy", "0306-2619", "1872-9118"),
    J("Progress in Photovoltaics: Research and Applications", "1062-7995", "1099-159X"),
    J("IEEE Journal of Photovoltaics", "2156-3381", "2156-3403"),
    _RENEWABLE,
    J("IEEE Access", None, "2169-3536", "연간 1만 건 이상인 대형 저널 - 수집 시간이 오래 걸림"),
    J("IEEE Transactions on Smart Grid", "1949-3053", "1949-3061"),
    J("IEEE Transactions on Sustainable Energy", "1949-3029", "1949-3037"),
    J("Energy", "0360-5442", "1873-6785"),
    J("한국유체기계학회지", "2287-9706", None),
    _NRE,
    J("한국지열·수열에너지학회지", "2799-4783", "2799-4805", _NOT_IN_OPENALEX),
    J("한국태양에너지학회 논문집", "1598-6411", "2508-3562"),
    J("전기학회논문지", "1975-8359", "2287-4364"),
])

MANAGEMENT = _domain("management", "경영", "경영연구소", [
    _KWRA, _KSWW,
    J("회계학연구", "1229-3288", "2508-7193"),
    J("회계저널", "1229-327X", "2508-7207"),
    _NRE,
    J("Applied Economics Letters", "1350-4851", "1466-4291"),
    J("자원·환경경제연구", "1229-9146", "2765-5253", _NOT_IN_OPENALEX),
])

DOMAINS: dict[str, Domain] = {d.key: d for d in (AI, WATER_ENV, INFRA, SEWERAGE, ENERGY, MANAGEMENT)}
DEFAULT_DOMAIN = WATER_ENV.key
DEFAULT = DOMAINS[DEFAULT_DOMAIN]   # 분석 코드를 인자 없이 단독 실행할 때 쓰는 분야

# OpenAI 모델 (커뮤니티 이름·해설 생성, 새 저널 분야 분류에 공통 사용).
# gpt-4o / gpt-4o-mini는 2027-01-14 API 종료 예정이라(2026-10-14 OpenAI 공지) 공식 대체 모델로 바꿨다.
# 이 계열 모델은 temperature를 지정할 수 없어(기본값 1만 허용) 코드에서 temperature를 넘기지 않는다.
OPENAI_MODEL = "gpt-6-luna"

# 서버에서 직접 돌리는 Gemma 4 12B(vLLM, OpenAI 호환 API). 'AI 해석 보기'와 'AI와 대화하기'(모델 선택 시)에 쓴다.
# 입력+출력 합계 8,192 토큰까지만 처리하며 API 키는 필요 없다(아무 문자열).
GEMMA_BASE_URL = "http://127.0.0.1:8001/v1"
GEMMA_MODEL = "gemma4-12b"
GEMMA_CONTEXT = 8192   # 입력+출력 합계 토큰 한도

# OpenAlex 요청에 붙이는 연락 이메일(웹앱 수집, 명령줄 수집 공통).
OPENALEX_EMAIL = "hjhj@kwater.or.kr"

# 예전 분야 key -> 이름. 이벤트 로그(logs/events)에 남아 있는 기록을 모니터링 화면에 표시할 때만 쓴다.
LEGACY_DOMAIN_NAMES = {"water": "수자원(논문·보관)"}

# 새 분야에 정규화 사전이 아직 없을 때 처음 한 번 복사해 쓰는 시작본(수자원 논문 때 만든 사전).
# 변환 155건 중 152건이 단·복수 통일(models -> model 등)이고, 나머지 3건은 machines -> machine,
# modelling -> modeling, flowing -> flow로 모두 표기 차이라 분야가 달라도 뜻을 잘못 합치지 않는다
# (2026-10-06 실측 확인).
SEED_NORMALIZATION_DICT = ROOT / "analysis_codes" / "term_normalization_seed.csv"


def domains_for_issn(issn: str) -> list[str]:
    """그 ISSN이 수집 대상 목록에 들어 있는 분야 key들 (여러 분야에 겹칠 수 있음)."""
    issn = issn.strip().upper()
    return [d.key for d in DOMAINS.values() if any(issn in (x.upper() for x in j.issns) for j in d.journals)]
