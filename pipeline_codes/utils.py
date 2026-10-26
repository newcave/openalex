# pipeline_codes/utils.py
import logging
from logging.handlers import TimedRotatingFileHandler
from numbers import Integral, Real
from pathlib import Path

# ── 로그 디렉터리 ──────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── 포맷 ───────────────────────────────────────────────────────
_FMT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_formatter = logging.Formatter(_FMT)

# ── 핸들러 1: 터미널 ───────────────────────────────────────────
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)

# ── 핸들러 2: 파일 (날짜별 로테이션, 무기한 보관) ─────────────
_file_handler = TimedRotatingFileHandler(
    filename=LOG_DIR / "journal_app.log",
    when="midnight",        # 매일 자정에 새 파일
    interval=1,
    backupCount=0,          # 0 = 오래된 로그 파일 삭제 안 함 (무기한 보관)
    encoding="utf-8",
)
_file_handler.setFormatter(_formatter)
_file_handler.suffix = "%Y-%m-%d"   # 예: journal_app.log.2025-06-09

# ── 루트 로거 설정 ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])

logger = logging.getLogger(__name__)
logger.info(f"로그 파일 경로: {LOG_DIR / 'journal_app.log'}")


def normalize_value(v):
    """
    numpy 타입 등을 Python 기본 타입으로 변환.
    메트릭 딕셔너리 값들을 JSON/UI에서 안전하게 쓰기 위해 사용.
    """
    try:
        import numpy as np
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
    except ImportError:
        pass
    if isinstance(v, Integral):
        return int(v)
    if isinstance(v, Real):
        return float(v)
    return v


def collect_metrics(**updates) -> dict:
    """
    파이프라인 각 단계의 처리 결과 통계를 딕셔너리로 반환.
    반환된 딕셔너리는 pipeline.py에서 합산되어 Reflex UI로 전달됨.

    사용 예시:
        metrics = collect_metrics(
            total_papers=5000,
            ror_missing_before=1234,
            ror_filled=1145,
            ror_missing_after=89,
        )
    """
    normalized = {k: normalize_value(v) for k, v in updates.items()}
    logger.info(f"메트릭 수집: {normalized}")
    return normalized