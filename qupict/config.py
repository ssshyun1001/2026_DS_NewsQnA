"""
전역 설정. .env 파일에서 API 키 및 튜닝 파라미터를 로드한다.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


@dataclass(frozen=True)
class Settings:
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    # 비워 두면 OpenAI 공식 API, 학교 팩트챗을 쓰면 https://factchat-cloud.mindlogic.ai/v1/gateway
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Naver Search API
    naver_client_id: str = os.getenv("NAVER_CLIENT_ID", "")
    naver_client_secret: str = os.getenv("NAVER_CLIENT_SECRET", "")

    # PostgreSQL + pgvector (로컬 Docker / 팀 공용 DB 모두 이 주소 하나만 바꾸면 된다)
    database_url: str = os.getenv("DATABASE_URL", "")

    # Pipeline tuning
    dedup_similarity_threshold: float = _get_float("DEDUP_SIMILARITY_THRESHOLD", 0.90)
    top_k_articles: int = _get_int("TOP_K_ARTICLES", 10)
    max_crawl_articles: int = _get_int("MAX_CRAWL_ARTICLES", 150)
    crawl_timeout_seconds: int = _get_int("CRAWL_TIMEOUT_SECONDS", 10)
    # 질문-청크(기사) 간 코사인 유사도가 이 값 미만이면 답변 근거로 쓰지 않는다.
    # 0.5: 팀 결정 — 확실히 관련된 근거만 쓰도록 엄격하게. (예전 0.2)
    #   실제 값 참고: 방송인 강남 질문에서 관련 청크가 0.48~0.70 → 0.5면 상위 일부만 남는다.
    #   근거가 하나도 안 남으면 답변은 "관련 뉴스를 찾지 못했습니다"로 끝난다.
    relevance_similarity_threshold: float = _get_float("RELEVANCE_SIMILARITY_THRESHOLD", 0.50)



settings = Settings()


TARGETS_PATH = Path(__file__).resolve().parent.parent / "targets.yaml"


def load_targets(path: Path | str = TARGETS_PATH) -> list[dict]:
    """targets.yaml(수집 대상 목록)을 읽는다. 각 항목에는 query와 sense가 꼭 있어야 한다."""
    import yaml

    with open(path, encoding="utf-8") as f:
        targets = yaml.safe_load(f) or []
    for i, t in enumerate(targets, 1):
        missing = [k for k in ("query", "sense") if not t.get(k)]
        if missing:
            raise ValueError(f"targets.yaml {i}번째 항목에 {', '.join(missing)} 값이 없습니다: {t}")
    return targets


def validate_settings() -> None:
    missing = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.naver_client_id:
        missing.append("NAVER_CLIENT_ID")
    if not settings.naver_client_secret:
        missing.append("NAVER_CLIENT_SECRET")
    if missing:
        raise EnvironmentError(
            f"다음 환경변수가 설정되지 않았습니다: {', '.join(missing)}. "
            f".env 파일을 확인하세요 (.env.example 참고)."
        )