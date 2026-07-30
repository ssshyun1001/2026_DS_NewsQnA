"""
전역 설정. .env 파일에서 API 키 및 튜닝 파라미터를 로드한다.
"""
import os
from dataclasses import dataclass
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
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Naver Search API
    naver_client_id: str = os.getenv("NAVER_CLIENT_ID", "")
    naver_client_secret: str = os.getenv("NAVER_CLIENT_SECRET", "")

    # Pipeline tuning
    dedup_similarity_threshold: float = _get_float("DEDUP_SIMILARITY_THRESHOLD", 0.90)
    top_k_articles: int = _get_int("TOP_K_ARTICLES", 5)
    max_crawl_articles: int = _get_int("MAX_CRAWL_ARTICLES", 150)
    crawl_timeout_seconds: int = _get_int("CRAWL_TIMEOUT_SECONDS", 10)
    # 질문-기사 간 코사인 유사도가 이 값 미만이면 출처로 채택하지 않는다.
    relevance_similarity_threshold: float = _get_float("RELEVANCE_SIMILARITY_THRESHOLD", 0.20)

    # 제외 키워드(LLM 자동 생성, topic 기준 혼동 개체) 임베딩 경계값.
    # high 이상이면 명확히 혼동 개체(동음이의어/유사 명칭) -> LLM 호출 없이 즉시 제외.
    # low 미만이면 명확히 무관한 혼동 -> 통과. 그 사이(경계)만 LLM 재확인.
    exclusion_low_threshold: float = _get_float("EXCLUSION_LOW_THRESHOLD", 0.20)
    exclusion_high_threshold: float = _get_float("EXCLUSION_HIGH_THRESHOLD", 0.35)


settings = Settings()


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