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
    max_crawl_articles: int = _get_int("MAX_CRAWL_ARTICLES", 100)
    crawl_timeout_seconds: int = _get_int("CRAWL_TIMEOUT_SECONDS", 8)
    # 질문-기사 간 코사인 유사도가 이 값 미만이면 출처로 채택하지 않는다.
    # top_k_articles는 이제 "최대 개수(cap)"일 뿐, 이 threshold를 통과한 기사만 실제로 반환된다.
    relevance_similarity_threshold: float = _get_float("RELEVANCE_SIMILARITY_THRESHOLD", 0.30)


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