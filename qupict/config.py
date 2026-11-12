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
    # 중복 제거 기준 (쉼표로 여러 개): "title"(제목 임베딩) | "content"(본문 앞 N문장 임베딩)
    #   | "noun"(제목의 드문 고유명사 겹침, TF-IDF). "both" = title,content / "all" = 셋 다
    dedup_by: str = os.getenv("DEDUP_BY", "all")
    # 기준별 코사인 유사도 임계값 (이 값 이상이면 그 기준에서 '비슷함')
    dedup_title_threshold: float = _get_float("DEDUP_TITLE_THRESHOLD", 0.7)
    # DEDUP_SIMILARITY_THRESHOLD는 예전 이름 (본문 기준 임계값으로 그대로 읽는다)
    dedup_content_threshold: float = _get_float("DEDUP_CONTENT_THRESHOLD",
                                                _get_float("DEDUP_SIMILARITY_THRESHOLD", 0.7))
    # 제목 고유명사 TF-IDF 코사인 유사도 임계값
    dedup_noun_threshold: float = _get_float("DEDUP_NOUN_THRESHOLD", 0.3)
    # content 기준에서 비교할 본문 앞 문장 수
    dedup_lead_sentences: int = _get_int("DEDUP_LEAD_SENTENCES", 5)
    # 기준이 여러 개일 때 묶는 방법: "and"(모두 임계값 이상) | "or"(하나라도 이상)
    dedup_rule: str = os.getenv("DEDUP_RULE", "or")

    top_k_articles: int = _get_int("TOP_K_ARTICLES", 10)
    max_crawl_articles: int = _get_int("MAX_CRAWL_ARTICLES", 150)
    crawl_timeout_seconds: int = _get_int("CRAWL_TIMEOUT_SECONDS", 10)
    # LLM · 임베딩 요청 하나를 기다리는 최대 시간(초). 넘으면 끊고 다시 보낸다 (최대 2번 더, qupict/llm.py).
    #   예전에는 제한이 없어서(라이브러리 기본 10분 × 3번) 게이트웨이가 요청 하나에 답을 안 하면
    #   준비 단계 전체가 수십 분 멈췄다 (측정 2026-11-11: 동시 8개로 보낸 임베딩 40건 중 1~2건이 답 없음).
    #   측정한 정상 응답 시간: 임베딩 0.7~4.6초 (기사 100건 한 요청이 가장 김), LLM 판정(기사 8건) 2~3초,
    #   짧은 질문 1초 안팎. 이미지 설명은 장당 수 초 (2026-11-10: 8장 22초).
    #   → 둘 다 15초 (가장 긴 정상 응답 4.6초의 약 3배). 3번 다 실패하면 약 46초 뒤 오류가 난다.
    #     타임아웃 오류가 자주 보이면 .env에서 올린다. 답변 생성은 글자가 나오는 대로 받으므로(스트리밍)
    #     긴 답변도 걸리지 않는다 (제한 시간은 "다음 글자가 올 때까지"에 적용된다).
    llm_timeout_seconds: float = _get_float("LLM_TIMEOUT_SECONDS", 15)
    embed_timeout_seconds: float = _get_float("EMBED_TIMEOUT_SECONDS", 15)
    # 질문-청크(기사) 간 코사인 유사도가 이 값 미만이면 답변 근거로 쓰지 않는다.
    # 0.5: 팀 결정 — 확실히 관련된 근거만 쓰도록 엄격하게. (예전 0.2)
    #   실제 값 참고: 방송인 강남 질문에서 관련 청크가 0.48~0.70 → 0.5면 상위 일부만 남는다.
    #   근거가 하나도 안 남으면 답변은 "관련 뉴스를 찾지 못했습니다"로 끝난다.
    relevance_similarity_threshold: float = _get_float("RELEVANCE_SIMILARITY_THRESHOLD", 0.50)
    # 서비스(run.py · web.py)에서 기사 이미지를 글로 바꾸는 방법. 기본: "ocr"
    #   "ocr" = EasyOCR로 이미지 속 글자 읽기 (LLM 비용 0, "준비 완료" 뒤 백그라운드에서 → 준비 시간에 영향 없음)
    #   "vlm" = LLM이 이미지 설명 (측정 2026-11-10, "삼성" 기사 100건: 39장에 60초 = 준비 시간 89초 중 67%)
    #   "off" = 새로 만들지 않음. 어느 쪽이든 이미 만들어 둔 이미지 글은 청크에 그대로 들어간다.
    #   예전 설정 DESCRIBE_IMAGES=1은 "vlm"으로 읽는다. (admin.py index-all은 이 설정과 상관없이 --no-images로 정한다)
    image_text: str = (os.getenv("IMAGE_TEXT")
                       or ("vlm" if os.getenv("DESCRIBE_IMAGES") == "1" else "ocr")).strip().lower()


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
