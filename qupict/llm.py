"""
LLM·임베딩 클라이언트를 만드는 곳 (프로젝트 전체에서 여기만 사용)

모든 모듈이 이 파일의 함수로 클라이언트를 받기 때문에,
어느 서비스로 연결할지는 .env만 바꾸면 된다.

  OPENAI_BASE_URL 비어 있음  → OpenAI 공식 API
  OPENAI_BASE_URL 지정       → 그 주소의 OpenAI 호환 API
                               (예: 학교 팩트챗 Gateway https://factchat-cloud.mindlogic.ai/v1/gateway)

임베딩 클라이언트는 하나만 만들어 공유한다
(중복 제거와 관련성 필터가 같은 기사를 두 번 임베딩하지 않도록).
"""
from __future__ import annotations

import warnings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from qupict.config import settings

_embeddings: OpenAIEmbeddings | None = None

# 임베딩 요청 한 번에 보내는 텍스트 수. 기본값(1000)이면 기사 300건(본문 2000자씩)이 한 요청에 들어가
# 게이트웨이 한도(요청당 30만 토큰)를 넘는다. 100건이면 기사 기준 약 13만 토큰.
EMBED_BATCH_SIZE = 100


def _base_url() -> str | None:
    return settings.openai_base_url or None


def get_chat_model(temperature: float = 0) -> ChatOpenAI:
    """채팅 모델 클라이언트. 모델 이름은 .env의 OPENAI_CHAT_MODEL."""
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        base_url=_base_url(),
        temperature=temperature,
    )


def get_structured_model(schema, temperature: float = 0):
    """정해진 형식(pydantic 모델)으로 답하는 채팅 모델.

    method="json_schema"(response_format)를 쓴다. 기본값(function_calling)은 tool_choice로 도구 호출을
    강제하는데, 게이트웨이의 Claude 모델은 이를 거부한다 (400 "tool_choice ... not supported").
    """
    return get_chat_model(temperature).with_structured_output(schema, method="json_schema")


# json_schema 방식으로 부를 때마다 langchain이 찍는 경고 (스트리밍을 쓰지 않으므로 상관없음)
warnings.filterwarnings("ignore", message="Streaming with Pydantic response_format not yet supported")


def get_embeddings() -> OpenAIEmbeddings:
    """임베딩 클라이언트 (공유 인스턴스). 모델 이름은 .env의 OPENAI_EMBEDDING_MODEL."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=_base_url(),
            chunk_size=EMBED_BATCH_SIZE,
        )
    return _embeddings
