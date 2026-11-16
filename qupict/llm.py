"""
LLM·임베딩 클라이언트를 만드는 곳 (프로젝트 전체에서 여기만 사용)

모든 모듈이 이 파일의 함수로 클라이언트를 받기 때문에,
어느 서비스로 연결할지는 .env만 바꾸면 된다.

  OPENAI_BASE_URL 비어 있음  → OpenAI 공식 API
  OPENAI_BASE_URL 지정       → 그 주소의 OpenAI 호환 API
                               (예: 학교 팩트챗 Gateway https://factchat-cloud.mindlogic.ai/v1/gateway)

임베딩 클라이언트는 이벤트 루프마다 하나씩 만들어 그 안에서 공유한다 (아래 get_embeddings).

여러 사용자가 동시에 쓸 때 (웹 서버)
  - 준비 작업마다 자기 이벤트 루프(asyncio.run)에서 돈다. 비동기 클라이언트(내부 httpx 연결)는 만든 루프에 묶여 있어서,
    다른 루프가 같이 쓰면 OpenAI 라이브러리가 "Connection error."를 낸다 → 루프마다 따로 만든다.
  - 서버 전체에서 동시에 보내는 LLM · 임베딩 요청 수에 상한을 둔다 (llm_slot). 사용자가 몰려도 게이트웨이에
    요청이 한꺼번에 쏟아지지 않는다. 사용자 한 명일 때는 상한에 닿지 않아 속도가 그대로다.
"""
from __future__ import annotations

import asyncio
import threading
import warnings
import weakref
from contextlib import asynccontextmanager

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from qupict.config import settings

_embeddings: OpenAIEmbeddings | None = None      # 이벤트 루프 밖(동기 호출)에서 쓰는 것
_embeddings_by_loop: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, OpenAIEmbeddings]" = weakref.WeakKeyDictionary()
_embeddings_lock = threading.Lock()

# 서버 전체(모든 사용자 · 모든 작업)에서 동시에 보내는 LLM · 임베딩 요청 수 상한.
# 작업 하나의 LLM 판정은 filtering.LLM_CONCURRENCY(8)개까지 동시에 보내므로, 그보다 넉넉하게 잡아
# 사용자 한 명일 때는 느려지지 않게 했다. 여러 명이면 넘는 요청은 차례를 기다린다.
LLM_GLOBAL_CONCURRENCY = 12
_llm_slots = threading.BoundedSemaphore(LLM_GLOBAL_CONCURRENCY)


@asynccontextmanager
async def llm_slot():
    """async with llm_slot(): … — 서버 전체의 동시 요청 상한(LLM_GLOBAL_CONCURRENCY) 안에서 요청 하나를 보낸다.
    작업마다 이벤트 루프가 달라서 asyncio.Semaphore로는 루프끼리 나눠 쓸 수 없다 → 스레드용 세마포어를
    다른 스레드에서 기다려(asyncio.to_thread) 이벤트 루프를 막지 않는다."""
    await asyncio.to_thread(_llm_slots.acquire)
    try:
        yield
    finally:
        _llm_slots.release()

# 임베딩 요청 한 번에 보내는 텍스트 수. 기본값(1000)이면 기사 300건(본문 2000자씩)이 한 요청에 들어가
# 게이트웨이 한도(요청당 30만 토큰)를 넘는다. 100건이면 기사 기준 약 13만 토큰.
EMBED_BATCH_SIZE = 100

# 요청이 실패하거나 제한 시간(config.py의 llm_timeout_seconds · embed_timeout_seconds)을 넘으면 다시 보내는 횟수.
# 라이브러리 기본값과 같은 2번 (처음 1번 + 재시도 2번 = 최대 3번). 제한 시간이 생겨서 값을 드러내 적었다.
# 가장 오래 기다리는 경우: 제한 시간 × 3 + 재시도 사이 쉬는 시간(합 약 1.5초).
MAX_RETRIES = 2


def _base_url() -> str | None:
    return settings.openai_base_url or None


def get_chat_model(temperature: float = 0) -> ChatOpenAI:
    """채팅 모델 클라이언트. 모델 이름은 .env의 OPENAI_CHAT_MODEL."""
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        base_url=_base_url(),
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,     # 답이 없으면 끊는다 (예전: 제한 없음 = 10분)
        max_retries=MAX_RETRIES,
    )


def get_structured_model(schema, temperature: float = 0):
    """정해진 형식(pydantic 모델)으로 답하는 채팅 모델.

    method="json_schema"(response_format)를 쓴다. 기본값(function_calling)은 tool_choice로 도구 호출을
    강제하는데, 게이트웨이의 Claude 모델은 이를 거부한다 (400 "tool_choice ... not supported").
    """
    return get_chat_model(temperature).with_structured_output(schema, method="json_schema")


# json_schema 방식으로 부를 때마다 langchain이 찍는 경고 (스트리밍을 쓰지 않으므로 상관없음)
warnings.filterwarnings("ignore", message="Streaming with Pydantic response_format not yet supported")


async def aembed_in_batches(embeddings, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    """texts를 batch_size개씩 나눠 임베딩 요청을 동시에 보낸다. 결과는 texts 순서 그대로.

    langchain의 aembed_documents도 chunk_size(=EMBED_BATCH_SIZE)개씩 나누지만, 나눈 요청을 하나씩 차례로 보낸다.
    여기서는 같은 크기·같은 순서로 나눈 요청을 한꺼번에 보낸다.
    → 요청마다 들어가는 글 묶음이 예전과 같아서 임베딩 값도 같다 (임베딩 API는 같이 보낸 글이 달라지면
      값이 아주 조금 달라진다). 요청당 크기(게이트웨이 한도)도 그대로고, 기다림만 준다.
    """
    if not texts:
        return []
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    async def one(batch: list[str]) -> list[list[float]]:
        async with llm_slot():                     # 서버 전체 동시 요청 상한 안에서
            return await embeddings.aembed_documents(batch)

    results = await asyncio.gather(*(one(b) for b in batches))
    return [vec for batch in results for vec in batch]


def _new_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=_base_url(),
        chunk_size=EMBED_BATCH_SIZE,
        timeout=settings.embed_timeout_seconds,   # 답이 없으면 끊는다 (예전: 제한 없음 = 10분)
        max_retries=MAX_RETRIES,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """임베딩 클라이언트. 모델 이름은 .env의 OPENAI_EMBEDDING_MODEL.

    - 이벤트 루프 안(async 함수)에서 부르면: 그 루프 전용 클라이언트 (같은 루프 안에서는 하나를 공유)
      → 동시에 도는 다른 작업(다른 루프)과 연결을 나눠 쓰지 않는다. 루프가 끝나면 클라이언트도 같이 사라진다.
    - 루프 밖(동기 함수: 질문할 때 검색 등)에서 부르면: 서버 전체가 공유하는 하나 (동기 연결은 여러 스레드가 같이 써도 된다)
    """
    global _embeddings
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    with _embeddings_lock:
        if loop is None:
            if _embeddings is None:
                _embeddings = _new_embeddings()
            return _embeddings
        client = _embeddings_by_loop.get(loop)
        if client is None:
            client = _embeddings_by_loop[loop] = _new_embeddings()
        return client
