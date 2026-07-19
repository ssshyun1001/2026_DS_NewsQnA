"""
공유 임베딩 클라이언트.

relevance_filter.py와 deduplication.py가 같은 기사를 두 번 임베딩하지 않도록
클라이언트 생성 로직을 이 모듈로 분리했다.
"""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from config import settings

_embeddings: OpenAIEmbeddings | None = None


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
    return _embeddings