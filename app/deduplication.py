"""
4. 임베딩 기반 실시간 중복 기사 제거 (Semantic Deduplication)

text-embedding-3-small로 기사 본문을 벡터화하고,
코사인 유사도가 threshold 이상인 기사 쌍 중 하나(더 짧은 쪽)를 제거하여
대표 기사만 남긴다.
"""
from __future__ import annotations

import numpy as np
from langchain_openai import OpenAIEmbeddings

from config import settings
from app.schemas import Article

_embeddings = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
    return _embeddings


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    normalized = vectors / norm
    return normalized @ normalized.T


async def embed_articles(articles: list[Article]) -> list[Article]:
    """기사 본문을 임베딩하여 Article.embedding에 채운다."""
    if not articles:
        return articles

    texts = [f"{a.title}\n{a.content}" for a in articles]
    vectors = await _get_embeddings().aembed_documents(texts)

    for article, vec in zip(articles, vectors):
        article.embedding = vec

    return articles


def deduplicate_articles(
    articles: list[Article], threshold: float | None = None
) -> list[Article]:
    """
    코사인 유사도 threshold 이상인 기사 그룹에서 대표 기사 1개만 남긴다.
    대표 기사는 본문 길이가 가장 긴 기사를 선택한다 (정보량이 가장 많다고 가정).
    """
    threshold = threshold if threshold is not None else settings.dedup_similarity_threshold

    embedded = [a for a in articles if a.embedding is not None]
    if len(embedded) <= 1:
        return articles

    vectors = np.array([a.embedding for a in embedded])
    sim_matrix = _cosine_similarity_matrix(vectors)

    n = len(embedded)
    kept = [True] * n

    for i in range(n):
        if not kept[i]:
            continue
        for j in range(i + 1, n):
            if not kept[j]:
                continue
            if sim_matrix[i, j] >= threshold:
                # 더 짧은 본문을 가진 쪽을 제거
                if len(embedded[i].content) >= len(embedded[j].content):
                    kept[j] = False
                else:
                    kept[i] = False
                    break

    return [article for article, keep in zip(embedded, kept) if keep]
