"""
4. 임베딩 기반 실시간 중복 기사 제거 (Semantic Deduplication)

text-embedding-3-small로 기사 본문을 벡터화하고,
코사인 유사도가 threshold 이상인 기사 쌍 중 하나(더 짧은 쪽)를 제거하여
대표 기사만 남긴다.

임베딩 클라이언트는 app.embeddings의 공유 인스턴스를 사용한다
(relevance_filter.py와 같은 임베딩을 중복 계산하지 않기 위함).
"""
from __future__ import annotations

import numpy as np

from app.embeddings import get_embeddings
from app.schemas import Article
from config import settings


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    normalized = vectors / norm
    return normalized @ normalized.T


async def embed_articles(articles: list[Article]) -> list[Article]:
    """기사 본문을 임베딩하여 Article.embedding에 채운다.

    이미 embedding이 채워진 기사는 다시 계산하지 않는다
    (relevance filter 단계에서 이미 채워졌을 수 있으므로).
    """
    if not articles:
        return articles

    to_embed = [a for a in articles if a.embedding is None]
    if not to_embed:
        return articles

    texts = [f"{a.title}\n{a.content}" for a in to_embed]
    vectors = await get_embeddings().aembed_documents(texts)
    for article, vec in zip(to_embed, vectors):
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
                if len(embedded[i].content) >= len(embedded[j].content):
                    kept[j] = False
                else:
                    kept[i] = False
                    break

    return [article for article, keep in zip(embedded, kept) if keep]