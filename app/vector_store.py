"""
5. 즉시형(In-Memory) Vector Store

디스크에 영구 저장하지 않고, 요청마다 FAISS 인메모리 인덱스를
새로 구축한 뒤 질문과 가장 관련도 높은 기사만 골라 컨텍스트로 사용한다.

임베딩 클라이언트는 app.embeddings의 공유 인스턴스를 사용한다
(relevance_filter.py / deduplication.py와 동일한 클라이언트를 재사용해
매번 새로 생성하지 않는다).
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.embeddings import get_embeddings
from config import settings
from app.schemas import Article


def build_in_memory_store(articles: list[Article]) -> FAISS:
    """중복 제거된 기사들로 임시 FAISS 인메모리 인덱스를 만든다."""
    documents = [
        Document(
            page_content=f"{a.title}\n{a.content}",
            metadata={"link": a.link, "title": a.title, "press": a.press or ""},
        )
        for a in articles
    ]

    if not documents:
        raise ValueError("벡터 스토어를 만들 기사가 없습니다. 검색/크롤링 결과를 확인하세요.")

    # 이미 계산된 임베딩이 있으면 재사용하여 API 호출을 절약한다.
    if all(a.embedding is not None for a in articles):
        text_embedding_pairs = [
            (f"{a.title}\n{a.content}", a.embedding) for a in articles
        ]
        store = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=get_embeddings(),
            metadatas=[doc.metadata for doc in documents],
        )
    else:
        store = FAISS.from_documents(documents, get_embeddings())

    return store


def retrieve_relevant_articles(
    store: FAISS,
    articles: list[Article],
    question: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list[Article]:
    """
    질문과 코사인 유사도가 similarity_threshold 이상인 기사만, 최대 top_k개까지 반환한다.

    top_k는 "정확히 이 개수를 채운다"가 아니라 상한(cap)이다. threshold를 통과한
    기사가 top_k보다 적으면 그만큼만, 하나도 없으면 빈 리스트를 반환한다.

    FAISS의 기본 인덱스(IndexFlatL2)는 유클리드 거리의 제곱을 점수로 반환한다.
    text-embedding-3-small은 단위 벡터(norm=1)를 반환하므로 다음 항등식이 성립한다:
        ||a - b||^2 = 2 - 2*cos_sim  =>  cos_sim = 1 - (score / 2)
    이를 이용해 L2 점수를 사람이 해석하기 쉬운 코사인 유사도(0~1)로 환산한다.
    """
    top_k = top_k or settings.top_k_articles
    similarity_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.relevance_similarity_threshold
    )

    # threshold 필터링 후에도 top_k를 채울 수 있도록, 넉넉하게 더 넓은 범위를 우선 검색한다.
    search_k = min(len(articles), max(top_k * 3, 10))
    docs_with_scores = store.similarity_search_with_score(question, k=search_k)

    link_to_article = {a.link: a for a in articles}
    scored: list[tuple[float, Article]] = []

    for doc, l2_score in docs_with_scores:
        cosine_sim = 1.0 - (l2_score / 2.0)
        if cosine_sim < similarity_threshold:
            continue

        link = doc.metadata.get("link")
        article = link_to_article.get(link)
        if article is not None:
            scored.append((cosine_sim, article))

    # 유사도 높은 순으로 정렬 후 top_k개까지만 채택
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [article for _, article in scored[:top_k]]