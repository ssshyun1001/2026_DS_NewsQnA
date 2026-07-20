"""
5. 즉시형(In-Memory) Vector Store

디스크에 영구 저장하지 않고, 요청마다 FAISS 인메모리 인덱스를
새로 구축한 뒤 질문과 가장 관련도 높은 기사만 골라 컨텍스트로 사용한다.

임베딩 클라이언트는 app.embeddings의 공유 인스턴스를 사용한다.

검색 방식 (HyDE, Hypothetical Document Embeddings):
  "제논의 대표는 누구야?" 같은 자연스러운 질문 문장은, 실제 뉴스 기사 본문
  스타일("제논의 고석태 대표는...")과 임베딩 공간에서 꽤 떨어져 있어서,
  질문 텍스트를 그대로 임베딩해 검색하면 관련 기사가 있어도 유사도 threshold를
  못 넘겨 검색이 실패하는 경우가 잦다 ("GEN 2.0"처럼 기사 제목과 비슷한
  키워드성 질문만 성공하고, 자연어 질문은 실패하는 비대칭이 발생).

  이를 완화하기 위해, 질문을 그대로 검색하지 않고 LLM으로 "이 질문에 답하는
  가상의 뉴스 문장"을 먼저 생성한 뒤 그 문장으로 검색한다. 가상의 문장은
  사실 여부와 무관하게 뉴스 기사와 문체가 비슷해서, 실제 관련 기사와의
  임베딩 유사도가 훨씬 높게 나온다.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from app.embeddings import get_embeddings
from config import settings
from app.schemas import Article

_HYDE_SYSTEM_PROMPT = """\
당신은 검색 품질을 높이기 위한 보조 도구입니다.
사용자의 질문에 대해, 만약 이 질문에 답하는 뉴스 기사가 실제로 존재한다면
어떤 문장으로 쓰여 있을지 가상의 문장을 1~2개 작성하세요.

규칙:
- 사실 여부는 중요하지 않습니다. 오직 검색 임베딩 품질을 높이기 위한 용도입니다.
- 뉴스 기사체("~다.", "~라고 밝혔다.")로, 질문의 핵심 키워드를 반드시 포함해 작성하세요.
- 주제 맥락이 주어지면 그 맥락에 맞는 문장을 작성하세요 (다른 동명이인이나 다른 대상으로
  빗나가지 않도록 주의).
- 다른 설명 없이 가상의 문장만 출력하세요.
"""

_hyde_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _HYDE_SYSTEM_PROMPT),
        ("human", "주제 맥락: {topic}\n질문: {question}"),
    ]
)

_hyde_chain = None


def _get_hyde_chain():
    global _hyde_chain
    if _hyde_chain is None:
        llm = ChatOpenAI(
            model=getattr(settings, "relevance_llm_model", None) or settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0.3,
        )
        _hyde_chain = _hyde_prompt | llm | StrOutputParser()
    return _hyde_chain


def _build_search_text(question: str, topic: str = "", use_hyde: bool = True) -> str:
    """HyDE: 질문을 가상의 기사 문장으로 확장해 검색용 텍스트를 만든다.

    use_hyde=False면 원래 질문 텍스트를 그대로 반환한다 (ablation 비교용).
    LLM 호출이 실패하면 원래 질문 텍스트로 안전하게 폴백한다.
    """
    if not use_hyde:
        return question
    try:
        hypothetical = _get_hyde_chain().invoke({"question": question, "topic": topic or "(없음)"})
        # 원래 질문도 함께 포함해, 가상의 문장이 핵심에서 벗어나도 원 질문의 정보가 남게 한다.
        return f"{question}\n{hypothetical}"
    except Exception:
        return question


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
    topic: str = "",
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    use_hyde: bool = True,
) -> list[Article]:
    """
    질문과 코사인 유사도가 similarity_threshold 이상인 기사만, 최대 top_k개까지 반환한다.

    top_k는 "정확히 이 개수를 채운다"가 아니라 상한(cap)이다. threshold를 통과한
    기사가 top_k보다 적으면 그만큼만, 하나도 없으면 빈 리스트를 반환한다.

    질문은 그대로 검색하지 않고, HyDE로 확장한 텍스트로 검색한다 (모듈 docstring 참고).

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

    search_text = _build_search_text(question, topic, use_hyde=use_hyde)

    # threshold 필터링 후에도 top_k를 채울 수 있도록, 넉넉하게 더 넓은 범위를 우선 검색한다.
    search_k = min(len(articles), max(top_k * 3, 10))
    docs_with_scores = store.similarity_search_with_score(search_text, k=search_k)

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