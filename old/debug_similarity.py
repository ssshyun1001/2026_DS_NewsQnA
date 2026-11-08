"""
debug_similarity.py — threshold 없이 실제 코사인 유사도 값을 확인하는 임시 디버그 스크립트

목적:
  질문 검색은 RELEVANCE_SIMILARITY_THRESHOLD(기본 0.5)를 넘는 결과만
  반환하기 때문에, "왜 검색이 실패했는지"(threshold 문제인지 아닌지)를 알 수 없다.
  이 스크립트는 threshold를 적용하지 않고 유사도 점수를 그대로 출력해서
  목표 기사가 실제로 몇 점을 받았는지 확인한다.

지금 구조에 맞춰 바뀐 점:
  - 검증 키워드 대신 고른 의미 이름으로 대상을 정한다
  - DB 모드(기본): DB에 그 topic이 있으면 실제 서비스처럼 청크(약 500자) 유사도를 본다 (수집 안 함)
  - 메모리 모드  : DB에 topic이 없거나 --memory를 주면, 예전처럼 그 자리에서 수집 → 필터링 →
                  FAISS 기사 유사도를 본다 (기간을 함께 줘야 함, LLM 판정 비용이 든다)
  - 유사도 앞의 O / - 는 지금 기준값을 넘는지 (O = 답변 근거로 쓰임)
  (검색 방식 4가지 비교·질문 세트 전체 보기는 python run.py sim)

사용법 (프로젝트 맨 위 폴더에서):
  python old/debug_similarity.py 강남 "방송인 강남" "강남의 아내는 누구야?"
  python old/debug_similarity.py 강남 "방송인 강남" "강남의 아내는 누구야?" 50 --memory
  (검색어, 의미 이름, 질문 [, 기간(일)] [--memory])
"""
from __future__ import annotations

import asyncio
import sys

from _common import get_db_topic, pick_sense

from qupict.config import settings
from qupict.retrieval import _build_search_text, topic_context

SHOW_LIMIT = 30     # DB 모드에서 보여 줄 청크 수 (유사도 높은 순)


def _mark(similarity: float) -> str:
    return "O" if similarity >= settings.relevance_similarity_threshold else "-"


def debug_db(record, question: str) -> None:
    from qupict.llm import get_embeddings
    from qupict.retrieval import search_vectors

    selected = record.selected
    print(f"\nDB topic '{selected.query} → {selected.sense.name}'의 청크 중 질문과의 유사도 "
          f"(threshold 미적용, 위 {SHOW_LIMIT}개, 지금 기준값 {settings.relevance_similarity_threshold:.2f})\n")
    for use_hyde in (False, True):
        label = "V1 (HyDE)" if use_hyde else "V0 (HyDE 없음)"
        search_text = _build_search_text(question, topic_context(selected), use_hyde=use_hyde)
        print(f"=== {label} ===")
        if use_hyde:
            print(f"[검색에 실제로 쓰인 텍스트]\n{search_text}\n")
        vector = get_embeddings().embed_documents([search_text])[0]
        hits = search_vectors(record.id, [vector], SHOW_LIMIT)
        if not hits:
            print("  (청크 없음 — 이 topic을 python run.py index-all 했는지 확인하세요)")
        for h in hits:
            image = " [이미지]" if h.source == "image" else ""
            print(f"  {h.similarity:.3f} {_mark(h.similarity)}  {h.article.title}{image}")
        print()


async def debug_memory(query: str, sense_name: str | None, period_days: int, question: str) -> None:
    import run

    selected = pick_sense(query, sense_name)
    session = await run.collect_news(selected, period_days)
    if session is None:
        print("세션 구축에 실패했습니다. 검색어/의미/기간을 확인해주세요.")
        return

    print(f"\n총 {len(session.articles)}건의 기사 중 질문과의 유사도 (threshold 미적용)\n")
    for use_hyde in (False, True):
        label = "V1 (HyDE)" if use_hyde else "V0 (HyDE 없음)"
        search_text = _build_search_text(question, topic_context(selected), use_hyde=use_hyde)

        print(f"=== {label} ===")
        if use_hyde:
            print(f"[검색에 실제로 쓰인 텍스트]\n{search_text}\n")

        docs_with_scores = session.store.similarity_search_with_score(
            search_text, k=len(session.articles)
        )
        for doc, l2_score in docs_with_scores:
            cosine_sim = 1.0 - (l2_score / 2.0)
            print(f"  {cosine_sim:.3f} {_mark(cosine_sim)}  {doc.metadata.get('title')}")
        print()


def main() -> None:
    use_memory = "--memory" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "--memory"]
    if len(args) >= 3:
        query, sense_name, question = args[0], args[1], args[2]
        period_days = int(args[3]) if len(args) >= 4 else 30
    else:
        query = input("검색어를 입력하세요: ").strip()
        sense_name = input("의미 이름 (모르면 Enter → 후보에서 고르기): ").strip() or None
        question = input("확인할 질문: ").strip()
        period_days = 30

    record = None if use_memory else get_db_topic(query, sense_name)
    if record is not None:
        debug_db(record, question)
        return
    if not use_memory:
        print(f"DB에 topic '{query} → {sense_name}'이 없어 그 자리에서 수집합니다 (최근 {period_days}일, 메모리 모드).")
    asyncio.run(debug_memory(query, sense_name, period_days, question))


if __name__ == "__main__":
    main()
