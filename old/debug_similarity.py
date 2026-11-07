"""
debug_similarity.py — threshold 없이 실제 코사인 유사도 값을 확인하는 임시 디버그 스크립트

목적:
  retrieve_relevant_articles()는 RELEVANCE_SIMILARITY_THRESHOLD를 넘는 기사만
  반환하기 때문에, "왜 검색이 실패했는지"(threshold 문제인지 아닌지)를 알 수 없다.
  이 스크립트는 threshold를 적용하지 않고 전체 기사의 유사도 점수를 그대로 출력해서
  목표 기사가 실제로 몇 점을 받았는지 확인한다.

사용법:
  python debug_similarity.py "강남" "연예인,방송인" 50 "강남의 아내는 누구야?"
  (topic, validation_terms(콤마구분), period_days, question)
"""
from __future__ import annotations

import asyncio
import sys

from main import collect_news
from app.vector_store import _build_search_text


async def debug_similarity(
    topic: str, validation_terms: list[str], period_days: int, question: str
) -> None:
    session = await collect_news(topic, validation_terms, period_days)
    if session is None:
        print("세션 구축에 실패했습니다. topic/검증어/기간을 확인해주세요.")
        return

    print(f"\n총 {len(session.articles)}건의 기사 중 질문과의 유사도 (threshold 미적용)\n")

    for use_hyde in (False, True):
        label = "V1 (HyDE)" if use_hyde else "V0 (HyDE 없음)"
        search_text = _build_search_text(question, topic=topic, use_hyde=use_hyde)

        print(f"=== {label} ===")
        if use_hyde:
            print(f"[검색에 실제로 쓰인 텍스트]\n{search_text}\n")

        docs_with_scores = session.store.similarity_search_with_score(
            search_text, k=len(session.articles)
        )
        for doc, l2_score in docs_with_scores:
            cosine_sim = 1.0 - (l2_score / 2.0)
            print(f"  {cosine_sim:.3f}  {doc.metadata.get('title')}")
        print()


def main() -> None:
    if len(sys.argv) >= 5:
        topic = sys.argv[1]
        validation_terms = [t.strip() for t in sys.argv[2].split(",") if t.strip()]
        period_days = int(sys.argv[3])
        question = sys.argv[4]
    else:
        topic = input("관심 분야를 입력하세요: ").strip()
        validation_terms = [
            t.strip() for t in input("검증 키워드(콤마 구분): ").strip().split(",") if t.strip()
        ]
        period_days = int(input("기간(일): ").strip())
        question = input("확인할 질문: ").strip()

    asyncio.run(debug_similarity(topic, validation_terms, period_days, question))


if __name__ == "__main__":
    main()