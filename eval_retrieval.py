"""
eval_retrieval.py — HyDE 검색 개선 효과 측정 스크립트

목적:
  질문을 그대로 임베딩해 검색하던 방식(V0)과, HyDE로 가상의 기사 문장을 만들어
  검색하는 방식(V1)의 검색 성공률을 비교한다.

골든셋:
  실제 대화에서 V0가 검색에 실패했던 질문들("제논의 대표는 누구야?", "고석태에
  대해 알려줘" 등)과, 원래도 성공했던 키워드성 질문("GEN 2.0" 등)을 함께 포함해
  HyDE가 실패 케이스를 개선하면서 기존 성공 케이스를 해치지는 않는지도 같이 본다.

판정 기준:
  각 질문마다 "이 답을 포함하는 기사라면 반드시 등장해야 하는 키워드"를 정해두고,
  검색된 기사(제목+본문) 중 하나라도 그 키워드를 포함하면 성공(hit)으로 센다.
  완전한 정답 여부(faithfulness)까지는 보지 않는, 순수 검색 성공률(retrieval hit rate) 지표다.

사용법:
  python eval_retrieval.py "제논" "AI기업" 30
  (topic, validation_terms(콤마구분), period_days — main.py에서 실제로 쓴 값과 맞추면 좋다)
"""
from __future__ import annotations

import asyncio
import sys

from main import collect_news
from app.vector_store import retrieve_relevant_articles

# 실제 대화에서 나왔던 질문들 + 각 질문에 맞는 기사라면 반드시 포함할 키워드
GOLDEN_QUESTIONS = [
    {"question": "제논의 대표는 누구야?", "expected_keyword": "고석태"},
    {"question": "고석태에 대해 알려줘", "expected_keyword": "제논 대표"},
    {"question": "제논의 최근 참여하거나 진행한 거 알려줘", "expected_keyword": "AIXperience"},
    {"question": "원에이전트에 대해 자세히 알려줘", "expected_keyword": "액셔너블 AI"},
    {"question": "GEN 2.0에 대해 설명해봐", "expected_keyword": "액셔너블 AI"}
]


def _is_hit(articles, expected_keyword: str) -> bool:
    if not articles:
        return False
    keyword = expected_keyword.lower()
    return any(
        keyword in f"{a.title} {a.content}".lower() for a in articles
    )


async def run_eval(topic: str, validation_terms: list[str], period_days: int) -> None:
    session = await collect_news(topic, validation_terms, period_days)
    if session is None:
        print("세션 구축에 실패했습니다. topic/검증어/기간을 확인해주세요.")
        return

    print(f"\n{'질문':<40} {'V0 (HyDE 없음)':<18} {'V1 (HyDE)':<10}")
    print("-" * 70)

    hits_v0 = 0
    hits_v1 = 0

    for item in GOLDEN_QUESTIONS:
        question = item["question"]
        expected = item["expected_keyword"]

        result_v0 = retrieve_relevant_articles(
            session.store, session.articles, question, topic=session.topic, use_hyde=False
        )
        result_v1 = retrieve_relevant_articles(
            session.store, session.articles, question, topic=session.topic, use_hyde=True
        )

        hit_v0 = _is_hit(result_v0, expected)
        hit_v1 = _is_hit(result_v1, expected)
        hits_v0 += hit_v0
        hits_v1 += hit_v1

        print(f"{question:<40} {'O' if hit_v0 else 'X':<18} {'O' if hit_v1 else 'X':<10}")

    n = len(GOLDEN_QUESTIONS)
    print("-" * 70)
    print(f"검색 성공률(Hit Rate): V0 = {hits_v0}/{n} ({hits_v0/n:.1%})  →  V1 = {hits_v1}/{n} ({hits_v1/n:.1%})")


def main() -> None:
    if len(sys.argv) >= 4:
        topic = sys.argv[1]
        validation_terms = [t.strip() for t in sys.argv[2].split(",") if t.strip()]
        period_days = int(sys.argv[3])
    else:
        topic = input("관심 분야를 입력하세요: ").strip()
        validation_terms = [
            t.strip() for t in input("검증 키워드(콤마 구분): ").strip().split(",") if t.strip()
        ]
        period_days = int(input("기간(일): ").strip())

    asyncio.run(run_eval(topic, validation_terms, period_days))


if __name__ == "__main__":
    main()
