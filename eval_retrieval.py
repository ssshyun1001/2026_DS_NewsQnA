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
    # ===== 인물 =====
    {
        "question": "제논의 대표는 누구야?",
        "expected_keywords": ["고석태"]
    },
    {
        "question": "고석태 대표는 어떤 이야기를 했어?",
        "expected_keywords": ["Gen AI 2.0", "생성형 AI 2.0"]
    },

    # ===== 행사 =====
    {
        "question": "제논이 최근 개최한 행사는 뭐야?",
        "expected_keywords": ["AIXperience Day", "AI 익스피리언스 데이"]
    },
    {
        "question": "AI 익스피리언스 데이에서 발표한 핵심 내용은 뭐야?",
        "expected_keywords": ["Gen AI 2.0", "생성형 AI 2.0"]
    },

    # ===== 생성형 AI 2.0 =====
    {
        "question": "Gen AI 2.0이 뭐야?",
        "expected_keywords": ["업무를 완결", "기업 데이터"]
    },
    {
        "question": "기존 생성형 AI와 Gen AI 2.0은 무엇이 달라?",
        "expected_keywords": [
            "업무를 완결",
            "기업 데이터",
            "기업 업무 시스템"
        ]
    },
    {
        "question": "제논이 앞으로 집중하려는 AI 방향은 뭐야?",
        "expected_keywords": [
            "Gen AI 2.0",
            "액셔너블 AI",
            "피지컬 AI"
        ]
    },

    # ===== 플랫폼 =====
    {
        "question": "GenOS 2.0은 어떤 플랫폼이야?",
        "expected_keywords": ["GenOS", "AX 플랫폼"]
    },
    {
        "question": "GenD는 어떤 기능이야?",
        "expected_keywords": [
            "기업 데이터",
            "데이터 분석"
        ]
    },
    {
        "question": "GenBuilder는 어떤 기능을 제공해?",
        "expected_keywords": [
            "업무 앱",
            "코드 생성",
            "배포"
        ]
    },
    {
        "question": "GenA는 어떤 서비스야?",
        "expected_keywords": [
            "AI 에이전트 포털",
            "개인",
            "업무 생산성"
        ]
    },

    # ===== Actionable AI =====
    {
        "question": "원에이전트는 무엇을 하는 AI야?",
        "expected_keywords": [
            "OneAgent",
            "액셔너블 AI",
            "업무를 완결"
        ]
    },
    {
        "question": "액셔너블 AI가 왜 중요한 거야?",
        "expected_keywords": [
            "업무를 완결",
            "상용화"
        ]
    },

    # ===== Physical AI =====
    {
        "question": "피지컬 AI는 어떤 의미야?",
        "expected_keywords": [
            "물리 세계",
            "휴머노이드"
        ]
    },
    {
        "question": "제논은 피지컬 AI를 어디에 활용하려고 해?",
        "expected_keywords": [
            "KB금융",
            "시니어 케어",
            "휴머노이드"
        ]
    },

    # ===== 상용화 =====
    {
        "question": "생성형 AI 시장은 올해 어떻게 변한다고 전망했어?",
        "expected_keywords": [
            "40%",
            "상용화",
            "프로덕션"
        ]
    },
    {
        "question": "왜 생성형 AI가 상용화되기 어려웠다고 했어?",
        "expected_keywords": [
            "PoC",
            "파일럿",
            "업무 프로세스"
        ]
    },
    {
        "question": "기업들이 생성형 AI를 실제로 도입하면서 달라진 점은 뭐야?",
        "expected_keywords": [
            "IT 예산",
            "프로덕션",
            "업무 자동화"
        ]
    },

    # ===== 추론형 (HyDE 검증용) =====
    {
        "question": "제논이 해결하려는 가장 큰 문제는 뭐야?",
        "expected_keywords": [
            "업무를 완결",
            "상용화",
            "업무 프로세스"
        ]
    },
    {
        "question": "제논의 핵심 기술을 한 문장으로 설명해줘.",
        "expected_keywords": [
            "Gen AI 2.0",
            "액셔너블 AI",
            "기업 데이터"
        ]
    }
]


def _is_hit(articles, expected_keywords: list[str]) -> bool:
    if not articles:
        return False
    text = " ".join(f"{a.title} {a.content}" for a in articles).lower()
    return any(keyword.lower() in text for keyword in expected_keywords)


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
        expected = item["expected_keywords"]

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