"""
eval_retrieval.py — HyDE 검색 개선 효과 측정 스크립트

목적:
  질문을 그대로 임베딩해 검색하던 방식(V0)과, HyDE로 가상의 기사 문장을 만들어
  검색하는 방식(V1)의 검색 성공률을 비교한다.

골든셋:
  eval_filter.py에서 사용한 것과 동일한 3개 주제(제논, 서울여대, 강남)에 대해
  각각 별도의 골든 질문셋을 둔다. 주제별로 실제 대화에서 V0가 검색에 실패했던
  질문들과, 원래도 성공했던 키워드성 질문을 함께 포함해 HyDE가 실패 케이스를
  개선하면서 기존 성공 케이스를 해치지는 않는지도 같이 본다.

  서울여대 골든셋은 2026-07-27/07-23에 실제로 크롤링된 기사(정의학원 제12대
  이사장 이/취임, 2027수시박람회 관련 기사)를 확인하고 만든 것이다.

  강남 골든셋은 2026-07-29에 실제로 크롤링된 기사(방송인 강남의 '라디오스타'
  출연 관련 기사)를 확인하고 만든 것이다.

  다만 두 골든셋 모두 수집 기간/시점에 따라 다른 기사가 섞일 수 있으므로,
  실행 결과가 낮게 나오면 그 시점에 크롤링된 기사가 골든셋과 다른 내용일
  가능성을 먼저 의심할 것.

판정 기준:
  각 질문마다 "이 답을 포함하는 기사라면 반드시 등장해야 하는 키워드"를 정해두고,
  검색된 기사(제목+본문) 중 하나라도 그 키워드를 포함하면 성공(hit)으로 센다.
  완전한 정답 여부(faithfulness)까지는 보지 않는, 순수 검색 성공률(retrieval hit rate) 지표다.

사용법:
  python eval_retrieval.py 제논
  python eval_retrieval.py 서울여대
  python eval_retrieval.py 강남
  (인자를 주지 않으면 GOLDEN_SETS에 등록된 모든 주제를 순서대로 평가하고,
   마지막에 주제별 + 전체 평균 결과를 요약해서 보여준다)
"""
from __future__ import annotations

import asyncio
import sys

from main import collect_news
from app.vector_store import retrieve_relevant_articles


# ============================================================
# 주제별 골든셋
# 각 주제는 {topic, validation_terms, period_days, questions} 를 가진다.
# ============================================================

GOLDEN_SETS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # 제논 — 실제 대화에서 나왔던 질문들을 바탕으로 구성 (검증 완료)
    # ------------------------------------------------------------------
    "제논": {
        "topic": "제논",
        "validation_terms": ["AI기업"],
        "period_days": 30,
        "questions": [
            # ===== 인물 =====
            {"question": "제논의 대표는 누구야?", "expected_keywords": ["고석태"]},
            {"question": "고석태 대표는 어떤 이야기를 했어?",
             "expected_keywords": ["Gen AI 2.0", "생성형 AI 2.0"]},

            # ===== 행사 =====
            {"question": "제논이 최근 개최한 행사는 뭐야?",
             "expected_keywords": ["AIXperience Day", "AI 익스피리언스 데이"]},
            {"question": "AI 익스피리언스 데이에서 발표한 핵심 내용은 뭐야?",
             "expected_keywords": ["Gen AI 2.0", "생성형 AI 2.0"]},

            # ===== 생성형 AI 2.0 =====
            {"question": "Gen AI 2.0이 뭐야?",
             "expected_keywords": ["업무를 완결", "기업 데이터"]},
            {"question": "기존 생성형 AI와 Gen AI 2.0은 무엇이 달라?",
             "expected_keywords": ["업무를 완결", "기업 데이터", "기업 업무 시스템"]},
            {"question": "제논이 앞으로 집중하려는 AI 방향은 뭐야?",
             "expected_keywords": ["Gen AI 2.0", "액셔너블 AI", "피지컬 AI"]},

            # ===== 플랫폼 =====
            {"question": "GenOS 2.0은 어떤 플랫폼이야?",
             "expected_keywords": ["GenOS", "AX 플랫폼"]},
            {"question": "GenD는 어떤 기능이야?",
             "expected_keywords": ["기업 데이터", "데이터 분석"]},
            {"question": "GenBuilder는 어떤 기능을 제공해?",
             "expected_keywords": ["업무 앱", "코드 생성", "배포"]},
            {"question": "GenA는 어떤 서비스야?",
             "expected_keywords": ["AI 에이전트 포털", "개인", "업무 생산성"]},

            # ===== Actionable AI =====
            {"question": "원에이전트는 무엇을 하는 AI야?",
             "expected_keywords": ["OneAgent", "액셔너블 AI", "업무를 완결"]},
            {"question": "액셔너블 AI가 왜 중요한 거야?",
             "expected_keywords": ["업무를 완결", "상용화"]},

            # ===== Physical AI =====
            {"question": "피지컬 AI는 어떤 의미야?",
             "expected_keywords": ["물리 세계", "휴머노이드"]},
            {"question": "제논은 피지컬 AI를 어디에 활용하려고 해?",
             "expected_keywords": ["KB금융", "시니어 케어", "휴머노이드"]},

            # ===== 상용화 =====
            {"question": "생성형 AI 시장은 올해 어떻게 변한다고 전망했어?",
             "expected_keywords": ["40%", "상용화", "프로덕션"]},
            {"question": "왜 생성형 AI가 상용화되기 어려웠다고 했어?",
             "expected_keywords": ["PoC", "파일럿", "업무 프로세스"]},
            {"question": "기업들이 생성형 AI를 실제로 도입하면서 달라진 점은 뭐야?",
             "expected_keywords": ["IT 예산", "프로덕션", "업무 자동화"]},

            # ===== 추론형 (HyDE 검증용) =====
            {"question": "제논이 해결하려는 가장 큰 문제는 뭐야?",
             "expected_keywords": ["업무를 완결", "상용화", "업무 프로세스"]},
            {"question": "제논의 핵심 기술을 한 문장으로 설명해줘.",
             "expected_keywords": ["Gen AI 2.0", "액셔너블 AI", "기업 데이터"]},
        ],
    },

    # ------------------------------------------------------------------
    # 서울여대 — 2026-07-27/23 크롤링된 실제 기사(제12대 이사장 이/취임,
    # 2027수시박람회) 기준으로 검증한 골든셋.
    # ------------------------------------------------------------------
    "서울여대": {
        "topic": "서울여대",
        "validation_terms": ["서울여자대학교"],
        "period_days": 50,
        "questions": [
            {"question": "서울여대 재단 이름이 뭐야?",
             "expected_keywords": ["정의학원"]},
            {"question": "서울여대 제12대 이사장은 누구야?",
             "expected_keywords": ["황성은"]},
            {"question": "서울여대 제11대 이사장은 누구였어?",
             "expected_keywords": ["김경진"]},
            {"question": "서울여대 총장은 누구야?",
             "expected_keywords": ["이윤선"]},
            {"question": "황성은 이사장은 어떤 경력을 가지고 있어?",
             "expected_keywords": ["창동염광교회", "장로회신학대"]},
            {"question": "서울여대가 올해 신설한 학과는 뭐야?",
             "expected_keywords": ["AI융합학부", "인공지능전공"]},
            {"question": "서울여대 2027학년도 수시 정원내 모집인원은 몇 명이야?",
             "expected_keywords": ["890명"]},
            {"question": "서울여대 2027수시 원서접수 기간은 언제야?",
             "expected_keywords": ["9월8일", "9월 8일"]},
            {"question": "서울여대는 어떤 구에 있어?",
             "expected_keywords": ["노원구"]},
            {"question": "서울여대와 같은 지역에 있는 대학교는?",
             "expected_keywords": ["광운대", "삼육대", "서울과학기술대", "인덕대", "한국성서대", "육군사관학교"]},
        ],
    },

    # ------------------------------------------------------------------
    # 강남 — 2026-07-29 크롤링된 실제 기사("강남, 아내 이상화 위해 스포츠
    # 마사지 자격증까지 땄다(라스)") 기준으로 검증한 골든셋.
    # 자녀 관련 질문처럼 이 기사에서 확인되지 않는 내용은 제외했다.
    # 이후 다른 기간/다른 기사가 섞이면 새로 확인된 사실을 추가할 것.
    # ------------------------------------------------------------------
    "강남": {
        "topic": "강남",
        "validation_terms": ["방송인","연예인"],
        "period_days": 50,
        "questions": [
            {"question": "강남의 아내는 누구야?",
             "expected_keywords": ["이상화"]},
            {"question": "강남은 최근 어떤 방송에 출연했어?",
             "expected_keywords": ["라디오스타", "라스"]},
            {"question": "강남이 아내 이상화를 위해 딴 자격증은 뭐야?",
             "expected_keywords": ["스포츠 마사지"]},
            {"question": "강남은 아내 이상화와 함께 어떤 특훈을 받았어?",
             "expected_keywords": ["마라톤", "태극기"]},
            {"question": "강남이 마라톤에 도전한 장소는 어디야?",
             "expected_keywords": ["북극"]},
            {"question": "강남이 최근 빠져 있는 취미는 뭐야?",
             "expected_keywords": ["포켓몬 카드"]},
        ],
    },
}

_DEBUG_PREVIEW_LIMIT = 5


def _is_hit(articles, expected_keywords: list[str]) -> bool:
    if not articles:
        return False
    text = " ".join(f"{a.title} {a.content}" for a in articles).lower()
    return any(keyword.lower() in text for keyword in expected_keywords)


async def run_eval(golden_set_name: str, golden_set: dict) -> tuple[int, int]:
    """골든셋 하나를 평가하고 (hits_v0, hits_v1, n)이 아니라 (hits_v0, hits_v1)을 반환한다.
    실패 시 (0, 0)을 반환하며, 이 경우 n=0이므로 run_all에서 합산에서 제외한다.
    """
    topic = golden_set["topic"]
    validation_terms = golden_set["validation_terms"]
    period_days = golden_set["period_days"]
    questions = golden_set["questions"]

    session = await collect_news(topic, validation_terms, period_days)
    if session is None:
        print(f"[{golden_set_name}] 세션 구축에 실패했습니다. topic/검증어/기간을 확인해주세요.")
        return 0, 0, 0

    # 디버그: 골든셋 관련 기사가 최종 세션(필터+중복제거 통과)에 남아있는지 확인.
    # 기사 제목을 전부 나열하지 않고, 개수 + 미리보기 몇 건만 보여준다.
    debug_keywords = {kw for item in questions for kw in item["expected_keywords"]}
    found_titles = [
        a.title for a in session.articles
        if any(kw.lower() in f"{a.title} {a.content}".lower() for kw in debug_keywords)
    ]
    if found_titles:
        print(f"[DEBUG] 최종 세션({len(session.articles)}건) 안에서 골든셋 관련 기사 {len(found_titles)}건 발견"
              f" (최대 {_DEBUG_PREVIEW_LIMIT}건만 미리보기):")
        for t in found_titles[:_DEBUG_PREVIEW_LIMIT]:
            print(f"  - {t}")
        if len(found_titles) > _DEBUG_PREVIEW_LIMIT:
            print(f"  ...외 {len(found_titles) - _DEBUG_PREVIEW_LIMIT}건")
    else:
        print(f"[DEBUG] 최종 세션({len(session.articles)}건) 안에 골든셋 관련 기사가 하나도 없음 "
              f"→ 관련성 필터 또는 크롤링 단계에서 탈락했을 가능성이 높음")

    print(f"\n=== 골든셋: {golden_set_name} (topic='{topic}', 검증어={validation_terms}) ===")
    print(f"{'질문':<40} {'V0 (HyDE 없음)':<18} {'V1 (HyDE)':<10}")
    print("-" * 70)

    hits_v0 = 0
    hits_v1 = 0

    for item in questions:
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

    n = len(questions)
    print("-" * 70)
    print(
        f"[{golden_set_name}] 검색 성공률(Hit Rate): "
        f"V0 = {hits_v0}/{n} ({hits_v0/n:.1%})  →  V1 = {hits_v1}/{n} ({hits_v1/n:.1%})\n"
    )
    return hits_v0, hits_v1, n


async def run_all() -> None:
    per_topic: list[tuple[str, int, int, int]] = []  # (name, hits_v0, hits_v1, n)

    for name, golden_set in GOLDEN_SETS.items():
        hits_v0, hits_v1, n = await run_eval(name, golden_set)
        if n > 0:
            per_topic.append((name, hits_v0, hits_v1, n))

    if not per_topic:
        print("평가된 골든셋이 없습니다.")
        return

    print("=" * 70)
    print("전체 요약")
    print("=" * 70)
    print(f"{'주제':<10} {'문항 수':<8} {'V0 Hit Rate':<14} {'V1 Hit Rate':<14}")
    print("-" * 70)
    for name, hits_v0, hits_v1, n in per_topic:
        print(f"{name:<10} {n:<8} {hits_v0}/{n} ({hits_v0/n:.1%})    {hits_v1}/{n} ({hits_v1/n:.1%})")
    print("-" * 70)

    # macro-average: 주제별 hit rate를 동일 가중치로 평균 (주제 수로 나눔)
    macro_v0 = sum(h0 / n for _, h0, _, n in per_topic) / len(per_topic)
    macro_v1 = sum(h1 / n for _, _, h1, n in per_topic) / len(per_topic)

    # micro-average: 전체 문항을 다 합쳐서 계산 (문항 수가 많은 주제에 더 큰 가중치)
    total_n = sum(n for _, _, _, n in per_topic)
    total_v0 = sum(h0 for _, h0, _, _ in per_topic)
    total_v1 = sum(h1 for _, _, h1, _ in per_topic)
    micro_v0 = total_v0 / total_n
    micro_v1 = total_v1 / total_n

    print(f"Macro 평균 (주제별 동일 가중치): V0 = {macro_v0:.1%}  →  V1 = {macro_v1:.1%}")
    print(f"Micro 평균 (전체 {total_n}문항 기준):   V0 = {total_v0}/{total_n} ({micro_v0:.1%})  "
          f"→  V1 = {total_v1}/{total_n} ({micro_v1:.1%})")


def main() -> None:
    if len(sys.argv) >= 2:
        name = sys.argv[1]
        if name not in GOLDEN_SETS:
            print(f"'{name}'에 해당하는 골든셋이 없습니다. 사용 가능: {list(GOLDEN_SETS.keys())}")
            return
        asyncio.run(run_eval(name, GOLDEN_SETS[name]))
    else:
        asyncio.run(run_all())


if __name__ == "__main__":
    main()