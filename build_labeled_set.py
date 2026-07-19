"""
build_labeled_set.py — 평가용 골드셋 라벨링 도구

실제 검색/크롤링 파이프라인으로 기사를 수집한 뒤, 콘솔에서 사람이 직접
y/n으로 "이 기사가 topic+validation_terms 기준 진짜 관련 있는가"를 라벨링하고
JSON으로 저장한다. 이 JSON을 eval_filter.py가 읽어서 Precision/Recall/F1을 계산한다.

사용법:
    python build_labeled_set.py "제논" "AI기업" 20
    (topic, validation_terms(콤마구분), period_days)

결과:
    labeled_set.json 파일이 생성/추가된다. (같은 파일에 계속 누적 가능)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from app.news_search import search_news
from app.crawler import crawl_articles
from app.query_understanding import analyze_topic
from app.schemas import Article

_OUTPUT_PATH = Path("labeled_set.json")


def _load_existing() -> list[dict]:
    if _OUTPUT_PATH.exists():
        return json.loads(_OUTPUT_PATH.read_text(encoding="utf-8"))
    return []


def _save(entries: list[dict]) -> None:
    _OUTPUT_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _ask_label(article: Article, topic: str, validation_terms: list[str]) -> bool | None:
    print("\n" + "=" * 60)
    print(f"제목: {article.title}")
    print(f"언론사: {article.press or '알 수 없음'}")
    print(f"본문 일부: {article.content[:300]}...")
    print("-" * 60)
    print(f"기준 → topic='{topic}', validation_terms={validation_terms}")
    while True:
        raw = input("이 기사가 실제로 관련 있습니까? (y/n, s=건너뛰기, q=중단): ").strip().lower()
        if raw == "y":
            return True
        if raw == "n":
            return False
        if raw == "s":
            return None
        if raw == "q":
            return "quit"  # type: ignore[return-value]
        print("y, n, s, q 중 하나를 입력하세요.")


def _read_args_interactively() -> tuple[str, list[str], int]:
    topic = input("관심 분야를 입력하세요: ").strip()
    raw_terms = input("검증 키워드를 입력하세요 (여러 개는 콤마로 구분): ").strip()
    validation_terms = [t.strip() for t in raw_terms.split(",") if t.strip()]
    while True:
        raw_days = input("기간을 숫자로 입력하세요 (최근 며칠, 예: 20): ").strip()
        try:
            period_days = int(raw_days)
            break
        except ValueError:
            print(f"'{raw_days}'는 숫자가 아닙니다. 다시 입력해주세요.")
    return topic, validation_terms, period_days


async def main() -> None:
    if len(sys.argv) >= 4:
        topic = sys.argv[1]
        validation_terms = [t.strip() for t in sys.argv[2].split(",") if t.strip()]
        period_days = int(sys.argv[3])
    else:
        # PyCharm 실행 버튼처럼 인자 없이 실행된 경우: main.py와 동일하게 인터랙티브로 입력받는다.
        topic, validation_terms, period_days = _read_args_interactively()

    search_query = analyze_topic(topic, period_days)
    search_query.filter_terms = validation_terms  # main.py와 동일한 방식으로 덮어씀

    metas = await search_news(search_query)
    print(f"뉴스 검색 완료 → {len(metas)}건 발견")
    if not metas:
        print("검색 결과가 없습니다.")
        return

    articles = await crawl_articles(metas)
    print(f"본문 크롤링 완료 → {len(articles)}건\n")
    if not articles:
        print("크롤링된 기사가 없습니다.")
        return

    entries = _load_existing()
    print(f"기존 라벨링된 데이터 {len(entries)}건에 이어서 진행합니다. (s=건너뛰기, q=중단 후 저장)\n")

    for article in articles:
        label = _ask_label(article, topic, validation_terms)
        if label == "quit":
            break
        if label is None:
            continue
        entries.append(
            {
                "article": article.model_dump(),
                "topic": topic,
                "validation_terms": validation_terms,
                "is_actually_relevant": label,
            }
        )
        _save(entries)  # 매 라벨링마다 저장 (중간에 중단해도 안전)

    print(f"\n완료. 총 {len(entries)}건이 {_OUTPUT_PATH}에 저장되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())