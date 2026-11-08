"""
build_labeled_set.py — 평가용 골드셋 라벨링 도구 (DB 없이 JSON 파일로)

실제 검색/크롤링 파이프라인으로 기사를 수집하고, 중복을 제거한 뒤, 콘솔에서 사람이 직접
"이 기사가 고른 의미(예: 방송인 강남) 기준으로 진짜 관련 있는가"를 라벨링해 JSON으로 저장한다.
이 JSON을 eval_filter.py가 읽어서 Precision/Recall/F1을 계산한다.

지금 구조에 맞춰 바뀐 점
  - 검증 키워드(validation_terms) 대신 0단계에서 고른 의미(SelectedSense)를 기준으로 쓴다
  - 라벨링 전에 중복 기사를 묶고(2단계와 같은 기준, 유사도 0.90) 대표 기사만 보여 준다
  - 대상당 최신 대표 기사 150건까지만 라벨링한다 (python evaluate.py label과 같은 기준)
  - 라벨은 1 관련 / 2 무관 / 3 보류 (예전 y/n → is_actually_relevant도 같이 저장해서 예전 형식과 호환)
  - 공용 DB에는 쓰지 않는다 (DB에 라벨을 붙이려면 python evaluate.py label)

사용법 (프로젝트 맨 위 폴더에서):
    python old/build_labeled_set.py 강남 "방송인 강남" 30
    (검색어, 의미 이름, 기간(일))
    인자 없이 실행하면 검색어·기간을 입력받고 의미는 후보에서 고른다.

결과:
    old/labeled_set.json 파일이 생성/추가된다. (같은 파일에 계속 누적, 이미 라벨 붙인 기사는 다시 안 물어봄)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from _common import describe, pick_sense

from qupict.collect import collect_articles
from qupict.filtering import embed_articles, find_duplicates
from qupict.pipeline import COLLECT_MAX_PER_TARGET, LABEL_PER_TARGET
from qupict.schemas import Article, SelectedSense

_OUTPUT_PATH = Path(__file__).resolve().parent / "labeled_set.json"
_LABEL_KEYS = {"1": "관련", "2": "무관", "3": "보류"}
_RELEVANT = {"관련": True, "무관": False, "보류": None}   # 예전 형식(is_actually_relevant)과 맞추기


def _load_existing() -> list[dict]:
    if _OUTPUT_PATH.exists():
        return json.loads(_OUTPUT_PATH.read_text(encoding="utf-8"))
    return []


def _save(entries: list[dict]) -> None:
    _OUTPUT_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _article_json(article: Article) -> dict:
    # 임베딩(숫자 1536개)과 이미지 파일 바이트는 저장하지 않는다 (파일이 너무 커짐)
    return article.model_dump(exclude={"embedding": True, "images": {"__all__": {"data"}}})


def _ask_label(article: Article, selected: SelectedSense, index: int, total: int) -> str | None:
    print("\n" + "=" * 60)
    date = article.pub_date.strftime("%Y-%m-%d") if article.pub_date else "날짜 없음"
    print(f"[{index}/{total}] 제목: {article.title}")
    print(f"언론사: {article.press or '알 수 없음'} · {date} · 찾은 검색어: {article.search_term}")
    print(f"본문 일부: {article.content[:300]}...")
    print("-" * 60)
    print(f"기준 → {selected.query} = {selected.sense.name}: {selected.sense.definition}")
    while True:
        raw = input("1 관련 / 2 무관 / 3 보류 (s=건너뛰기, q=중단): ").strip().lower()
        if raw in _LABEL_KEYS:
            return _LABEL_KEYS[raw]
        if raw == "s":
            return None
        if raw == "q":
            return "quit"
        print("1, 2, 3, s, q 중 하나를 입력하세요.")


def _read_args_interactively() -> tuple[str, str | None, int]:
    query = input("검색어를 입력하세요: ").strip()
    while True:
        raw_days = input("기간을 숫자로 입력하세요 (최근 며칠, 예: 30): ").strip()
        try:
            period_days = int(raw_days)
            break
        except ValueError:
            print(f"'{raw_days}'는 숫자가 아닙니다. 다시 입력해주세요.")
    return query, None, period_days


async def main() -> None:
    if len(sys.argv) >= 4:
        query, sense_name, period_days = sys.argv[1], sys.argv[2], int(sys.argv[3])
    else:
        # PyCharm 실행 버튼처럼 인자 없이 실행된 경우: run.py와 동일하게 인터랙티브로 입력받는다.
        query, sense_name, period_days = _read_args_interactively()

    selected = pick_sense(query, sense_name)
    print(f"기준: {describe(selected)}")

    metas, articles = await collect_articles(selected, period_days, COLLECT_MAX_PER_TARGET)
    print(f"뉴스 검색 완료 → {len(metas)}건 발견, 본문 크롤링 완료 → {len(articles)}건")
    if not articles:
        print("크롤링된 기사가 없습니다.")
        return

    # [2단계 중복 제거] 비슷한 기사는 대표 기사 하나만 라벨링한다 (임베딩만, LLM 판정 없음)
    await embed_articles(articles)
    duplicates = find_duplicates(articles)
    reps = [a for a in articles if a.link not in duplicates]
    reps.sort(key=lambda a: a.pub_date.timestamp() if a.pub_date else 0, reverse=True)   # 최신순
    print(f"중복 제거 → {len(articles)}건 중 중복 {len(duplicates)}건, 대표 기사 {len(reps)}건")

    entries = _load_existing()
    mine = [e for e in entries if e.get("query") == selected.query and e.get("sense") == selected.sense.name]
    done_links = {e["article"]["link"] for e in mine}
    remaining = max(0, LABEL_PER_TARGET - len(mine))
    todo = [a for a in reps if a.link not in done_links][:remaining]
    print(f"기존 라벨 {len(mine)}건 (목표 {LABEL_PER_TARGET}건) → 최신 대표 기사부터 {len(todo)}건 진행합니다."
          " (s=건너뛰기, q=중단 후 저장)\n")
    if len(todo) < remaining:
        print(f"  ※ 대표 기사가 모자라 목표보다 {remaining - len(todo)}건 적습니다. 기간을 늘려 다시 실행하세요.")

    for i, article in enumerate(todo, 1):
        label = _ask_label(article, selected, i, len(todo))
        if label == "quit":
            break
        if label is None:
            continue
        entries.append(
            {
                "article": _article_json(article),
                "query": selected.query,
                "sense": selected.sense.name,
                "label": label,
                "is_actually_relevant": _RELEVANT[label],
            }
        )
        _save(entries)  # 매 라벨링마다 저장 (중간에 중단해도 안전)

    print(f"\n완료. 총 {len(entries)}건이 {_OUTPUT_PATH}에 저장되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())
