"""
build_labeled_set.py — 평가용 정답 라벨링 도구 (DB 없이 JSON 파일에 계속 쌓는다)

한 번 실행하면 아래를 반복한다. 검색어에 q를 입력하면 끝난다.

  검색어 입력 → 기간 입력 → 의미 고르기(0단계) → 수집(1단계) → 중복 제거(2단계, 대표 기사만)
  → 최신 대표 기사부터 1 관련 / 2 무관 라벨 → 다시 검색어 입력 ...

  - 결과는 evalset/labeled_set_1.json, labeled_set_2.json, ... 에 검색어 단위로 나눠 저장한다
    (파일 하나가 너무 커지지 않게. 나누는 규칙은 evalset/labeled_store.py).
    라벨 하나 붙일 때마다 바로 저장하므로 중간에 꺼져도 그때까지 붙인 라벨은 남는다.
  - 다시 실행하면 이어서 쓴다. 이미 있는 검색어는 그 검색어가 든 파일에, 새 검색어는 마지막 파일
    (자리가 모자라면 다음 번호 파일)에 쓴다. 이미 라벨을 붙인 기사(검색어 × 의미 × 주소)는 다시 묻지 않는다.
  - 검색어 × 의미 하나당 최신 대표 기사 LABEL_PER_TARGET(150)건까지만 묻는다.
  - 라벨링 중 s = 이 기사 건너뛰기, q = 이 검색어 그만하고 다음 검색어로.
  - 의미·연관어·별칭은 LLM이 만든 후보만 쓴다.
    같은 검색어는 처음 만든 후보를 DB(query_senses)에 저장해 두고 다시 쓴다 (재현성).
  - 라벨은 공용 DB에 쓰지 않는다.

저장 형식 (항목 하나 = 기사 하나, old/eval_filter.py가 읽는다):
  {"title", "url", "description", "content", "query",
   "sense": {"name", "definition", "aliases", "related_terms", "excluded_terms", "other_senses": [...]},
   "label": true(관련) / false(무관)}

사용법 (프로젝트 맨 위 폴더에서):
    python evalset/build_labeled_set.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:            # evalset/ 안의 파일을 바로 실행해도 qupict를 찾을 수 있게
    sys.path.insert(0, str(ROOT))

from qupict.cli import read_line, setup_console                                  # noqa: E402
from qupict.collect import collect_articles                                      # noqa: E402
from qupict.filtering import embed_for_dedup, find_duplicates                    # noqa: E402
from qupict.pipeline import COLLECT_MAX_PER_TARGET, LABEL_PER_TARGET             # noqa: E402
from qupict.schemas import Article, SelectedSense                                # noqa: E402
from qupict.sense import choose_sense_interactively, load_sense_candidates      # noqa: E402
from evalset.labeled_store import LabeledStore                                   # noqa: E402

_OUTPUT_DIR = Path(__file__).resolve().parent      # labeled_set_N.json 이 쌓이는 폴더
DEFAULT_PERIOD_DAYS = 30
_LABEL_KEYS = {"1": True, "2": False}      # 1 관련 / 2 무관
_QUIT = "q"


# ---------------------------------------------------------------------------
# 파일
# ---------------------------------------------------------------------------

def _sense_json(selected: SelectedSense) -> dict:
    s = selected.sense
    return {
        "name": s.name,
        "definition": s.definition,
        "aliases": s.aliases,
        "related_terms": s.related_terms,
        "excluded_terms": selected.excluded_terms,
        "other_senses": [{"name": o.name, "definition": o.definition, "related_terms": o.related_terms}
                         for o in selected.other_senses],
    }


def _entry(article: Article, selected: SelectedSense, label: bool) -> dict:
    return {
        "title": article.title,
        "url": article.link,
        "description": article.description,
        "content": article.content,
        "query": selected.query,
        "sense": _sense_json(selected),
        "label": label,
    }


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------

def describe(selected: SelectedSense) -> str:
    """출력용 한 줄: 예) 강남 → 방송인 강남 (배제어: 강남역, 재건축)"""
    excluded = ", ".join(selected.excluded_terms) or "없음"
    return f"{selected.query} → {selected.sense.name} (배제어: {excluded})"


def _read_period_days() -> int:
    while True:
        raw = input(f"기간(최근 며칠, Enter = {DEFAULT_PERIOD_DAYS})> ").strip()
        if not raw:
            return DEFAULT_PERIOD_DAYS
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("기간은 1 이상의 정수로 입력해주세요.")


def _pick_sense(query: str) -> SelectedSense:
    """LLM이 만든 의미 후보에서 고른다 (DB에 캐시가 있으면 그 후보)."""
    candidates, source = load_sense_candidates(query)
    print(f"[의미 후보] {source}")
    return choose_sense_interactively(candidates)


def _ask_label(article: Article, selected: SelectedSense, index: int, total: int) -> bool | str | None:
    """True 관련 / False 무관 / None 건너뛰기 / "q" 이 검색어 중단"""
    print("\n" + "=" * 60)
    date = article.pub_date.strftime("%Y-%m-%d") if article.pub_date else "날짜 없음"
    print(f"[{index}/{total}] 제목: {article.title}")
    print(f"언론사: {article.press or '알 수 없음'} · {date} · 찾은 검색어: {article.search_term}")
    print(f"본문 일부: {article.content[:300]}...")
    print("-" * 60)
    print(f"기준 → {selected.query} = {selected.sense.name}: {selected.sense.definition}")
    while True:
        raw = input("1 관련 / 2 무관 (s=건너뛰기, q=이 검색어 그만)> ").strip().lower()
        if raw in _LABEL_KEYS:
            return _LABEL_KEYS[raw]
        if raw == "s":
            return None
        if raw == _QUIT:
            return _QUIT
        print("1, 2, s, q 중 하나를 입력하세요.")


# ---------------------------------------------------------------------------
# 검색어 하나 라벨링
# ---------------------------------------------------------------------------

async def label_query(query: str, store: LabeledStore) -> int:
    """검색어 하나를 수집·중복 제거하고 라벨을 붙여 store에 추가·저장한다. 붙인 라벨 수를 돌려준다."""
    period_days = _read_period_days()
    selected = _pick_sense(query)
    print(f"기준: {describe(selected)}")

    metas, articles = await collect_articles(selected, period_days, COLLECT_MAX_PER_TARGET)
    print(f"뉴스 검색 → {len(metas)}건, 본문 크롤링 → {len(articles)}건")
    if not articles:
        print("크롤링된 기사가 없습니다.")
        return 0

    # [2단계 중복 제거] 비슷한 기사는 대표 기사 하나만 라벨링한다 (임베딩만, LLM 판정 없음)
    await embed_for_dedup(articles, selected.match_terms)
    duplicates = find_duplicates(articles, terms=selected.match_terms)
    reps = [a for a in articles if a.link not in duplicates]
    reps.sort(key=lambda a: a.pub_date.timestamp() if a.pub_date else 0, reverse=True)   # 최신순
    print(f"중복 제거 → 중복 {len(duplicates)}건, 대표 기사 {len(reps)}건")

    mine = [e for e in store.entries if e.get("query") == selected.query
            and isinstance(e.get("sense"), dict) and e["sense"].get("name") == selected.sense.name]
    done_urls = {e["url"] for e in mine}
    remaining = max(0, LABEL_PER_TARGET - len(mine))
    todo = [a for a in reps if a.link not in done_urls][:remaining]
    print(f"이미 라벨 {len(mine)}건 (목표 {LABEL_PER_TARGET}건) → 최신 대표 기사부터 {len(todo)}건 진행합니다.")
    if len(todo) < remaining:
        print(f"  ※ 새 대표 기사가 모자라 목표보다 {remaining - len(todo)}건 적습니다. 기간을 늘려 다시 해 보세요.")

    print(f"저장 파일: {store.path_for(selected.query).name}")
    added = 0
    for i, article in enumerate(todo, 1):
        label = _ask_label(article, selected, i, len(todo))
        if label == _QUIT:
            break
        if label is None:
            continue
        store.add(_entry(article, selected, label))      # 매 라벨마다 저장 (중간에 꺼져도 안전)
        added += 1
    print(f"\n'{describe(selected)}' 라벨 {added}건 추가 (이 기준 누적 {len(mine) + added}건)")
    return added


async def main() -> None:
    store = LabeledStore(_OUTPUT_DIR, per_query=LABEL_PER_TARGET)
    print(f"라벨 폴더: {_OUTPUT_DIR} (파일 {len(store.files)}개, 기존 {len(store.entries)}건)")
    added = 0
    while True:
        query = read_line("\n검색어 (q=종료)> ")
        if query.lower() == _QUIT:
            break
        added += await label_query(query, store)
    print(f"\n종료. 이번에 {added}건 추가, 총 {len(store.entries)}건이 "
          f"{', '.join(p.name for p in store.files)}에 저장되어 있습니다.")


if __name__ == "__main__":
    setup_console()          # asyncio.run 전에 불러야 Windows 루프 설정이 적용된다
    asyncio.run(main())
