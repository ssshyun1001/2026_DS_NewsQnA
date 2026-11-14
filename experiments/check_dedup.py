"""
중복 제거 확인 — 기사를 새로 검색·크롤링하고, 지금 설정한 중복 제거 기준으로 분류한 결과를 보여 준다

  1) 검색어 입력 → 의미 고르기 (run.py와 같은 의미 후보)
  2) 네이버 검색 + 본문 크롤링 (서비스와 같은 수집)
  3) 중복 제거 (서비스와 같은 기준·임계값: .env의 DEDUP_* 값)
  4) 결과: 중복으로 걸린 기사(대표 기사별로 묶음) / 중복 안 걸린 기사

DB에는 아무것도 저장하지 않는다. LLM 판정도 하지 않는다 (의미 후보만 DB 캐시 또는 LLM에서 가져옴).
결과는 터미널에 출력하고, 링크·본문 앞부분을 붙여 experiments/results/check_dedup_검색어_의미_날짜시각.md로도 저장한다.

실행 (프로젝트 맨 위 폴더에서, 인자 없이 실행하면 입력을 물어본다):
    python experiments/check_dedup.py
    python experiments/check_dedup.py --target 강남 --sense "방송인 강남" --days 30 --max 100
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qupict.cli import setup_console                                              # noqa: E402
from qupict.collect import collect_articles                                       # noqa: E402
from qupict.config import settings                                                # noqa: E402
from qupict.filtering import embed_for_dedup, find_duplicates                     # noqa: E402
from qupict.schemas import Article, SelectedSense                                 # noqa: E402
from qupict.sense import choose_sense_interactively, load_sense_candidates, select_sense_by_name  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_DAYS, DEFAULT_MAX = 30, 100
SNIPPET = 200                        # 결과 파일에 붙일 본문 앞부분 길이


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} (Enter = {default})> ").strip()
        if not raw:
            return default
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("1 이상의 정수로 입력해 주세요.")


def pick_sense(query: str, name: str | None) -> SelectedSense:
    candidates, source = load_sense_candidates(query)
    print(f"[의미 후보] {source}")
    if name:
        selected = select_sense_by_name(candidates, name)
        if selected:
            return selected
        print(f"'{name}' 의미를 찾지 못했습니다. 번호로 골라 주세요.")
    return choose_sense_interactively(candidates)


def group_duplicates(articles: list[Article], duplicates: dict[str, str]):
    """→ (묶음 [(대표, [중복...])], 중복 안 걸린 기사 [...])  묶음은 큰 것부터."""
    by_link = {a.link: a for a in articles}
    groups: dict[str, list[Article]] = {}
    for dup, rep in duplicates.items():
        groups.setdefault(rep, []).append(by_link[dup])
    grouped = sorted(((by_link[rep], dups) for rep, dups in groups.items()), key=lambda g: -len(g[1]))
    alone = [a for a in articles if a.link not in duplicates and a.link not in groups]
    return grouped, alone


def _date(a: Article) -> str:
    return a.pub_date.strftime("%m-%d %H:%M") if a.pub_date else "날짜 없음"


def _line(a: Article) -> str:
    return f"{a.title}  ({a.press or '언론사 모름'}, {_date(a)})"


def print_result(articles, duplicates, grouped, alone) -> None:
    print(f"\n기사 {len(articles)}건 → 중복으로 걸린 기사 {len(duplicates)}건 / "
          f"남은 기사 {len(articles) - len(duplicates)}건 (대표 {len(grouped)}건 + 중복 안 걸린 기사 {len(alone)}건)")
    print(f"\n=== 중복으로 걸린 기사 {len(duplicates)}건 (대표 기사 {len(grouped)}건에 묶임) ===")
    for n, (rep, dups) in enumerate(grouped, 1):
        print(f"\n[{n}] 대표: {_line(rep)}")
        for d in dups:
            print(f"     └ 중복: {_line(d)}")
    print(f"\n=== 중복 안 걸린 기사 {len(alone)}건 ===")
    for a in alone:
        print(f"  - {_line(a)}")


def write_result(path: Path, selected: SelectedSense, settings_line: str, articles, duplicates, grouped, alone) -> None:
    md = lambda t: (t or "").replace("\n", " ")
    item = lambda a: (f"**{md(a.title)}** — {a.press or '언론사 모름'}, {_date(a)} · [링크]({a.link})\n"
                      f"  > {md(a.content[:SNIPPET])}…")
    L = [f"# 중복 제거 확인 — {selected.query} → {selected.sense.name}", "",
         f"- 실행: {datetime.now():%Y-%m-%d %H:%M}",
         f"- 설정: {settings_line}",
         f"- 기사 {len(articles)}건 → 중복으로 걸린 기사 {len(duplicates)}건 / 남은 기사 {len(articles) - len(duplicates)}건",
         "", f"## 중복으로 걸린 기사 {len(duplicates)}건 (대표 기사 {len(grouped)}건에 묶임)", ""]
    for n, (rep, dups) in enumerate(grouped, 1):
        L += [f"### 묶음 {n} ({len(dups) + 1}건)", "", f"- 대표: {item(rep)}"]
        L += [f"- 중복: {item(d)}" for d in dups]
        L += [""]
    L += [f"## 중복 안 걸린 기사 {len(alone)}건", ""]
    L += [f"- {item(a)}" for a in alone]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


async def run(args) -> None:
    query = args.target or input("검색어> ").strip()
    if not query:
        print("검색어를 입력해 주세요.")
        return
    selected = pick_sense(query, args.sense)
    days = args.days or ask_int("기간(최근 며칠)", DEFAULT_DAYS)
    max_articles = args.max or ask_int("최대 수집 기사 수", DEFAULT_MAX)

    print(f"\n검색·크롤링 중… ({selected.query} → {selected.sense.name}, 최근 {days}일, 최대 {max_articles}건)")
    metas, articles = await collect_articles(selected, days, max_articles)
    print(f"뉴스 검색 {len(metas)}건 → 본문 크롤링 {len(articles)}건")
    if not articles:
        print("크롤링된 기사가 없습니다.")
        return

    settings_line = (f"DEDUP_BY={settings.dedup_by}, DEDUP_RULE={settings.dedup_rule}, "
                     f"임계값 title={settings.dedup_title_threshold} content={settings.dedup_content_threshold} "
                     f"noun={settings.dedup_noun_threshold} jaccard={settings.dedup_jaccard_threshold}, "
                     f"검색어 문단 ±{settings.dedup_passage_window}문장, 발행일 창 {settings.dedup_date_window_days}일")
    print(f"중복 제거 중… ({settings_line})")
    await embed_for_dedup(articles, selected.match_terms)
    duplicates = find_duplicates(articles, terms=selected.match_terms)

    grouped, alone = group_duplicates(articles, duplicates)
    print_result(articles, duplicates, grouped, alone)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in f"{selected.query}_{selected.sense.name}")
    path = RESULTS_DIR / f"check_dedup_{safe}_{datetime.now():%Y%m%d_%H%M}.md"
    write_result(path, selected, settings_line, articles, duplicates, grouped, alone)
    print(f"\n결과 파일 (링크·본문 앞부분 포함): {path}")


def main() -> None:
    setup_console()
    p = argparse.ArgumentParser(description="기사를 크롤링하고 지금 기준으로 중복 제거한 결과를 보여 준다 (DB 저장 없음)")
    p.add_argument("--target", help="검색어 (비우면 입력을 물어봄)")
    p.add_argument("--sense", help="의미 이름 (예: '방송인 강남'). 비우면 번호로 고름")
    p.add_argument("--days", type=int, help=f"최근 며칠 기사 (비우면 물어봄, 기본 {DEFAULT_DAYS})")
    p.add_argument("--max", type=int, help=f"최대 수집 기사 수 (비우면 물어봄, 기본 {DEFAULT_MAX})")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
