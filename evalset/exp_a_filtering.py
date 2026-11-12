"""
실험 A — 관련성 필터링 성능 비교 (기획안 7단계), 정답 JSON 기준

evalset/labeled_set_*.json(python evalset/build_labeled_set.py로 만든 정답, 검색어 단위로 나뉜 파일 전부)을 읽어, 방법별로
"관련 기사를 얼마나 정확히 골라내는지" 비교한다. DB는 쓰지 않는다.
기준(고른 의미·별칭·연관어·배제어·다른 의미)은 라벨링할 때 JSON에 함께 저장한 값을 그대로 쓴다.

  방법                  기사를 '관련'으로 통과시키는 조건
  ───────────────────  ─────────────────────────────────────────────────────────────
  baseline             전부 (네이버 API 결과 그대로)
  1a_alias             검색어·별칭이 기사에 있다 (정식 명칭·별칭 매칭)
  1b_validation        1a + 검증어(VALIDATION_TERMS)가 기사에 있다 (기존 검증 키워드 방식 재현)
  1c_rule              1a + 규칙 점수(연관어 − 배제어, 제목 2점·본문 1점) ≥ RULE_ACCEPT_SCORE
  2b_llm_only          LLM 판정이 관련·보류 (모든 기사를 LLM으로)
  proposed             1차 규칙 + 2차 LLM: 규칙이 '무관'이면 무관, 아니면 LLM 판정 (서비스 파이프라인과 같은 순서)

지표: Precision · Recall · F1 · Accuracy · 무관 기사 유입률, LLM 호출 수(2b vs 제안 방식)

- 정답은 label true(관련) / false(무관). 라벨링 때 건너뛴 기사는 파일에 없다.
- LLM 판정 '보류'는 '관련'으로 센다. 실제 서비스가 보류 기사도 답변에 쓰기 때문이다.
- LLM 판정은 기사마다 한 번만 하고 evalset/results/llm_cache.json에 저장한다 (다시 돌려도 비용 0).
  2b는 그 판정을 그대로, 제안 방식은 규칙을 통과한 기사에만 그 판정을 쓴다.
  LLM 호출 수는 "실제로 그 방법만 돌렸다면" 기준으로 센다 (2b = 전체 기사, 제안 = 규칙 통과 기사, 8건당 1번).
  LLM이 답하지 못한 기사는 캐시에 넣지 않고, 그 방법에서는 건너뛴다 (skipped). 다시 돌리면 그 기사만 다시 판정.
- 1c는 기준값별 결과도 함께 낸다 (규칙 기준값을 정할 때 참고).

실행 (프로젝트 맨 위 폴더에서):
    python evalset/exp_a_filtering.py [--target 강남] [--file evalset/labeled_set_2.json]   (--file 없으면 전부)
    python evaluate.py exp-a [--target 강남]          (같은 실험)
결과: evalset/results/exp_a_filtering_날짜시각.json / .csv
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:            # evalset/ 안의 파일을 바로 실행해도 qupict를 찾을 수 있게
    sys.path.insert(0, str(ROOT))

from experiments.common import BinaryScores, print_table, save_results           # noqa: E402
from qupict.filtering import LLM_BATCH_SIZE, llm_verify, rule_check, rule_verdict  # noqa: E402
from qupict.schemas import Article, SelectedSense, Sense                          # noqa: E402
from evalset.labeled_store import LABELED_DIR, load_entries                      # noqa: E402

EVALSET_DIR = Path(__file__).resolve().parent
LABELED_SET_PATH = LABELED_DIR             # 폴더면 labeled_set_*.json 전부, 파일이면 그 파일만
RESULTS_DIR = EVALSET_DIR / "results"
LLM_CACHE = RESULTS_DIR / "llm_cache.json"

RULE_ACCEPT_SCORE = 1                     # 1c: 이 점수 이상이면 관련
RULE_SWEEP = [-2, -1, 0, 1, 2, 3, 4]      # 1c 기준값별 결과
METHODS = ["baseline", "1a_alias", "1b_validation", "1c_rule", "2b_llm_only", "proposed"]
PASS_STATUSES = ("관련", "보류")          # 서비스가 답변에 쓰는 판정 = "관련 있다고 예측"

# 1b: 예전 "검증 키워드" 방식에서 사용자가 입력하던 검증어. 방법 1b를 재현하는 데만 쓴다
#     (의미·연관어 기준에는 쓰지 않음). 키는 (검색어, 라벨링 때 고른 의미 이름).
#     여기에 없는 대상은 1b를 건너뛴다 (skipped에 집계).
VALIDATION_TERMS: dict[tuple[str, str], str] = {
    ("서울여대", "서울여자대학교"): "대학",
    ("삼전", "기업 삼성전자"): "반도체",
    ("베라", "아이스크림 브랜드 배스킨라빈스"): "아이스크림",
    ("강남", "방송인 강남"): "방송인",
    ("애플", "기업 Apple"): "아이폰",
    ("당근", "중고거래 플랫폼 당근"): "중고거래",
    ("눈", "기상 눈"): "폭설",
    ("눈", "신체 눈"): "시력",
    ("후드", "의류 후드"): "패션",
    ("후드", "주방 후드"): "주방",
    ("마스크", "방역용 마스크"): "미세먼지",
    ("마스크", "마스크 팩 화장품"): "피부",
}


@dataclass
class LabeledArticle:
    article: Article
    relevant: bool              # 사람 정답: True 관련 / False 무관


@dataclass
class Group:
    """검색어 × 의미 하나 = 대상 하나"""
    selected: SelectedSense
    items: list[LabeledArticle]

    @property
    def name(self) -> str:
        return f"{self.selected.query} → {self.selected.sense.name}"


# ---------------------------------------------------------------------------
# 정답 JSON 읽기
# ---------------------------------------------------------------------------

def _selected_from_entry(entry: dict) -> SelectedSense:
    s = entry["sense"]
    return SelectedSense(
        query=str(entry["query"]).strip(),
        sense=Sense(name=s["name"], definition=s.get("definition", ""),
                    aliases=s.get("aliases", []), related_terms=s.get("related_terms", [])),
        excluded_terms=s.get("excluded_terms", []),
        other_senses=[Sense(name=o["name"], definition=o.get("definition", ""),
                            related_terms=o.get("related_terms", [])) for o in s.get("other_senses", [])],
    )


def load_groups(path: Path = LABELED_SET_PATH, query: str | None = None) -> tuple[list[Group], int]:
    """정답 JSON → 대상별 묶음. (묶음 목록, 읽지 못해 건너뛴 항목 수)
    같은 대상의 기준은 그 대상의 첫 항목에 저장된 값을 쓴다 (라벨링 때 고른 기준)."""
    entries = load_entries(path)
    groups: dict[tuple[str, str], Group] = {}
    skipped = 0
    for entry in entries:
        try:
            if not isinstance(entry.get("label"), bool):
                raise ValueError("label이 true/false가 아님")
            selected = _selected_from_entry(entry)
            article = Article(title=entry["title"], link=entry["url"],
                              description=entry.get("description") or "", content=entry.get("content") or "")
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if query and selected.query != query:
            continue
        key = (selected.query, selected.sense.name)
        group = groups.setdefault(key, Group(selected, []))
        if all(x.article.link != article.link for x in group.items):       # 같은 기사가 두 번 있으면 처음 것만
            group.items.append(LabeledArticle(article, entry["label"]))
    return list(groups.values()), skipped


# ---------------------------------------------------------------------------
# LLM 판정 (캐시)
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    try:
        return json.loads(LLM_CACHE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    LLM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LLM_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


async def llm_statuses(group: Group, cache: dict) -> dict[str, str]:
    """대상의 모든 기사에 대한 LLM 판정 {url: 관련/무관/보류}. 캐시에 없는 기사만 새로 판정한다."""
    topic_cache = cache.setdefault(group.name, {})
    todo = [x.article for x in group.items if x.article.link not in topic_cache]
    if todo:
        print(f"  {group.name}: LLM 판정 {len(todo)}건 (약 {math.ceil(len(todo) / LLM_BATCH_SIZE)}번 호출)")
        verdicts, _ = await llm_verify(todo, group.selected)
        for v in verdicts:
            topic_cache[v.link] = v.status
        _save_cache(cache)
    return {x.article.link: topic_cache[x.article.link] for x in group.items if x.article.link in topic_cache}


# ---------------------------------------------------------------------------
# 방법별 예측
# ---------------------------------------------------------------------------

def _contains(article: Article, term: str) -> bool:
    from qupict.filtering import _normalize
    t = _normalize(term)
    return bool(t) and t in _normalize(f"{article.title} {article.content}")


def predict(method: str, x: LabeledArticle, selected: SelectedSense,
            llm: dict[str, str], accept_score: int = RULE_ACCEPT_SCORE) -> bool | None:
    """방법별 예측 (True = 관련으로 통과). None이면 예측할 수 없음 (지표에서 뺌)."""
    a = x.article
    if method == "baseline":
        return True
    r = rule_check(a, selected)
    if method == "1a_alias":
        return r.mentioned
    if method == "1b_validation":
        term = VALIDATION_TERMS.get((selected.query, selected.sense.name))
        return None if not term else (r.mentioned and _contains(a, term))
    if method == "1c_rule":
        return r.mentioned and r.score >= accept_score
    if method == "2b_llm_only":
        s = llm.get(a.link)
        return None if s is None else s in PASS_STATUSES
    if method == "proposed":
        if rule_verdict(a, selected) is not None:        # 1차 규칙에서 무관
            return False
        s = llm.get(a.link)
        return None if s is None else s in PASS_STATUSES
    raise ValueError(method)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

async def run(query: str | None = None, path: Path = LABELED_SET_PATH) -> list[dict]:
    groups, bad = load_groups(path, query)
    if bad:
        print(f"[경고] 읽지 못한 항목 {bad}건은 건너뜁니다 (label이 true/false가 아니거나 필드가 빠짐).")
    cache = _load_cache()
    rows, sweep_rows = [], []
    totals = {m: BinaryScores() for m in METHODS}
    calls = {"2b_llm_only": 0, "proposed": 0}
    for group in groups:
        selected = group.selected
        llm = await llm_statuses(group, cache)
        to_llm = sum(1 for x in group.items if rule_verdict(x.article, selected) is None)
        calls["proposed"] += math.ceil(to_llm / LLM_BATCH_SIZE)
        calls["2b_llm_only"] += math.ceil(len(group.items) / LLM_BATCH_SIZE)

        for method in METHODS:
            scores, skipped = BinaryScores(), 0
            for x in group.items:
                p = predict(method, x, selected, llm)
                if p is None:
                    skipped += 1
                    continue
                scores.add(p, x.relevant)
            totals[method].merge(scores)
            rows.append({"target": group.name, "method": method, **scores.as_row(), "skipped": skipped})

        for s in RULE_SWEEP:
            sc = BinaryScores()
            for x in group.items:
                sc.add(predict("1c_rule", x, selected, llm, accept_score=s), x.relevant)
            sweep_rows.append({"target": group.name, "accept_score": s, **sc.as_row()})

    for method in METHODS:
        rows.append({"target": "전체(micro)", "method": method, **totals[method].as_row(),
                     "llm_calls": calls.get(method, 0)})
    return rows + [{"section": "1c 기준값별", **r} for r in sweep_rows]   # target 칸은 그대로 두고 구분 칸을 따로 둔다


def main(query: str | None = None, path: Path = LABELED_SET_PATH) -> None:
    rows = asyncio.run(run(query, path))
    overall = [r for r in rows if r.get("target") == "전체(micro)"]
    if not overall or overall[0]["n"] == 0:
        print(f"정답 라벨이 없습니다 ({path}). 먼저 python evalset/build_labeled_set.py 로 라벨을 붙이세요.")
        return
    per_target = [r for r in rows if r.get("method") == "proposed" and r.get("target") != "전체(micro)"]
    print("\n[대상별 · 제안 방식]")
    print_table(per_target, ["target", "n", "precision", "recall", "f1", "accuracy", "skipped"])
    print("\n[전체(micro)]")
    print_table(overall, ["method", "n", "precision", "recall", "f1", "accuracy", "irrelevant_inflow", "llm_calls"])
    json_path, csv_path = save_results(
        "exp_a_filtering", rows,
        {"labeled_set": str(path), "rule_accept_score": RULE_ACCEPT_SCORE, "llm_batch_size": LLM_BATCH_SIZE},
        results_dir=RESULTS_DIR)
    print(f"\n대상별·기준값별 결과 저장: {csv_path}")


if __name__ == "__main__":
    from qupict.cli import setup_console
    setup_console()          # asyncio.run 전에 (Windows 루프 설정)
    parser = argparse.ArgumentParser(prog="python evalset/exp_a_filtering.py",
                                     description="실험 A: 정답 JSON으로 필터링 방법 비교")
    parser.add_argument("--target", help="이 검색어만 (예: 강남)")
    parser.add_argument("--file", type=Path, default=LABELED_SET_PATH, help="정답 JSON 파일 하나만 쓸 때 (기본: evalset/labeled_set_*.json 전부)")
    args = parser.parse_args()
    main(args.target, args.file)
