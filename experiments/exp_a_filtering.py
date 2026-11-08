"""
실험 A — 관련성 필터링 성능 비교 (기획안 7단계)

사람이 정답 라벨을 붙인 기사(중복 제외)로, 방법별로 "관련 기사를 얼마나 정확히 골라내는지" 비교한다.

  방법                  기사를 '관련'으로 통과시키는 조건
  ───────────────────  ─────────────────────────────────────────────────────────────
  baseline             전부 (네이버 API 결과 그대로)
  1a_alias             검색어·별칭이 기사에 있다 (정식 명칭·별칭 매칭)
  1b_validation        1a + targets.yaml의 검증어(validation_term)가 기사에 있다 (기존 검증 키워드 방식 재현)
  1c_rule              1a + 규칙 점수(연관어 − 배제어, 제목 2점·본문 1점) ≥ RULE_ACCEPT_SCORE
  2b_llm_only          LLM 판정이 관련·보류 (모든 기사를 LLM으로)
  proposed             1차 규칙 + 2차 LLM (실제 파이프라인 판정, DB에 저장된 것)

지표: Precision · Recall · F1 · Accuracy · 무관 기사 유입률, LLM 호출 수(2b vs 제안 방식)

- 정답 라벨 '보류'인 기사는 정답이 애매하므로 지표에서 뺀다 (개수만 보고).
- LLM 판정 '보류'는 '관련'으로 센다. 실제 서비스가 보류 기사도 답변에 쓰기 때문이다.
- 2b(LLM만)는 규칙으로 걸러진 기사만 LLM을 추가로 부른다 (나머지는 제안 방식의 LLM 판정과 같은 호출).
  추가 판정은 experiments/results/llm_only_cache.json에 저장해서 다시 돌려도 비용이 들지 않는다.
- 1c는 기준값별 결과도 함께 낸다 (규칙 기준값을 정할 때 참고).

실행: python run.py exp-a [--target 강남]
"""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass

from experiments.common import RESULTS_DIR, BinaryScores, print_table, save_results
from qupict import db
from qupict.filtering import LLM_BATCH_SIZE, llm_verify, rule_check
from qupict.pipeline import pick_targets, resolve_target, target_label
from qupict.schemas import Article, SelectedSense

RULE_ACCEPT_SCORE = 1                     # 1c: 이 점수 이상이면 관련
RULE_SWEEP = [-2, -1, 0, 1, 2, 3, 4]      # 1c 기준값별 결과
METHODS = ["baseline", "1a_alias", "1b_validation", "1c_rule", "2b_llm_only", "proposed"]
LLM_ONLY_CACHE = RESULTS_DIR / "llm_only_cache.json"


@dataclass
class LabeledArticle:
    article: Article
    label: str                  # 사람 정답: 관련 / 무관 / 보류 (/ 동음이의어)
    status: str | None          # 시스템 판정 (제안 방식)
    stage: str | None


def load_labeled(topic_id: int) -> list[LabeledArticle]:
    """정답 라벨이 있는 기사 (중복 제외). 읽기 전용 조회."""
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT a.link, a.title, a.description, a.content, at.label, at.status, at.stage
               FROM article_topics at JOIN articles a ON a.id = at.article_id
               WHERE at.topic_id = %s AND at.label IS NOT NULL AND at.duplicate_of IS NULL""",
            (topic_id,),
        ).fetchall()
    return [LabeledArticle(Article(title=r["title"], link=r["link"], description=r["description"],
                                   content=r["content"]), r["label"], r["status"], r["stage"]) for r in rows]


def _contains(article: Article, term: str) -> bool:
    from qupict.filtering import _normalize
    t = _normalize(term)
    return bool(t) and t in _normalize(f"{article.title} {article.content}")


def _load_cache() -> dict:
    try:
        return json.loads(LLM_ONLY_CACHE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


async def llm_only_status(topic_id: int, selected: SelectedSense, items: list[LabeledArticle]) -> dict[str, str]:
    """2b: 모든 기사의 LLM 판정. 제안 방식에서 LLM이 판정한 기사는 그 결과를 쓰고,
    규칙으로 걸러져 LLM이 안 본 기사만 새로 판정한다 (결과는 캐시 파일에 저장)."""
    cache = _load_cache()
    topic_cache = cache.setdefault(str(topic_id), {})
    status = {x.article.link: x.status for x in items if x.stage == "llm" and x.status}
    todo = [x.article for x in items if x.article.link not in status and x.article.link not in topic_cache]
    if todo:
        verdicts, _ = await llm_verify(todo, selected)
        for v in verdicts:
            topic_cache[v.link] = v.status
        LLM_ONLY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LLM_ONLY_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    for x in items:
        status.setdefault(x.article.link, topic_cache.get(x.article.link))
    return status


def predict(method: str, x: LabeledArticle, target: dict, selected: SelectedSense,
            llm_only: dict[str, str], accept_score: int = RULE_ACCEPT_SCORE) -> bool | None:
    """방법별 예측 (True = 관련으로 통과). None이면 예측할 수 없음 (지표에서 뺌)."""
    a = x.article
    if method == "baseline":
        return True
    r = rule_check(a, selected)
    if method == "1a_alias":
        return r.mentioned
    if method == "1b_validation":
        term = target.get("validation_term")
        return None if not term else (r.mentioned and _contains(a, term))
    if method == "1c_rule":
        return r.mentioned and r.score >= accept_score
    if method == "2b_llm_only":
        s = llm_only.get(a.link)
        return None if s is None else s in ("관련", "보류")
    if method == "proposed":
        return None if x.status is None else x.status in ("관련", "보류")
    raise ValueError(method)


async def run(query: str | None = None) -> list[dict]:
    rows, sweep_rows = [], []
    totals = {m: BinaryScores() for m in METHODS}
    calls = {"2b_llm_only": 0, "proposed": 0}
    for target in pick_targets(query):
        topic_id, selected = resolve_target(target)
        items = load_labeled(topic_id)
        usable = [x for x in items if x.label in ("관련", "무관", "동음이의어")]
        if not usable:
            print(f"- {target_label(target)}: 정답 라벨이 없어 건너뜀")
            continue
        llm_only = await llm_only_status(topic_id, selected, usable)
        n_llm_proposed = sum(1 for x in usable if x.stage == "llm")
        calls["proposed"] += math.ceil(n_llm_proposed / LLM_BATCH_SIZE)
        calls["2b_llm_only"] += math.ceil(len(usable) / LLM_BATCH_SIZE)
        homonyms = sum(1 for x in usable if x.label == "동음이의어")

        for method in METHODS:
            scores, skipped, homonym_in = BinaryScores(), 0, 0
            for x in usable:
                p = predict(method, x, target, selected, llm_only)
                if p is None:
                    skipped += 1
                    continue
                scores.add(p, x.label == "관련")
                homonym_in += int(p and x.label == "동음이의어")
            totals[method].merge(scores)
            row = {"target": target_label(target), "method": method, **scores.as_row(), "skipped": skipped,
                   "label_hold_excluded": len(items) - len(usable)}
            if homonyms:
                row["homonym_inflow"] = round(homonym_in / (scores.tp + scores.fp), 3) if scores.tp + scores.fp else 0.0
            rows.append(row)

        for s in RULE_SWEEP:
            sc = BinaryScores()
            for x in usable:
                sc.add(predict("1c_rule", x, target, selected, llm_only, accept_score=s), x.label == "관련")
            sweep_rows.append({"target": target_label(target), "accept_score": s, **sc.as_row()})

    for method in METHODS:
        rows.append({"target": "전체(micro)", "method": method, **totals[method].as_row(),
                     "llm_calls": calls.get(method, 0)})
    return rows + [{"section": "1c 기준값별", **r} for r in sweep_rows]   # target 칸은 그대로 두고 구분 칸을 따로 둔다


def main(query: str | None = None) -> None:
    rows = asyncio.run(run(query))
    overall = [r for r in rows if r.get("target") == "전체(micro)"]
    if not overall or overall[0]["n"] == 0:
        print("정답 라벨이 있는 기사가 없습니다. 먼저 python run.py label 로 라벨을 붙이세요.")
        return
    print("\n[전체(micro)]")
    print_table(overall, ["method", "n", "precision", "recall", "f1", "accuracy", "irrelevant_inflow", "llm_calls"])
    json_path, csv_path = save_results("exp_a_filtering", rows, {"rule_accept_score": RULE_ACCEPT_SCORE,
                                                                 "llm_batch_size": LLM_BATCH_SIZE})
    print(f"\n대상별·기준값별 결과 저장: {csv_path}")
