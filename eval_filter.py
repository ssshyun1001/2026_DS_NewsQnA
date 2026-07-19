"""
eval_filter.py — Relevance Filter 효과 측정 스크립트

목적:
  "검증어(validation_terms)를 넣었을 때 관련성 필터가 실제로 얼마나 효과적인가"를
  가장 널리 쓰이는 이진 분류 지표(Precision / Recall / F1 / Accuracy)로 수치화한다.

전제:
  - 사람이 직접 라벨링한 골드셋(gold set)이 있어야 한다.
  - 골드셋 형식: 기사 제목/본문 + "이 topic+검증어 기준으로 진짜 관련 있는가(True/False)"

사용법:
  1) 아래 GOLD_SET 형태로 라벨링 데이터를 준비 (CSV/JSON으로 분리해도 됨)
  2) filter_by_relevance()를 그대로 불러와서 통과/탈락 여부를 예측값으로 사용
  3) 사람 라벨(정답) vs 필터 예측을 비교해 지표 산출

  python eval_filter.py
"""
from __future__ import annotations

from dataclasses import dataclass

from app.deduplication import embed_articles
from app.relevance_filter import filter_by_relevance
from app.schemas import Article


@dataclass
class LabeledArticle:
    article: Article
    is_actually_relevant: bool  # 사람이 직접 판단한 정답(gold label)


async def confusion_matrix(labeled: list[LabeledArticle], topic: str, validation_terms: list[str]):
    """필터를 통과한 기사 집합과 사람이 매긴 정답을 비교해 TP/FP/FN/TN을 센다."""
    all_articles = [la.article for la in labeled]
    # 하이브리드 필터는 embedding이 이미 채워져 있어야 동작한다.
    all_articles = await embed_articles(all_articles)
    passed = await filter_by_relevance(all_articles, topic, validation_terms)
    passed_ids = {a.link for a in passed}  # Article에 고유 id/link가 있다고 가정

    tp = fp = fn = tn = 0
    for la in labeled:
        predicted_relevant = la.article.link in passed_ids
        actual_relevant = la.is_actually_relevant

        if predicted_relevant and actual_relevant:
            tp += 1
        elif predicted_relevant and not actual_relevant:
            fp += 1  # 필터가 통과시켰지만 실제로는 무관 (동음이의어 오염 등)
        elif not predicted_relevant and actual_relevant:
            fn += 1  # 필터가 걸러냈지만 사실은 관련 있었던 기사 (과잉 필터링)
        else:
            tn += 1

    return tp, fp, fn, tn


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
    }


async def run_ablation(labeled: list[LabeledArticle], topic: str, validation_terms: list[str]) -> None:
    """검증어 필터 적용 전(V0)과 후(V1)를 비교 출력한다."""
    print(f"라벨링된 기사 수: {len(labeled)}건")
    print(f"주제: {topic} / 검증어: {validation_terms}\n")

    # V0: 필터 없음 -> 모든 기사를 '관련 있음'으로 예측했다고 가정
    tp0 = sum(1 for la in labeled if la.is_actually_relevant)
    fp0 = sum(1 for la in labeled if not la.is_actually_relevant)
    fn0 = 0
    tn0 = 0
    metrics_v0 = compute_metrics(tp0, fp0, fn0, tn0)

    # V1: 검증어 필터 적용 (임베딩 1차 + LLM 경계 재판정 하이브리드)
    tp1, fp1, fn1, tn1 = await confusion_matrix(labeled, topic, validation_terms)
    metrics_v1 = compute_metrics(tp1, fp1, fn1, tn1)

    print("=== V0: 필터 없음 (크롤링 결과 그대로 사용) ===")
    for k, v in metrics_v0.items():
        print(f"  {k}: {v}")

    print("\n=== V1: 검증어 기반 relevance filter 적용 ===")
    for k, v in metrics_v1.items():
        print(f"  {k}: {v}")

    print("\n=== 개선폭 ===")
    print(f"  Precision: {metrics_v0['precision']} → {metrics_v1['precision']} "
          f"({metrics_v1['precision'] - metrics_v0['precision']:+.3f})")
    print(f"  F1: {metrics_v0['f1']} → {metrics_v1['f1']} "
          f"({metrics_v1['f1'] - metrics_v0['f1']:+.3f})")
    print(f"  Accuracy: {metrics_v0['accuracy']} → {metrics_v1['accuracy']} "
          f"({metrics_v1['accuracy'] - metrics_v0['accuracy']:+.3f})")


def _load_labeled_set(path: str = "labeled_set.json") -> list[dict]:
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


async def _main() -> None:
    raw_entries = _load_labeled_set()
    if not raw_entries:
        print(
            "labeled_set.json이 없거나 비어 있습니다. "
            "먼저 `python build_labeled_set.py \"topic\" \"검증어1,검증어2\" period_days`로 "
            "라벨링 데이터를 만들어주세요."
        )
        return

    # topic + validation_terms 조합별로 묶어서 각각 ablation을 돌린다.
    groups: dict[tuple[str, tuple[str, ...]], list[LabeledArticle]] = {}
    for entry in raw_entries:
        key = (entry["topic"], tuple(entry["validation_terms"]))
        article = Article(**entry["article"])
        groups.setdefault(key, []).append(
            LabeledArticle(article=article, is_actually_relevant=entry["is_actually_relevant"])
        )

    for (topic, validation_terms), labeled_articles in groups.items():
        await run_ablation(labeled_articles, topic=topic, validation_terms=list(validation_terms))
        print()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())