"""
eval_filter.py — Relevance Filter 효과 측정 스크립트

목적:
  "제외 키워드(exclusion_terms, LLM 자동 생성)"와 "직접 언급 보호 규칙"을
  각각 그리고 함께 적용했을 때 관련성 필터가 얼마나 효과적인지 Precision /
  Recall / F1 / Accuracy로 수치화한다.

비교 대상:
  - V0: 필터 없음
  - V1: 순수 제외어(동음이의어) 필터만 적용 (직접 언급 가드는 끔)
  - V2: V1 + 직접 언급 가드(topic 문자열이 기사에 통째로 등장하면 무조건 통과)

  V1과 V2는 모두 relevance_filter.py의 filter_by_relevance()를 그대로 호출
  하고, apply_direct_mention_guard 파라미터만 다르게 준다 (V1=False, V2=True).
  즉 이 스크립트가 별도의 가드 로직을 구현하지 않고, 실제 서비스가 쓰는
  코드와 동일한 함수로 두 조건을 재현한다.

제외어 최소 1개 보장(best-effort):
  relevance_filter.py의 generate_exclusion_terms()는 1차 시도에서 제외어가
  하나도 안 나오면, 더 관대한 프롬프트로 한 번 더 시도해 최소 1개는 뽑아내려고
  한다. 제외어가 하나도 없으면 필터가 사실상 아무 것도 걸러내지 못해 V1의
  의미가 없어지기 때문이다. (그래도 정말 마땅한 후보가 없으면 빈 리스트로
  남을 수 있다 — 없는 동음이의어를 억지로 지어내지는 않는다.)

sanitize 역할:
  LLM이 생성한 제외어 중 "topic과 완전히 동일하거나, topic의 일부만 지나치게
  짧게 잘라낸 표현"만 사후에 한 번 더 제거한다 (예: topic='서울여대'인데
  '서울'만 나온 경우). topic+validation_terms 조합(예: '방송인 강남')을
  걸러내는 것은 generate_exclusion_terms() 내부에서 이미 처리하므로 여기서는
  다루지 않는다.

출력:
  - topic별로 V0/V1/V2 지표와 V0→V1, V1→V2, V0→V2 개선폭을 모두 출력한다.
  - 모든 topic을 마친 후, 전체를 합산한 micro-average 통합 결과도 마지막에
    한 번 더 출력해 전체적인 효과를 한눈에 파악할 수 있게 한다.
  - 오분류 기사 목록(False Positive/Negative 상세)은 출력하지 않는다.
    Precision/Recall/F1/Accuracy 수치만으로 충분히 판단 가능하다.

사용법:
  1) build_labeled_set.py로 labeled_set.json 생성
  2) 사람이 is_actually_relevant 값을 라벨링
  3) python eval_filter.py 실행
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.deduplication import embed_articles
from app.relevance_filter import filter_by_relevance, generate_exclusion_terms
from app.schemas import Article


@dataclass
class LabeledArticle:
    article: Article
    is_actually_relevant: bool


@dataclass
class PredictionResult:
    tp: int
    fp: int
    fn: int
    tn: int


def normalize_text(text: str | None) -> str:
    """비교용 정규화: HTML 태그 제거, 소문자화, 한글/영문/숫자만 남김."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def unique_terms(terms: Iterable[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term:
            continue
        stripped = term.strip()
        normalized = normalize_text(stripped)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(stripped)
    return result


async def call_generate_exclusion_terms(topic: str, validation_terms: list[str]) -> list[str]:
    """generate_exclusion_terms(topic, validation_terms)를 호출하고, 실패 시
    빈 리스트로 안전하게 폴백한다.

    validation_terms를 반드시 함께 전달해야 한다. relevance_filter.py의
    generate_exclusion_terms 내부에서 "topic+validation_terms 조합"과 일치하는
    항목(예: topic='강남', validation_terms=['방송인']일 때 '방송인 강남')을
    걸러내는데, validation_terms를 안 넘기면 이 방어 로직이 무력화된다.
    """
    try:
        result = await generate_exclusion_terms(topic, validation_terms)
    except Exception:
        result = []
    return unique_terms(result)


def sanitize_exclusion_terms(
    exclusion_terms: list[str],
    intended_terms: list[str],
) -> list[str]:
    """생성된 제외어 중 명백히 위험한 표현만 제거한다 (완화된 규칙).

    제거 대상:
      1. topic/의도 표현과 완전히 동일한 경우 (예: exclusion == topic)
      2. exclusion이 intended의 부분 문자열이면서, 그 자체로 지나치게
         짧게 잘려나온 경우 (예: topic='서울여대'인데 exclusion='서울'만
         남아 원래 표현보다 2글자 넘게 짧은 경우)

    반대로 "exclusion이 intended를 포함하는" 방향(예: '강남' → '강남구',
    '강남역')은 제거하지 않는다. 동음이의어 제외어는 topic 글자를 포함하는
    게 정상이므로, 이 방향까지 제거하면 제외어가 거의 남지 않게 된다.
    """
    normalized_intended = [normalize_text(t) for t in intended_terms if normalize_text(t)]

    safe_terms: list[str] = []
    for exclusion_term in exclusion_terms:
        normalized_exclusion = normalize_text(exclusion_term)
        if not normalized_exclusion:
            continue

        is_unsafe = any(
            normalized_exclusion == intended
            or (
                normalized_exclusion in intended
                and len(intended) - len(normalized_exclusion) <= 2
            )
            for intended in normalized_intended
        )
        if not is_unsafe:
            safe_terms.append(exclusion_term)

    return unique_terms(safe_terms)


def calculate_confusion_matrix(
    labeled: list[LabeledArticle],
    passed_articles: list[Article],
) -> PredictionResult:
    passed_ids = {a.link for a in passed_articles}
    tp = fp = fn = tn = 0

    for la in labeled:
        predicted = la.article.link in passed_ids
        actual = la.is_actually_relevant
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    return PredictionResult(tp, fp, fn, tn)


async def run_filters(
    labeled: list[LabeledArticle],
    topic: str,
    exclusion_terms: list[str],
) -> tuple[PredictionResult, PredictionResult]:
    """V1(순수 제외어 필터, 직접언급 가드 끔)과 V2(+직접언급 가드 켬)를 계산한다.

    두 버전 모두 relevance_filter.py의 filter_by_relevance()를 그대로 호출하되,
    apply_direct_mention_guard 스위치만 다르게 준다. V1은 "topic이 기사에
    통째로 있는지"조차 보지 않고 순수하게 동음이의어 제외어만으로 판정하고,
    V2는 거기에 "topic 문자열이 통째로 등장하면 무조건 통과"라는 가드를 더한다.
    """
    all_articles = [la.article for la in labeled]
    embedded_articles = await embed_articles(all_articles)  # 그룹당 1회

    v1_articles = await filter_by_relevance(
        embedded_articles, topic, exclusion_terms=exclusion_terms,
        apply_direct_mention_guard=False,
    )
    v2_articles = await filter_by_relevance(
        embedded_articles, topic, exclusion_terms=exclusion_terms,
        apply_direct_mention_guard=True,
    )

    result_v1 = calculate_confusion_matrix(labeled, v1_articles)
    result_v2 = calculate_confusion_matrix(labeled, v2_articles)

    return result_v1, result_v2


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, int | float]:
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


def result_to_metrics(result: PredictionResult) -> dict[str, int | float]:
    return compute_metrics(result.tp, result.fp, result.fn, result.tn)


def print_metric_change(
    label: str,
    before: dict[str, int | float],
    after: dict[str, int | float],
) -> None:
    parts = []
    for name in ("precision", "recall", "f1", "accuracy"):
        b, a = float(before[name]), float(after[name])
        parts.append(f"{name[:4].capitalize()} {b:.3f}→{a:.3f}({a - b:+.3f})")
    print(f"  {label}: " + " | ".join(parts))


async def run_ablation(
    labeled: list[LabeledArticle],
    topic: str,
    validation_terms: list[str],
) -> tuple[PredictionResult, PredictionResult, PredictionResult]:
    print("=" * 70)
    print(f"주제: {topic} ({len(labeled)}건) | 검증어: {validation_terms or '없음'}")

    # V0: 모든 기사를 관련 있다고 예측했다고 가정
    tp0 = sum(1 for la in labeled if la.is_actually_relevant)
    fp0 = sum(1 for la in labeled if not la.is_actually_relevant)
    result_v0 = PredictionResult(tp0, fp0, 0, 0)
    metrics_v0 = result_to_metrics(result_v0)

    raw_exclusion_terms = await call_generate_exclusion_terms(topic, validation_terms)
    safe_exclusion_terms = sanitize_exclusion_terms(raw_exclusion_terms, [topic])
    removed_terms = [t for t in raw_exclusion_terms if t not in safe_exclusion_terms]

    print(f"제외어(원본→안전화): {raw_exclusion_terms or '없음'} → {safe_exclusion_terms or '없음'}"
          + (f" (제거됨: {removed_terms})" if removed_terms else ""))

    # V1, V2 — apply_direct_mention_guard 스위치로 가드 유무만 다르게 준다.
    result_v1, result_v2 = await run_filters(labeled, topic, safe_exclusion_terms)
    metrics_v1 = result_to_metrics(result_v1)
    metrics_v2 = result_to_metrics(result_v2)

    # 지표 한 줄 표로 압축
    print(f"\n{'':10} {'TP':>4}{'FP':>5}{'FN':>5}{'TN':>5} | {'Prec':>7}{'Rec':>7}{'F1':>7}{'Acc':>7}")
    for label, m in (("V0", metrics_v0), ("V1", metrics_v1), ("V2", metrics_v2)):
        print(f"{label:10} {m['TP']:>4}{m['FP']:>5}{m['FN']:>5}{m['TN']:>5} | "
              f"{m['precision']:>7}{m['recall']:>7}{m['f1']:>7}{m['accuracy']:>7}")
    print()

    print_metric_change("V0 → V1 (제외어 필터 효과)", metrics_v0, metrics_v1)
    print_metric_change("V1 → V2 (직접언급 보호 효과)", metrics_v1, metrics_v2)
    print_metric_change("V0 → V2 (전체 개선폭)", metrics_v0, metrics_v2)
    print()

    return result_v0, result_v1, result_v2


def load_labeled_set(path: str = "labeled_set.json") -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        print(f"[오류] {path} JSON 파싱 실패: {error}")
        return []


def parse_validation_terms(entry: dict) -> list[str]:
    """labeled_set.json의 validation_terms를 list/str 두 형태 모두 지원해 읽는다."""
    raw = entry.get("validation_terms", entry.get("validation_term", []))
    if isinstance(raw, str):
        return unique_terms(t.strip() for t in raw.split(","))
    if isinstance(raw, list):
        return unique_terms(str(t) for t in raw)
    return []


async def main() -> None:
    raw_entries = load_labeled_set()
    if not raw_entries:
        print(
            "labeled_set.json이 없거나 비어 있습니다.\n"
            "먼저 build_labeled_set.py로 라벨링 데이터를 만들어주세요."
        )
        return

    required_keys = {"article", "topic", "is_actually_relevant"}
    groups: dict[tuple[str, tuple[str, ...]], list[LabeledArticle]] = {}
    skipped = 0

    for i, entry in enumerate(raw_entries):
        missing = required_keys - entry.keys()
        if missing:
            title = entry.get("article", {}).get("title", "(제목 확인 불가)")
            print(f"[경고] {i}번째 항목에 필수 키 누락 {missing} → 건너뜁니다. (제목: {title})")
            skipped += 1
            continue

        try:
            article = Article(**entry["article"])
        except Exception as error:
            print(f"[경고] {i}번째 기사 변환 실패 → 건너뜁니다: {error}")
            skipped += 1
            continue

        topic = str(entry["topic"]).strip()
        validation_terms = parse_validation_terms(entry)
        group_key = (topic, tuple(validation_terms))

        groups.setdefault(group_key, []).append(
            LabeledArticle(article=article, is_actually_relevant=bool(entry["is_actually_relevant"]))
        )

    if skipped:
        print(f"\n총 {skipped}건의 항목을 형식 오류로 건너뛰었습니다.\n")

    total_v0 = PredictionResult(0, 0, 0, 0)
    total_v1 = PredictionResult(0, 0, 0, 0)
    total_v2 = PredictionResult(0, 0, 0, 0)

    def _add(acc: PredictionResult, r: PredictionResult) -> PredictionResult:
        return PredictionResult(acc.tp + r.tp, acc.fp + r.fp, acc.fn + r.fn, acc.tn + r.tn)

    for (topic, validation_terms_tuple), labeled_articles in groups.items():
        result_v0, result_v1, result_v2 = await run_ablation(
            labeled_articles, topic=topic, validation_terms=list(validation_terms_tuple)
        )
        total_v0 = _add(total_v0, result_v0)
        total_v1 = _add(total_v1, result_v1)
        total_v2 = _add(total_v2, result_v2)

    # 전체 topic을 합친 통합(micro-average) 요약 — 개별 topic 결과만으로는
    # 전체적으로 얼마나 개선됐는지 한눈에 파악하기 어려우므로 마지막에 정리해서 보여준다.
    print("=" * 70)
    print(f"전체 통합 결과 ({len(groups)}개 topic 합산)")
    metrics_v0 = result_to_metrics(total_v0)
    metrics_v1 = result_to_metrics(total_v1)
    metrics_v2 = result_to_metrics(total_v2)

    print(f"\n{'':10} {'TP':>4}{'FP':>5}{'FN':>5}{'TN':>5} | {'Prec':>7}{'Rec':>7}{'F1':>7}{'Acc':>7}")
    for label, m in (("V0", metrics_v0), ("V1", metrics_v1), ("V2", metrics_v2)):
        print(f"{label:10} {m['TP']:>4}{m['FP']:>5}{m['FN']:>5}{m['TN']:>5} | "
              f"{m['precision']:>7}{m['recall']:>7}{m['f1']:>7}{m['accuracy']:>7}")
    print()

    print_metric_change("V0 → V1 (제외어 필터 효과, 전체)", metrics_v0, metrics_v1)
    print_metric_change("V1 → V2 (직접언급 보호 효과, 전체)", metrics_v1, metrics_v2)
    print_metric_change("V0 → V2 (전체 개선폭, 전체)", metrics_v0, metrics_v2)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())