"""
eval_filter.py — Relevance Filter 효과 측정 스크립트

목적:
  2·3단계 관련성 필터(1차 규칙 필터, 2차 LLM 검증)를 각각, 그리고 함께 적용했을 때
  얼마나 효과적인지 Precision / Recall / F1 / Accuracy로 수치화한다.

비교 대상 (지금 구조에 맞춰 바뀜):
  - V0: 필터 없음 (모든 기사를 관련으로 예측)
  - V1: 1차 규칙 필터만 — 검색어·별칭이 기사에 아예 없거나, 관련어(별칭·연관어) 점수 − 배제어
        (다른 의미의 연관어) 점수가 기준(RULE_REJECT_SCORE, -2) 이하이면 무관, 나머지는 통과
        (예전 V1 "LLM이 만든 제외어 필터"의 자리. 제외어는 이제 0단계 의미 선택에서 자동으로 정해진다)
  - V2: V1 + 2차 LLM 검증 — 규칙을 통과한 기사를 LLM이 고른 의미 기준으로 관련/무관/보류 판정
        (실제 서비스 설정. '보류'는 답변에 쓰므로 통과로 센다)
        (예전 V2 "직접 언급 가드"는 없어졌다. 규칙 점수가 검색어·별칭 언급을 이미 반영한다)

  V1·V2 모두 qupict/filtering.py의 rule_verdict()·llm_verify()를 그대로 호출한다.
  즉 이 스크립트가 별도의 필터 로직을 구현하지 않고, 실제 서비스가 쓰는 코드와 동일한 함수로 재현한다.

출력:
  - 대상(검색어 × 의미)별로 V0/V1/V2 지표와 V0→V1, V1→V2, V0→V2 개선폭을 모두 출력한다.
  - 모든 대상을 마친 후, 전체를 합산한 micro-average 통합 결과도 마지막에 한 번 더 출력한다.
  - 정답이 '보류'인 기사는 맞고 틀림을 정할 수 없어서 지표에서 뺀다.

사용법 (프로젝트 맨 위 폴더에서):
  1) python evalset/build_labeled_set.py 로 evalset/labeled_set.json 생성 (사람이 라벨링)
  2) python old/eval_filter.py 실행
  비용: V2에서 규칙을 통과한 기사 8건당 LLM 1번 (약 0.3 크레딧)

(같은 JSON으로 방법 6가지를 비교하고 결과 파일까지 저장하려면 python evaluate.py exp-a = evalset/exp_a_filtering.py)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from _common import describe, pick_sense

from qupict.filtering import llm_verify, rule_verdict
from qupict.schemas import Article, SelectedSense, Sense

LABELED_SET_PATH = Path(__file__).resolve().parents[1] / "evalset" / "labeled_set.json"
PASS_STATUSES = ("관련", "보류")   # 서비스가 답변에 쓰는 판정 = "관련 있다고 예측"


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


def calculate_confusion_matrix(
    labeled: list[LabeledArticle],
    passed_links: set[str],
) -> PredictionResult:
    tp = fp = fn = tn = 0

    for la in labeled:
        predicted = la.article.link in passed_links
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
    selected: SelectedSense,
) -> tuple[PredictionResult, PredictionResult, int]:
    """V1(1차 규칙 필터만)과 V2(규칙 + LLM 검증)를 계산한다. (V1, V2, LLM이 판정한 기사 수)

    V1: rule_verdict()가 '무관'으로 거른 기사만 빼고 나머지는 모두 통과.
    V2: V1을 통과한 기사(규칙이 판단을 미룬 기사)를 llm_verify()로 판정해 관련·보류만 통과.
        LLM이 답하지 못한 기사는 서비스처럼 통과시키지 않는다.
    """
    articles = [la.article for la in labeled]
    rule = {a.link: rule_verdict(a, selected) for a in articles}

    v1_links = {a.link for a in articles if rule[a.link] is None or rule[a.link].status in PASS_STATUSES}
    to_llm = [a for a in articles if rule[a.link] is None]
    llm_verdicts, _ = await llm_verify(to_llm, selected) if to_llm else ([], [])
    v2_links = ({v.link for v in llm_verdicts if v.status in PASS_STATUSES}
                | {link for link, v in rule.items() if v is not None and v.status in PASS_STATUSES})

    return calculate_confusion_matrix(labeled, v1_links), calculate_confusion_matrix(labeled, v2_links), len(to_llm)


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


def print_table(metrics: list[tuple[str, dict]]) -> None:
    print(f"\n{'':10} {'TP':>4}{'FP':>5}{'FN':>5}{'TN':>5} | {'Prec':>7}{'Rec':>7}{'F1':>7}{'Acc':>7}")
    for label, m in metrics:
        print(f"{label:10} {m['TP']:>4}{m['FP']:>5}{m['FN']:>5}{m['TN']:>5} | "
              f"{m['precision']:>7}{m['recall']:>7}{m['f1']:>7}{m['accuracy']:>7}")
    print()


async def run_ablation(
    labeled: list[LabeledArticle],
    selected: SelectedSense,
) -> tuple[PredictionResult, PredictionResult, PredictionResult]:
    print("=" * 70)
    print(f"대상: {describe(selected)} ({len(labeled)}건)")

    # V0: 모든 기사를 관련 있다고 예측했다고 가정
    tp0 = sum(1 for la in labeled if la.is_actually_relevant)
    fp0 = sum(1 for la in labeled if not la.is_actually_relevant)
    result_v0 = PredictionResult(tp0, fp0, 0, 0)

    result_v1, result_v2, llm_count = await run_filters(labeled, selected)
    print(f"LLM 검증 대상(규칙이 판단을 미룬 기사): {llm_count}건")

    metrics_v0, metrics_v1, metrics_v2 = (result_to_metrics(r) for r in (result_v0, result_v1, result_v2))
    print_table([("V0", metrics_v0), ("V1", metrics_v1), ("V2", metrics_v2)])

    print_metric_change("V0 → V1 (규칙 필터 효과)", metrics_v0, metrics_v1)
    print_metric_change("V1 → V2 (LLM 검증 효과)", metrics_v1, metrics_v2)
    print_metric_change("V0 → V2 (전체 개선폭)", metrics_v0, metrics_v2)
    print()

    return result_v0, result_v1, result_v2


def load_labeled_set(path: Path | None = None) -> list[dict]:
    path = path or LABELED_SET_PATH      # 기본 경로는 부를 때 정한다 (정의할 때 정하면 경로를 바꿔도 반영 안 됨)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        print(f"[오류] {path} JSON 파싱 실패: {error}")
        return []


def entry_relevance(entry: dict) -> bool | None:
    """정답: 관련 True / 무관 False / 보류·알 수 없음 None.
    evalset 형식(label: true/false)과 예전 형식(label: 관련/무관/보류, is_actually_relevant) 모두 읽는다."""
    label = entry.get("label")
    if isinstance(label, bool):
        return label
    if label in ("관련", "무관"):
        return label == "관련"
    if label == "보류":
        return None
    value = entry.get("is_actually_relevant")
    return None if value is None else bool(value)


def parse_evalset_entry(entry: dict) -> tuple[str, str, Article, SelectedSense]:
    """evalset/build_labeled_set.py 형식 → (검색어, 의미 이름, 기사, 고른 의미)"""
    query = str(entry["query"]).strip()
    s = entry["sense"]
    article = Article(title=entry["title"], link=entry["url"],
                      description=entry.get("description") or "", content=entry.get("content") or "")
    selected = SelectedSense(
        query=query,
        sense=Sense(name=s["name"], definition=s.get("definition", ""),
                    aliases=s.get("aliases", []), related_terms=s.get("related_terms", [])),
        excluded_terms=s.get("excluded_terms", []),
        other_senses=[Sense(name=o["name"], definition=o.get("definition", ""),
                            related_terms=o.get("related_terms", []))
                      for o in s.get("other_senses", [])],
    )
    return query, selected.sense.name, article, selected


async def main() -> None:
    raw_entries = load_labeled_set()
    if not raw_entries:
        print(
            f"{LABELED_SET_PATH}이 없거나 비어 있습니다.\n"
            "먼저 python evalset/build_labeled_set.py 로 라벨링 데이터를 만들어주세요."
        )
        return

    groups: dict[tuple[str, str], list[LabeledArticle]] = {}
    stored_senses: dict[tuple[str, str], SelectedSense] = {}   # 라벨링 때 저장한 기준 (evalset/build_labeled_set.py)
    skipped = held = 0

    for i, entry in enumerate(raw_entries):
        if "url" in entry and isinstance(entry.get("sense"), dict):    # evalset 형식: 기준이 항목 안에 있다
            relevant = entry_relevance(entry)
            try:
                query, sense_name, article, selected = parse_evalset_entry(entry)
            except Exception as error:
                print(f"[경고] {i}번째 항목 변환 실패 → 건너뜁니다: {error}")
                skipped += 1
                continue
            if relevant is None:
                held += 1
                continue
            groups.setdefault((query, sense_name), []).append(LabeledArticle(article, relevant))
            stored_senses.setdefault((query, sense_name), selected)
            continue

        query = str(entry.get("query") or entry.get("topic") or "").strip()
        sense_name = str(entry.get("sense") or "").strip()
        if "article" not in entry or not query or not sense_name:
            title = entry.get("article", {}).get("title", "(제목 확인 불가)")
            print(f"[경고] {i}번째 항목에 검색어·의미(query, sense)가 없음 → 건너뜁니다. (제목: {title})"
                  "  ※ 예전 형식(검증 키워드)은 의미가 없어서 지금 필터로 평가할 수 없습니다.")
            skipped += 1
            continue

        relevant = entry_relevance(entry)
        if relevant is None:
            held += 1
            continue

        try:
            article = Article(**entry["article"])
        except Exception as error:
            print(f"[경고] {i}번째 기사 변환 실패 → 건너뜁니다: {error}")
            skipped += 1
            continue

        groups.setdefault((query, sense_name), []).append(LabeledArticle(article, relevant))

    if skipped:
        print(f"\n총 {skipped}건의 항목을 형식 오류로 건너뛰었습니다.")
    if held:
        print(f"정답이 '보류'인 {held}건은 지표에서 뺐습니다.\n")

    total_v0 = PredictionResult(0, 0, 0, 0)
    total_v1 = PredictionResult(0, 0, 0, 0)
    total_v2 = PredictionResult(0, 0, 0, 0)

    def _add(acc: PredictionResult, r: PredictionResult) -> PredictionResult:
        return PredictionResult(acc.tp + r.tp, acc.fp + r.fp, acc.fn + r.fn, acc.tn + r.tn)

    for (query, sense_name), labeled_articles in groups.items():
        selected = stored_senses.get((query, sense_name))
        if selected is None:                          # 예전 파일: 저장된 기준이 없으면 다시 정한다
            selected = pick_sense(query, sense_name)
        result_v0, result_v1, result_v2 = await run_ablation(labeled_articles, selected)
        total_v0 = _add(total_v0, result_v0)
        total_v1 = _add(total_v1, result_v1)
        total_v2 = _add(total_v2, result_v2)

    if not groups:
        print("평가할 기사가 없습니다.")
        return

    # 전체 대상을 합친 통합(micro-average) 요약 — 개별 결과만으로는
    # 전체적으로 얼마나 개선됐는지 한눈에 파악하기 어려우므로 마지막에 정리해서 보여준다.
    print("=" * 70)
    print(f"전체 통합 결과 ({len(groups)}개 대상 합산)")
    metrics_v0, metrics_v1, metrics_v2 = (result_to_metrics(r) for r in (total_v0, total_v1, total_v2))
    print_table([("V0", metrics_v0), ("V1", metrics_v1), ("V2", metrics_v2)])

    print_metric_change("V0 → V1 (규칙 필터 효과, 전체)", metrics_v0, metrics_v1)
    print_metric_change("V1 → V2 (LLM 검증 효과, 전체)", metrics_v1, metrics_v2)
    print_metric_change("V0 → V2 (전체 개선폭, 전체)", metrics_v0, metrics_v2)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
