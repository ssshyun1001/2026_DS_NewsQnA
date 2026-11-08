"""
유사도 확인 도구 — 기준값 없이 실제 유사도 보기  (old/debug_similarity.py를 지금 구조로 옮김)

왜 필요한가
  질문 답변은 유사도가 기준값(RELEVANCE_SIMILARITY_THRESHOLD, 기본 0.5) 미만인 청크를 버린다.
  그래서 답이 안 나왔을 때 "근거 기사가 아예 없었는지, 있었는데 기준값에 걸렸는지" 알 수 없다.
  이 도구는 기준값을 끄고 청크 유사도를 그대로 보여 준다.

  python run.py sim --target 강남 "강남이 부른 OST 제목이 뭐야?"
      질문 하나: 검색 방식(base · hyde · multi · both)마다 위쪽 청크와 유사도
      (질문 세트에 있는 질문이면 정답 근거 기사에 ★ 표시)

  python run.py sim [--target 강남]
      질문 세트(qa_set.yaml) 전체: 질문마다 "정답 근거 기사 청크의 최고 유사도"
      → 기준값 후보(0.40 / 0.45 / 0.50)마다 정답 근거가 살아남는 질문 수 (회의에서 기준값 정할 때)

비용: 질문 1개당 LLM 2번(HyDE 1 + Multi-Query 1) + 임베딩. 검색 방식 4가지가 같은 결과를 나눠 쓴다.
DB는 읽기만 한다.
"""
from __future__ import annotations

from qupict.config import settings
from qupict.llm import get_embeddings
from qupict.retrieval import (
    _build_search_text,
    _generate_query_variants,
    search_vectors,
    topic_context,
)
from qupict.schemas import SelectedSense

VARIANT_NAMES = {
    "base": "질문 그대로",
    "hyde": "질문 + HyDE 가상 기사 문장",
    "multi": "질문 + Multi-Query 변형 질문",
    "both": "HyDE + Multi-Query (실제 서비스 설정)",
}
THRESHOLD_CANDIDATES = (0.40, 0.45, 0.50)
PER_VECTOR = 30       # 검색 문장 하나당 가져올 청크 수 (정답 근거가 아래쪽에 있어도 찾게 넉넉히)
SHOW_CHUNKS = 10      # 질문 하나 볼 때 방식마다 보여 줄 청크 수


# ---------------------------------------------------------------------------
# 검색 (기준값 없이)
# ---------------------------------------------------------------------------

def variant_texts(question: str, selected: SelectedSense) -> dict[str, list[str]]:
    """검색 방식별 검색 문장. HyDE·변형 질문은 한 번만 만들어 네 방식이 나눠 쓴다.
    (retrieval.build_search_texts와 같은 문장을 만든다)"""
    context = topic_context(selected)
    hyde = _build_search_text(question, context, use_hyde=True)   # 실패하면 질문 그대로
    extra = _generate_query_variants(question, context)            # 실패하면 []
    return {"base": [question], "hyde": [hyde], "multi": [question, *extra], "both": [hyde, *extra]}


def variant_hits(topic_id: int, question: str, selected: SelectedSense,
                 per_vector: int = PER_VECTOR) -> tuple[dict[str, list], dict[str, list[str]]]:
    """검색 방식별 청크 목록 (기준값 없이, 유사도 높은 순)과 그때 쓴 검색 문장."""
    texts = variant_texts(question, selected)
    unique = list(dict.fromkeys(t for ts in texts.values() for t in ts))   # 같은 문장은 한 번만 임베딩
    vectors = dict(zip(unique, get_embeddings().embed_documents(unique)))
    hits = {name: search_vectors(topic_id, [vectors[t] for t in ts], per_vector)
            for name, ts in texts.items()}
    return hits, texts


def evidence_best(hits: list, evidence: list[str]) -> tuple[float | None, int | None]:
    """정답 근거 기사 청크 중 가장 높은 유사도와, 그 기사의 순위(기사가 처음 나온 순서 기준)."""
    ranking: list[str] = []
    for h in hits:
        if h.article.link not in ranking:
            ranking.append(h.article.link)
        if h.article.link in evidence:
            return h.similarity, len(ranking)
    return None, None


# ---------------------------------------------------------------------------
# 질문 하나
# ---------------------------------------------------------------------------

def show_question(topic_id: int, selected: SelectedSense, question: str,
                  evidence: list[str] | None = None, top: int = SHOW_CHUNKS) -> None:
    threshold = settings.relevance_similarity_threshold
    evidence = evidence or []
    hits, texts = variant_hits(topic_id, question, selected)

    print(f"\n[{selected.query} → {selected.sense.name}] 질문: {question}")
    print(f"지금 기준값: {threshold:.2f} (O = 기준값 이상이라 답변 근거로 쓰임, - = 버려짐)")
    if evidence:
        print("★ = 질문 세트의 정답 근거 기사")
    if texts["hyde"][0] != question:
        print(f"\n[HyDE 검색 문장]\n{texts['hyde'][0]}")
    if len(texts["multi"]) > 1:
        print("[Multi-Query 변형 질문]\n" + "\n".join(f"  - {t}" for t in texts["multi"][1:]))

    for name, label in VARIANT_NAMES.items():
        print(f"\n=== {name}: {label} ===")
        if not hits[name]:
            print("  (청크 없음 — 이 topic을 index-all 했는지 확인하세요)")
            continue
        for h in hits[name][:top]:
            mark = "O" if h.similarity >= threshold else "-"
            star = " ★" if h.article.link in evidence else ""
            image = " [이미지]" if h.source == "image" else ""
            print(f"  {h.similarity:.3f}  {mark}  {h.article.title[:45]} ({h.article.press or '?'}){image}{star}")
            print(f"                 {h.content[:70].replace(chr(10), ' ')}…")
        if evidence:
            sim, rank = evidence_best(hits[name], evidence)
            print(f"  → 정답 근거: " + (f"유사도 {sim:.3f}, 기사 순위 {rank}위" if sim is not None
                                     else f"위 {PER_VECTOR}개 청크 안에 없음"))


def main_question(query: str, sense_name: str, question: str) -> None:
    from experiments.qa_set import load_qa
    from qupict import db

    record = db.get_topic(query, sense_name)
    if record is None:
        print(f"DB에 topic '{query} → {sense_name}'이 없습니다. 먼저 collect-all → filter-all → index-all을 실행하세요.")
        return
    same = [x for x in load_qa(query=query) if x.sense == sense_name and x.question.strip() == question.strip()]
    show_question(record.id, record.selected, question, same[0].evidence if same else None)


# ---------------------------------------------------------------------------
# 질문 세트 전체 → 기준값 후보 비교
# ---------------------------------------------------------------------------

def qa_report(query: str | None = None) -> tuple[list[dict], list[dict]]:
    """(요약 행, 질문별 행). 질문별: 방식마다 정답 근거 최고 유사도·순위."""
    from experiments.qa_set import load_qa
    from qupict import db

    per_question: list[dict] = []
    topics: dict[tuple[str, str], object] = {}
    for item in load_qa(query=query):
        key = (item.query, item.sense)
        if key not in topics:
            topics[key] = db.get_topic(item.query, item.sense)
            if topics[key] is None:
                print(f"- DB에 topic '{item.query} → {item.sense}'이 없어 이 topic 질문은 건너뜀")
        record = topics[key]
        if record is None:
            continue
        hits, _ = variant_hits(record.id, item.question, record.selected)
        for name in VARIANT_NAMES:
            sim, rank = evidence_best(hits[name], item.evidence)
            top = hits[name][0].similarity if hits[name] else None
            per_question.append({"id": item.id, "question": item.question, "variant": name,
                                 "evidence_similarity": None if sim is None else round(sim, 3),
                                 "evidence_rank": rank,
                                 "top_similarity": None if top is None else round(top, 3)})

    summary: list[dict] = []
    for name in VARIANT_NAMES:
        rows = [r for r in per_question if r["variant"] == name]
        if not rows:
            continue
        row = {"variant": name, "questions": len(rows)}
        for t in THRESHOLD_CANDIDATES:      # 정답 근거가 기준값을 넘는 질문 수
            row[f"pass@{t:.2f}"] = sum(1 for r in rows
                                       if r["evidence_similarity"] is not None and r["evidence_similarity"] >= t)
        summary.append(row)
    return summary, per_question


def main_qa(query: str | None = None) -> None:
    from experiments.common import print_table, save_results
    from experiments.qa_set import load_qa

    if not load_qa(query=query):
        print("질문 세트가 없습니다. 먼저 python run.py make-qa 로 질문을 만드세요.")
        return
    summary, per_question = qa_report(query)
    if not per_question:
        return

    print("\n질문별 정답 근거 기사의 최고 유사도 (없음 = 위쪽 청크 안에 정답 근거가 없음)")
    table = []
    for qid in dict.fromkeys(r["id"] for r in per_question):
        rows = {r["variant"]: r for r in per_question if r["id"] == qid}
        table.append({"id": qid, **{name: _fmt(rows[name]["evidence_similarity"]) for name in rows},
                      "question": next(iter(rows.values()))["question"][:30]})
    print_table(table, ["id", *VARIANT_NAMES, "question"])

    print(f"\n기준값 후보별: 정답 근거가 살아남는 질문 수 (지금 설정 {settings.relevance_similarity_threshold:.2f})")
    print_table(summary, ["variant", "questions", *[f"pass@{t:.2f}" for t in THRESHOLD_CANDIDATES]])
    _, csv_path = save_results("similarity", summary + per_question,
                               {"thresholds": THRESHOLD_CANDIDATES, "per_vector": PER_VECTOR,
                                "current_threshold": settings.relevance_similarity_threshold})
    print(f"\n질문별 결과 저장: {csv_path}")


def _fmt(value: float | None) -> str:
    return "없음" if value is None else f"{value:.3f}"
