"""
실험 C — 답변 신뢰성 비교: LLM 단독 vs 기본 RAG vs QuPicT (기획안 7단계)

질문 세트(experiments/qa_set.yaml)의 질문마다 세 방식으로 답을 만들고, LLM 채점자가 정답 근거 기사와 대조해 채점한다.

  방식         답변에 쓰는 기사
  ──────────  ─────────────────────────────────────────────────────────────
  llm_only     없음 (질문만)
  basic_rag    필터링 없이 수집한 기사 전부 (중복·무관 포함) → FAISS 기사 검색 상위 5건
  qupict       의미 선택 + 관련성 검증을 통과한 기사(관련·보류) → pgvector 청크 검색 (실제 서비스)
  세 방식 모두 같은 LLM·같은 답변 프롬프트를 쓴다.

지표
  정답률        질문에 맞는 답을 한 비율 (정답과 같은 사실을 말했는가)
  근거 일치율    답변 속 주장 중 정답 근거 기사로 뒷받침되는 주장의 비율 (질문별 평균)
  환각 비율      근거 기사에 없거나 다른 내용을 사실처럼 말한 답변의 비율
  출처 정확도    답변이 인용한 기사 중 정답 근거 기사의 비율 (RAG 방식만)

채점은 LLM이 하고(근거를 먼저 쓰고 판정하게 함), 결과 파일에 주장별 판정을 남긴다 → 일부는 사람이 검수한다.
출처 정확도는 "정답 근거로 지정한 기사"만 맞다고 보므로, 같은 내용을 담은 다른 기사를 인용하면 낮게 나올 수 있다.

비용: 질문 1개당 LLM 약 10번 (답변 3 + 검색 문장 4 + 채점 3). 질문 50개면 수백 크레딧 → --limit로 나눠 돌릴 수 있다.

실행: python evaluate.py exp-c [--target 강남] [--limit 10]
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from experiments.common import print_table, save_results
from experiments.qa_set import QAItem, load_qa
from qupict import db
from qupict.llm import get_chat_model
from qupict.schemas import Answer

METHODS = ["llm_only", "basic_rag", "qupict"]
EVIDENCE_CHARS = 3000      # 채점자에게 보여 줄 근거 기사 길이 (기사당)
GRADE_ATTEMPTS = 2         # 채점 실패 시 다시 시도하는 횟수 포함 총 시도 수


# ---------------------------------------------------------------------------
# 세 가지 답변
# ---------------------------------------------------------------------------

def answer_llm_only(item: QAItem) -> Answer:
    from qupict.qa import generate_answer
    return generate_answer(item.question, [])


def answer_basic_rag(item: QAItem, store, articles) -> Answer:
    from qupict.qa import generate_answer
    from qupict.retrieval import retrieve_relevant_articles
    found = retrieve_relevant_articles(store, articles, item.question, topic=item.query, top_k=5) if store else []
    return generate_answer(item.question, found)


def answer_qupict(item: QAItem, topic_id: int, selected) -> Answer:
    from qupict.qa import generate_answer_from_chunks
    from qupict.retrieval import search_topic
    return generate_answer_from_chunks(item.question, search_topic(topic_id, item.question, selected))


def cited_links(answer: Answer) -> list[str]:
    from qupict.qa import cited_numbers
    numbers = answer.cited or cited_numbers(answer.answer, len(answer.sources))
    return [answer.sources[n - 1].link for n in numbers]


# ---------------------------------------------------------------------------
# 채점 (LLM)
# ---------------------------------------------------------------------------

class _Claim(BaseModel):
    claim: str = Field(description="답변에서 사실로 말한 주장 하나")
    evidence: str = Field(description="근거 기사에서 이 주장을 뒷받침하거나 반박하는 부분 (없으면 '없음')")
    supported: bool = Field(description="근거 기사로 뒷받침되면 true")


class _Grade(BaseModel):
    # 근거를 먼저 쓰고 판정하게 필드 순서를 정했다 (filtering.py와 같은 이유)
    claims: list[_Claim] = Field(description="답변의 사실 주장들 ('확인할 수 없다'는 말은 주장이 아니므로 뺀다)")
    reasoning: str = Field(description="정답과 비교한 판단 근거 한 줄")
    correct: bool = Field(description="질문에 대해 정답과 같은 사실을 답했으면 true")
    hallucination: bool = Field(description="근거 기사에 없거나 다른 내용을 사실처럼 말했으면 true")


_GRADE_PROMPT = """\
너는 뉴스 Q&A 답변을 채점한다. 정답 근거 기사만을 기준으로 판단하라 (너의 배경지식으로 판단하지 마라).

- claims: 답변에서 사실로 말한 주장을 나눠 적고, 근거 기사에서 뒷받침되는지 판단한다.
  "기사로는 확인할 수 없다", "모른다" 같은 말은 주장이 아니므로 넣지 않는다.
- correct: 정답과 같은 사실을 답했으면 true. 표현이 달라도 사실이 같으면 true.
- hallucination: 근거 기사에 없거나 근거 기사와 다른 내용을 사실처럼 말했으면 true.
"""


def grade(item: QAItem, answer_text: str, evidence_texts: list[str]) -> _Grade | None:
    evidence = "\n\n".join(f"[근거 기사 {i}]\n{t[:EVIDENCE_CHARS]}" for i, t in enumerate(evidence_texts, 1))
    llm = get_chat_model(temperature=0).with_structured_output(_Grade)
    messages = [("system", _GRADE_PROMPT),
                ("human", f"질문: {item.question}\n정답: {item.answer}\n\n{evidence}\n\n[채점할 답변]\n{answer_text}")]
    for attempt in range(GRADE_ATTEMPTS):      # 가끔 한 번씩 실패한다 (실제: 같은 채점이 재시도에서 성공)
        try:
            return llm.invoke(messages)
        except Exception as e:
            if attempt == GRADE_ATTEMPTS - 1:
                print(f"  [경고] 채점 실패 ({type(e).__name__}): {item.id}")
    return None


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def _evidence_texts(links: list[str]) -> list[str]:
    with db.connect() as conn:
        rows = conn.execute("SELECT link, title, content FROM articles WHERE link = ANY(%s)", (links,)).fetchall()
    return [f"제목: {r['title']}\n{r['content']}" for r in rows]


def run(query: str | None = None, limit: int | None = None) -> tuple[list[dict], list[dict]]:
    from qupict.retrieval import build_in_memory_store

    items = load_qa(query=query)[:limit] if limit else load_qa(query=query)
    topics: dict[tuple[str, str], tuple | None] = {}
    per_answer: list[dict] = []
    for item in items:
        key = (item.query, item.sense)
        if key not in topics:
            record = db.get_topic(item.query, item.sense)
            if record is None:
                topics[key] = None
            else:
                everything = db.get_topic_articles(record.id)        # 기본 RAG: 필터링 없이 전부
                topics[key] = (record, build_in_memory_store(everything) if everything else None, everything)
        if topics[key] is None:
            print(f"- {item.id}: DB에 topic이 없어 건너뜀")
            continue
        record, store, everything = topics[key]
        evidence = _evidence_texts(item.evidence)

        answers = {"llm_only": answer_llm_only(item),
                   "basic_rag": answer_basic_rag(item, store, everything),
                   "qupict": answer_qupict(item, record.id, record.selected)}
        for method, ans in answers.items():
            g = grade(item, ans.answer, evidence)
            cited = cited_links(ans) if method != "llm_only" else []
            claims = g.claims if g else []
            per_answer.append({
                "id": item.id, "method": method, "question": item.question, "answer": ans.answer,
                "graded": g is not None,
                "correct": g.correct if g else None,
                "hallucination": g.hallucination if g else None,
                "claims": len(claims), "supported_claims": sum(c.supported for c in claims),
                "cited": len(cited), "cited_evidence": sum(1 for l in cited if l in item.evidence),
                "grade_reason": g.reasoning if g else "",
                "claim_details": [c.model_dump() for c in claims],
            })
        print(f"- {item.id}: 채점 완료")

    summary = []
    for method in METHODS:
        rs = [r for r in per_answer if r["method"] == method and r["graded"]]
        if not rs:
            continue
        with_claims = [r for r in rs if r["claims"]]
        with_cites = [r for r in rs if r["cited"]]
        summary.append({
            "method": method, "answers": len(rs),
            "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 3),
            "grounding_rate": round(sum(r["supported_claims"] / r["claims"] for r in with_claims) / len(with_claims), 3)
                              if with_claims else None,
            "hallucination_rate": round(sum(r["hallucination"] for r in rs) / len(rs), 3),
            "citation_accuracy": round(sum(r["cited_evidence"] / r["cited"] for r in with_cites) / len(with_cites), 3)
                                 if with_cites else None,
        })
    return summary, per_answer


def main(query: str | None = None, limit: int | None = None) -> None:
    if not load_qa(query=query):
        print("질문 세트가 없습니다. 먼저 python evaluate.py make-qa 로 질문을 만드세요.")
        return
    summary, per_answer = run(query, limit)
    print()
    print_table(summary, ["method", "answers", "accuracy", "grounding_rate", "hallucination_rate", "citation_accuracy"])
    json_path, csv_path = save_results("exp_c_answers", summary + [{k: v for k, v in r.items() if k != "claim_details"}
                                                                   for r in per_answer],
                                       {"methods": METHODS, "per_answer_with_claims": per_answer})
    print(f"\n답변·채점 결과 저장: {csv_path}\n주장별 판정(사람 검수용): {json_path}")
