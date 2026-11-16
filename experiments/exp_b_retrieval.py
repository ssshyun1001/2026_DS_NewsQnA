"""
실험 B — RAG 검색 성능 비교 (기획안 7단계)

질문 세트(experiments/qa_set.csv, --qa로 다른 CSV)의 질문마다, 검색 방식별로 "정답 근거 기사를 위쪽에 찾아오는지" 잰다.

  방식        검색 문장
  ─────────  ──────────────────────────────────────
  base        질문 그대로
  hyde        질문 + HyDE 가상 기사 문장
  multi       질문 + Multi-Query 변형 질문 2개
  both        HyDE + Multi-Query
  hybrid      질문 그대로 벡터 검색 + BM25 키워드 검색, RRF로 합침  (실제 서비스 설정, qupict/bm25.py)
              위 4가지는 벡터 검색만 (use_hybrid=False). HyDE + hybrid 비교는 experiments/exp_b_hyde_hybrid.py

  검색 엔진: db (Firestore 벡터 검색: topic · 관련/보류 · 기간 조건 안에서 청크 검색, 실제 서비스.
             2026-11-16 이전 결과 파일에는 예전 이름 pgvector로 적혀 있다)
            --faiss를 주면 예전 FAISS 메모리 검색(기사 단위, 관련/보류 기사)도 base~both 4가지로 비교 (hybrid는 DB만)

지표
  Recall@5   정답 근거 기사가 위에서 5개 기사 안에 있는 질문의 비율
  MRR        정답 근거 기사가 처음 나온 순위의 역수 평균 (1위 = 1, 2위 = 0.5, 없으면 0, 10위까지 봄)
  검색 시간   질문 1개당 평균 초 (HyDE·Multi-Query LLM 시간 포함)

DB 검색은 청크를 찾으므로 "청크가 처음 나온 순서"대로 기사 순위를 매긴다.

실행: python evaluate.py exp-b [--target 강남] [--faiss] [--qa 골든셋.csv]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 이 파일을 바로 실행할 때도 프로젝트 폴더를 찾게

import time

from experiments.common import print_table, save_results
from experiments.qa_set import QAItem, load_qa
from qupict import db

# 방식 이름 → (HyDE, Multi-Query, 하이브리드)
VARIANTS = {"base": (False, False, False), "hyde": (True, False, False), "multi": (False, True, False),
            "both": (True, True, False), "hybrid": (False, False, True)}
RANK_DEPTH = 10            # MRR을 볼 기사 순위 깊이
CHUNKS_PER_SEARCH = 20     # DB 벡터 검색에서 가져올 청크 수 (기사 10개 순위를 만들기에 충분하게)


def article_ranking_db(topic_id: int, item: QAItem, selected, hyde: bool, multi: bool,
                       hybrid: bool = False) -> list[str]:
    """hybrid=False(기본)면 벡터 검색만. experiments/exp_b_hyde_hybrid.py도 벡터 쪽 순위를 이걸로 만든다."""
    from qupict.retrieval import search_topic
    hits = search_topic(topic_id, item.question, selected, k=CHUNKS_PER_SEARCH,
                        use_hyde=hyde, use_multi_query=multi, use_hybrid=hybrid)
    ranking: list[str] = []
    for h in hits:                              # 유사도 높은 청크 순서 → 기사가 처음 나온 순서
        if h.article.link not in ranking:
            ranking.append(h.article.link)
    return ranking[:RANK_DEPTH]


def article_ranking_faiss(store, articles, item: QAItem, selected, hyde: bool, multi: bool) -> list[str]:
    from qupict.retrieval import retrieve_relevant_articles, topic_context
    found = retrieve_relevant_articles(store, articles, item.question, topic=topic_context(selected),
                                       top_k=RANK_DEPTH, use_hyde=hyde, use_multi_query=multi)
    return [a.link for a in found]


def score(ranking: list[str], evidence: list[str]) -> tuple[bool, float, int | None]:
    """(Recall@5 맞춤 여부, 역순위, 순위)"""
    for rank, link in enumerate(ranking, 1):
        if link in evidence:
            return rank <= 5, 1.0 / rank, rank
    return False, 0.0, None


def run(query: str | None = None, with_faiss: bool = False, qa_path=None,
        items: list[QAItem] | None = None) -> tuple[list[dict], list[dict]]:
    items = load_qa(qa_path, query=query) if items is None else items
    per_question: list[dict] = []
    engines = ["db"] + (["faiss"] if with_faiss else [])
    topics: dict[tuple[str, str], tuple] = {}

    no_evidence = [x.id for x in items if not x.evidence]
    if no_evidence:
        print(f"- 근거 기사 link가 없는 질문 {len(no_evidence)}개는 건너뜀 (Recall · MRR을 잴 수 없음): {', '.join(no_evidence)}")
    todo = [x for x in items if x.evidence]
    run_start = time.time()
    for done, item in enumerate(todo):
        if done:
            left = (time.time() - run_start) / done * (len(todo) - done)
            print(f"  [{done + 1}/{len(todo)}] {item.id}  (남은 시간 약 {left / 60:.0f}분)", flush=True)
        else:
            print(f"  [1/{len(todo)}] {item.id}  (검색 방식 {len(VARIANTS)}가지 × 질문 {len(todo)}개)", flush=True)
        key = (item.query, item.sense)
        if key not in topics:
            record = db.get_topic(item.query, item.sense)
            if record is None:
                print(f"- {item.id}: DB에 topic '{item.query} → {item.sense}'이 없어 건너뜀")
                topics[key] = None
                continue
            store = articles = None
            if with_faiss:
                from qupict.retrieval import build_in_memory_store
                articles = db.get_topic_articles(record.id, statuses=("관련", "보류"))
                store = build_in_memory_store(articles) if articles else None
            topics[key] = (record, store, articles)
        if topics[key] is None:
            continue
        record, store, articles = topics[key]

        for engine in engines:
            for name, (hyde, multi, hybrid) in VARIANTS.items():
                if engine == "faiss" and hybrid:
                    continue                        # 하이브리드는 DB 검색에만 있다
                start = time.time()
                if engine == "db":
                    ranking = article_ranking_db(record.id, item, record.selected, hyde, multi, hybrid)
                else:
                    ranking = article_ranking_faiss(store, articles, item, record.selected, hyde, multi) if store else []
                elapsed = time.time() - start
                hit5, rr, rank = score(ranking, item.evidence)
                per_question.append({"id": item.id, "engine": engine, "variant": name, "hit@5": hit5,
                                     "reciprocal_rank": round(rr, 3), "rank": rank, "seconds": round(elapsed, 2)})

    summary = []
    for engine in engines:
        for name in VARIANTS:
            rs = [r for r in per_question if r["engine"] == engine and r["variant"] == name]
            if rs:
                summary.append({"engine": engine, "variant": name, "questions": len(rs),
                                "recall@5": round(sum(r["hit@5"] for r in rs) / len(rs), 3),
                                "mrr": round(sum(r["reciprocal_rank"] for r in rs) / len(rs), 3),
                                "avg_seconds": round(sum(r["seconds"] for r in rs) / len(rs), 2)})
    return summary, per_question


def main(query: str | None = None, with_faiss: bool = False, qa_path=None) -> None:
    items = load_qa(qa_path, query=query)
    if not items:
        print("질문 세트가 없습니다. experiments/qa_set.csv에 질문을 넣거나 python evaluate.py make-qa 로 만드세요.")
        return
    summary, per_question = run(query, with_faiss, qa_path, items=items)
    print()
    print_table(summary, ["engine", "variant", "questions", "recall@5", "mrr", "avg_seconds"])
    _, csv_path = save_results("exp_b_retrieval", summary + per_question,
                               {"rank_depth": RANK_DEPTH, "chunks_per_search": CHUNKS_PER_SEARCH})
    print(f"\n질문별 결과 저장: {csv_path}")


if __name__ == "__main__":   # PyCharm 등에서 이 파일을 바로 실행해도 python evaluate.py exp-b 와 같게
    import sys
    import evaluate
    evaluate.main(["exp-b", *sys.argv[1:]])
