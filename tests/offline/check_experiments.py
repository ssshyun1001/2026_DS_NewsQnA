"""
실험 코드: 지표 계산, 실험 A 방법별 예측, 실험 B 순위 점수, 실험 C 요약, 질문 세트 만들기 (가짜 DB·LLM)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import asyncio
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments.common as C
import experiments.exp_a_filtering as A
import experiments.exp_b_retrieval as B
import experiments.exp_c_answers as X
import experiments.qa_set as Q
from qupict import db
from qupict.schemas import Answer, Article, SelectedSense, Sense

tmp = Path(tempfile.mkdtemp())

# 1) 지표
s = C.BinaryScores()
for p, a in [(True, True), (True, True), (True, False), (False, True), (False, False)]:
    s.add(p, a)
assert (s.tp, s.fp, s.fn, s.tn) == (2, 1, 1, 1)
assert round(s.precision, 3) == 0.667 and round(s.recall, 3) == 0.667 and round(s.f1, 3) == 0.667
assert s.accuracy == 0.6 and round(s.irrelevant_inflow, 3) == 0.333
assert C.BinaryScores().f1 == 0.0                                  # 0으로 나누지 않음
C.RESULTS_DIR = tmp / "results"
_, csv_path = C.save_results("t", [{"a": 1}, {"a": 2, "b": 3}])    # 행마다 칸이 달라도 저장
assert list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))[1] == {"a": "2", "b": "3"}
print("1 ok: 지표·결과 저장")

# 2) 실험 A: 방법별 예측
sel = SelectedSense(query="강남", sense=Sense(name="방송인 강남", definition="d", related_terms=["이상화", "예능"]),
                    excluded_terms=["재건축", "강남역"])
target = {"query": "강남", "sense": "방송인 강남", "validation_term": "방송인"}
art = lambda link, title, content="": Article(title=title, link=link, content=content)
items = [
    A.LabeledArticle(art("1", "강남 이상화 예능", "방송인 강남"), "관련", "관련", "llm"),
    A.LabeledArticle(art("2", "강남역 재건축", "강남 아파트"), "무관", "무관", "rule"),
    A.LabeledArticle(art("3", "성신여대 축제"), "무관", "무관", "rule"),
    A.LabeledArticle(art("4", "강남 근황", "예능 출연"), "관련", "보류", "llm"),
]
llm_only = {"1": "관련", "2": "무관", "3": "무관", "4": "관련"}
pred = lambda m, x: A.predict(m, x, target, sel, llm_only)
assert [pred("baseline", x) for x in items] == [True, True, True, True]
assert [pred("1a_alias", x) for x in items] == [True, True, False, True]          # 3번엔 '강남' 없음
assert [pred("1b_validation", x) for x in items] == [True, False, False, False]   # 검증어 '방송인'
assert [pred("1c_rule", x) for x in items] == [True, False, False, True]          # 점수 ≥ 1
assert [pred("proposed", x) for x in items] == [True, False, False, True]         # 보류 = 관련으로 셈
assert A.predict("1b_validation", items[0], {"query": "강남", "sense": "x"}, sel, llm_only) is None  # 검증어 없음
print("2 ok: 실험 A 방법별 예측")

# 3) 실험 A 전체 흐름: 라벨 보류 제외, 2b는 규칙으로 걸러진 기사만 LLM 추가 판정(캐시)
A.LLM_ONLY_CACHE = tmp / "cache.json"
A.pick_targets = lambda q=None: [target]
A.resolve_target = lambda t: (7, sel)
A.load_labeled = lambda tid: items + [A.LabeledArticle(art("5", "강남"), "보류", None, None)]
asked = []
async def fake_verify(articles, selected):
    asked.extend(a.link for a in articles)
    from qupict.filtering import Verdict
    return [Verdict(a.link, "무관", "llm") for a in articles], []
A.llm_verify = fake_verify
rows = asyncio.run(A.run())
assert sorted(asked) == ["2", "3"]                                  # LLM이 안 본 기사만
asked.clear(); asyncio.run(A.run()); assert asked == []             # 두 번째는 캐시 → 호출 0
overall = {r["method"]: r for r in rows if r.get("target") == "전체(micro)"}
assert overall["proposed"]["f1"] == 1.0 and overall["baseline"]["precision"] == 0.5
assert overall["1b_validation"]["recall"] == 0.5
assert next(r for r in rows if r.get("method") == "baseline" and r["target"] != "전체(micro)")["label_hold_excluded"] == 1
assert any(r.get("section") == "1c 기준값별" and r.get("target") == "강남 → 방송인 강남" for r in rows)
print("3 ok: 실험 A 흐름 (라벨 보류 제외, LLM 추가 판정 캐시)")

# 4) 실험 B: 순위 점수, pgvector 청크 → 기사 순위
assert B.score(["a", "b", "c"], ["b"]) == (True, 0.5, 2)
assert B.score(["a"] * 5 + ["x"], ["x"]) == (False, 1 / 6, 6)       # 6위: Recall@5 실패, MRR엔 반영
assert B.score(["a"], ["x"]) == (False, 0.0, None)
import qupict.retrieval as R
R.search_topic = lambda tid, q, s, k=5, use_hyde=True, use_multi_query=True: [
    db.ChunkHit("c", "body", 0.9, art("A", "")), db.ChunkHit("c2", "body", 0.8, art("A", "")),
    db.ChunkHit("c3", "body", 0.7, art("B", ""))]
assert B.article_ranking_pgvector(1, Q.QAItem("i", "q", "s", "질문", "답"), sel, True, True) == ["A", "B"]
print("4 ok: 실험 B 순위·점수")

# 5) 실험 C: 인용 링크, 요약 계산
ans = Answer(question="q", answer="x(기사 2)", sources=[art("A", ""), art("B", "")], cited=[2])
assert X.cited_links(ans) == ["B"]
Q.QA_PATH = tmp / "qa.yaml"
Q.save_qa([Q.QAItem("강남-001", "강남", "방송인 강남", "질문", "정답", ["B"])])
X.load_qa = lambda query=None: Q.load_qa(Q.QA_PATH, query)
class Rec:
    id = 3
    selected = sel
db.get_topic = lambda q, s: Rec()
db.get_topic_articles = lambda tid, **k: []
X._evidence_texts = lambda links: ["근거"]
X.answer_llm_only = lambda item: Answer(question="q", answer="모름", sources=[])
X.answer_basic_rag = lambda item, store, arts: Answer(question="q", answer="x(기사 1)", sources=[art("A", "")])
X.answer_qupict = lambda item, tid, s: Answer(question="q", answer="y(기사 1)", sources=[art("B", "")], cited=[1])
grades = {"모름": X._Grade(claims=[], reasoning="r", correct=False, hallucination=False),
          "x(기사 1)": X._Grade(claims=[X._Claim(claim="c", evidence="e", supported=False)], reasoning="r", correct=False, hallucination=True),
          "y(기사 1)": X._Grade(claims=[X._Claim(claim="c", evidence="e", supported=True)], reasoning="r", correct=True, hallucination=False)}
X.grade = lambda item, text, ev: grades[text]
summary, per = X.run()
by = {r["method"]: r for r in summary}
assert by["llm_only"]["accuracy"] == 0.0 and by["llm_only"]["grounding_rate"] is None
assert by["basic_rag"]["hallucination_rate"] == 1.0 and by["basic_rag"]["citation_accuracy"] == 0.0   # A는 근거 아님
assert by["qupict"]["accuracy"] == 1.0 and by["qupict"]["grounding_rate"] == 1.0 and by["qupict"]["citation_accuracy"] == 1.0
print("5 ok: 실험 C 요약 (정답률·근거 일치율·환각·출처 정확도)")

# 6) 질문 세트: 번호 매기기, 만들기(직접 입력 / 초안 받기 / 건너뛰기 / 끝내기)
items_qa = Q.load_qa(Q.QA_PATH)
assert Q.next_id(items_qa, "강남") == "강남-002" and Q.next_id(items_qa, "눈") == "눈-001"
arts = [art("B", "이미 질문 있음"), art("C", "기사 C", "본문"), art("D", "기사 D", "본문"), art("E", "기사 E", "본문")]
keys = iter(["질문C", "정답C", "", "q"])                           # C 직접 입력, D 건너뜀(빈 질문), E에서 끝
assert Q.make_qa("강남", "방송인 강남", arts, suggest=False, read=lambda p="": next(keys)) == 1
Q.draft_qa = lambda title, content, target: (f"초안-{title}", "초안정답")
keys = iter(["", "s", "q"])                                         # D 초안 그대로, E 건너뜀
assert Q.make_qa("강남", "방송인 강남", arts, suggest=True, read=lambda p="": next(keys)) == 1
saved = Q.load_qa(Q.QA_PATH)
assert [(x.id, x.question, x.evidence) for x in saved[1:]] == [("강남-002", "질문C", ["C"]), ("강남-003", "초안-기사 D", ["D"])]
print("6 ok: 질문 세트 만들기")
print("ALL PASS")
