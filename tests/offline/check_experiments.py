"""
실험 코드: 지표 계산, 실험 A(정답 JSON) 방법별 예측, 실험 B 순위 점수, 실험 C 요약, 질문 세트 만들기 (가짜 DB·LLM)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import asyncio
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments.common as C
import evalset.exp_a_filtering as A
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

# 2) 실험 A (evalset, 정답 JSON 기준): 방법별 예측
import json
sel = SelectedSense(query="강남", sense=Sense(name="방송인 강남", definition="d", related_terms=["이상화", "예능"]),
                    excluded_terms=["재건축", "강남역"])
art = lambda link, title, content="": Article(title=title, link=link, content=content)
items = [
    A.LabeledArticle(art("1", "강남 이상화 예능", "방송인 강남"), True),
    A.LabeledArticle(art("2", "강남역 재건축", "강남 아파트"), False),
    A.LabeledArticle(art("3", "성신여대 축제"), False),
    A.LabeledArticle(art("4", "강남 근황", "예능 출연"), True),
]
llm = {"1": "관련", "2": "관련", "3": "무관", "4": "보류"}
pred = lambda m, x: A.predict(m, x, sel, llm)
assert [pred("baseline", x) for x in items] == [True, True, True, True]
assert [pred("1a_alias", x) for x in items] == [True, True, False, True]          # 3번엔 '강남' 없음
assert [pred("1b_validation", x) for x in items] == [True, False, False, False]   # 검증어 '방송인' (VALIDATION_TERMS)
assert [pred("1c_rule", x) for x in items] == [True, False, False, True]          # 점수 ≥ 1
assert [pred("2b_llm_only", x) for x in items] == [True, True, False, True]       # LLM만: 2번을 관련으로 잘못 봄
assert [pred("proposed", x) for x in items] == [True, False, False, True]         # 규칙이 2·3번을 먼저 거름, 보류 = 관련
other = sel.model_copy(update={"sense": sel.sense.model_copy(update={"name": "x"})})
assert A.predict("1b_validation", items[0], other, llm) is None                  # 검증어 없는 대상
assert A.predict("2b_llm_only", items[0], sel, {}) is None                       # LLM이 답 못 한 기사
print("2 ok: 실험 A 방법별 예측")

# 3) 실험 A 전체 흐름: 정답 JSON 읽기 (기준은 항목에 저장된 값), LLM 판정 캐시, 결과 행
sense_json = {"name": "방송인 강남", "definition": "d", "aliases": [], "related_terms": ["이상화", "예능"],
              "excluded_terms": ["재건축", "강남역"], "other_senses": [{"name": "지역 강남", "definition": "구", "related_terms": ["강남역"]}]}
entry = lambda url, title, content, label: {"title": title, "url": url, "description": "", "content": content,
                                            "query": "강남", "sense": sense_json, "label": label}
labeled = tmp / "labeled_set.json"
labeled.write_text(json.dumps([entry("1", "강남 이상화 예능", "방송인 강남", True),
                               entry("2", "강남역 재건축", "강남 아파트", False),
                               entry("3", "성신여대 축제", "", False),
                               entry("4", "강남 근황", "예능 출연", True),
                               entry("4", "같은 기사 두 번", "", False),                     # 중복 url → 처음 것만
                               {"title": "예전 형식", "url": "x", "label": "관련"}], ensure_ascii=False), encoding="utf-8")
groups, bad = A.load_groups(labeled)
assert bad == 1 and len(groups) == 1 and [x.relevant for x in groups[0].items] == [True, False, False, True]
assert groups[0].selected.other_senses[0].name == "지역 강남" and groups[0].selected.excluded_terms == ["재건축", "강남역"]
assert A.load_groups(labeled, "애플") == ([], 1)
A.LLM_CACHE = tmp / "llm_cache.json"
asked = []
async def fake_verify(articles, selected):
    asked.extend(a.link for a in articles)
    from qupict.filtering import Verdict
    return [Verdict(a.link, "무관" if a.link == "3" else "관련", "llm") for a in articles if a.link != "2"], ["2"]
A.llm_verify = fake_verify
rows = asyncio.run(A.run(path=labeled))
assert sorted(asked) == ["1", "2", "3", "4"]                         # 처음엔 모든 기사 판정
asked.clear(); asyncio.run(A.run(path=labeled)); assert asked == ["2"]   # 다시: 답 못 한 기사만
overall = {r["method"]: r for r in rows if r.get("target") == "전체(micro)"}
assert overall["proposed"]["f1"] == 1.0 and overall["proposed"]["n"] == 4   # 2번은 규칙이 거름 → LLM 답 없어도 예측
assert overall["2b_llm_only"]["n"] == 3                              # 2번은 LLM 답이 없어 건너뜀
assert overall["baseline"]["precision"] == 0.5 and overall["1b_validation"]["recall"] == 0.5
assert overall["2b_llm_only"]["llm_calls"] == 1 and overall["proposed"]["llm_calls"] == 1
assert any(r.get("section") == "1c 기준값별" and r.get("target") == "강남 → 방송인 강남" for r in rows)
A.RESULTS_DIR = tmp / "results"
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    A.main(path=labeled)
assert list((tmp / "results").glob("exp_a_filtering_*.csv"))           # 결과는 evalset/results (여기선 임시 폴더)
print("3 ok: 실험 A 흐름 (정답 JSON, 저장된 기준, LLM 캐시)")

# 4) 실험 B: 순위 점수, DB 청크 → 기사 순위
assert B.score(["a", "b", "c"], ["b"]) == (True, 0.5, 2)
assert B.score(["a"] * 5 + ["x"], ["x"]) == (False, 1 / 6, 6)       # 6위: Recall@5 실패, MRR엔 반영
assert B.score(["a"], ["x"]) == (False, 0.0, None)
import qupict.retrieval as R
R.search_topic = lambda tid, q, s, k=5, use_hyde=True, use_multi_query=True: [
    db.ChunkHit("c", "body", 0.9, art("A", "")), db.ChunkHit("c2", "body", 0.8, art("A", "")),
    db.ChunkHit("c3", "body", 0.7, art("B", ""))]
assert B.article_ranking_db(1, Q.QAItem("i", "q", "s", "질문", "답"), sel, True, True) == ["A", "B"]
print("4 ok: 실험 B 순위·점수")

# 5) 실험 C: 인용 링크, 요약 계산
ans = Answer(question="q", answer="x(기사 2)", sources=[art("A", ""), art("B", "")], cited=[2])
assert X.cited_links(ans) == ["B"]
Q.QA_PATH = tmp / "qa.csv"
Q.save_qa([Q.QAItem("강남-001", "강남", "방송인 강남", "질문", "정답", ["B"])])
X.load_qa = lambda path=None, query=None: Q.load_qa(Q.QA_PATH, query, resolve=False)
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
items_qa = Q.load_qa(Q.QA_PATH, resolve=False)
assert Q.next_id(items_qa, "강남") == "강남-002" and Q.next_id(items_qa, "눈") == "눈-001"
arts = [art("B", "이미 질문 있음"), art("C", "기사 C", "본문"), art("D", "기사 D", "본문"), art("E", "기사 E", "본문")]
keys = iter(["질문C", "정답C", "", "q"])                           # C 직접 입력, D 건너뜀(빈 질문), E에서 끝
assert Q.make_qa("강남", "방송인 강남", arts, suggest=False, read=lambda p="": next(keys)) == 1
Q.draft_qa = lambda title, content, target: (f"초안-{title}", "초안정답")
keys = iter(["", "s", "q"])                                         # D 초안 그대로, E 건너뜀
assert Q.make_qa("강남", "방송인 강남", arts, suggest=True, read=lambda p="": next(keys)) == 1
saved = Q.load_qa(Q.QA_PATH, resolve=False)
assert [(x.id, x.question, x.evidence) for x in saved[1:]] == [("강남-002", "질문C", ["C"]), ("강남-003", "초안-기사 D", ["D"])]
print("6 ok: 질문 세트 만들기")

# 7) 질문 세트 CSV 읽기: 열 이름(한글) · 엑셀(cp949) · 의미 채우기 · 중복 기사 → 대표 기사 · 건너뛰기
csv_path = tmp / "golden.csv"
csv_path.write_text(
    "검색어,의미,질문,정답,근거 기사\n"
    "강남,,강남 새 질문?,답1,A; DUP\n"                     # 강남 topic 둘 → 근거 기사가 '관련'인 topic으로
    "눈,,눈 질문?,,\n"                                      # 눈 topic 하나 → 자동, 정답 · 근거 없음 → 확인
    "후드,,후드 질문?,답,\n"                                # DB에 없음 → 건너뜀
    "강남,지역  강남,강남역 질문?,답,X\n"                   # 의미 띄어쓰기 달라도 맞춤, X는 topic에 없음 → 확인
    "강남,,근거 없는 강남 질문?,답,\n"                      # 의미 여러 개 + 근거 없음 → 건너뜀
    "강남,강남 방송인,이미지 질문?,답,IMG?type=w\n"         # 이름 다른 의미 → 같은 단어 많은 의미, 이미지 주소 → 기사 A
    "강남,연예인,무관 근거 질문?,답,A\n",                   # 이름이 안 겹침 → 근거가 '관련'인 topic ('무관' topic은 안 봄)
    encoding="cp949")
def fake_lookup(query, links):
    senses = {"강남": ["방송인 강남", "지역 강남"], "눈": ["날씨 눈"]}.get(query, [])
    rows = [Q.EvidenceRow("A", "방송인 강남", "관련", None), Q.EvidenceRow("DUP", "방송인 강남", "관련", "A2"),
            Q.EvidenceRow("A", "지역 강남", "무관", None)]
    found = [r for r in rows if r.link in links]
    if "IMG?type=w" in links:
        found += [Q.EvidenceRow("A", "방송인 강남", "관련", None, "IMG?type=w"),
                  Q.EvidenceRow("A", "지역 강남", "무관", None, "IMG?type=w")]
    return senses, found
items_i, rep = Q.resolve_rows(Q.read_csv_rows(csv_path), lookup=fake_lookup)
new_q = {x.question: x for x in items_i}
assert list(new_q) == ["강남 새 질문?", "눈 질문?", "강남역 질문?", "이미지 질문?", "무관 근거 질문?"] and len(rep.skipped) == 2
assert new_q["강남 새 질문?"].sense == "방송인 강남" and new_q["강남 새 질문?"].evidence == ["A", "DUP", "A2"]
assert [x.id for x in items_i] == ["강남-001", "눈-001", "강남-002", "강남-003", "강남-004"]
assert new_q["이미지 질문?"].sense == "방송인 강남" and new_q["이미지 질문?"].evidence == ["A"]
assert any("7번 줄" in m and "'방송인 강남'로 봤습니다" in m for m in rep.warnings)
assert not any("7번 줄" in m and "에 없습니다" in m for m in rep.warnings)
assert new_q["무관 근거 질문?"].sense == "방송인 강남"
assert Q._closest_names("기업 애플", ["기업 Apple", "과일 사과"]) == ["기업 Apple"]
assert Q._closest_names("삼성전자(기업)", ["기업 삼성전자"]) == ["기업 삼성전자"]
assert new_q["눈 질문?"].sense == "날씨 눈" and new_q["눈 질문?"].answer == "" and new_q["눈 질문?"].evidence == []
assert new_q["강남역 질문?"].sense == "지역 강남"
assert any("후드" in m and "collect" in m for m in rep.skipped) and any("sense 열" in m for m in rep.skipped)
assert any("X" in m for m in rep.warnings) and any("모두 없어" in m for m in rep.warnings)
# load_qa는 기본으로 DB를 보고 채운다 (db_lookup), 같은 안내는 한 번만 출력
Q.db_lookup = fake_lookup
import contextlib, io
out = io.StringIO()
with contextlib.redirect_stdout(out):
    assert [x.sense for x in Q.load_qa(csv_path, query="강남")] == ["방송인 강남", "지역 강남", "방송인 강남", "방송인 강남"]
    Q.load_qa(csv_path)
assert out.getvalue().count("[질문 세트 건너뜀] 4번 줄") == 1
# 저장 → 다시 읽기 (make-qa가 쓰는 형식, 근거 link는 ; 로)
Q.save_qa(items_i, tmp / "saved.csv")
assert Q.load_qa(tmp / "saved.csv", resolve=False) == items_i
assert "A; DUP; A2" in (tmp / "saved.csv").read_text(encoding="utf-8-sig")
assert all(r["query"] and r["question"]                                                   # 실제 질문 세트도 읽힌다
           for r in Q.read_csv_rows(Path(__file__).resolve().parents[2] / "experiments" / "qa_set.csv"))
print("7 ok: 질문 세트 CSV 읽기 · 저장")

# 8) 정답 · 근거가 비어 있는 질문: 실험 B는 근거 없는 질문을 건너뛰고, 실험 C는 정답 · 근거 둘 다 없을 때만 건너뜀
Q.save_qa([Q.QAItem("강남-001", "강남", "방송인 강남", "질문", "정답", ["B"]),
           Q.QAItem("강남-002", "강남", "방송인 강남", "정답만", "정답", []),
           Q.QAItem("강남-003", "강남", "방송인 강남", "둘 다 없음", "", [])])
B.load_qa = X.load_qa
B.article_ranking_db = lambda tid, item, s, h, m: ["B"]
_, per_b = B.run()
assert {r["id"] for r in per_b} == {"강남-001"}
X.grade = lambda item, text, ev: grades[text]
_, per_c = X.run()
assert {r["id"] for r in per_c} == {"강남-001", "강남-002"}
assert [r["has_evidence"] for r in per_c if r["method"] == "qupict"] == [True, False]
print("8 ok: 정답 · 근거 빈 질문 처리")
print("ALL PASS")
