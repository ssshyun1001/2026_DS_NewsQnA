"""
유사도 확인 도구 (python run.py sim): 기준값 없이 유사도, 정답 근거 순위, 기준값별 통과 수 (가짜 LLM·DB)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments.common as C
import experiments.qa_set as Q
import experiments.similarity as S
from qupict import db, retrieval as R
from qupict.schemas import Article, SelectedSense, Sense

sel = SelectedSense(query="강남", sense=Sense(name="방송인 강남", definition="가수 겸 방송인"))

# 가짜 LLM: HyDE 문장 1개, 변형 질문 2개 (부른 횟수를 센다)
calls = {"hyde": 0, "multi": 0, "embed": []}
def fake_hyde(question, context, use_hyde=True):
    calls["hyde"] += 1
    return f"{question}\n가상 기사"
def fake_variants(question, context):
    calls["multi"] += 1
    return ["변형1", "변형2"]
S._build_search_text, S._generate_query_variants = fake_hyde, fake_variants

# 가짜 임베딩: 문장마다 번호 벡터. 가짜 DB: 벡터 번호에 따라 유사도가 다른 청크
class FakeEmb:
    def embed_documents(self, texts):
        calls["embed"].append(list(texts))
        return [[float(i)] for i in range(len(texts))]
S.get_embeddings = lambda: FakeEmb()

def hit(link, content, sim):
    return db.ChunkHit(content=content, source="body", similarity=sim, article=Article(title=f"제목{link}", link=link))
TABLE = {   # 벡터 번호 → 찾은 청크
    0.0: [hit("A", "a1", 0.42), hit("B", "b1", 0.30)],          # 질문 그대로
    1.0: [hit("B", "b1", 0.55), hit("A", "a1", 0.47)],          # HyDE 문장
    2.0: [hit("C", "c1", 0.60)],                                # 변형1
    3.0: [hit("A", "a2", 0.51)],                                # 변형2
}
def fake_search(topic_id, vec, k=5, period_days=None, statuses=("관련",)):
    return TABLE[vec[0]][:k]
db.search_chunks = fake_search

# 1) 방식별 검색 문장: HyDE·변형은 한 번만 만들고, 같은 문장은 한 번만 임베딩
hits, texts = S.variant_hits(7, "질문", sel)
assert calls["hyde"] == 1 and calls["multi"] == 1
assert texts == {"base": ["질문"], "hyde": ["질문\n가상 기사"], "multi": ["질문", "변형1", "변형2"],
                 "both": ["질문\n가상 기사", "변형1", "변형2"]}
assert calls["embed"] == [["질문", "질문\n가상 기사", "변형1", "변형2"]]
print("1 ok: 검색 문장 한 번만 만들고 한 번만 임베딩")

# 2) 기준값 없이 전부, 같은 청크는 최고 유사도 (max-pooling), 높은 순
assert [(h.article.link, h.similarity) for h in hits["base"]] == [("A", 0.42), ("B", 0.30)]   # 0.5 미만도 보임
assert [(h.content, h.similarity) for h in hits["hyde"]] == [("b1", 0.55), ("a1", 0.47)]
assert [(h.content, h.similarity) for h in hits["multi"]] == [("c1", 0.60), ("a2", 0.51), ("a1", 0.42), ("b1", 0.30)]
assert [h.content for h in hits["both"]] == ["c1", "b1", "a2", "a1"]
print("2 ok: 기준값 없이 유사도 높은 순 (max-pooling)")

# 3) 정답 근거 최고 유사도와 기사 순위
assert S.evidence_best(hits["both"], ["A"]) == (0.51, 3)      # C, B 다음 세 번째 기사
assert S.evidence_best(hits["base"], ["A"]) == (0.42, 1)
assert S.evidence_best(hits["base"], ["Z"]) == (None, None)
print("3 ok: 정답 근거 유사도·순위")

# 4) search_topic은 그대로 기준값을 적용한다 (search_vectors로 나눈 뒤에도)
R._build_search_text = lambda q, c, use_hyde=True: q
R._generate_query_variants = lambda q, c: []
R.get_embeddings = lambda: FakeEmb()
assert [h.similarity for h in R.search_topic(7, "질문", sel, use_multi_query=False, similarity_threshold=0.4)] == [0.42]
assert R.search_topic(7, "질문", sel, use_multi_query=False) == []          # 기본 0.5: 0.42도 빠짐
print("4 ok: search_topic 기준값 유지")

# 5) 질문 세트 보고서: 방식별 정답 근거 유사도, 기준값 후보별 통과 수, 결과 저장
tmp = Path(tempfile.mkdtemp())
Q.QA_PATH = tmp / "qa.yaml"
Q.save_qa([Q.QAItem("강남-001", "강남", "방송인 강남", "질문", "정답", ["A"]),
           Q.QAItem("강남-002", "강남", "방송인 강남", "질문", "정답", ["Z"]),
           Q.QAItem("눈-001", "눈", "날씨 눈", "질문", "정답", ["A"])])
db.get_topic = lambda q, s: db.TopicRecord(id=7, selected=sel, last_collected_at=None) if q == "강남" else None
out = io.StringIO()
with contextlib.redirect_stdout(out):
    summary, rows = S.qa_report()
assert "눈 → 날씨 눈" in out.getvalue()                                  # DB에 없는 topic은 건너뜀
first = {r["variant"]: r for r in rows if r["id"] == "강남-001"}
assert first["base"]["evidence_similarity"] == 0.42 and first["both"]["evidence_similarity"] == 0.51
assert first["both"]["evidence_rank"] == 3 and first["multi"]["top_similarity"] == 0.6
assert all(r["evidence_similarity"] is None for r in rows if r["id"] == "강남-002")
by = {r["variant"]: r for r in summary}
assert by["base"]["questions"] == 2
assert (by["base"]["pass@0.40"], by["base"]["pass@0.45"], by["base"]["pass@0.50"]) == (1, 0, 0)
assert (by["both"]["pass@0.40"], by["both"]["pass@0.50"]) == (1, 1)
C.RESULTS_DIR = tmp / "results"
with contextlib.redirect_stdout(io.StringIO()):
    S.main_qa("강남")
assert list((tmp / "results").glob("similarity_*.csv"))
print("5 ok: 질문 세트 보고서 · 기준값별 통과 수 · 결과 저장")

# 6) 질문 하나 보기: 질문 세트에 같은 질문이 있으면 정답 근거 ★
out = io.StringIO()
with contextlib.redirect_stdout(out):
    S.main_question("강남", "방송인 강남", "질문")
text = out.getvalue()
assert "★" in text and "지금 기준값: 0.50" in text and "0.420  -" in text and "0.600  O" in text
with contextlib.redirect_stdout(io.StringIO()) as o:
    S.main_question("눈", "날씨 눈", "질문")
assert "DB에 topic" in o.getvalue()
print("6 ok: 질문 하나 보기")
