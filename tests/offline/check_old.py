"""
old/ 스크립트: 라벨링(JSON) → 필터 평가 V0/V1/V2, 골든셋 검색 성공률, 유사도 디버그 (가짜 LLM·DB·네이버)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import asyncio
import builtins
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "old"))     # old/ 스크립트는 "from _common import ..."를 쓴다
sys.path.insert(0, str(ROOT / "evalset"))
sys.path.insert(0, str(ROOT))
import _common
import build_labeled_set as B
import debug_similarity as D
import eval_filter as E
import eval_retrieval as R
from qupict import db
from qupict.filtering import Verdict
from qupict.schemas import Article, SelectedSense, Sense

tmp = Path(tempfile.mkdtemp())
sel = SelectedSense(query="강남", sense=Sense(name="방송인 강남", definition="가수 겸 방송인",
                                            related_terms=["이상화", "예능"]),
                    excluded_terms=["재건축", "강남역"])
quiet = lambda: contextlib.redirect_stdout(io.StringIO())

# 1) evalset/build_labeled_set: 의미 후보 선택 → 수집 → 중복 제거 → 최신순 → 1/2 라벨 → JSON (임베딩·이미지 바이트는 저장 안 함)
from datetime import datetime, timezone
day = lambda d: datetime(2026, 11, d, tzinfo=timezone.utc)
arts = [Article(title="강남 이상화 예능", link="new", content="방송인 강남 " * 20, pub_date=day(8)),
        Article(title="강남 이상화 예능 (같은 기사)", link="dup", content="방송인 강남", pub_date=day(8)),
        Article(title="강남역 재건축", link="mid", content="강남역 재건축 아파트", pub_date=day(5)),
        Article(title="강남 근황", link="old", content="강남 예능 출연", pub_date=day(1))]
async def fake_collect(selected, days, mx): return ["m"] * 4, [a.model_copy() for a in arts]
class SameForDup:
    async def aembed_documents(self, texts):
        return [[1.0, 0.0] if "같은 기사" in t or t.startswith(("강남 이상화 예능", "방송인 강남")) else [0.0, 1.0] if "재건축" in t else [0.0, -1.0]
                for t in texts]
import qupict.filtering as F
F.get_embeddings = lambda: SameForDup()
# 라벨링 흐름만 확인한다: 중복 기준은 가짜 임베딩이 맞춰진 제목·본문으로 고정 (고유명사 기준·.env 값과 무관하게)
import dataclasses
F.settings = dataclasses.replace(F.settings, dedup_by="title,content", dedup_rule="or",
                                 dedup_title_threshold=0.7, dedup_content_threshold=0.7)
from qupict.schemas import SenseCandidates
cands = SenseCandidates(query="강남", model="m", senses=[
    Sense(name="지역 강남", definition="서울 강남구 일대", related_terms=["재건축", "강남역"]),
    Sense(name="방송인 강남", definition="가수 겸 방송인", related_terms=["이상화", "예능"])])
B.collect_articles = fake_collect
B.load_sense_candidates = lambda q: (cands.model_copy(update={"query": q}), "DB")
B._OUTPUT_PATH = tmp / "labeled_set.json"
# 검색어 → 기간(Enter=30) → 의미 2번 → 라벨 3건 → 다음 검색어 "서울여대"는 라벨 도중 q → 검색어 q로 종료
keys = iter(["강남", "", "2", "1", "2", "2", "서울여대", "7", "2", "q", "q"])
builtins_input = builtins.input
builtins.input = lambda prompt="": next(keys)
with quiet():
    asyncio.run(B.main())
assert next(keys, None) is None                                                  # 입력을 남기지 않고 끝남
entries = json.loads(B._OUTPUT_PATH.read_text(encoding="utf-8"))
assert [(e["url"], e["label"]) for e in entries] == [("new", True), ("mid", False), ("old", False)], entries  # dup 빠짐, 최신순
assert set(entries[0]) == {"title", "url", "description", "content", "query", "sense", "label"}
assert entries[0]["query"] == "강남" and entries[0]["content"].startswith("방송인 강남")
sense = entries[0]["sense"]                                                      # 고른 의미 + 배제어 + 고르지 않은 후보
assert sense["name"] == "방송인 강남" and "재건축" in sense["excluded_terms"]
assert sense["other_senses"] == [{"name": "지역 강남", "definition": "서울 강남구 일대", "related_terms": ["재건축", "강남역"]}]
asked = []                                                                       # 다시 실행: 같은 파일에 이어서, 이미 붙인 기사는 안 물어봄
keys = iter(["강남", "30", "2", "q"])
builtins.input = lambda prompt="": (asked.append(prompt), next(keys))[1]
with quiet():
    asyncio.run(B.main())
assert [p.strip() for p in asked] == ["검색어 (q=종료)>", "기간(최근 며칠, Enter = 30)>", "번호>", "검색어 (q=종료)>"], asked
assert len(json.loads(B._OUTPUT_PATH.read_text(encoding="utf-8"))) == 3
builtins.input = builtins_input
print("1 ok: build_labeled_set (의미 후보 선택 → 중복 제거 → 최신순 → 라벨 → JSON, 이어서 하기)")

# 2) eval_filter: V0(필터 없음) / V1(규칙) / V2(규칙 + LLM), 보류 정답 제외, 예전 형식 건너뜀
labeled = [
    {"article": {"title": "강남 이상화 예능", "link": "a", "content": "방송인 강남"}, "query": "강남", "sense": "방송인 강남", "label": "관련"},
    {"article": {"title": "강남역 재건축", "link": "b", "content": "강남역 재건축 강남 아파트"}, "query": "강남", "sense": "방송인 강남", "label": "무관"},
    {"article": {"title": "강남 맛집", "link": "c", "content": "강남 맛집 소개"}, "query": "강남", "sense": "방송인 강남", "label": "무관"},
    {"article": {"title": "강남 근황", "link": "d", "content": "예능 출연"}, "query": "강남", "sense": "방송인 강남", "label": "관련"},
    {"article": {"title": "애매", "link": "e", "content": "강남"}, "query": "강남", "sense": "방송인 강남", "label": "보류"},
    {"article": {"title": "예전 형식", "link": "f", "content": "x"}, "topic": "강남", "validation_terms": ["방송인"], "is_actually_relevant": True},
]
E.LABELED_SET_PATH = tmp / "eval.json"
E.LABELED_SET_PATH.write_text(json.dumps(labeled, ensure_ascii=False), encoding="utf-8")
E.pick_sense = lambda q, s=None: sel
llm_seen = []
async def fake_llm(articles, selected):
    llm_seen.extend(a.link for a in articles)
    return [Verdict(a.link, "무관" if a.link == "c" else "관련", "llm") for a in articles], []
E.llm_verify = fake_llm
v0, v1, v2 = None, None, None
orig = E.run_ablation
async def spy(labeled_articles, selected):
    global v0, v1, v2
    v0, v1, v2 = await orig(labeled_articles, selected)
    return v0, v1, v2
E.run_ablation = spy
out = io.StringIO()
with contextlib.redirect_stdout(out):
    asyncio.run(E.main())
text = out.getvalue()
assert "보류" in text and "예전 형식" in text and "전체 통합 결과" in text
assert (v0.tp, v0.fp, v0.fn, v0.tn) == (2, 2, 0, 0)                 # 모두 관련으로 예측
assert (v1.tp, v1.fp, v1.fn, v1.tn) == (2, 1, 0, 1)                 # 규칙이 b(재건축·강남역)만 거름
assert sorted(llm_seen) == ["a", "c", "d"]                          # 규칙이 미룬 기사만 LLM
assert (v2.tp, v2.fp, v2.fn, v2.tn) == (2, 0, 0, 2)                 # LLM이 c(맛집)도 거름
# evalset 형식(라벨 true/false, 기준이 항목 안에 있음)을 읽으면 저장된 기준을 그대로 쓴다
E.LABELED_SET_PATH = B._OUTPUT_PATH
E.pick_sense = lambda q, s=None: (_ for _ in ()).throw(AssertionError("저장된 기준을 써야 함"))
seen_sel = []
async def spy2(labeled_articles, selected):
    seen_sel.append(selected)
    return await orig(labeled_articles, selected)
E.run_ablation = spy2
llm_seen.clear()
with quiet():
    asyncio.run(E.main())
assert seen_sel[0].sense.name == "방송인 강남" and seen_sel[0].excluded_terms == sense["excluded_terms"]
assert seen_sel[0].other_senses[0].name == "지역 강남"
print("2 ok: eval_filter V0/V1/V2 (예전 형식 + evalset 형식)")

# 3) eval_retrieval: DB topic이 있으면 서비스 검색(search_topic), 키워드가 결과에 있으면 hit
class Rec: id = 7; selected = sel
R.get_db_topic = lambda q, s: Rec() if q == "강남" else None
db.get_topic_articles = lambda tid, statuses=None, **k: [Article(title="강남 라스", link="x", content="라디오스타 이상화")]
import qupict.retrieval as RT
calls = []
def fake_search_topic(topic_id, question, selected, k=5, use_hyde=True, use_multi_query=True, **kw):
    calls.append((use_hyde, use_multi_query))
    hit = db.ChunkHit(content="아내 이상화와 함께", source="body", similarity=0.6, article=Article(title="강남", link="x"))
    return [hit] if use_hyde else []                                 # HyDE를 켰을 때만 찾는다고 가정
RT.search_topic = fake_search_topic
gs = {"topic": "강남", "sense": "방송인 강남", "period_days": 50,
      "questions": [{"question": "강남의 아내는?", "expected_keywords": ["이상화"]},
                    {"question": "취미는?", "expected_keywords": ["포켓몬 카드"]}]}
with quiet():
    res = asyncio.run(R.run_eval("강남", gs))
assert res == {"Base": (0, 2), "HyDE only": (1, 2), "MultiQ only": (0, 2), "Both": (1, 2)}, res
assert len(calls) == 8 and set(calls) == {(False, False), (True, False), (False, True), (True, True)}
assert all("sense" in g and "validation_terms" not in g for g in R.GOLDEN_SETS.values())
assert sum(len(g["questions"]) for g in R.GOLDEN_SETS.values()) == 39            # 예전 골든 질문 그대로
print("3 ok: eval_retrieval (DB 모드, 조건 4가지, 골든셋 39문항 유지)")

# 4) debug_similarity: DB 모드, 기준값 없이 유사도와 O/- 표시
D._build_search_text = lambda q, c, use_hyde=True: q + (" 가상" if use_hyde else "")
class Emb:
    def embed_documents(self, texts): return [[0.1]] * len(texts)
import qupict.llm as L
L.get_embeddings = lambda: Emb()
RT.search_vectors = lambda tid, vecs, per: [
    db.ChunkHit(content="c", source="body", similarity=0.62, article=Article(title="높은 기사", link="1")),
    db.ChunkHit(content="c", source="image", similarity=0.31, article=Article(title="낮은 기사", link="2"))]
out = io.StringIO()
with contextlib.redirect_stdout(out):
    D.debug_db(Rec(), "질문")
text = out.getvalue()
assert "0.620 O  높은 기사" in text and "0.310 -  낮은 기사 [이미지]" in text
assert "V0 (HyDE 없음)" in text and "질문 가상" in text
print("4 ok: debug_similarity (DB 모드)")

# 5) _common.pick_sense: LLM 의미 후보에서 이름으로 찾는다 (DB topic·사람이 적은 값은 쓰지 않음)
import qupict.sense as S
S.load_sense_candidates = lambda q: (cands, "DB")
_common.get_db_topic = lambda q, s: (_ for _ in ()).throw(AssertionError("DB topic을 읽으면 안 됨"))
with quiet():
    picked = _common.pick_sense("강남", "방송인 강남")
assert picked.sense.name == "방송인 강남" and picked.excluded_terms == ["재건축", "강남역"]
assert _common.describe(sel) == B.describe(sel) == "강남 → 방송인 강남 (배제어: 재건축, 강남역)"
print("5 ok: 의미 정하기")
print("ALL PASS")
