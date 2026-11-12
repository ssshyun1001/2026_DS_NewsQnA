"""
2·3단계: 중복 제거·규칙 점수·LLM 판정 (가짜 LLM·DB)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import asyncio, dataclasses, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from qupict import filtering as F, db
from qupict.schemas import Article, SelectedSense, Sense

gangnam = SelectedSense(
    query="강남",
    sense=Sense(name="방송인 강남", definition="M.I.B 출신 방송인", aliases=[], related_terms=["이상화", "M.I.B", "예능"]),
    excluded_terms=["강남역", "재건축", "압구정"],
    other_senses=[Sense(name="지역 강남", definition="서울 강남구 일대", related_terms=["강남역", "재건축", "압구정"])],
)
# emb는 DB용 임베딩(제목+본문)과 중복 제거용 본문 앞 문장 임베딩(lead_embedding)에 같이 넣는다
A = lambda link, title, content="", emb=None: Article(title=title, link=link, content=content, embedding=emb, lead_embedding=emb)

# 1) 규칙 점수: 제목 2점 / 본문 1점, 배제어는 빼기, 정규화(M.I.B ↔ MIB)
r = F.rule_check(A("1", "강남, 아내 이상화와 예능 출연", "MIB 시절 이야기"), gangnam)
assert (r.score, r.mentioned, r.related_hits, r.excluded_hits) == (5, True, ["이상화", "M.I.B", "예능"], []), r
r = F.rule_check(A("2", "강남역 재건축 단지 시세", "압구정 아파트값도 올라 강남 집값"), gangnam)
assert (r.score, r.excluded_hits) == (-5, ["강남역", "재건축", "압구정"]), r
print("1 ok: 규칙 점수", r)

# 2) 규칙 판정: 검색어 없음 → 무관 / 점수 -2 이하 → 무관 / 나머지 → LLM으로(None)
assert F.rule_verdict(A("3", "성신여대 축제", "서울 여대들이"), gangnam).reason.startswith("검색어·별칭")
v = F.rule_verdict(A("2", "강남역 재건축 단지 시세", "압구정 아파트값도 올라 강남 집값"), gangnam)
assert (v.status, v.stage) == ("무관", "rule") and "점수 -5" in v.reason
assert F.rule_verdict(A("4", "강남 맛집", "강남역 근처"), gangnam) is None          # 점수 -1 → LLM
assert F.rule_verdict(A("5", "가수 강남 근황", ""), gangnam) is None                # 점수 0 → LLM
print("2 ok: 규칙 판정")

# 3) 중복 제거: 비슷한 기사 중 가장 긴 것이 대표, 이미 판정한 기사(existing)는 항상 대표
e1, e2 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
near = [0.99, 0.05, 0.0]
arts = [A("short", "t", "x" * 10, near), A("long", "t", "x" * 50, e1), A("other", "t", "x" * 20, e2)]
assert F.find_duplicates(arts, content_threshold=0.9, by="content") == {"short": "long"}
old = A("old", "t", "x", e1)   # 이미 판정됨 (본문은 짧아도 대표 유지)
assert F.find_duplicates(arts, existing=[old], content_threshold=0.9, by="content") == {"long": "old", "short": "old"}
# 라벨 붙은 기사(keep)는 짧아도 대표로 남고, 이미 판정한 기사와 비슷해도 중복이 되지 않는다
assert F.find_duplicates(arts, content_threshold=0.9, keep={"short"}, by="content") == {"long": "short"}
assert F.find_duplicates(arts, existing=[old], content_threshold=0.9, keep={"short"}, by="content") == {"long": "old"}
# 제목 기준(by="title"): title_embedding으로 비교한다 (본문 앞 문장 임베딩이 달라도 제목이 비슷하면 중복)
T = lambda link, content, emb, temb: Article(title="t", link=link, content=content, lead_embedding=emb, title_embedding=temb)
tarts = [T("a", "x" * 10, e1, e1), T("b", "x" * 50, e2, near), T("c", "x" * 20, e2, e2)]
assert F.find_duplicates(tarts, title_threshold=0.9, by="title") == {"a": "b"}
assert F.find_duplicates(tarts, content_threshold=0.9, by="content") == {"c": "b"}
# both: and = 제목·본문 둘 다 넘어야 중복 / or = 하나만 넘어도 중복, 기준별 임계값은 따로
both = lambda **kw: F.find_duplicates(tarts, by="both", title_threshold=0.9, content_threshold=0.9, **kw)
assert both(rule="and") == {}                       # a-b는 제목만, c-b는 본문만 비슷 → and면 중복 없음
assert both(rule="or") == {"a": "b", "c": "b"}
tb = [T("p", "x" * 50, e1, e1), T("q", "x" * 10, near, near)]
assert F.find_duplicates(tb, by="both", rule="and", title_threshold=0.9, content_threshold=0.9) == {"q": "p"}
assert F.find_duplicates(tb, by="both", rule="and", title_threshold=0.9, content_threshold=0.999) == {}  # 본문 임계값만 올림
try:
    F.find_duplicates(tarts, rule="xor"); raise AssertionError("잘못된 rule이 통과함")
except ValueError:
    pass
try:
    F.find_duplicates(tarts, by="body"); raise AssertionError("잘못된 by가 통과함")
except ValueError:
    pass
# embed_titles는 제목만 임베딩한다
class RecEmb:
    seen = []
    async def aembed_documents(self, texts): RecEmb.seen += texts; return [[1.0, 0.0, 0.0] for _ in texts]
_orig_emb = F.get_embeddings
F.get_embeddings = lambda: RecEmb()
x = A("x", "제목만", "본문은 안 들어감")
asyncio.run(F.embed_for_dedup([x], by="title"))
assert RecEmb.seen == ["제목만"] and x.title_embedding == [1.0, 0.0, 0.0] and x.embedding is None
# content 기준은 본문 앞 5문장만 임베딩한다 (제목·6번째 문장 이후는 안 들어감, DB용 embedding은 건드리지 않음)
body = '금리가 3.5%로 올랐다. 한은은 "물가가 걱정된다."고 밝혔다! 정말?\n둘째 문단. 다섯째. 여섯째는 빠진다.'
lead5 = '금리가 3.5%로 올랐다. 한은은 "물가가 걱정된다."고 밝혔다! 정말? 둘째 문단. 다섯째.'
assert F.lead_text(Article(title="제목", link="l", content=body)) == lead5
assert F.lead_text(Article(title="제목", link="l", content=body), n=2) == '금리가 3.5%로 올랐다. 한은은 "물가가 걱정된다."고 밝혔다!'
assert F.lead_text(Article(title="본문 없음", link="l")) == "본문 없음"          # 본문이 비면 제목
RecEmb.seen = []
y = Article(title="제목", link="y", content=body)
asyncio.run(F.embed_for_dedup([y], by="content"))
assert RecEmb.seen == [lead5] and y.lead_embedding is not None and y.title_embedding is None and y.embedding is None
RecEmb.seen = []
z = Article(title="제목", link="z", content=body)
asyncio.run(F.embed_for_dedup([z], by="both"))
assert sorted(RecEmb.seen) == sorted(["제목", lead5])
F.get_embeddings = _orig_emb

# noun 기준: 드문 고유명사를 같이 쓰는 기사끼리 비슷함 (모든 기사에 나오는 "강남"은 가중치가 낮음)
_orig_nouns = F.extract_nouns
F.extract_nouns = lambda text: text.split()          # 형태소 분석 대신 공백으로 (결정적인 테스트)
# 고유명사는 제목에서만 뽑는다 (본문 단어는 무시되는지도 확인: 본문에 같은 단어를 넣어도 영향 없음)
N = lambda link, words, n=10: Article(title=words, link=link, content="편스토랑 도쿄 " + "x" * n)
nouns = [N("n1", "강남 이상화 편스토랑 도쿄", 50), N("n2", "강남 이상화 도쿄 편스토랑", 10),
         N("n3", "강남 유튜브 강나미", 30), N("n4", "강남 압구정 재건축", 20)]
vec = F.noun_tfidf(nouns)
cos = lambda x, y: float((F._unit(np.array([vec[x]])) @ F._unit(np.array([vec[y]])).T)[0, 0])
assert cos("n1", "n2") > 0.9 and cos("n1", "n3") < 0.2 and cos("n3", "n4") < 0.2
assert F.find_duplicates(nouns, by="noun", noun_threshold=0.4) == {"n2": "n1"}
# 임베딩 기준과 섞기: noun은 임베딩이 없어도 되고, and면 임베딩 기준도 넘어야 한다
for a_, t_ in zip(nouns, ([1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.0, -1.0])): a_.title_embedding = t_
assert F.find_duplicates(nouns, by="title,noun", rule="and", title_threshold=0.9, noun_threshold=0.4) == {}
assert F.find_duplicates(nouns, by="title,noun", rule="or", title_threshold=0.9, noun_threshold=0.4) == {"n2": "n1"}
assert F._dedup_criteria("all") == ["title", "content", "noun"] and F._dedup_criteria("both") == ["title", "content"]
assert F._dedup_criteria(" Title , noun ") == ["title", "noun"]
F.extract_nouns = _orig_nouns
print("3 ok: 중복 제거 (라벨 붙은 기사 지키기, 제목/본문 앞 문장/고유명사 기준 포함)")

# 아래 흐름 테스트(5~7)는 가짜 임베딩이 본문 기준·0.9에 맞춰져 있으므로 고정 (.env 값과 무관하게)
F.settings = dataclasses.replace(F.settings, dedup_by="content", dedup_content_threshold=0.9)

# 4) LLM 검증: 배치 나누기, 번호 매칭, 빠진 번호·범위 밖 번호, 호출 실패
class FakeLLM:
    def __init__(self, fail_on=None): self.fail_on, self.calls = fail_on, []
    def with_structured_output(self, schema): return self
    async def ainvoke(self, msgs):
        self.calls.append(msgs)
        body = msgs[1][1]
        n = body.count("\n제목: ") + body.startswith("아래") * 0
        n = body.count("] 제목: ")
        if self.fail_on is not None and len(self.calls) - 1 == self.fail_on: raise TimeoutError()
        vs = [F._LLMVerdict(index=i, status="관련" if i % 2 == 0 else "무관", reason=f"근거{i}") for i in range(n) if i != 1]
        vs.append(F._LLMVerdict(index=99, status="관련", reason="없는 번호"))
        return F._LLMVerdictBatch(verdicts=vs)
fake = FakeLLM()
F.get_structured_model = lambda schema, temperature=0: fake
F.LLM_BATCH_SIZE = 3
F.LLM_RETRY_WAIT_SECONDS = 0                                    # 테스트에서는 쉬지 않고 바로 다시 묻는다
items = [A(f"l{i}", f"강남 기사 {i}") for i in range(7)]       # 3 + 3 + 1
verdicts, unjudged = asyncio.run(F.llm_verify(items, gangnam))
# 첫 판정 3번(배치) + 빠진 l1·l4를 1건씩 다시 묻기. 1건짜리 요청에서는 0번 답이 오므로 둘 다 판정된다
assert len(fake.calls) == 3 + 2, len(fake.calls)
assert {v.link for v in verdicts} == {f"l{i}" for i in range(7)} and unjudged == []
assert all(v.stage == "llm" and v.model for v in verdicts)
system = fake.calls[0][0][1]
assert "방송인 강남" in system and "지역 강남" in system and "강남역" in system   # 다른 의미와 대조
fake2 = FakeLLM(fail_on=1); F.get_structured_model = lambda schema, temperature=0: fake2
verdicts, unjudged = asyncio.run(F.llm_verify(items, gangnam))
assert {v.link for v in verdicts} == {f"l{i}" for i in range(7)} and unjudged == []   # 실패한 배치도 다시 물어 판정
assert all(v.status in ("관련", "무관") and v.model == F.settings.chat_model for v in verdicts)   # 모두 LLM이 직접 고른 판정


class StubbornLLM(FakeLLM):
    """1건짜리 요청에 처음 4번은 빈 답을 주다가 그 뒤에 답한다 → 답할 때까지 계속 다시 묻는지 확인."""
    single = 0
    async def ainvoke(self, msgs):
        result = await super().ainvoke(msgs)
        if msgs[1][1].count("] 제목: ") == 1:
            StubbornLLM.single += 1
            if StubbornLLM.single <= 4 * 3:            # 답 없는 기사 3건 × 4회차 동안 빈 답
                return F._LLMVerdictBatch(verdicts=[])
        return result
stubborn = StubbornLLM(); F.get_structured_model = lambda schema, temperature=0: stubborn
verdicts, unjudged = asyncio.run(F.llm_verify(items, gangnam))
# l1·l4: 배치 답에서 빠짐, l6: 마지막 배치가 1건이라 처음부터 빈 답 → 5회차에서 답을 받아 모두 판정
assert unjudged == [] and {v.link for v in verdicts} == {f"l{i}" for i in range(7)}, (unjudged, len(verdicts))
assert all(v.status in ("관련", "무관") and v.model == F.settings.chat_model for v in verdicts)
assert len(stubborn.calls) >= 3 + 3 * 4, len(stubborn.calls)                         # 빈 답이 계속돼도 포기하지 않고 여러 회차 다시 물음
print("4 ok: LLM 검증 (배치, 누락·실패는 답할 때까지 다시 묻기, 판정 없이 남기지 않음)")

# 5) 전체 흐름(메모리): 중복 → 규칙 → LLM
F.get_structured_model = lambda schema, temperature=0: FakeLLM()
F.LLM_BATCH_SIZE = 8
class FakeEmb:
    async def aembed_documents(self, texts): return [[float(len(t) % 7), 1.0, 0.0] for t in texts]
F.get_embeddings = lambda: FakeEmb()
arts = [A("dup", "강남 이상화 예능", "x" * 5, e1), A("rep", "강남 이상화 예능", "x" * 99, e1),
        A("area", "강남역 재건축", "압구정 강남", e2), A("none", "성신여대", "여대", [0, 0, 1.0]),
        A("ok", "강남 근황", "예능", [0.5, 0.5, 0.7])]
out = asyncio.run(F.filter_articles(arts, gangnam))
assert out.duplicates == {"dup": "rep"}
stages = {v.link: (v.stage, v.status) for v in out.verdicts}
assert stages["area"] == ("rule", "무관") and stages["none"] == ("rule", "무관")
assert stages["rep"][0] == "llm" and stages["ok"][0] == "llm" and out.unjudged == []   # 빠진 답은 다시 물어 판정
print("5 ok:", out.summary())

# 6) DB 버전: 새 기사만, 임베딩 없는 기사는 계산해서 저장, 결과 저장
class DistinctEmb:   # 기사마다 서로 다른(직교) 벡터 → 중복 없음
    n = 0
    async def aembed_documents(self, texts):
        out = []
        for _ in texts:
            v = [0.0] * 8; v[DistinctEmb.n % 8] = 1.0; DistinctEmb.n += 1; out.append(v)
        return out
F.get_embeddings = lambda: DistinctEmb()
saved = {}
new = [A("n1", "강남 이상화 예능", "x" * 30), A("n2", "강남역 재건축", "압구정 강남", [0.0] * 7 + [1.0])]
existing = [A("e1", "강남 예능", "x" * 10, None)]
db.reset_topic_filter = lambda tid: saved.__setitem__("reset", tid)
db.get_topic_articles = lambda tid, period_days=None, statuses=None, include_duplicates=True, unjudged_only=False: new if unjudged_only else existing
db.save_article_embeddings = lambda arts: saved.__setitem__("emb", sorted(a.link for a in arts))
db.save_duplicates = lambda tid, d: saved.__setitem__("dups", d)
db.save_judgments = lambda tid, js: saved.__setitem__("judg", [(j.link, j.status, j.stage) for j in js])
copied = []
db.copy_duplicate_judgments = lambda tid: copied.append(tid) or 0    # 실제 DB에 쓰지 않게
F._labeled_links = lambda tid: set()
out = asyncio.run(F.filter_topic(5, gangnam, redo=True))
assert saved["reset"] == 5 and saved["emb"] == ["e1", "n1"]   # n2는 이미 임베딩이 있어서 다시 계산·저장 안 함
assert saved["dups"] == {}
assert ("n2", "무관", "rule") in saved["judg"] and any(l == "n1" and s == "llm" for l, _, s in saved["judg"])
assert copied == [5]                                          # 판정 저장 뒤 중복 기사에 대표 판정 복사
db.get_topic_articles = lambda *a, **k: []
assert asyncio.run(F.filter_topic(5, gangnam)).verdicts == []      # 새 기사 없음
assert copied == [5, 5]                                       # 새 기사가 없어도 빈 중복 판정은 채운다
print("6 ok: DB 버전 (재판정·임베딩 저장·결과 저장·새 기사 없음)")

# 7) dedup_topic: 판정 없이 중복 표시만 (LLM 0번), 라벨 붙은 기사는 지킴, 다시 돌려도 같음
v = [1.0, 0.0, 0.0]
long_, short_, lab = A("d_long", "t", "x" * 50, v), A("d_short", "t", "x" * 10, v), A("d_lab", "t", "x" * 5, None)
saved.clear()
db.get_topic_articles = lambda tid, period_days=None, statuses=None, include_duplicates=True, unjudged_only=False: [long_, short_, lab] if unjudged_only else []
F._labeled_links = lambda tid: {"d_lab"}
emb_calls = []
class SameEmb:
    async def aembed_documents(self, texts): emb_calls.append(len(texts)); return [[1.0, 0.0, 0.0] for _ in texts]
F.get_embeddings = lambda: SameEmb()
F.llm_verify = None                          # 부르면 바로 오류 → LLM을 안 부르는지 확인
dups = asyncio.run(F.dedup_topic(9))
assert dups == {"d_long": "d_lab", "d_short": "d_lab"}, dups          # 라벨 붙은 기사가 대표
assert saved["dups"] == dups and saved["emb"] == ["d_lab"]
assert emb_calls == [1, 1]          # DB용 임베딩 1건(d_lab) + 중복 제거용 본문 앞 문장 임베딩 1건(d_lab)
assert "judg" not in saved                                            # 판정은 저장하지 않음
assert copied[-1] == 9                                                # 이미 판정된 대표의 중복이면 판정을 바로 받게
db.get_topic_articles = lambda *a, **k: []
assert asyncio.run(F.dedup_topic(9)) == {}
print("7 ok: dedup_topic (판정 없이 중복만, 라벨 기사 지킴)")
print("ALL PASS")
