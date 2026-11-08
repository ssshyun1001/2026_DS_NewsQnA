"""
2·3단계: 중복 제거·규칙 점수·LLM 판정 (가짜 LLM·DB)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import asyncio, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from qupict import filtering as F, db
from qupict.schemas import Article, SelectedSense, Sense

gangnam = SelectedSense(
    query="강남",
    sense=Sense(name="방송인 강남", definition="M.I.B 출신 방송인", aliases=[], related_terms=["이상화", "M.I.B", "예능"]),
    excluded_terms=["강남역", "재건축", "압구정"],
    other_senses=[Sense(name="지역 강남", definition="서울 강남구 일대", related_terms=["강남역", "재건축", "압구정"])],
)
A = lambda link, title, content="", emb=None: Article(title=title, link=link, content=content, embedding=emb)

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
assert F.find_duplicates(arts, threshold=0.9) == {"short": "long"}
old = A("old", "t", "x", e1)   # 이미 판정됨 (본문은 짧아도 대표 유지)
assert F.find_duplicates(arts, existing=[old], threshold=0.9) == {"long": "old", "short": "old"}
# 라벨 붙은 기사(keep)는 짧아도 대표로 남고, 이미 판정한 기사와 비슷해도 중복이 되지 않는다
assert F.find_duplicates(arts, threshold=0.9, keep={"short"}) == {"long": "short"}
assert F.find_duplicates(arts, existing=[old], threshold=0.9, keep={"short"}) == {"long": "old"}
print("3 ok: 중복 제거 (라벨 붙은 기사 지키기 포함)")

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
F.get_chat_model = lambda temperature=0: fake
F.LLM_BATCH_SIZE = 3
items = [A(f"l{i}", f"강남 기사 {i}") for i in range(7)]       # 3 + 3 + 1
verdicts, unjudged = asyncio.run(F.llm_verify(items, gangnam))
assert len(fake.calls) == 3
assert {v.link for v in verdicts} == {"l0", "l2", "l3", "l5", "l6"} and unjudged == ["l1", "l4"]   # 각 배치의 1번 누락
assert all(v.stage == "llm" and v.model for v in verdicts)
system = fake.calls[0][0][1]
assert "방송인 강남" in system and "지역 강남" in system and "강남역" in system   # 다른 의미와 대조
fake2 = FakeLLM(fail_on=1); F.get_chat_model = lambda temperature=0: fake2
verdicts, unjudged = asyncio.run(F.llm_verify(items, gangnam))
assert set(unjudged) >= {"l3", "l4", "l5"}                     # 실패한 배치는 판정하지 않음
print("4 ok: LLM 검증 (배치, 누락, 범위 밖, 실패)")

# 5) 전체 흐름(메모리): 중복 → 규칙 → LLM
F.get_chat_model = lambda temperature=0: FakeLLM()
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
assert stages["rep"][0] == "llm" and out.unjudged == ["ok"]   # 가짜 LLM은 1번 답을 일부러 빼먹음 → 판정 못 함
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
F._labeled_links = lambda tid: set()
out = asyncio.run(F.filter_topic(5, gangnam, redo=True))
assert saved["reset"] == 5 and saved["emb"] == ["e1", "n1"]   # n2는 이미 임베딩이 있어서 다시 계산·저장 안 함
assert saved["dups"] == {}
assert ("n2", "무관", "rule") in saved["judg"] and any(l == "n1" and s == "llm" for l, _, s in saved["judg"])
db.get_topic_articles = lambda *a, **k: []
assert asyncio.run(F.filter_topic(5, gangnam)).verdicts == []      # 새 기사 없음
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
assert saved["dups"] == dups and saved["emb"] == ["d_lab"] and emb_calls == [1]
assert "judg" not in saved                                            # 판정은 저장하지 않음
db.get_topic_articles = lambda *a, **k: []
assert asyncio.run(F.dedup_topic(9)) == {}
print("7 ok: dedup_topic (판정 없이 중복만, 라벨 기사 지킴)")
print("ALL PASS")
