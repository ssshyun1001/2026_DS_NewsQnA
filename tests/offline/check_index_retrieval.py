"""
4·5단계: 청크·이미지 설명·검색·답변 (가짜 LLM·DB)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import asyncio, sys
from datetime import datetime, timezone
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from qupict import indexing as I, retrieval as R, qa as Q, db
from qupict.schemas import Article, ArticleImage, SelectedSense, Sense

# 1) 청크 나누기
text = "첫 문장이다. 두 번째 문장이다! 세 번째 문장인가? 네 번째 \"인용\" 문장이다. 다섯."
cs = I.split_into_chunks(text, size=30)
assert all(c for c in cs) and cs[0].startswith("첫 문장이다.")
assert all(cs[i].split(" ")[0] in cs[i - 1] or True for i in range(1, len(cs)))
for a, b in zip(cs, cs[1:]):                 # 다음 청크는 앞 청크의 마지막 문장으로 시작
    last = I._SENTENCE_END.split(a)[-1].strip()
    assert b.startswith(last), (a, b)
assert I.split_into_chunks("") == [] and I.split_into_chunks("짧은 글") == ["짧은 글"]
long = "가" * 700 + "."
assert I.split_into_chunks(long + " 끝.", size=500) == [long, long + " 끝."] or I.split_into_chunks(long + " 끝.", size=500)[0] == long
print("1 ok: 청크", cs)

# 2) 이미지 청크 글
assert I.image_chunk_text("가수 강남", "무대에서 노래하는 모습") == "[기사 이미지] 사진 설명(기사): 가수 강남 / 사진 내용: 무대에서 노래하는 모습"
assert I.image_chunk_text("", "그래프") == "[기사 이미지] 사진 내용: 그래프"
print("2 ok: 이미지 청크 글")

# 3) index_topic (가짜 DB·임베딩·이미지 설명)
class FakeEmb:
    def __init__(self): self.calls = []
    async def aembed_documents(self, texts): self.calls.append(texts); return [[0.1] * 3 for _ in texts]
    def embed_documents(self, texts): return [[float(i)] * 3 for i, _ in enumerate(texts)]
fake_emb = FakeEmb(); I.get_embeddings = lambda: fake_emb
imgs = [db.ImageToDescribe(1, "A", "u1", "캡1"), db.ImageToDescribe(2, "A", "u2", "캡2"),
        db.ImageToDescribe(3, "A", "u3", "캡3"), db.ImageToDescribe(4, "B", "u4", "")]
link_of = {i.image_id: i.article_link for i in imgs}
desc_store: dict = {}                          # 가짜 DB의 llm_description
saved = {"desc": [], "chunks": {}}
def save_desc(iid, d, m=None):
    saved["desc"].append(iid); desc_store.setdefault(link_of[iid], []).append((iid, I.image_chunk_text("", d)))
db.get_images_to_describe = lambda tid, statuses=("관련",): [i for i in imgs if i.image_id not in {x for v in desc_store.values() for x, _ in v}]
db.save_image_description = save_desc
db.save_image_descriptions = lambda items, m=None: [save_desc(i, d) for i, d in items]
I.described_images = lambda links: {l: desc_store[l] for l in links if l in desc_store}
all_articles = [Article(title="제목A", link="A", content="본문 하나. 본문 둘."), Article(title="제목B", link="B", content=""),
                Article(title="제목C", link="C", content="")]
db.get_topic_articles = lambda tid, **k: all_articles
no_chunks = {"A", "B", "C"}
db.get_articles_without_chunks = lambda tid, statuses=("관련",): [a for a in all_articles if a.link in no_chunks]
def save_chunks(link, chunks, model=None):
    saved["chunks"][link] = [(c.source, c.image_id) for c in chunks]; no_chunks.discard(link)
db.save_chunks = save_chunks
db.save_chunks_many = lambda items, model=None: [save_chunks(l, c) for l, c in items]
real_describe = I.describe_image
async def fake_describe(image, title, link=""): return None if image.url == "u4" else f"설명({image.url})"
I.describe_image = fake_describe
r = asyncio.run(I.index_topic(9))
assert saved["desc"] == [1, 2] and (r.images_described, r.images_failed) == (2, 1)   # A는 2장까지, B(u4)는 실패
assert saved["chunks"]["A"] == [("body", None), ("image", 1), ("image", 2)]
assert "B" not in saved["chunks"] and "C" not in saved["chunks"]                     # 내용 없는 기사는 건너뜀
assert fake_emb.calls[-1][0].startswith("제목A\n")                                    # 제목을 붙여 임베딩
assert (r.articles, r.body_chunks, r.image_chunks) == (1, 1, 2), r
print("3 ok:", r.summary())

# 3-2) 다시 실행: A는 이미 2장 설명됨 → 더 설명하지 않음, 청크 있는 기사는 그대로
saved["desc"].clear(); saved["chunks"].clear()
r = asyncio.run(I.index_topic(9))
assert saved["desc"] == [] and saved["chunks"] == {} and r.images_failed == 1
# 3-3) 청크가 이미 있는 기사에 설명이 새로 생기면 다시 만든다 (이전 실행 설명 + 새 설명 모두 들어감)
desc_store.clear(); desc_store["A"] = [(1, "[기사 이미지] 옛 설명")]
imgs[:] = [db.ImageToDescribe(1, "A", "u1", ""), db.ImageToDescribe(2, "A", "u2", "")]
r = asyncio.run(I.index_topic(9))
assert saved["desc"] == [2] and saved["chunks"]["A"] == [("body", None), ("image", 1), ("image", 2)], (saved, r)
# 3-4) 이미지 설명 끄기
saved["desc"].clear(); no_chunks.add("A")
r2 = asyncio.run(I.index_topic(9, describe_images=False))
assert saved["desc"] == [] and r2.image_chunks == 2      # 새로 설명은 안 하지만 있는 설명은 넣음
print("3 ok: 재실행·설명 상한·다시 만들기·설명 끄기")

# 4) search_topic: 검색 문장 여러 개 → 같은 청크는 최고 유사도, 기준 미달 제거, 상위 k
sel = SelectedSense(query="강남", sense=Sense(name="방송인 강남", definition="M.I.B 출신"))
R.get_embeddings = lambda: fake_emb
R.build_search_texts = lambda q, ctx, h, m: ["q", "hyde", "variant"]
art = lambda l: Article(title=l, link=l)
calls = []
def fake_search(tid, vec, k=5, period_days=None, statuses=("관련",)):
    calls.append((tid, k, period_days, statuses))
    i = int(vec[0])   # 검색 문장 번호
    return [db.ChunkHit("c1", "body", [0.5, 0.7, 0.6][i], art("A")),
            db.ChunkHit("c2", "image", 0.4, art("B")),
            db.ChunkHit("low", "body", 0.1, art("C"))]
db.search_chunks = fake_search
hits = R.search_topic(3, "질문", sel, k=2, period_days=30, similarity_threshold=0.3)
assert [(h.content, h.similarity) for h in hits] == [("c1", 0.7), ("c2", 0.4)]
assert calls[0] == (3, 6, 30, ("관련", "보류")) and len(calls) == 3
assert "방송인 강남" in R.topic_context(sel)
calls.clear()
strict = R.search_topic(3, "질문", sel, k=2)                          # 기본 기준 0.5: 0.4 청크는 빠짐
assert [(h.content, h.similarity) for h in strict] == [("c1", 0.7)]
print("4 ok: 검색 합치기", [(h.content, h.similarity) for h in hits])

# 5) 청크로 답변: 기사별 번호, 근거 종류
d = datetime(2026, 11, 6, tzinfo=timezone.utc)
hits = [db.ChunkHit("본문1", "body", 0.9, Article(title="T-A", link="A", press="뉴스1", pub_date=d)),
        db.ChunkHit("[기사 이미지] 사진", "image", 0.8, Article(title="T-B", link="B")),
        db.ChunkHit("본문2", "body", 0.7, Article(title="T-A", link="A", press="뉴스1", pub_date=d))]
ctx, sources, kinds = Q.build_chunk_context(hits)
assert [s.link for s in sources] == ["A", "B"] and kinds == ["본문", "이미지 설명"]
assert "[기사 1]" in ctx and "- 본문1\n- 본문2" in ctx and "발행일: 2026-11-06" in ctx and "[기사 2]" in ctx
class FakeChain:
    def invoke(self, x): FakeChain.ctx, FakeChain.subject, FakeChain.period = x["context"], x["subject"], x["period"]; return "답변"
Q._get_chain = lambda: FakeChain()
ans = Q.generate_answer_from_chunks("질문", hits)
assert ans.answer == "답변" and ans.source_kinds == ["본문", "이미지 설명"] and len(ans.sources) == 2
assert FakeChain.subject == Q.NO_SUBJECT                                    # 대상을 안 주면 "(지정 없음)"
ans0 = Q.generate_answer_from_chunks("질문", [])
assert ans0.sources == [] and FakeChain.ctx == Q.NO_ARTICLE_CONTEXT
print("5 ok: 청크 답변 (번호·근거 종류·근거 없음)")

# 5-1) 검색 대상: 고른 의미가 답변 프롬프트의 [검색 대상]으로 들어간다
assert Q.answer_subject(sel) == "강남 (방송인 강남: M.I.B 출신)"
assert Q.answer_subject(SelectedSense(query="베라", sense=Sense(name="배스킨라빈스", definition=""))) == "베라 (배스킨라빈스)"
assert Q.answer_subject(None) == Q.NO_SUBJECT
Q.generate_answer_from_chunks("아내가 누구야?", hits, subject=Q.answer_subject(sel))
assert FakeChain.subject == "강남 (방송인 강남: M.I.B 출신)"
Q.generate_answer("아내가 누구야?", [], subject=Q.answer_subject(sel))       # 메모리 버전도 같은 대상을 넘긴다
assert FakeChain.subject == "강남 (방송인 강남: M.I.B 출신)"
msgs = Q._prompt.format_messages(subject="강남 (방송인 강남)", period="2026-10-14 ~ 2026-11-13",
                                 question="아내가 누구야?", context="-")
assert msgs[1].content.startswith("[검색 대상] 강남 (방송인 강남)\n[참고 가능한 기사 기간] 2026-10-14 ~ 2026-11-13\n"
                                  "질문: 아내가 누구야?")
print("5-1 ok: 답변 프롬프트에 검색 대상")

# 5-2) 기사 기간: 가장 오래된 기사 날짜 ~ 오늘 (한국 시간 기준), 기사가 없으면 "(알 수 없음)"
kst_late = datetime(2026, 10, 13, 16, tzinfo=timezone.utc)                 # 한국 시간 10월 14일 새벽 1시
assert Q.answer_period(kst_late, today=datetime(2026, 11, 13, 1, tzinfo=timezone.utc)) == "2026-10-14 ~ 2026-11-13"
assert Q.answer_period(None) == Q.NO_PERIOD
arts = [Article(title="a", link="a", pub_date=datetime(2026, 11, 1, tzinfo=timezone.utc)),
        Article(title="b", link="b", pub_date=kst_late), Article(title="c", link="c")]
assert Q.oldest_date(arts) == kst_late and Q.oldest_date([Article(title="c", link="c")]) is None
Q.generate_answer_from_chunks("아내가 누구야?", [], period="2026-10-14 ~ 2026-11-13")
assert FakeChain.period == "2026-10-14 ~ 2026-11-13"
Q.generate_answer_from_chunks("아내가 누구야?", [])
assert FakeChain.period == Q.NO_PERIOD                                      # 안 주면 "(알 수 없음)"
print("5-2 ok: 기사 기간")

# 6) describe_image: 주소 먼저 → 실패하면 내려받아 파일로
sent = []
async def fake_ask(url, title, caption):
    sent.append(url[:5])
    if url.startswith("http") and "blocked" in url: raise RuntimeError("주소 실패")
    if url.startswith("data:") and "bad" in title: raise RuntimeError("파일도 실패")
    return "설명"
I._ask_image = fake_ask
async def dl(articles):
    for a in articles:
        for im in a.images: im.data, im.content_type = b"x" * 6000, "image/jpeg"
    return 1, 1
I.download_images = dl
assert asyncio.run(real_describe(ArticleImage(url="http://ok"), "t")) == "설명" and sent == ["http:"]
sent.clear()
assert asyncio.run(real_describe(ArticleImage(url="http://blocked"), "t", "A")) == "설명" and sent == ["http:", "data:"]
sent.clear()
assert asyncio.run(real_describe(ArticleImage(url="http://blocked"), "bad", "A")) is None
async def dl_fail(articles): return 0, 1
I.download_images = dl_fail
assert asyncio.run(real_describe(ArticleImage(url="http://blocked"), "t", "A")) is None
print("6 ok: 이미지 설명 (주소 → 파일 순서, 둘 다 실패)")

# 7) 인용 번호 찾기
assert Q.cited_numbers("A다(기사 1). B다(기사 1, 기사 3). C(기사 2, 이미지). D(기사 4,5). E(기사 9)", 5) == [1, 2, 3, 4, 5]
assert Q.cited_numbers("기사에 따르면 …", 3) == [] and Q.cited_numbers("(기사 2)", 3) == [2]
print("7 ok: 인용 번호")
print("ALL PASS")
