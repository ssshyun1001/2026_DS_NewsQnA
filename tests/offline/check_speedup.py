"""
속도 최적화가 결과를 바꾸지 않는지 — 예전 방식(차례로 실행)과 새 방식(동시 실행·한꺼번에)의 결과 비교

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import asyncio, hashlib, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from qupict import collect, db, indexing as I, qa as Q, retrieval as R
from qupict.concurrency import map_parallel, run_parallel
from qupict.llm import aembed_in_batches
from qupict.schemas import Article, NewsMeta, Sense, SelectedSense


def vec_of(text: str) -> list[float]:          # 글마다 정해진 가짜 임베딩 (같은 글 → 같은 벡터)
    h = hashlib.sha1(text.encode()).digest()
    return [b / 255 for b in h[:4]]


class FakeEmb:
    def __init__(self, delay=0.0): self.calls, self.delay = [], delay
    async def aembed_documents(self, texts):
        self.calls.append(list(texts)); await asyncio.sleep(self.delay); return [vec_of(t) for t in texts]
    def embed_documents(self, texts): return [vec_of(t) for t in texts]


# 1) 동시 실행 도우미: 결과는 넣은 순서, 예외는 그대로, 실제로 동시에
assert run_parallel(lambda: 1, lambda: 2, lambda: 3) == [1, 2, 3]
assert map_parallel(lambda x: x * 10, [3, 1, 2]) == [30, 10, 20]
try:
    run_parallel(lambda: 1, lambda: 1 / 0); raise AssertionError("예외가 안 남")
except ZeroDivisionError:
    pass
start = time.perf_counter(); map_parallel(lambda _: time.sleep(0.2), range(4)); took = time.perf_counter() - start
assert took < 0.5, took                                     # 차례로면 0.8초
print("1 ok: run_parallel / map_parallel")

# 2) 임베딩 나눠 보내기: 결과 순서 그대로, 요청 크기는 batch_size 이하, 요청은 동시에
emb = FakeEmb(delay=0.2)
texts = [f"글 {i}" for i in range(250)]
start = time.perf_counter(); vecs = asyncio.run(aembed_in_batches(emb, texts, batch_size=100)); took = time.perf_counter() - start
assert vecs == [vec_of(t) for t in texts] and [len(c) for c in emb.calls] == [100, 100, 50]
assert took < 0.4, took                                     # 차례로면 0.6초
assert asyncio.run(aembed_in_batches(emb, [])) == []
print("2 ok: aembed_in_batches")

# 3) 청크 만들기: 예전(기사마다 임베딩 요청) == 지금(한꺼번에) — 청크 내용·순서·벡터·이미지 id 모두 같다
I.get_embeddings = lambda: FakeEmb()
arts = [Article(title="제목A", link="A", content="첫 문장이다. " * 60),
        Article(title="제목B", link="B", content=""),                       # 본문 없음 + 이미지 있음
        Article(title="제목C", link="C", content=""),                       # 아무것도 없음 → 빠짐
        Article(title="제목D", link="D", content="짧은 본문. 둘째 문장.")]
image_chunks = {"A": [(11, "[기사 이미지] a1"), (12, "[기사 이미지] a2")], "B": [(21, "[기사 이미지] b1")]}


async def old_way():
    out = {}
    for a in arts:
        bodies = I.split_into_chunks(a.content); images = image_chunks.get(a.link, [])
        texts = bodies + [t for _, t in images]
        if not texts:
            continue
        vectors = await I.embed_chunks(a.title, texts)
        out[a.link] = ([db.Chunk(t, v) for t, v in zip(bodies, vectors)]
                       + [db.Chunk(t, v, source="image", image_id=i) for (i, t), v in zip(images, vectors[len(bodies):])])
    return out

assert asyncio.run(I.build_article_chunks(arts, image_chunks)) == asyncio.run(old_way())
assert list(asyncio.run(I.build_article_chunks(arts, image_chunks))) == ["A", "B", "D"]
# 임베딩 요청 묶음도 예전과 같다 (기사마다 1번): 묶음이 바뀌면 실제 API는 값이 아주 조금 달라진다
slow = FakeEmb(delay=0.2); I.get_embeddings = lambda: slow
start = time.perf_counter(); asyncio.run(I.build_article_chunks(arts, image_chunks)); took = time.perf_counter() - start
old = FakeEmb(); I.get_embeddings = lambda: old; asyncio.run(old_way())
assert sorted(slow.calls) == sorted(old.calls) and len(slow.calls) == 3, slow.calls
assert took < 0.4, took                                      # 기사 3건 요청을 동시에 (차례로면 0.6초)
print("3 ok: build_article_chunks (예전과 같은 결과·같은 요청 묶음, 동시에)")

# 4) 벡터 검색: 문장별 DB 검색을 동시에 해도 합친 결과(유사도·순서)가 예전과 같다
from qupict.db import ChunkHit
def hit(link, content, sim): return ChunkHit(content, "body", sim, Article(title=link, link=link))
table = {0: [hit("a", "x", 0.9), hit("b", "y", 0.5)], 1: [hit("a", "x", 0.95), hit("c", "z", 0.7)],
         2: [hit("b", "y", 0.6), hit("d", "w", 0.6)]}
def fake_search(topic_id, vec, k=5, period_days=None, statuses=("관련",)):
    time.sleep(0.2); return table[int(vec[0])]
db.search_chunks = fake_search
def old_merge(vectors):
    best = {}
    for v in vectors:
        for h in table[int(v[0])]:
            key = (h.article.link, h.content)
            if key not in best or h.similarity > best[key].similarity:
                best[key] = h
    return sorted(best.values(), key=lambda h: -h.similarity)
start = time.perf_counter(); got = R.search_vectors(1, [[0.0], [1.0], [2.0]], 5); took = time.perf_counter() - start
assert [(h.article.link, h.similarity) for h in got] == [(h.article.link, h.similarity) for h in old_merge([[0.0], [1.0], [2.0]])]
assert [(h.article.link, h.similarity) for h in got] == [("a", 0.95), ("c", 0.7), ("b", 0.6), ("d", 0.6)]
assert took < 0.5, took                                      # 차례로면 0.6초 이상
print("4 ok: search_vectors (동시 검색, 결과 같음)")

# 5) HyDE와 Multi-Query 동시: 문장 목록과 순서는 예전과 같다 [HyDE 문장, 변형들]
def slow_hyde(q, c, use_hyde=True): time.sleep(0.3); return f"{q}\n가상" if use_hyde else q
def slow_mq(q, c): time.sleep(0.3); return ["변형1", "변형2"]
R._build_search_text, R._generate_query_variants = slow_hyde, slow_mq
start = time.perf_counter(); texts = R.build_search_texts("질문", "맥락"); took = time.perf_counter() - start
assert texts == ["질문\n가상", "변형1", "변형2"] and took < 0.5, (texts, took)   # 차례로면 0.6초
assert R.build_search_texts("질문", "맥락", use_hyde=False) == ["질문", "변형1", "변형2"]
assert R.build_search_texts("질문", "맥락", use_multi_query=False) == ["질문\n가상"]
print("5 ok: build_search_texts (동시 호출, 순서 같음)")

# 6) 답변 스트리밍: 완성된 답변 글은 한 번에 받을 때와 같다. 스트리밍이 처음부터 실패하면 한 번에 받는다
class FakeChain:
    def __init__(self, fail_stream=False): self.fail_stream = fail_stream
    def invoke(self, inputs): return "강남은 (기사 1) 이렇게 했다."
    def stream(self, inputs):
        if self.fail_stream:
            raise RuntimeError("스트리밍 안 됨")
        for piece in ["강남은 ", "(기사 1) ", "이렇게 했다."]:
            yield piece
hits = [hit("a", "청크", 0.9)]
Q._get_chain = lambda: FakeChain()
pieces = []
streamed = Q.generate_answer_from_chunks("질문", hits, on_token=pieces.append)
plain = Q.generate_answer_from_chunks("질문", hits)
assert streamed.answer == plain.answer and pieces == ["강남은 ", "(기사 1) ", "이렇게 했다."]
assert streamed.cited == plain.cited == [1]
Q._get_chain = lambda: FakeChain(fail_stream=True)
pieces.clear()
assert Q.generate_answer_from_chunks("질문", hits, on_token=pieces.append).answer == plain.answer
assert pieces == [plain.answer]
print("6 ok: 답변 스트리밍 (같은 답변, 실패 시 한 번에)")

# 7) 크롤링: 해석 순서·결과는 예전과 같다 (여러 프로세스에서 해석하는 경우는 check_parse_pool.py에서 따로 확인)
#    이 파일은 if __name__ == "__main__" 없이 위에서부터 실행되므로 프로세스를 쓰지 않게 끈다
collect.PARSE_PROCESSES = 0
html ="<html><body><div id='dic_area'>" + "본문 문장입니다. " * 20 + "</div></body></html>"
async def fake_get_html(session, url):
    await asyncio.sleep(0.01); return None if "fail" in url else html
collect._get_html = fake_get_html
metas = [NewsMeta(title=f"t{i}", link=f"https://press.com/{'fail' if i == 2 else i}") for i in range(5)]
got = asyncio.run(collect.crawl_articles(metas))
assert [a.link for a in got] == [m.link for m in metas if "fail" not in m.link]
assert all(a.content == collect.parse_article_html(html)[0] for a in got)
print("7 ok: crawl_articles (다운로드·해석 분리, 결과 같음)")
print("ALL PASS")
