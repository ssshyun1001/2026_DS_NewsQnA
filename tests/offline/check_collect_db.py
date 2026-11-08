"""
1단계: collect_to_db (가짜 DB)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import asyncio, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from qupict import collect, db
from qupict.schemas import Article, ArticleImage, NewsMeta, SelectedSense, Sense

calls = {}
sel = SelectedSense(query="강남", sense=Sense(name="방송인 강남", definition="d", related_terms=["이상화"]))
metas = [NewsMeta(title=f"t{i}", link=f"https://press.com/{i}", naver_link=(f"https://n.news.naver.com/{i}" if i == 0 else None), search_term="강남 이상화") for i in range(5)]

async def fake_search(s, period, max_articles): calls["search"] = (s.query, period, max_articles); return metas
async def fake_crawl(ms):
    calls["crawl"] = [m.link for m in ms]
    # 3건 중 1건은 본문 실패
    return [Article(title=m.title, link=m.link, search_term=m.search_term,
                    images=[ArticleImage(url="https://img/1.jpg"), ArticleImage(url="https://img/2.jpg")]) for m in ms[:2]]
collect.search_news, collect.crawl_articles = fake_search, fake_crawl
db.save_topic = lambda s: calls.setdefault("topic", s.sense.name) and 7
db.link_known_articles = lambda tid, ms: (calls.__setitem__("link_known", tid), ms[2:])[1]   # 앞 2건은 이미 DB에 있음
db.save_articles = lambda tid, arts: calls.__setitem__("saved", (tid, [(a.link, len(a.images)) for a in arts]))
db.mark_topic_collected = lambda tid: calls.__setitem__("marked", tid)

r = asyncio.run(collect.collect_to_db(sel, 30, max_articles=50))
assert calls["search"] == ("강남", 30, 50)
assert calls["topic"] == "방송인 강남" and calls["link_known"] == 7 and calls["marked"] == 7
assert calls["crawl"] == ["https://press.com/2", "https://press.com/3", "https://press.com/4"]   # DB에 없는 것만 크롤링
assert calls["saved"][0] == 7 and len(calls["saved"][1]) == 2
assert (r.topic_id, r.searched, r.reused, r.crawled, r.crawl_failed, r.images) == (7, 5, 2, 2, 1, 4), r
print("ok:", r.summary())

# naver_link가 Article까지 전달되는지 (크롤러 쪽)
import qupict.collect as c2
assert "naver_link" not in Article.model_fields and "naver_link" in NewsMeta.model_fields
print("ok: naver_link는 검색 결과(NewsMeta)에만 있음")
