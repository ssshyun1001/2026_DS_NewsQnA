"""
HTML 해석을 여러 프로세스에서 해도 결과가 같은지 (qupict/collect.py의 PARSE_PROCESSES)

Windows는 새 프로세스가 이 파일을 다시 읽으므로, 실행할 코드는 모두 if __name__ == "__main__": 안에 둔다.
LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qupict import collect  # noqa: E402
from qupict.schemas import NewsMeta  # noqa: E402

PAGES = {
    "https://n.news.naver.com/1": (
        "<html><head><meta property='og:article:author' content='연합뉴스 | 네이버'></head><body>"
        "<div id='dic_area'>" + "강남이 예능에 출연했다. " * 20
        + "<span class='end_photo_org'><img data-src='https://img/a.jpg'><em class='img_desc'>사진 설명</em></span>"
        "</div></body></html>"),
    "https://press.com/2": "<html><body><article>" + "원문 사이트 본문입니다. " * 15 + "</article></body></html>",
    "https://press.com/3": "<html><body><p>짧음</p></body></html>",          # 본문이 짧아 실패
}


async def fake_get_html(session, url):
    await asyncio.sleep(0.01)
    return PAGES.get(url)


def main() -> None:
    assert collect.PARSE_PROCESSES > 1, "이 컴퓨터는 CPU가 1개라 프로세스 해석을 쓰지 않음"

    # 1) 프로세스에서 해석한 결과 == 이 프로세스에서 해석한 결과
    async def parse_all():
        return [await collect._parse_html(h) for h in PAGES.values()]
    via_pool = asyncio.run(parse_all())
    assert collect._parse_pool is not None, "프로세스를 쓰지 않았음"
    assert via_pool == [collect.parse_article_html(h) for h in PAGES.values()]
    print("1 ok: 프로세스 해석 결과 같음")

    # 2) crawl_articles 전체: 결과·순서가 프로세스를 끈 경우와 같다
    collect._get_html = fake_get_html
    metas = [NewsMeta(title="t1", link="https://press.com/1", naver_link="https://n.news.naver.com/1"),
             NewsMeta(title="t2", link="https://press.com/2"),
             NewsMeta(title="t3", link="https://press.com/3")]
    with_pool = asyncio.run(collect.crawl_articles(metas))
    collect._disable_parse_pool()
    without_pool = asyncio.run(collect.crawl_articles(metas))
    strip = lambda arts: [a.model_dump(exclude={"collected_at"}) for a in arts]
    assert strip(with_pool) == strip(without_pool) and [a.link for a in with_pool] == ["https://press.com/1",
                                                                                         "https://press.com/2"]
    assert with_pool[0].press == "연합뉴스" and with_pool[0].images[0].caption == "사진 설명"
    print("2 ok: crawl_articles (프로세스 사용 / 안 씀 결과 같음)")
    print("ALL PASS")


if __name__ == "__main__":
    main()
