"""
1단계: 검색어·결과 섞기·HTML 파싱·인코딩·언론사

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import sys
from datetime import datetime, timedelta, timezone
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from qupict import collect
from qupict.schemas import NewsMeta, SelectedSense, Sense

# 1) 검색어 만들기
sel = SelectedSense(query="눈", sense=Sense(name="신체 눈", definition="d", aliases=["안구"], related_terms=["안과", "시력", "각막", "백내장"]))
assert collect.build_search_terms(sel) == ["눈 안과", "눈 시력", "눈 각막", "눈 백내장", "안구"], collect.build_search_terms(sel)
g = SelectedSense(query="강남", sense=Sense(name="방송인 강남", definition="d", aliases=[], related_terms=["이상화"]))
assert collect.build_search_terms(g) == ["강남 이상화"]                      # 검색어 단독은 검색하지 않음
empty = SelectedSense(query="베라", sense=Sense(name="x", definition="d"))
assert collect.build_search_terms(empty) == ["베라"]                          # 좁힐 단어가 없을 때만 단독
print("1 ok: 검색어", collect.build_search_terms(sel))

# 2) 섞기: 돌아가며, 중복 링크 제거, 상한
m = lambda link: NewsMeta(title=link, link=link)
A, B = [m("a1"), m("a2"), m("a3")], [m("b1"), m("a2"), m("b3")]
assert [x.link for x in collect._interleave([A, B], 10)] == ["a1", "b1", "a2", "a3", "b3"]
assert [x.link for x in collect._interleave([A, B], 3)] == ["a1", "b1", "a2"]
assert collect._interleave([], 5) == []
print("2 ok: 섞기")

# 2-1) 쪽 넘기기: 100건씩 최대 3쪽, 100건 미만이거나 쪽 마지막 기사가 기간 밖이면 멈춤
import asyncio as _aio
_kst = timezone(timedelta(hours=9))
_day = lambda d: (datetime.now(_kst) - timedelta(days=d)).strftime("%a, %d %b %Y %H:%M:%S %z")
def _pages(*sizes_and_age):
    calls = []
    async def fake_page(session, keyword, start):
        calls.append(start)
        n, age = sizes_and_age[len(calls) - 1]
        return [{"pubDate": _day(age)}] * n
    return fake_page, calls
cut = datetime.now(timezone.utc) - timedelta(days=30)
collect._fetch_page, calls = _pages((100, 1), (100, 5), (100, 9))
assert len(_aio.run(collect._fetch_one_keyword(None, "k", cut))) == 300 and calls == [1, 101, 201]   # 3쪽까지
collect._fetch_page, calls = _pages((100, 1), (40, 3))
assert len(_aio.run(collect._fetch_one_keyword(None, "k", cut))) == 140 and calls == [1, 101]        # 결과가 끝남
collect._fetch_page, calls = _pages((100, 1), (100, 40), (100, 50))
assert len(_aio.run(collect._fetch_one_keyword(None, "k", cut))) == 200 and calls == [1, 101]        # 기간 밖으로 나감
print("2-1 ok: 쪽 넘기기")

# 3) API 결과 → NewsMeta: 기간 밖 제외, 네이버 링크 구분, HTML 태그 제거
now = datetime.now(timezone(timedelta(hours=9)))
fmt = lambda d: d.strftime("%a, %d %b %Y %H:%M:%S %z")
items = [
    {"title": "<b>눈</b> 건강", "originallink": "https://press.com/1", "link": "https://n.news.naver.com/mnews/article/001/1", "pubDate": fmt(now), "description": "요약 &quot;a&quot;"},
    {"title": "옛날 기사", "originallink": "https://press.com/2", "link": "https://press.com/2", "pubDate": fmt(now - timedelta(days=40))},
    {"title": "원문만", "originallink": "https://press.com/3", "link": "https://press.com/3", "pubDate": fmt(now)},
]
metas = collect._to_metas(items, "눈 안과", datetime.now(timezone.utc) - timedelta(days=30))
assert [x.link for x in metas] == ["https://press.com/1", "https://press.com/3"]
assert metas[0].title == "눈 건강" and metas[0].description == '요약 "a"' and metas[0].search_term == "눈 안과"
assert metas[0].naver_link.startswith("https://n.news.naver.com") and metas[1].naver_link is None
print("3 ok: API 결과 변환")

# 4) 네이버 뉴스 형태 HTML 파싱: 본문, 사진 설명, 언론사
body_text = "안과 전문의에 따르면 겨울철 안구건조증 환자가 늘고 있다. " * 5
html = f"""<html><head><meta property="og:article:author" content="연합뉴스 | 네이버"></head><body>
<div class="media_end_head_top_logo"><img alt="연합뉴스" src="https://x/logo.png"></div>
<article id="dic_area">
  <span class="end_photo_org"><img data-src="https://imgnews.pstatic.net/a.jpg" src="data:image/gif;base64,xx"><em class="img_desc">안과 진료 모습</em></span>
  {body_text}
  <img src="https://imgnews.pstatic.net/icon_share.png">
  <figure><img src="https://imgnews.pstatic.net/b.jpg"><figcaption>두 번째 사진</figcaption></figure>
</article></body></html>"""
text, images, press = collect.parse_article_html(html)
assert press == "연합뉴스", press
assert [(i.url, i.caption) for i in images] == [("https://imgnews.pstatic.net/a.jpg", "안과 진료 모습"), ("https://imgnews.pstatic.net/b.jpg", "두 번째 사진")], images
assert "안구건조증" in text and "안과 진료 모습" not in text and "두 번째 사진" not in text
# 로고가 없으면 og 태그에서 언론사 찾기
_, _, press2 = collect.parse_article_html(html.replace('<div class="media_end_head_top_logo"><img alt="연합뉴스" src="https://x/logo.png"></div>', ""))
assert press2 == "연합뉴스", press2
# 본문 영역을 못 찾으면 <p> 모음, 이미지 없음
t3, i3, _ = collect.parse_article_html("<html><body><p>문단1</p><p>문단2</p></body></html>")
assert t3 == "문단1 문단2" and i3 == []
print("4 ok: HTML 파싱 (본문·사진 설명·언론사)")

# 5) 인코딩: EUC-KR 페이지 (서버가 문자셋을 안 알려 준 경우 / 잘못 알려 준 경우)
page = '<html><head><meta charset="euc-kr"><meta property="og:site_name" content="동아일보"></head><body><p>강남이 이상화와 함께 출연했다</p></body></html>'
raw = page.encode("cp949")
assert "강남이 이상화" in collect._decode_html(raw, None)
assert "강남이 이상화" in collect._decode_html(raw, "utf-8")           # utf-8로 안 읽히면 추정으로 넘어감
assert collect._decode_html("한글".encode("utf-8"), "utf-8") == "한글"
print("5 ok: EUC-KR 인코딩")

# 6) 언론사: 기자 이름(og:article:author)은 쓰지 않음
reporter = '<html><head><meta property="og:article:author" content="김경은 기자"><meta property="og:site_name" content="스포츠조선"></head><body></body></html>'
assert collect.parse_article_html(reporter)[2] == "스포츠조선"
only_reporter = '<html><head><meta property="og:article:author" content="장윤우"></head><body></body></html>'
assert collect.parse_article_html(only_reporter)[2] is None
print("6 ok: 기자 이름은 언론사로 쓰지 않음")

# 7) 사이트 이름 자리에 페이지 제목이 들어간 경우
long_site = '<html><head><meta property="og:site_name" content="“♥이상화 돈 펑펑 쓰나?” 강남, 이서진 내기 도발(비서진) - 손에 잡히는 뉴스 눈에 보이는 뉴스 - 뉴스엔"></head><body></body></html>'
assert collect.parse_article_html(long_site)[2] == "뉴스엔"
# 8) 국내 CMS 본문 영역이 사이드바(<article>)보다 먼저 잡힘
cms = "<html><body><article><p>다른 기사 제목 목록 " + "x" * 120 + "</p></article><div id='article-view-content-div'>" + "강남이 출연했다. " * 20 + "</div></body></html>"
assert collect.parse_article_html(cms)[0].startswith("강남이 출연했다")
from bs4 import BeautifulSoup
b = BeautifulSoup('<div><img src="http://cdn.queen.co.kr/image/newsroom/default-user.png"><img src="https://x.com/photo.jpg"></div>', "lxml")
assert [i.url for i in collect._extract_images(b)] == ["https://x.com/photo.jpg"]
print("7·8 ok: 긴 사이트 이름, CMS 본문 영역")

# 9) 추가한 본문 영역: 프라임경제(#news_body_area), bnt뉴스(div.newsview div.content, <p> 없음)
prime = "<html><body><p>짧은 안내</p><div id='news_body_area'>" + "삼성물산 실적이 늘었다. " * 20 + "</div></body></html>"
assert collect.parse_article_html(prime)[0].startswith("삼성물산 실적이 늘었다")
bnt = ("<html><body><article>메뉴</article><div class='din newsview'><h1>제목</h1><div class='content'>"
       + "신혜선이 팬미팅을 연다. " * 20 + "</div></div></body></html>")
assert collect.parse_article_html(bnt)[0].startswith("신혜선이 팬미팅을 연다")
print("9 ok: 프라임경제 · bnt뉴스 본문 영역")

# 10) 보안 연결(SSL) 실패 → 느슨한 설정으로 한 번 더. 다른 오류 · 200이 아닌 응답은 다시 받지 않는다
import aiohttp


class FakeSSLError(aiohttp.ClientSSLError):
    def __init__(self): pass
    def __str__(self): return "ssl"


class FakeResponse:
    def __init__(self, status): self.status, self.charset = status, "utf-8"
    async def read(self): return "<p>본문</p>".encode("utf-8")
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False


class FakeSession:
    """outcomes를 차례로 돌려준다 (예외면 던진다). 받은 ssl 옵션을 calls에 적는다."""
    def __init__(self, outcomes): self.outcomes, self.calls = list(outcomes), []
    def get(self, url, headers=None, timeout=None, **options):
        self.calls.append((options.get("ssl"), timeout.total))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


get = lambda session: _aio.run(collect._get_html(session, "https://press.com/1"))
s = FakeSession([FakeSSLError(), 200])
assert get(s) == "<p>본문</p>"
assert s.calls[0][0] is None and s.calls[1][0] is collect._lenient_ssl_context()      # 두 번째만 느슨한 설정
assert s.calls[1][0].verify_mode.name == "CERT_NONE"
assert s.calls[0][1] == min(collect.settings.crawl_timeout_seconds, collect.CRAWL_PAGE_TIMEOUT_SECONDS)
s = FakeSession([FakeSSLError(), FakeSSLError()])
assert get(s) is None and len(s.calls) == 2                                          # 두 번 다 실패하면 끝
s = FakeSession([aiohttp.ClientConnectionError(), 200])
assert get(s) is None and len(s.calls) == 1                                          # 다른 연결 오류는 다시 안 받음
s = FakeSession([403, 200])
assert get(s) is None and len(s.calls) == 1                                          # 403도 다시 안 받음
s = FakeSession([200])
assert get(s) == "<p>본문</p>" and s.calls[0][0] is None                              # 정상 사이트는 검사한 채로 한 번
print("10 ok: 보안 연결 실패 때만 느슨한 설정으로 재시도")
print("ALL PASS")
