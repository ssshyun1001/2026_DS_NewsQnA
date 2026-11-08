"""
[1단계] 뉴스 수집 — 네이버 뉴스 검색 API + 본문·이미지 크롤링

0단계에서 고른 의미(SelectedSense)를 기준으로 검색어를 만들고, 기사 본문·언론사·이미지를 가져온다.

  search_news(selected, period_days)   → list[NewsMeta]   검색 (기사 제목·링크·날짜·요약)
  crawl_articles(metas)                → list[Article]    본문·언론사·이미지 크롤링
  collect_articles(selected, period)   → list[Article]    위 둘을 한 번에
  collect_to_db(selected, period_days) → topic id         DB 버전: 이미 있는 기사는 재사용, 새 기사만 크롤링

## 검색어 만들기 (build_search_terms)

  1순위  검색어 + 연관어      "눈 안과", "눈 시력", "눈 각막"     ← 의미가 좁혀진 검색
  2순위  별칭 단독             "서울여자대학교", "배스킨라빈스"     ← 같은 대상의 다른 이름
  3순위  검색어 단독           "눈"                                ← 놓치는 기사 줄이기

  "눈"처럼 뜻이 여러 개인 검색어를 그대로 최신순 검색하면, 그날 기사가 많은 쪽(날씨)이
  결과를 다 채워 버려서 사용자가 고른 의미(신체 눈)의 기사가 거의 안 들어온다.
  그래서 의미가 좁혀진 "검색어 + 연관어"를 먼저 검색한다. (기존 "검색어 + 검증어"를 대체)
  검색어 단독 검색도 남겨 두는 이유: 연관어가 안 들어간 관련 기사를 놓치지 않기 위해서다.
  거기 섞여 들어오는 다른 의미의 기사는 2·3단계 필터가 걸러 낸다.

## 결과 섞기 (_interleave)

  검색어마다 최대 100건씩 받은 뒤, 검색어별로 돌아가며 하나씩 뽑아 상한(기본 100건)을 채운다.
  첫 검색어 결과만으로 상한이 차 버리는 것을 막아, 여러 검색어의 기사가 고르게 들어오게 한다.

## 기간

  네이버 API는 기간 조건을 지원하지 않으므로, 받은 뒤 게시일(pubDate)로 거른다.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone

import aiohttp
from bs4 import BeautifulSoup, Tag, UnicodeDammit
from tenacity import retry, stop_after_attempt, wait_exponential

from qupict.config import settings
from qupict.schemas import Article, ArticleImage, NewsMeta, SelectedSense, unique_terms

_NAVER_NEWS_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
_TAG_RE = re.compile(r"<.*?>")

_NAVER_MAX_DISPLAY = 100        # 네이버 뉴스 검색 API가 한 번에 주는 최대 건수
DEFAULT_MAX_ARTICLES = 100      # 대상 하나당 수집할 기사 수 상한
MAX_RELATED_IN_QUERY = 3        # "검색어 + 연관어" 조합에 쓸 연관어 개수 (API 호출 수 = 이 값 + 별칭 수 + 1)


# ---------------------------------------------------------------------------
# 검색
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    return _TAG_RE.sub("", text).replace("&quot;", '"').replace("&amp;", "&")


def _parse_pubdate(raw: str) -> datetime | None:
    # 예: 'Mon, 01 Jan 2024 12:00:00 +0900'
    try:
        return datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %z")
    except (ValueError, TypeError):
        return None


def build_search_terms(selected: SelectedSense) -> list[str]:
    """고른 의미로 네이버 검색어 목록을 만든다 (우선순위 순)."""
    query = selected.query
    combos = [f"{query} {t}" for t in selected.sense.related_terms[:MAX_RELATED_IN_QUERY]]
    return unique_terms([*combos, *selected.sense.aliases, query])


def _interleave(results_by_term: list[list[NewsMeta]], limit: int) -> list[NewsMeta]:
    """검색어별 결과를 돌아가며 하나씩 뽑아 합친다. 같은 기사(링크)는 한 번만 넣는다.

    예: A=[a1,a2,a3], B=[b1,b2] → a1, b1, a2, b2, a3
    """
    merged: list[NewsMeta] = []
    seen: set[str] = set()
    for i in range(max((len(r) for r in results_by_term), default=0)):
        for results in results_by_term:
            if len(merged) >= limit:
                return merged
            if i < len(results) and results[i].link not in seen:
                seen.add(results[i].link)
                merged.append(results[i])
    return merged


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _fetch_one_keyword(session: aiohttp.ClientSession, keyword: str) -> list[dict]:
    headers = {
        "X-NCP-APIGW-API-KEY-ID": settings.naver_client_id,
        "X-NCP-APIGW-API-KEY": settings.naver_client_secret,
    }
    params = {"query": keyword, "display": _NAVER_MAX_DISPLAY, "start": 1, "sort": "date"}
    async with session.get(_NAVER_NEWS_URL, headers=headers, params=params) as resp:
        resp.raise_for_status()
        data = await resp.json(content_type=None)
        return data.get("items", [])


def _to_metas(items: list[dict], search_term: str, cutoff: datetime) -> list[NewsMeta]:
    metas = []
    for item in items:
        original = item.get("originallink") or item.get("link")
        if not original:
            continue
        pub_date = _parse_pubdate(item.get("pubDate", ""))
        if pub_date and pub_date < cutoff:
            continue
        naver = item.get("link") or ""
        metas.append(NewsMeta(
            title=_strip_html(item.get("title", "")),
            link=original,
            naver_link=naver if "news.naver.com" in naver else None,
            pub_date=pub_date,
            description=_strip_html(item.get("description", "")),
            search_term=search_term,
        ))
    return metas


async def search_news(
    selected: SelectedSense,
    period_days: int,
    max_articles: int = DEFAULT_MAX_ARTICLES,
) -> list[NewsMeta]:
    """고른 의미로 검색어들을 만들어 동시에 검색하고, 기간 안의 기사를 최대 max_articles건 돌려준다."""
    terms = build_search_terms(selected)
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)

    async with aiohttp.ClientSession() as session:
        responses = await asyncio.gather(
            *(_fetch_one_keyword(session, t) for t in terms), return_exceptions=True
        )

    results_by_term = []
    for term, items in zip(terms, responses):
        if isinstance(items, Exception):  # 검색어 하나가 실패해도 나머지는 쓴다
            print(f"  [검색 실패] '{term}': {type(items).__name__}")
            continue
        results_by_term.append(_to_metas(items, term, cutoff))
    return _interleave(results_by_term, max_articles)


# ---------------------------------------------------------------------------
# 본문·언론사·이미지 크롤링
#
# 네이버 뉴스 주소(n.news.naver.com)가 있으면 거기서 먼저 가져온다. 언론사마다 HTML이 다른
# 원문 사이트보다 구조가 일정해서 본문·사진 설명·언론사 이름을 안정적으로 찾을 수 있다.
# 실패하면 언론사 원문 주소로 다시 시도한다.
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# 본문 영역을 찾는 셀렉터 (위에서부터 순서대로 시도, 필요하면 언론사별로 추가)
_SELECTOR_CANDIDATES = [
    "#dic_area",          # 네이버 뉴스 (일반)
    "#newsct_article",
    "div.article_body",
    "div#articleBodyContents",
    "div.news_end",
    "[itemprop='articleBody']",       # 표준 표기 (여러 언론사)
    "#article-view-content-div",       # 국내 지역·전문지에서 많이 쓰는 기사 관리 시스템
    "#articleBody",
    "#article_body",
    "div.article_view",
    "article",                         # 마지막 수단 (사이드바 기사 목록까지 잡힐 수 있음)
]

_MIN_CONTENT_LENGTH = 100
_MAX_CONTENT_LENGTH = 6000       # 과도한 컨텍스트 방지
MAX_IMAGES_PER_ARTICLE = 5
_SKIP_IMAGE_RE = re.compile(r"(icon|logo|btn|banner|blank|spacer|\.gif($|\?))", re.I)


def _find_body(soup: BeautifulSoup) -> Tag | None:
    for selector in _SELECTOR_CANDIDATES:
        node = soup.select_one(selector)
        if node and len(node.get_text(strip=True)) >= _MIN_CONTENT_LENGTH:
            return node
    return None


def _image_caption(img: Tag) -> str:
    """사진 설명 찾기: <figure>의 <figcaption> → 네이버 뉴스의 em.img_desc → alt 순서."""
    figure = img.find_parent("figure")
    if figure and figure.find("figcaption"):
        return figure.find("figcaption").get_text(" ", strip=True)
    wrapper = img.find_parent(class_="end_photo_org")
    desc = wrapper.find(class_="img_desc") if wrapper else img.find_next_sibling(class_="img_desc")
    if desc:
        return desc.get_text(" ", strip=True)
    return (img.get("alt") or "").strip()


def _extract_images(body: Tag) -> list[ArticleImage]:
    images: list[ArticleImage] = []
    seen: set[str] = set()
    for img in body.find_all("img"):
        url = img.get("data-src") or img.get("src") or ""  # 네이버는 지연 로딩이라 data-src에 실제 주소가 있다
        if not url.startswith("http") or url in seen or _SKIP_IMAGE_RE.search(url):
            continue
        seen.add(url)
        images.append(ArticleImage(url=url, caption=_image_caption(img)))
        if len(images) >= MAX_IMAGES_PER_ARTICLE:
            break
    return images


_MAX_PRESS_NAME = 20


def _meta_content(soup: BeautifulSoup, prop: str) -> str:
    tag = soup.find("meta", attrs={"property": prop})
    return (tag.get("content") or "").strip() if tag else ""


def _extract_press(soup: BeautifulSoup) -> str | None:
    """언론사 이름 찾기.

    1) 네이버 뉴스 상단 로고의 alt                    예: "연합뉴스"
    2) 네이버 뉴스의 og:article:author "언론사 | 네이버"  예: "연합뉴스 | 네이버" → "연합뉴스"
    3) 언론사 사이트의 og:site_name                   예: "동아일보"
    og:article:author는 언론사 사이트에서는 기자 이름("홍길동 기자")인 경우가 많아서
    '|'가 들어 있는 네이버 형식일 때만 쓴다.
    """
    logo = soup.select_one(".media_end_head_top_logo img[alt]")
    if logo and logo.get("alt", "").strip():
        return logo["alt"].strip()
    author = _meta_content(soup, "og:article:author")
    if "|" in author:
        return author.split("|")[0].strip() or None
    site = _meta_content(soup, "og:site_name")
    if len(site) > _MAX_PRESS_NAME:
        # 사이트 이름 자리에 페이지 제목을 넣는 언론사가 있다 → "제목 - 슬로건 - 뉴스엔"의 마지막 부분만 쓴다
        last = site.rsplit(" - ", 1)[-1].strip()
        site = last if len(last) <= _MAX_PRESS_NAME else ""
    if site and site not in ("네이버 뉴스", "NAVER"):
        return site
    return None


def parse_article_html(html: str) -> tuple[str, list[ArticleImage], str | None]:
    """기사 HTML → (본문, 이미지 목록, 언론사)."""
    soup = BeautifulSoup(html, "lxml")
    press = _extract_press(soup)
    body = _find_body(soup)
    if body is None:  # 본문 영역을 못 찾으면 <p> 태그를 모두 모아 본다 (이미지는 포기)
        text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
        return text, [], press
    images = _extract_images(body)
    for caption in body.select(".img_desc, figcaption"):  # 사진 설명은 본문 텍스트에서 뺀다
        caption.decompose()
    return body.get_text(separator=" ", strip=True), images, press


def _decode_html(raw: bytes, header_charset: str | None) -> str:
    """페이지 문자셋에 맞게 글자로 바꾼다.

    국내 언론사 중에는 아직 EUC-KR로 된 페이지가 있다. 이걸 UTF-8로 읽으면 글자가 깨진다
    (예: "̻ȭ ?"). 기존 코드는 깨진 글자를 무시하고 그대로 써서 본문이 깨진 채 저장됐다.
    서버가 알려 준 문자셋으로 먼저 읽어 보고, 안 맞으면 페이지 안의 <meta charset>과
    내용을 보고 문자셋을 추정한다 (UnicodeDammit).
    """
    if header_charset:
        try:
            return raw.decode(header_charset)
        except (UnicodeDecodeError, LookupError):
            pass
    dammit = UnicodeDammit(raw, ["utf-8", "cp949", "euc-kr"], is_html=True)
    return dammit.unicode_markup or raw.decode("utf-8", errors="replace")


async def _get_html(session: aiohttp.ClientSession, url: str) -> str | None:
    try:
        timeout = aiohttp.ClientTimeout(total=settings.crawl_timeout_seconds)
        async with session.get(url, headers=_HEADERS, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return _decode_html(await resp.read(), resp.charset)
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None


async def _fetch_and_parse(
    session: aiohttp.ClientSession, meta: NewsMeta, semaphore: asyncio.Semaphore
) -> Article | None:
    urls = [u for u in (meta.naver_link, meta.link) if u]
    async with semaphore:
        for url in urls:
            html = await _get_html(session, url)
            if not html:
                continue
            content, images, press = parse_article_html(html)
            if len(content) >= _MIN_CONTENT_LENGTH:
                return Article(
                    title=meta.title,
                    link=meta.link,
                    press=press or meta.press,
                    pub_date=meta.pub_date,
                    content=content[:_MAX_CONTENT_LENGTH],
                    description=meta.description,
                    search_term=meta.search_term,
                    images=images,
                    collected_at=datetime.now(timezone.utc),
                )
    return None


async def crawl_articles(metas: list[NewsMeta], concurrency: int = 8) -> list[Article]:
    """기사 정보 목록을 받아 병렬로 본문을 가져온다. 본문을 못 가져온 기사는 빠진다."""
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(_fetch_and_parse(session, m, semaphore) for m in metas))
    return [a for a in results if a is not None]


async def collect_articles(
    selected: SelectedSense,
    period_days: int,
    max_articles: int = DEFAULT_MAX_ARTICLES,
) -> tuple[list[NewsMeta], list[Article]]:
    """검색 + 크롤링을 한 번에. 검색 결과(metas)도 같이 돌려준다 (실험의 Baseline = API 결과 그대로)."""
    metas = await search_news(selected, period_days, max_articles)
    articles = await crawl_articles(metas)
    return metas, articles


async def collect_to_db(
    selected: SelectedSense,
    period_days: int,
    max_articles: int = DEFAULT_MAX_ARTICLES,
) -> int:
    """
    [1단계 DB 버전] 고른 의미의 topic을 만들거나 재사용하고, 기간 안 기사를 DB에 채운다. topic id를 돌려준다.

    같은 topic을 다시 요청하면: 검색은 다시 하지만(API 호출은 저렴), DB에 이미 있는 기사는
    크롤링하지 않고 topic에 연결만 한다. DB에 없는 기사(주로 지난 수집 이후 새로 나온 기사)만 크롤링해 저장한다.
    이후 단계는 db.get_topic_articles(topic_id, period_days=...)로 발행일 기준 기간 안 기사를 꺼내 쓴다.
    """
    from qupict import db

    topic_id = db.save_topic(selected)
    metas = await search_news(selected, period_days, max_articles)
    new_metas = db.link_known_articles(topic_id, metas)
    articles = await crawl_articles(new_metas)
    db.save_articles(topic_id, articles)
    db.mark_topic_collected(topic_id)
    return topic_id
