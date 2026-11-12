"""
[1단계] 뉴스 수집 — 네이버 뉴스 검색 API + 본문·이미지 크롤링

0단계에서 고른 의미(SelectedSense)를 기준으로 검색어를 만들고, 기사 본문·언론사·이미지를 가져온다.

  search_news(selected, period_days)   → list[NewsMeta]   검색 (기사 제목·링크·날짜·요약)
  crawl_articles(metas)                → list[Article]    본문·언론사·이미지 크롤링
  collect_articles(selected, period)   → list[Article]    위 둘을 한 번에
  collect_to_db(selected, period_days) → CollectResult    DB 버전: 이미 있는 기사는 재사용, 새 기사만 크롤링
  download_images(articles)            → 이미지 파일 내려받기 (4단계에서 관련 기사의 이미지 설명을 만들 때 사용)

## 검색어 만들기 (build_search_terms)

  검색어 + 연관어      "눈 안과", "눈 시력", "눈 각막", ...  ← 고른 의미로 좁힌 검색 (연관어 전부, 최대 8개)
  별칭 단독             "서울여자대학교", "배스킨라빈스"       ← 같은 대상의 다른 이름

  검색어 단독("눈", "강남")은 검색하지 않는다. 뜻이 여러 개인 검색어를 그대로 최신순 검색하면
  그날 기사가 많은 쪽(날씨 눈, 지역 강남)이 결과를 채워 버려서, 사용자가 고른 의미의 기사가
  상한(300건) 안에 잘 들어오지 않는다. (예전에는 검색어 단독도 섞어서 강남 → 방송인 강남 300건 중
  100건이 "강남" 단독 검색 결과였다)
  고른 의미에 연관어·별칭이 하나도 없을 때만 검색어 단독으로 검색한다 (결과가 0건이 되지 않게).

## 결과 섞기 (_interleave)

  1순위 관련성, 2순위 최신성 (search_news)
  검색어마다 관련도순 상위 100건과 최신순 최대 300건(100건씩 3쪽, 기간 밖으로 나가면 일찍 멈춤)을 받는다.
  관련도순 결과(기간 안)를 "관련 후보"로 삼고, 그 안에서 최신 기사부터 검색어별로 돌아가며 하나씩 뽑아
  상한(ask 100건, collect-all 300건)을 채운다. 모자라면 최신순 결과를 같은 방식으로 뽑아 나머지를 채운다.
  돌아가며 뽑는 이유: 첫 검색어 결과만으로 상한이 차 버리는 것을 막아, 여러 검색어의 기사가 고르게 들어오게.

## 기간

  네이버 API는 기간 조건을 지원하지 않으므로, 받은 뒤 게시일(pubDate)로 거른다.
"""
from __future__ import annotations

import asyncio
import os
import re
import ssl
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import aiohttp
from bs4 import BeautifulSoup, Tag, UnicodeDammit
from tenacity import retry, stop_after_attempt, wait_exponential

from qupict import timing
from qupict.concurrency import timed_call
from qupict.config import settings
from qupict.schemas import Article, ArticleImage, NewsMeta, SelectedSense, unique_terms

_NAVER_NEWS_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
_TAG_RE = re.compile(r"<.*?>")

_NAVER_MAX_DISPLAY = 100        # 네이버 뉴스 검색 API가 한 번에 주는 최대 건수
DEFAULT_MAX_ARTICLES = 100      # 대상 하나당 수집할 기사 수 상한 (ask 기본값. 모이는 날 collect-all은 300)
SEARCH_PAGES = 3                # 검색어 하나당 최대 몇 쪽(1쪽 = 100건)까지 받을지. 기간 밖으로 나가면 일찍 멈춘다
RELEVANCE_PAGES = 1             # 관련도순으로 받을 쪽수 (검색어당 상위 100건). 뒤쪽으로 갈수록 관련성이 떨어지고,
                                #   관련도순은 일찍 멈출 수 없어 쪽수만큼 호출이 늘어난다
MAX_RELATED_IN_QUERY = 8        # "검색어 + 연관어" 조합에 쓸 연관어 개수 (= 연관어 최대 개수, API 호출 수 = 이 값 + 별칭 수)


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
    """고른 의미로 네이버 검색어 목록을 만든다: 검색어 + 연관어, 별칭. (검색어 단독은 좁힐 단어가 없을 때만)"""
    query = selected.query
    combos = [f"{query} {t}" for t in selected.sense.related_terms[:MAX_RELATED_IN_QUERY]]
    return unique_terms([*combos, *selected.sense.aliases]) or [query]


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
async def _fetch_page(session: aiohttp.ClientSession, keyword: str, start: int, sort: str = "date") -> list[dict]:
    """네이버 뉴스 검색 한 쪽. sort: "date"(최신순) 또는 "sim"(관련도순)."""
    headers = {
        "X-NCP-APIGW-API-KEY-ID": settings.naver_client_id,
        "X-NCP-APIGW-API-KEY": settings.naver_client_secret,
    }
    params = {"query": keyword, "display": _NAVER_MAX_DISPLAY, "start": start, "sort": sort}
    async with session.get(_NAVER_NEWS_URL, headers=headers, params=params) as resp:
        resp.raise_for_status()
        data = await resp.json(content_type=None)
        return data.get("items", [])


async def _fetch_one_keyword(session: aiohttp.ClientSession, keyword: str,
                             cutoff: datetime | None = None, pages: int = None,
                             sort: str = "date") -> list[dict]:
    """검색어 하나를 최대 pages쪽(1쪽 = 100건)까지 받는다. sort: "date"(최신순) 또는 "sim"(관련도순).

    네이버는 한 번에 100건만 주므로, 인기 검색어는 첫 쪽이 하루치밖에 안 될 수 있다.
    다음 쪽을 이어 받되, 결과가 100건보다 적으면 멈춘다.
    최신순일 때는 쪽의 마지막 기사가 기간(cutoff) 밖이면 멈춘다 (그 뒤는 전부 더 오래된 기사).
    관련도순은 날짜가 섞여 있어서 이 규칙을 쓸 수 없다 → 정한 쪽수만큼 받는다.
    """
    pages = SEARCH_PAGES if pages is None else pages
    items: list[dict] = []
    for page in range(pages):
        batch = await _fetch_page(session, keyword, 1 + page * _NAVER_MAX_DISPLAY, sort)
        items.extend(batch)
        if len(batch) < _NAVER_MAX_DISPLAY:
            break
        last = _parse_pubdate(batch[-1].get("pubDate", ""))
        if sort == "date" and cutoff and last and last < cutoff:
            break
    return items


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
    """고른 의미로 검색어들을 만들어 동시에 검색하고, 기간 안의 기사를 최대 max_articles건 돌려준다.

    고르는 순서: 1순위 관련성, 2순위 최신성
      1) 검색어마다 관련도순 상위 결과(기간 안)를 "관련 후보"로 삼는다. → 관련 없는 기사가 걸러진다
      2) 관련 후보 안에서 최신 기사부터, 검색어별로 돌아가며 뽑아 상한을 채운다.
      3) 그래도 상한에 모자라면 최신순 결과(기간 안)로 나머지를 채운다.
    관련도순 결과에는 오래된 기사가 섞여 있어서 기간으로 거르면 상한에 못 미칠 수 있다 → 3)이 필요하다.
    네이버는 관련도 "점수"를 주지 않고 순서만 주므로, 관련도순 상위에 들었는지(후보인지)만 관련성으로 쓴다.
    """
    terms = build_search_terms(selected)
    cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)

    async with aiohttp.ClientSession() as session:
        # 검색어마다 관련도순 · 최신순 두 가지를 모두, 전부 동시에 받는다 (API 호출은 무료)
        responses = await asyncio.gather(
            *(_fetch_one_keyword(session, t, cutoff, pages=RELEVANCE_PAGES, sort="sim") for t in terms),
            *(_fetch_one_keyword(session, t, cutoff) for t in terms),
            return_exceptions=True,
        )

    by_relevance, by_date = [], []
    for kind, results, part in (("관련도순", by_relevance, responses[:len(terms)]),
                                ("최신순", by_date, responses[len(terms):])):
        for term, items in zip(terms, part):
            if isinstance(items, Exception):  # 검색 하나가 실패해도 나머지는 쓴다
                print(f"  [검색 실패] '{term}' ({kind}): {type(items).__name__}")
                continue
            results.append(_to_metas(items, term, cutoff))

    # 관련 후보 안에서는 최신 기사부터 (게시일을 모르는 기사는 맨 뒤). 같은 날짜면 관련도순 순서가 유지된다
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    by_relevance = [sorted(results, key=lambda m: m.pub_date or oldest, reverse=True) for results in by_relevance]

    picked = _interleave(by_relevance, max_articles)
    if len(picked) < max_articles:              # 모자란 만큼 최신순 결과로 채운다 (이미 뽑은 기사는 건너뜀)
        seen = {m.link for m in picked}
        rest = [[m for m in results if m.link not in seen] for results in by_date]
        picked += _interleave(rest, max_articles - len(picked))
    return picked


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
    "#news_body_area",                 # 프라임경제 (측정 2026-11-11: 본문 영역을 못 찾아 61자만 잡혔다)
    "div.newsview div.content",        # bnt뉴스 (<p> 없이 div 안에 바로 본문이 있어서 예전에는 0자였다)
    "article",                      # 마지막 수단 (사이드바 기사 목록까지 잡힐 수 있음)
]

_MIN_CONTENT_LENGTH = 100
_MAX_CONTENT_LENGTH = 6000       # 과도한 컨텍스트 방지
MAX_IMAGES_PER_ARTICLE = 5
# 아이콘·로고·기자 프로필 사진 등 기사 내용과 상관없는 이미지 주소
_SKIP_IMAGE_RE = re.compile(r"(icon|logo|btn|banner|blank|spacer|avatar|profile|default-user|byline|\.gif($|\?))", re.I)


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


# 기사 페이지 하나를 기다리는 최대 시간(초). 설정값(CRAWL_TIMEOUT_SECONDS, 기본 10초)이 더 짧으면 그 값을 쓴다.
#   측정(2026-11-11, 기사 257건): 본문을 가져온 요청 중 가장 느린 것이 1.93초였다 (5초를 넘긴 성공 0건).
#   crawl_articles는 모든 기사가 끝나야 끝나므로, 응답 없는 사이트 하나가 전체를 타임아웃만큼 붙잡는다
#   (같은 기사 목록을 반복해 받을 때 가끔 한 건이 멈춰 전체가 1.3초 → 10.9초가 됐다). 5초면 그 손해가 절반.
#   이미지 내려받기(download_images)는 파일이 커서 예전 설정값을 그대로 쓴다.
CRAWL_PAGE_TIMEOUT_SECONDS = 5

_lenient_ssl: ssl.SSLContext | None = None


def _lenient_ssl_context() -> ssl.SSLContext:
    """인증서 검사를 하지 않고 오래된 암호화 방식도 받아 주는 연결 설정. 보안 연결이 실패한 사이트에만 쓴다.

    측정(2026-11-11, 기사 257건): 실패 13건 중 7건이 보안 연결(SSL) 실패였다. 다시 받아도 똑같이 실패한다.
      - 인증서 검증 실패 5건 (viva100 · kbmaeil · metroseoul): 사이트가 중간 인증서를 빼고 보내서
        파이썬이 검증하지 못한다 (브라우저는 빠진 인증서를 알아서 받아 오므로 문제없이 열린다).
      - 연결 협상 실패 2건 (kbench): 사이트가 오래된 암호화 방식만 지원한다.
    이 설정으로 받으면 7건 모두 본문을 가져온다.

    검사를 끄면 중간에서 누가 내용을 바꿔도 알 수 없다. 공개된 기사 본문만 읽고 비밀번호 같은 것을
    보내지 않으므로 감수한다. 처음부터 끄지 않고, 정상 연결이 실패했을 때만 쓴다.
    """
    global _lenient_ssl
    if _lenient_ssl is None:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        context.set_ciphers("DEFAULT:@SECLEVEL=0")                 # 오래된 암호화 방식 허용
        context.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
        with warnings.catch_warnings():                             # TLS 1.0은 "오래됨" 경고가 뜬다
            warnings.simplefilter("ignore", DeprecationWarning)
            context.minimum_version = ssl.TLSVersion.TLSv1
        _lenient_ssl = context
    return _lenient_ssl


async def _get_html(session: aiohttp.ClientSession, url: str) -> str | None:
    """기사 페이지를 받는다. 보안 연결(SSL) 문제로 실패하면 느슨한 설정으로 한 번 더 받는다."""
    timeout = aiohttp.ClientTimeout(total=min(settings.crawl_timeout_seconds, CRAWL_PAGE_TIMEOUT_SECONDS))
    for lenient in (False, True):
        options = {"ssl": _lenient_ssl_context()} if lenient else {}
        try:
            async with session.get(url, headers=_HEADERS, timeout=timeout, **options) as resp:
                if resp.status != 200:
                    return None
                return _decode_html(await resp.read(), resp.charset)
        except aiohttp.ClientSSLError:          # 인증서 검증 실패 · 연결 협상 실패 → 느슨한 설정으로 다시
            continue
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None
    return None


# ---------------------------------------------------------------------------
# HTML 해석을 여러 프로세스에서
#
# 측정(2026-11-10, 기사 91건): 크롤링 시간의 대부분은 네트워크가 아니라 BeautifulSoup 해석이다
#   (한 건 평균 77ms, 91건 합 7.0초. 다운로드는 동시에 되므로 2~3초면 끝남).
# 파이썬은 한 프로세스 안에서 계산을 한 번에 하나만 하므로(GIL) 스레드로 옮기면 오히려 느려졌다
#   (측정: 7.0초 → 12.1초). 그래서 프로세스 여러 개(CPU 코어)에 나눠 해석한다.
# 해석 함수(parse_article_html)는 그대로라 결과는 예전과 같다.
#
# 주의: Windows는 새 프로세스가 실행 파일을 다시 읽는다. crawl_articles를 부르는 실행 파일은
#   if __name__ == "__main__": 안에서 실행해야 한다 (run.py · admin.py · evaluate.py는 이미 그렇다).
#   프로세스를 못 쓰는 상황이면 자동으로 예전처럼 이 프로세스에서 해석한다.
# ---------------------------------------------------------------------------

PARSE_PROCESSES = min(8, os.cpu_count() or 1)   # 0 또는 1이면 프로세스를 쓰지 않는다 (예전 방식)
_parse_pool: ProcessPoolExecutor | None = None


def _get_parse_pool() -> ProcessPoolExecutor | None:
    global _parse_pool
    if _parse_pool is None and PARSE_PROCESSES > 1:
        _parse_pool = ProcessPoolExecutor(max_workers=PARSE_PROCESSES)
    return _parse_pool


_parse_pool_warmed = False


def warm_up_parser() -> None:
    """해석용 프로세스를 미리 띄워 둔다 (프로세스 시작에 약 1초). 검색하는 동안 불러 두면 기다림이 겹친다.
    여러 번 불러도 처음 한 번만 띄운다."""
    global _parse_pool_warmed
    pool = _get_parse_pool()
    if pool is None or _parse_pool_warmed:
        return
    _parse_pool_warmed = True
    try:
        for _ in range(PARSE_PROCESSES):
            pool.submit(parse_article_html, "")
    except (BrokenProcessPool, RuntimeError):
        _disable_parse_pool()


def _disable_parse_pool() -> None:
    global _parse_pool, PARSE_PROCESSES
    PARSE_PROCESSES = 0
    if _parse_pool is not None:
        _parse_pool.shutdown(wait=False, cancel_futures=True)
    _parse_pool = None


async def _parse_html(html: str) -> tuple[str, list[ArticleImage], str | None]:
    pool = _get_parse_pool()
    if pool is not None:
        try:
            return await asyncio.get_running_loop().run_in_executor(pool, parse_article_html, html)
        except (BrokenProcessPool, RuntimeError):
            _disable_parse_pool()       # 프로세스를 못 쓰면 이후로는 예전 방식으로
    return parse_article_html(html)


async def _fetch_and_parse(
    session: aiohttp.ClientSession, meta: NewsMeta, semaphore: asyncio.Semaphore
) -> Article | None:
    urls = [u for u in (meta.naver_link, meta.link) if u]
    for url in urls:
        async with semaphore:                   # 동시 다운로드 수만 제한한다 (해석하는 동안 자리를 비켜 줌)
            html = await _get_html(session, url)
        if not html:
            continue
        content, images, press = await _parse_html(html)
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


# 동시에 내려받는 기사 수. 측정(2026-11-10, 처음 받는 기사 100건): 10% 정도가 3초 넘게 걸린다
#   (처음 접속하는 언론사 사이트의 주소 찾기·연결 재시도로 보임. 같은 기사를 두 번째 받으면 0.3초 안쪽).
#   동시 8개면 느린 다운로드가 자리를 막아 나머지가 줄을 선다 → 24개로. 해석은 별도 프로세스라 부딪히지 않는다.
# 측정(2026-11-11): 24개일 때 기사 한 건이 자리를 기다리는 시간(중앙 0.45초)이 다운로드(중앙 0.28초)보다 길었다.
#   같은 기사 목록으로 비교: 100건은 차이가 오차 범위(1.2~1.7초), 257건은 24개 3.3~3.8초 → 64개 2.7~3.5초.
#   한 사이트에 여는 연결 수(CRAWL_PER_HOST)는 그대로라 사이트 하나에 몰리지는 않는다.
CRAWL_CONCURRENCY = 64
CRAWL_PER_HOST = 8          # 한 사이트에 동시에 여는 연결 수 (네이버 뉴스에는 예전과 같은 8개까지만)


async def crawl_articles(metas: list[NewsMeta], concurrency: int = CRAWL_CONCURRENCY) -> list[Article]:
    """기사 정보 목록을 받아 병렬로 본문을 가져온다. 본문을 못 가져온 기사는 빠진다. (결과는 metas 순서)"""
    warm_up_parser()
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit_per_host=CRAWL_PER_HOST)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = await asyncio.gather(*(_fetch_and_parse(session, m, semaphore) for m in metas))
    return [a for a in results if a is not None]


# ---------------------------------------------------------------------------
# 이미지 파일 내려받기
#
# 1단계(수집)에서는 이미지 주소·사진 설명만 모아 DB에 저장한다. 이미지 파일은 DB에 저장하지 않는다
# (팀 결정: Supabase 무료 용량 500MB를 아끼기 위해. article_images에는 주소·캡션·LLM 설명만 있다).
#
# 파일은 4단계에서 "관련" 판정을 받은 기사의 이미지 설명을 만들 때만, 그때 이 함수로 받아
# 메모리에서 LLM에 보내고 버린다. 걸러질 기사의 이미지는 받지 않는다.
# ---------------------------------------------------------------------------

MAX_IMAGE_BYTES = 5 * 1024 * 1024   # 5MB보다 큰 파일은 받지 않는다 (DB 용량·LLM 입력 한도)
# 5KB보다 작은 파일은 버린다. 실제로 받아 보니 이런 파일은 기자 프로필 사진, 작은 그래픽,
# 자리만 차지하는 미리보기 이미지였다 (기사 사진은 보통 수십 KB 이상). 4단계 이미지 설명에 쓸모가 없다.
MIN_IMAGE_BYTES = 5 * 1024


async def _download_one(
    session: aiohttp.ClientSession, image: ArticleImage, referer: str, semaphore: asyncio.Semaphore
) -> bool:
    if image.data is not None:          # 이미 받은 이미지는 건너뛴다
        return True
    # 일부 사이트는 다른 사이트에서 이미지를 직접 가져가는 것을 막아서, 기사 주소를 Referer로 같이 보낸다
    headers = {**_HEADERS, "Referer": referer}
    async with semaphore:
        try:
            timeout = aiohttp.ClientTimeout(total=settings.crawl_timeout_seconds)
            async with session.get(image.url, headers=headers, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
                if resp.status != 200 or not content_type.startswith("image/"):
                    return False
                if resp.content_length and resp.content_length > MAX_IMAGE_BYTES:
                    return False
                # 끝까지 조각조각 읽는다. 예전 resp.content.read(n)은 "n바이트까지 전부"가 아니라
                # "지금 도착해 있는 만큼(n 이하)"만 돌려줘서, 큰 이미지가 중간에 잘린 채 저장됐다
                # (측정: 1200×995 JPEG가 16KB로 잘림 → OCR·이미지 해석 실패).
                parts, size = [], 0
                async for part in resp.content.iter_chunked(64 * 1024):
                    size += len(part)
                    if size > MAX_IMAGE_BYTES:
                        return False
                    parts.append(part)
                data = b"".join(parts)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False
    if not data or not (MIN_IMAGE_BYTES <= len(data) <= MAX_IMAGE_BYTES):
        return False
    image.data, image.content_type = data, content_type
    return True


async def download_images(articles: list[Article], concurrency: int = 8) -> tuple[int, int]:
    """기사들의 이미지 파일을 내려받아 ArticleImage.data에 채운다.

    받지 못한 이미지(주소 오류, 이미지가 아님, 5KB 미만·5MB 초과)는 data가 None으로 남는다.
    돌려주는 값: (받은 개수, 전체 이미지 개수)
    """
    semaphore = asyncio.Semaphore(concurrency)
    jobs = [(img, a.link) for a in articles for img in a.images]
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(_download_one(session, img, ref, semaphore) for img, ref in jobs))
    return sum(results), len(jobs)


async def collect_articles(
    selected: SelectedSense,
    period_days: int,
    max_articles: int = DEFAULT_MAX_ARTICLES,
) -> tuple[list[NewsMeta], list[Article]]:
    """검색 + 크롤링을 한 번에. 검색 결과(metas)도 같이 돌려준다 (실험의 Baseline = API 결과 그대로)."""
    metas = await search_news(selected, period_days, max_articles)
    articles = await crawl_articles(metas)
    return metas, articles


@dataclass
class CollectResult:
    """collect_to_db 결과. topic_id로 이후 단계가 DB에서 기사를 꺼내 쓴다."""
    topic_id: int
    searched: int        # 네이버 검색으로 찾은 기사 수 (기간 안, 중복 링크 제외)
    reused: int          # DB에 이미 있어서 크롤링 없이 topic에 연결만 한 기사 수
    crawled: int         # 새로 크롤링해 저장한 기사 수
    crawl_failed: int    # 본문을 못 가져와서 저장하지 못한 기사 수 (다음 수집 때 다시 시도됨)
    images: int          # 새로 저장한 기사의 이미지 수 (주소·캡션)

    def summary(self) -> str:
        return (f"검색 {self.searched}건 → DB 재사용 {self.reused}건 + 새로 저장 {self.crawled}건"
                f" (본문 실패 {self.crawl_failed}건, 이미지 {self.images}개)")


async def collect_to_db(
    selected: SelectedSense,
    period_days: int,
    max_articles: int = DEFAULT_MAX_ARTICLES,
) -> CollectResult:
    """
    [1단계 DB 버전] 고른 의미의 topic을 만들거나 재사용하고, 기간 안 기사를 DB에 채운다.

    같은 topic을 다시 요청하면: 검색은 다시 하지만(API 호출은 무료), DB에 이미 있는 기사는
    크롤링하지 않고 topic에 연결만 한다. DB에 없는 기사(주로 지난 수집 이후 새로 나온 기사)만 크롤링해 저장한다.
    이후 단계는 db.get_topic_articles(topic_id, period_days=...)로 발행일 기준 기간 안 기사를 꺼내 쓴다.

    저장되는 곳 (schema.sql)
      topics          고른 의미 (정의·별칭·연관어·배제어·다른 의미)
      articles        제목·링크·언론사·발행일·요약·본문·수집 시각
                      (네이버 뉴스 링크는 크롤링할 때만 쓰고 저장하지 않는다. 출처·중복 확인은 link 기준)
      article_images  이미지 주소·캡션·기사 안 순서 (이미지 파일은 저장하지 않음)
      article_topics  기사 ↔ topic 연결 + 이 기사를 찾은 검색어
    """
    from qupict import db

    warm_up_parser()        # 해석용 프로세스는 검색하는 동안 미리 띄운다
    # topic 저장(DB 연결 약 1.6초)과 네이버 검색은 서로 상관없어서 동시에 한다
    with timing.timed("topic 저장(DB) + 검색(네이버 API), 동시") as t:
        (topic_id, topic_sec), metas = await asyncio.gather(
            asyncio.to_thread(timed_call, db.save_topic, selected),
            search_news(selected, period_days, max_articles),
        )
        t.note = (f"topic 저장 {topic_sec:.2f}초, 검색어 {len(build_search_terms(selected))}개 → "
                  f"기사 {len(metas)}건")
    with timing.timed("이미 있는 기사 연결(DB)") as t:
        new_metas = db.link_known_articles(topic_id, metas)
        t.note = f"재사용 {len(metas) - len(new_metas)}건"
    with timing.timed("본문 크롤링") as t:
        articles = await crawl_articles(new_metas)
        t.note = f"성공 {len(articles)} / 시도 {len(new_metas)}건"
    with timing.timed("기사 저장(DB)") as t:
        db.save_articles(topic_id, articles)
        db.mark_topic_collected(topic_id)
        t.note = f"{len(articles)}건"
    return CollectResult(
        topic_id=topic_id,
        searched=len(metas),
        reused=len(metas) - len(new_metas),
        crawled=len(articles),
        crawl_failed=len(new_metas) - len(articles),
        images=sum(len(a.images) for a in articles),
    )
