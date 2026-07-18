"""
3. 비동기 HTML 크롤링 - 뉴스 링크에서 본문 텍스트를 추출한다.

언론사마다 HTML 구조가 다르므로, 실무에서는 언론사별 셀렉터 매핑을 늘려가는 것을 권장한다.
여기서는 다수 언론사에 공통적으로 잘 맞는 범용 셀렉터 + 폴백 전략을 사용한다.
"""
from __future__ import annotations

import asyncio

import aiohttp
from bs4 import BeautifulSoup

from config import settings
from app.schemas import Article, NewsMeta

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# 언론사별 본문 셀렉터 (필요에 따라 확장)
_SELECTOR_CANDIDATES = [
    "#dic_area",  # 네이버 뉴스 (연예/스포츠 제외 일반)
    "#newsct_article",
    "div.article_body",
    "div#articleBodyContents",
    "div.news_end",
    "article",
]

_MIN_CONTENT_LENGTH = 100


def _extract_body(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for selector in _SELECTOR_CANDIDATES:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(separator=" ", strip=True)
            if len(text) >= _MIN_CONTENT_LENGTH:
                return text

    # 폴백: <p> 태그를 모두 모아본다.
    paragraphs = soup.find_all("p")
    fallback = " ".join(p.get_text(strip=True) for p in paragraphs)
    return fallback


async def _fetch_and_parse(
    session: aiohttp.ClientSession, meta: NewsMeta, semaphore: asyncio.Semaphore
) -> Article | None:
    async with semaphore:
        try:
            timeout = aiohttp.ClientTimeout(total=settings.crawl_timeout_seconds)
            async with session.get(meta.link, headers=_HEADERS, timeout=timeout) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text(errors="ignore")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

    content = _extract_body(html)
    if len(content) < _MIN_CONTENT_LENGTH:
        return None

    return Article(
        title=meta.title,
        link=meta.link,
        press=meta.press,
        pub_date=meta.pub_date,
        content=content[:6000],  # 과도한 컨텍스트 방지를 위한 상한
    )


async def crawl_articles(metas: list[NewsMeta], concurrency: int = 8) -> list[Article]:
    """뉴스 메타데이터 목록을 받아 병렬로 본문을 크롤링한다. 실패한 기사는 제외한다."""
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_and_parse(session, meta, semaphore) for meta in metas]
        results = await asyncio.gather(*tasks)

    return [article for article in results if article is not None]
