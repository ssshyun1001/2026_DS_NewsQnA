"""
2. 실시간 뉴스 수집 (메타데이터) - Naver News Search API

지정된 기간 필터링은 API 응답의 pubDate를 기준으로 후처리한다.
(Naver 검색 API는 기간 파라미터를 직접 지원하지 않으므로 결과를 받은 뒤 필터링한다.)
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from app.schemas import NewsMeta, SearchQuery

_NAVER_NEWS_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
_TAG_RE = re.compile(r"<.*?>")


def _strip_html(text: str) -> str:
    return _TAG_RE.sub("", text).replace("&quot;", '"').replace("&amp;", "&")


def _parse_pubdate(raw: str) -> datetime | None:
    # 예: 'Mon, 01 Jan 2024 12:00:00 +0900'
    try:
        return datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %z")
    except (ValueError, TypeError):
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _fetch_one_keyword(
    session: aiohttp.ClientSession, keyword: str, display: int
) -> list[dict]:
    headers = {
        "X-NCP-APIGW-API-KEY-ID": settings.naver_client_id,
        "X-NCP-APIGW-API-KEY": settings.naver_client_secret,
    }
    params = {
        "query": keyword,
        "display": display,
        "start": 1,
        "sort": "date",
        "format": "json",
    }

    async with session.get(_NAVER_NEWS_URL, headers=headers, params=params) as resp:
        resp.raise_for_status()
        data = await resp.json(content_type=None)
        return data.get("items", [])


async def search_news(query: SearchQuery) -> list[NewsMeta]:
    """SearchQuery의 최적화된 키워드들로 뉴스를 검색하고 기간 내 결과만 반환한다."""
    display_per_keyword = max(5, settings.max_crawl_articles // max(len(query.optimized_keywords), 1))
    cutoff = datetime.now(timezone.utc) - timedelta(days=query.period_days)

    results: list[NewsMeta] = []
    seen_links: set[str] = set()

    async with aiohttp.ClientSession() as session:
        for keyword in query.optimized_keywords:
            items = await _fetch_one_keyword(session, keyword, display_per_keyword)
            for item in items:
                link = item.get("originallink") or item.get("link")
                if not link or link in seen_links:
                    continue

                pub_date = _parse_pubdate(item.get("pubDate", ""))
                if pub_date and pub_date < cutoff:
                    continue

                seen_links.add(link)
                results.append(
                    NewsMeta(
                        title=_strip_html(item.get("title", "")),
                        link=link,
                        pub_date=pub_date,
                        description=_strip_html(item.get("description", "")),
                    )
                )

    return results[: settings.max_crawl_articles]
