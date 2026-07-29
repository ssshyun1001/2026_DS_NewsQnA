"""
2. 실시간 뉴스 수집 (메타데이터) - Naver News Search API

지정된 기간 필터링은 API 응답의 pubDate를 기준으로 후처리한다.
(Naver 검색 API는 기간 파라미터를 직접 지원하지 않으므로 결과를 받은 뒤 필터링한다.)

수집 개수 정책:
  키워드별로 개수를 나눠 배분하지 않는다. 검색 우선순위(search_terms)를 순서대로
  검색하며, 전체 누적 기사 수가 TOTAL_ARTICLE_CAP(기본 100)에 도달하면 더 이상
  검색하지 않는다.

검색어 우선순위:
  query.filter_terms(사용자가 지정한 검증 키워드)가 있으면, "원문 키워드 + 검증어"
  조합 쿼리를 최우선으로 검색한다. 그 다음 원문 키워드(query.optimized_keywords)만
  단독으로 검색한다.

  이렇게 하는 이유: "강남"처럼 동음이의어(지역명/인물명)인 주제어를 단독으로
  최신순(sort=date) 검색하면, 그날그날 발행량이 많은 쪽(예: 강남구 행정 기사)이
  전체 cap(기본 100건)을 채워버려서, 사용자가 실제로 찾는 대상(예: 방송인 강남)의
  기사가 검색 결과에 아예 안 들어올 수 있다. "강남 방송인"처럼 검증어를 쿼리에
  같이 넣으면 그 시점에 이미 의미가 좁혀진 상태로 검색되므로 이 문제를 줄일 수 있다.
  (완전한 해결책은 아니다 — 조합 쿼리도 결과가 0건이면 여전히 원문 키워드 검색에
  의존하게 된다.)
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

_NAVER_MAX_DISPLAY = 100  # Naver 뉴스 검색 API의 display 파라미터 최대 허용값
_TOTAL_ARTICLE_CAP_DEFAULT = 100  # 전체 파이프라인에서 수집할 기사 수 상한


def _strip_html(text: str) -> str:
    return _TAG_RE.sub("", text).replace("&quot;", '"').replace("&amp;", "&")


def _parse_pubdate(raw: str) -> datetime | None:
    # 예: 'Mon, 01 Jan 2024 12:00:00 +0900'
    try:
        return datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %z")
    except (ValueError, TypeError):
        return None


def _build_search_terms(query: SearchQuery) -> list[str]:
    """검색에 사용할 쿼리 문자열들을 우선순위 순서로 만든다.

    1순위: "원문 키워드 + 검증어" 조합 (동음이의어 노이즈를 줄이기 위해 최우선)
    2순위: 원문 키워드 단독 (query.optimized_keywords 순서 그대로)

    조합 쿼리는 원문 키워드 리스트의 1순위 키워드(가장 신뢰도 높은 표기)에만
    검증어를 붙인다. 모든 키워드 x 모든 검증어 조합을 다 만들면 쿼리 수가
    과도하게 늘어나기 때문이다.
    """
    optimized_keywords = list(query.optimized_keywords)
    filter_terms = [
        t.strip() for t in (getattr(query, "filter_terms", None) or []) if t.strip()
    ]

    combined_terms: list[str] = []
    if filter_terms and optimized_keywords:
        primary_keyword = optimized_keywords[0]
        combined_terms = [f"{primary_keyword} {term}" for term in filter_terms]

    # 조합 쿼리 우선, 그 다음 원문 키워드. 중복 문자열은 제거하되 순서는 유지한다.
    ordered = combined_terms + optimized_keywords
    seen: set[str] = set()
    deduped: list[str] = []
    for term in ordered:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


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
    """SearchQuery의 최적화된 키워드들로 뉴스를 검색하고 기간 내 결과만 반환한다.

    검증어(filter_terms)가 있으면 "키워드+검증어" 조합 쿼리를 먼저 검색해
    동음이의어 노이즈를 줄이고, 그 다음 원문 키워드 단독 검색으로 나머지
    cap을 채운다. 전체 누적 기사 수가 total_cap에 도달하면 검색을 중단한다.
    """
    total_cap = getattr(settings, "max_total_articles", _TOTAL_ARTICLE_CAP_DEFAULT)
    cutoff = datetime.now(timezone.utc) - timedelta(days=query.period_days)

    results: list[NewsMeta] = []
    seen_links: set[str] = set()

    search_terms = _build_search_terms(query)

    async with aiohttp.ClientSession() as session:
        for keyword in search_terms:
            if len(results) >= total_cap:
                break

            display = min(_NAVER_MAX_DISPLAY, total_cap)
            items = await _fetch_one_keyword(session, keyword, display)

            for item in items:
                if len(results) >= total_cap:
                    break

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

    return results[:total_cap]