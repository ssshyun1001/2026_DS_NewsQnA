"""
[5단계] 하이브리드 검색의 키워드(BM25) 쪽 + 두 순위 합치기(RRF)

experiments/exp_b_hyde_hybrid.py의 "hybrid" 방식을 서비스(search_topic)에 쓰도록 옮겼다.
  실험 결과 (골든셋 87문항, 2026-11-16): Recall@5 / MRR / 질문당 시간
    base    0.333 / 0.280 / 0.93초      hyde         0.402 / 0.337 / 2.38초
    hybrid  0.609 / 0.508 / 0.93초      hyde_hybrid  0.598 / 0.501 / 2.39초
  → HyDE 없이 벡터 + BM25를 합치는 hybrid가 가장 정확하고 빠르다.

  BM25    검색 대상 청크(topic · 판정 관련/보류, 발행일 기간)를 형태소(kiwi)로 나눠 키워드 점수를 매긴다.
          토큰 = 체언·용언·외국어·숫자·어근 (실험과 같음), k1 = 1.2, b = 0.75
  RRF     벡터 순위와 BM25 순위를 Reciprocal Rank Fusion으로 합친다: 점수 = Σ 1 / (RRF_K + 순위)

실험과 다른 점: 실험은 기사 단위로 합쳤고, 서비스는 답변에 청크를 넣으므로 청크 단위로 합친다.

BM25 인덱스는 topic마다 메모리에 만들어 두고, 검색 대상 청크 수가 바뀌면(새 기사 인덱싱 · 이미지 글자 추가 ·
판정 변경) 다시 만든다. 처음 만들 때 topic 크기에 따라 0.3~6초 걸린다 (실험 측정).
"""
from __future__ import annotations

import math
import threading
from collections import Counter
from datetime import datetime
from typing import Optional

RRF_K = 60
BM25_K1 = 1.2
BM25_B = 0.75
_TOKEN_TAGS = ("N", "V", "SL", "SN", "XR")     # 체언 · 용언 · 외국어 · 숫자 · 어근


def tokens(text: str) -> list[str]:
    """BM25에 쓰는 토큰 (실험 exp_b_hyde_hybrid.tokens와 같은 규칙)."""
    from qupict.filtering import get_kiwi

    return [t.form.casefold() for t in get_kiwi().tokenize(text or "") if t.tag.startswith(_TOKEN_TAGS)]


class BM25Index:
    """청크(db.ChunkHit) 목록으로 만든 BM25 인덱스."""

    def __init__(self, hits: list):
        self.docs: list[tuple[object, Counter, int]] = []
        frequencies: Counter = Counter()
        total = 0
        for h in hits:
            terms = tokens(f"{h.article.title} {h.content}")
            counts = Counter(terms)
            self.docs.append((h, counts, len(terms)))
            frequencies.update(counts.keys())
            total += len(terms)
        n = len(self.docs)
        self.average_length = total / n if total else 1.0
        self.idf = {term: math.log1p((n - f + 0.5) / (f + 0.5)) for term, f in frequencies.items()}

    def rank(self, question: str, limit: int, since: Optional[datetime] = None,
             until: Optional[datetime] = None) -> list:
        """질문 키워드 점수가 0보다 큰 청크를 점수 높은 순으로 최대 limit개.
        since · until을 주면 그 발행일 기간의 청크만 (발행일이 없는 청크는 빠진다, 벡터 검색과 같게)."""
        terms = sorted(set(tokens(question)).intersection(self.idf))
        if not terms:
            return []
        scored = []
        for h, counts, length in self.docs:
            if since is not None:
                pub = h.article.pub_date
                if pub is None or pub < since or pub > until:
                    continue
            norm = BM25_K1 * (1 - BM25_B + BM25_B * length / self.average_length)
            score = sum(self.idf[t] * c * (BM25_K1 + 1) / (c + norm) for t in terms if (c := counts[t]))
            if score > 0:
                scored.append((score, h))
        scored.sort(key=lambda x: -x[0])                     # 안정 정렬: 동점이면 저장 순서
        return [h for _, h in scored[:limit]]


def rrf_fuse(*rankings: list) -> list:
    """여러 청크 순위를 RRF로 합친다. 같은 청크(기사 link + 내용)는 점수를 더하고,
    처음 나온 순위 목록의 hit을 쓴다 (벡터 순위를 먼저 넣으면 유사도 값이 남는다). 동점이면 먼저 나온 순서."""
    scores: dict[tuple[str, str], float] = {}
    first: dict[tuple[str, str], tuple[int, object]] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, 1):
            key = (h.article.link, h.content)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key not in first:
                first[key] = (len(first), h)
    return [first[key][1] for key in sorted(scores, key=lambda key: (-scores[key], first[key][0]))]


# ---------------------------------------------------------------------------
# topic별 인덱스 (메모리 캐시)
# ---------------------------------------------------------------------------

_cache: dict[tuple[int, tuple[str, ...]], tuple[int, BM25Index]] = {}
_lock = threading.Lock()


def topic_index(topic_id: int, statuses: tuple[str, ...]) -> BM25Index:
    """topic의 검색 대상 청크로 만든 BM25 인덱스. 청크 수가 그대로면 만들어 둔 것을 쓴다."""
    from qupict import db

    key = (topic_id, tuple(statuses))
    count = db.count_search_chunks(topic_id, statuses)
    cached = _cache.get(key)
    if cached is not None and cached[0] == count:
        return cached[1]
    with _lock:                                   # 동시에 들어온 질문이 같은 인덱스를 두 번 만들지 않게
        cached = _cache.get(key)
        if cached is not None and cached[0] == count:
            return cached[1]
        index = BM25Index(db.search_chunk_rows(topic_id, statuses))
        _cache[key] = (count, index)
        return index


def search(topic_id: int, question: str, limit: int, period_days: Optional[int] = None,
           statuses: tuple[str, ...] = ("관련", "보류")) -> list:
    """topic 안에서 질문 키워드(BM25)로 청크를 최대 limit개 찾는다 (점수 높은 순)."""
    from qupict import db

    since, until = db._window(period_days, None)
    return topic_index(topic_id, statuses).rank(question, limit, since, until)
