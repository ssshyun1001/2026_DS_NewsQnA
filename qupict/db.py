"""
DB 연결 및 저장·조회 (PostgreSQL + pgvector)

파이프라인 코드는 기사 id 대신 link(언론사 원문 주소)로 기사를 가리킨다.
articles.link가 UNIQUE라서 link만 알면 된다. id는 이 파일 안에서만 다룬다.

단계별로 쓰는 함수
  초기화       init_db()
  [0단계]      get_sense_candidates / save_sense_candidates
               save_topic / get_topic / mark_topic_collected
  [1단계]      link_known_articles  → DB에 이미 있는 기사는 topic에 연결만 하고, 새로 크롤링할 것만 돌려준다
               save_articles
  [2·3단계]    get_topic_articles / save_article_embeddings
               reset_topic_filter / save_duplicates / save_judgments / save_labels
  [4단계]      get_images_to_describe / save_image_description
               get_articles_without_chunks / save_chunks
  [5단계]      search_chunks

기간은 topic에 저장하지 않는다. 조회할 때 period_days를 주면 기준 시각(now, 기본 = 지금 = 질문한 시각)에서
period_days일 전 ~ now 사이에 발행된(pub_date) 기사만 돌려준다.
"""
from __future__ import annotations

import atexit
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from qupict.config import settings
from qupict.schemas import Article, ArticleImage, NewsMeta, SelectedSense, Sense, SenseCandidates

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"
EMBEDDING_DIM = 1536  # schema.sql의 vector(1536)과 같아야 한다

STATUSES = ("관련", "무관", "보류")


# ---------------------------------------------------------------- 연결

def _open(**kwargs) -> psycopg.Connection:
    if not settings.database_url:
        raise EnvironmentError("DATABASE_URL이 설정되지 않았습니다. .env 파일을 확인하세요 (.env.example 참고).")
    # prepare_threshold=None: 공용 DB의 커넥션 풀러(Supabase pooler 등)는 prepared statement를
    # 지원하지 않는 경우가 있어서 끈다. 로컬 Docker에서도 문제없다.
    return psycopg.connect(settings.database_url, prepare_threshold=None, connect_timeout=10, **kwargs)


# 커넥션 풀: 연결을 매번 새로 열지 않고 열어 둔 연결을 돌려 쓴다.
#   측정(2026-11-10, 팀 Supabase): 새 연결 1번 = 약 1.4~1.6초 (연결 + 인증), 열린 연결에서 쿼리 1번 = 약 80ms.
#   예전에는 db 함수를 부를 때마다 새로 연결해서, 함수 하나에 1.5초 이상이 걸렸다.
# connect()의 사용법·동작(정상 종료면 commit, 예외면 rollback, 결과는 dict 행)은 예전과 같다.
POOL_MIN_SIZE = 2           # 항상 열어 두는 연결 수
POOL_MAX_SIZE = 8           # 동시에 쓸 수 있는 최대 연결 수 (Supabase 무료 플랜 동시 연결 한도보다 넉넉히 낮게)
POOL_CHECK_AFTER = 30.0     # 이 초보다 오래 쉬던 연결은 쓰기 전에 살아 있는지 확인한다 (서버가 끊었을 수 있음)

_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def _check(conn: psycopg.Connection) -> None:
    """빌려주기 전 확인. 최근에 쓴 연결은 확인을 건너뛴다 (확인도 왕복 1번 = 약 80ms라서).
    오래 쉬던 연결이 끊겨 있으면 예외가 나고, 풀이 그 연결을 버리고 새 연결을 준다."""
    if time.monotonic() - getattr(conn, "_qupict_last_used", 0.0) > POOL_CHECK_AFTER:
        conn.execute("SELECT 1")


def _reset(conn: psycopg.Connection) -> None:
    conn._qupict_last_used = time.monotonic()    # 풀에 돌아온 시각 (_check에서 씀)


def _get_pool() -> ConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            if not settings.database_url:
                raise EnvironmentError("DATABASE_URL이 설정되지 않았습니다. .env 파일을 확인하세요 (.env.example 참고).")
            _pool = ConnectionPool(
                settings.database_url,
                min_size=POOL_MIN_SIZE, max_size=POOL_MAX_SIZE,
                # _open()과 같은 설정 (prepare_threshold=None: Supabase 풀러는 prepared statement를 못 쓸 수 있음)
                kwargs={"prepare_threshold": None, "connect_timeout": 10, "row_factory": dict_row},
                configure=_configure, check=_check, reset=_reset,
                max_idle=300, max_lifetime=1800,   # 5분 넘게 놀거나 30분 지난 연결은 닫는다 (min_size 넘는 것만)
                name="qupict", open=True,
            )
            atexit.register(close_pool)
        return _pool


def warm_up() -> None:
    """풀을 미리 열어 둔다 (첫 연결 약 1.4초가 뒤에서 진행되고 바로 돌아온다). 웹 서버를 켤 때 부른다."""
    if settings.database_url:
        _get_pool()


def close_pool() -> None:
    """열어 둔 연결을 모두 닫는다 (프로그램이 끝날 때 자동으로 불린다)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close(timeout=2)
            _pool = None


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """DB 연결 (풀에서 빌려 쓰고 돌려준다). with 블록이 정상 종료되면 commit, 예외가 나면 rollback 된다."""
    with _get_pool().connection() as conn:
        yield conn


def init_db() -> None:
    """schema.sql을 실행해 확장과 테이블을 만든다. 여러 번 실행해도 안전하다."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with _open() as conn:  # vector 확장이 생기기 전이라 register_vector를 하지 않는다
        conn.execute(sql)


def _vec(embedding: Optional[Iterable[float]]) -> Optional[np.ndarray]:
    if embedding is None:
        return None
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.shape != (EMBEDDING_DIM,):
        raise ValueError(f"임베딩 차원이 {arr.shape}입니다. schema.sql은 {EMBEDDING_DIM}차원으로 되어 있습니다.")
    return arr


def _window(period_days: Optional[int], now: Optional[datetime]) -> tuple[Optional[datetime], Optional[datetime]]:
    """발행일 기간 [now - period_days, now]. period_days가 없으면 조건 없음."""
    if period_days is None:
        return None, None
    now = now or datetime.now(timezone.utc)
    return now - timedelta(days=period_days), now


def _article_ids(conn: psycopg.Connection, links: Iterable[str]) -> dict[str, int]:
    links = list(set(links))
    if not links:
        return {}
    rows = conn.execute("SELECT id, link FROM articles WHERE link = ANY(%s)", (links,)).fetchall()
    return {r["link"]: r["id"] for r in rows}


# ---------------------------------------------------------------- [0단계] 의미 후보

def get_sense_candidates(query: str) -> Optional[SenseCandidates]:
    """저장된 LLM 의미 후보를 돌려준다. 없으면 None (→ LLM을 불러 만들고 save_sense_candidates)."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT name, definition, aliases, related_terms, model FROM query_senses
               WHERE query = %s AND NOT is_custom ORDER BY rank""",
            (query,),
        ).fetchall()
    if not rows:
        return None
    senses = [Sense(name=r["name"], definition=r["definition"], aliases=r["aliases"],
                    related_terms=r["related_terms"]) for r in rows]
    return SenseCandidates(query=query, senses=senses, model=rows[0]["model"])


def save_sense_candidates(candidates: SenseCandidates) -> None:
    """LLM 의미 후보를 저장한다. 같은 (검색어, 이름)이 있으면 내용을 갱신한다."""
    with connect() as conn:
        for rank, s in enumerate(candidates.senses, start=1):
            _upsert_sense(conn, candidates.query, s, rank, candidates.model)


def _upsert_sense(conn: psycopg.Connection, query: str, s: Sense, rank: Optional[int], model: str) -> int:
    row = conn.execute(
        """INSERT INTO query_senses (query, rank, name, definition, aliases, related_terms, is_custom, model)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (query, name) DO UPDATE SET
               rank = EXCLUDED.rank, definition = EXCLUDED.definition, aliases = EXCLUDED.aliases,
               related_terms = EXCLUDED.related_terms, is_custom = EXCLUDED.is_custom, model = EXCLUDED.model
           RETURNING id""",
        (query, rank, s.name, s.definition, s.aliases, s.related_terms, s.is_custom, model),
    ).fetchone()
    return row["id"]


# ---------------------------------------------------------------- topics

def save_topic(selected: SelectedSense) -> int:
    """고른 의미를 topic으로 저장하고 id를 돌려준다. 같은 (검색어, 의미 이름)이 있으면 그 topic을 재사용한다."""
    s = selected.sense
    with connect() as conn:
        sense_row = conn.execute(
            "SELECT id FROM query_senses WHERE query = %s AND name = %s", (selected.query, s.name)
        ).fetchone()
        if sense_row is None and s.is_custom:
            sense_id = _upsert_sense(conn, selected.query, s, None, "")
        else:
            sense_id = sense_row["id"] if sense_row else None
        row = conn.execute(
            """INSERT INTO topics (query, sense_id, sense_name, definition, aliases, related_terms,
                                   is_custom, excluded_terms, other_senses)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (query, sense_name) DO UPDATE SET
                   sense_id = EXCLUDED.sense_id, definition = EXCLUDED.definition,
                   aliases = EXCLUDED.aliases, related_terms = EXCLUDED.related_terms,
                   is_custom = EXCLUDED.is_custom, excluded_terms = EXCLUDED.excluded_terms,
                   other_senses = EXCLUDED.other_senses, updated_at = now()
               RETURNING id""",
            (selected.query, sense_id, s.name, s.definition, s.aliases, s.related_terms, s.is_custom,
             selected.excluded_terms, Jsonb([o.model_dump() for o in selected.other_senses])),
        ).fetchone()
    return row["id"]


@dataclass
class TopicRecord:
    id: int
    selected: SelectedSense
    last_collected_at: Optional[datetime]


def get_topic(query: str, sense_name: str) -> Optional[TopicRecord]:
    with connect() as conn:
        r = conn.execute("SELECT * FROM topics WHERE query = %s AND sense_name = %s",
                         (query, sense_name)).fetchone()
    if r is None:
        return None
    selected = SelectedSense(
        query=r["query"],
        sense=Sense(name=r["sense_name"], definition=r["definition"], aliases=r["aliases"],
                    related_terms=r["related_terms"], is_custom=r["is_custom"]),
        excluded_terms=r["excluded_terms"],
        other_senses=[Sense(**o) for o in r["other_senses"]],
    )
    return TopicRecord(id=r["id"], selected=selected, last_collected_at=r["last_collected_at"])


def mark_topic_collected(topic_id: int, at: Optional[datetime] = None) -> None:
    with connect() as conn:
        conn.execute("UPDATE topics SET last_collected_at = %s WHERE id = %s",
                     (at or datetime.now(timezone.utc), topic_id))


# ---------------------------------------------------------------- [1단계] 기사

def link_known_articles(topic_id: int, metas: list[NewsMeta]) -> list[NewsMeta]:
    """
    검색 결과 중 DB에 이미 있는 기사는 이 topic에 연결만 하고(크롤링 생략),
    DB에 없는 기사만 돌려준다. 돌려받은 것만 crawl_articles → save_articles 하면 된다.
    """
    with connect() as conn:
        known = _article_ids(conn, (m.link for m in metas))
        rows = [(known[m.link], topic_id, m.search_term) for m in metas if m.link in known]
        if rows:
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO article_topics (article_id, topic_id, search_term) VALUES (%s, %s, %s)
                       ON CONFLICT (article_id, topic_id) DO NOTHING""",
                    rows,
                )
    return [m for m in metas if m.link not in known]


def save_articles(topic_id: int, articles: list[Article]) -> dict[str, int]:
    """기사·이미지를 저장하고 topic에 연결한다. 이미 있는 기사는 본문 등을 갱신한다. {link: id}를 돌려준다.

    기사 전체를 한꺼번에 보낸다 (executemany: 여러 행을 묶어 한 번에 전송). 예전에는 기사마다 쿼리 3번
    (기사 · 이미지 · topic 연결)을 따로 보내서, 기사 86건이면 왕복 약 260번(× 약 80ms ≈ 20초)이었다.
    지금은 기사 전체 → 이미지 전체 → topic 연결 전체 순서로 3묶음. 행마다 하는 일(SQL)은 예전과 같아서
    저장되는 내용도 같다.
    """
    if not articles:
        return {}
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO articles (link, title, press, pub_date, description, content, collected_at,
                                     embedding, embedding_model)
               VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s, now()), %s, %s)
               ON CONFLICT (link) DO UPDATE SET
                   title = EXCLUDED.title, press = COALESCE(EXCLUDED.press, articles.press),
                   pub_date = COALESCE(EXCLUDED.pub_date, articles.pub_date),
                   description = EXCLUDED.description, content = EXCLUDED.content,
                   collected_at = EXCLUDED.collected_at,
                   embedding = COALESCE(EXCLUDED.embedding, articles.embedding),
                   embedding_model = COALESCE(EXCLUDED.embedding_model, articles.embedding_model)
               RETURNING id""",
            [(a.link, a.title, a.press, a.pub_date, a.description, a.content, a.collected_at,
              _vec(a.embedding), settings.embedding_model if a.embedding is not None else None)
             for a in articles],
            returning=True,
        )
        article_ids = []                     # 넣은 순서대로 기사 id (RETURNING 결과가 행마다 하나씩 온다)
        while True:
            article_ids.append(cur.fetchone()["id"])
            if not cur.nextset():
                break
        ids = {a.link: article_id for a, article_id in zip(articles, article_ids)}
        cur.executemany(
            """INSERT INTO article_images (article_id, position, url, caption) VALUES (%s, %s, %s, %s)
               ON CONFLICT (article_id, url) DO UPDATE SET
                   position = EXCLUDED.position, caption = EXCLUDED.caption""",
            [(article_id, i, img.url, img.caption)
             for a, article_id in zip(articles, article_ids) for i, img in enumerate(a.images)],
        )
        cur.executemany(
            """INSERT INTO article_topics (article_id, topic_id, search_term) VALUES (%s, %s, %s)
               ON CONFLICT (article_id, topic_id) DO NOTHING""",
            [(article_id, topic_id, a.search_term) for a, article_id in zip(articles, article_ids)],
        )
    return ids


def get_topic_articles(
    topic_id: int,
    period_days: Optional[int] = None,
    now: Optional[datetime] = None,
    statuses: Optional[Iterable[str]] = None,
    include_duplicates: bool = True,
    unjudged_only: bool = False,
) -> list[Article]:
    """
    topic에 연결된 기사를 발행일 최신순으로 돌려준다 (임베딩·이미지 포함).
      period_days        : 기준 시각(now, 기본 = 지금)까지 며칠 이내에 발행된 기사만
      statuses           : 예) ["관련"] — 해당 판정만
      include_duplicates : False면 중복(duplicate_of 있음)으로 표시된 기사 제외
      unjudged_only      : 아직 중복 표시도 판정도 없는 기사만 (새로 들어온 기사 처리용)
    """
    where = ["at.topic_id = %(topic_id)s"]
    params: dict = {"topic_id": topic_id}
    since, until = _window(period_days, now)
    if since is not None:
        where.append("a.pub_date BETWEEN %(since)s AND %(until)s")
        params.update(since=since, until=until)
    if statuses is not None:
        where.append("at.status = ANY(%(statuses)s)")
        params["statuses"] = list(statuses)
    if not include_duplicates:
        where.append("at.duplicate_of IS NULL")
    if unjudged_only:
        where.append("at.duplicate_of IS NULL AND at.status IS NULL")

    with connect() as conn:
        rows = conn.execute(
            f"""SELECT a.*, at.search_term FROM articles a
                JOIN article_topics at ON at.article_id = a.id
                WHERE {' AND '.join(where)}
                ORDER BY a.pub_date DESC NULLS LAST, a.id""",
            params,
        ).fetchall()
        images: dict[int, list[ArticleImage]] = {}
        if rows:
            for img in conn.execute(
                "SELECT article_id, url, caption FROM article_images WHERE article_id = ANY(%s) ORDER BY position",
                ([r["id"] for r in rows],),
            ).fetchall():
                images.setdefault(img["article_id"], []).append(ArticleImage(url=img["url"], caption=img["caption"]))

    return [
        Article(
            title=r["title"], link=r["link"], press=r["press"], pub_date=r["pub_date"],
            content=r["content"], description=r["description"], search_term=r["search_term"],
            images=images.get(r["id"], []), collected_at=r["collected_at"],
            embedding=r["embedding"].tolist() if r["embedding"] is not None else None,
        )
        for r in rows
    ]


def save_article_embeddings(articles: list[Article], model: Optional[str] = None) -> None:
    """기사 전체 임베딩을 저장한다 (다음 실행 때 다시 계산하지 않도록)."""
    rows = [(_vec(a.embedding), model or settings.embedding_model, a.link)
            for a in articles if a.embedding is not None]
    with connect() as conn, conn.cursor() as cur:
        cur.executemany("UPDATE articles SET embedding = %s, embedding_model = %s WHERE link = %s", rows)


# ---------------------------------------------------------------- [2·3단계] 판정

def reset_topic_filter(topic_id: int) -> None:
    """규칙을 바꿔 필터링을 처음부터 다시 돌릴 때: 중복 표시와 판정을 지운다 (정답 라벨은 남긴다)."""
    with connect() as conn:
        conn.execute(
            """UPDATE article_topics SET duplicate_of = NULL, status = NULL, stage = NULL,
                   reason = NULL, judged_model = NULL, judged_at = NULL
               WHERE topic_id = %s""",
            (topic_id,),
        )


def save_duplicates(topic_id: int, duplicates: dict[str, str]) -> None:
    """2단계 결과 저장. duplicates = {중복 기사 link: 대표 기사 link}. 중복 기사는 판정을 비운다."""
    with connect() as conn:
        ids = _article_ids(conn, [*duplicates.keys(), *duplicates.values()])
        rows = [(ids[rep], ids[dup], topic_id) for dup, rep in duplicates.items()
                if dup in ids and rep in ids and dup != rep]
        with conn.cursor() as cur:
            cur.executemany(
                """UPDATE article_topics SET duplicate_of = %s, status = NULL, stage = NULL, reason = NULL,
                       judged_model = NULL, judged_at = NULL
                   WHERE article_id = %s AND topic_id = %s""",
                rows,
            )


@dataclass
class Judgment:
    link: str
    status: str                 # '관련' | '무관' | '보류'
    stage: str                  # 'rule' | 'llm'
    reason: str = ""
    model: Optional[str] = None  # LLM 판정이면 모델명


def save_judgments(topic_id: int, judgments: list[Judgment]) -> None:
    """2단계 규칙 / 3단계 LLM 판정 저장. 이전 판정은 덮어쓴다."""
    for j in judgments:
        if j.status not in STATUSES:
            raise ValueError(f"판정 값은 {STATUSES} 중 하나여야 합니다: {j.status!r}")
    with connect() as conn:
        ids = _article_ids(conn, (j.link for j in judgments))
        with conn.cursor() as cur:
            cur.executemany(
                """UPDATE article_topics SET status = %s, stage = %s, reason = %s, judged_model = %s,
                       judged_at = now(), duplicate_of = NULL
                   WHERE article_id = %s AND topic_id = %s""",
                [(j.status, j.stage, j.reason, j.model, ids[j.link], topic_id)
                 for j in judgments if j.link in ids],
            )


def save_labels(topic_id: int, labels: dict[str, str]) -> None:
    """사람이 붙인 정답 라벨 저장. labels = {link: '관련' | '무관' | '보류'}"""
    for v in labels.values():
        if v not in STATUSES:
            raise ValueError(f"라벨 값은 {STATUSES} 중 하나여야 합니다: {v!r}")
    with connect() as conn:
        ids = _article_ids(conn, labels.keys())
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE article_topics SET label = %s, labeled_at = now() WHERE article_id = %s AND topic_id = %s",
                [(v, ids[k], topic_id) for k, v in labels.items() if k in ids],
            )


# ---------------------------------------------------------------- [4단계] 이미지 설명 · 청크

@dataclass
class ImageToDescribe:
    image_id: int
    article_link: str
    url: str
    caption: str


def get_images_to_describe(topic_id: int, statuses: Iterable[str] = ("관련",)) -> list[ImageToDescribe]:
    """판정을 통과한 기사 중 아직 LLM 설명이 없는 이미지."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT i.id, a.link, i.url, i.caption FROM article_images i
               JOIN articles a ON a.id = i.article_id
               JOIN article_topics at ON at.article_id = a.id
               WHERE at.topic_id = %s AND at.status = ANY(%s) AND i.llm_description IS NULL
               ORDER BY a.id, i.position""",
            (topic_id, list(statuses)),
        ).fetchall()
    return [ImageToDescribe(r["id"], r["link"], r["url"], r["caption"]) for r in rows]


def save_image_description(image_id: int, description: str, model: Optional[str] = None) -> None:
    save_image_descriptions([(image_id, description)], model)


def save_image_descriptions(items: list[tuple[int, str]], model: Optional[str] = None) -> None:
    """여러 이미지의 LLM 설명을 한 번에 저장한다. items = [(image_id, 설명)]"""
    if not items:
        return
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """UPDATE article_images SET llm_description = %s, described_model = %s, described_at = now()
               WHERE id = %s""",
            [(description, model or settings.chat_model, image_id) for image_id, description in items],
        )


def get_articles_without_chunks(topic_id: int, statuses: Iterable[str] = ("관련",)) -> list[Article]:
    """판정을 통과했는데 아직 청크가 없는 기사 (4단계 대상)."""
    with connect() as conn:
        links = [r["link"] for r in conn.execute(
            """SELECT a.link FROM articles a JOIN article_topics at ON at.article_id = a.id
               WHERE at.topic_id = %s AND at.status = ANY(%s)
                 AND NOT EXISTS (SELECT 1 FROM article_chunks c WHERE c.article_id = a.id)""",
            (topic_id, list(statuses)),
        ).fetchall()]
    wanted = set(links)
    return [a for a in get_topic_articles(topic_id) if a.link in wanted]


@dataclass
class Chunk:
    content: str
    embedding: list[float]
    source: str = "body"            # 'body' | 'image'
    image_id: Optional[int] = None  # source = 'image'일 때 article_images.id


def save_chunks(article_link: str, chunks: list[Chunk], model: Optional[str] = None) -> None:
    """기사 하나의 청크를 통째로 교체 저장한다 (청크 규칙을 바꿔 다시 돌려도 안전)."""
    save_chunks_many([(article_link, chunks)], model)


def save_chunks_many(items: list[tuple[str, list[Chunk]]], model: Optional[str] = None) -> None:
    """여러 기사의 청크를 한 번에 통째로 교체 저장한다. items = [(기사 link, 청크 목록)]

    예전에는 기사마다 save_chunks(연결 1번 + 쿼리 3번)를 불렀다. 여기서는 연결 1번에
    기사 id 조회 1번 · 이전 청크 지우기 1번 · 새 청크 넣기 1묶음으로 끝낸다. 저장되는 내용은 같다.
    DB에 없는 기사가 있으면 아무것도 저장하지 않고 ValueError를 낸다.
    """
    if not items:
        return
    with connect() as conn:
        ids = _article_ids(conn, [link for link, _ in items])
        missing = [link for link, _ in items if link not in ids]
        if missing:
            raise ValueError(f"DB에 없는 기사입니다: {missing[0]}" + (f" 외 {len(missing) - 1}건" if len(missing) > 1 else ""))
        conn.execute("DELETE FROM article_chunks WHERE article_id = ANY(%s)", ([ids[link] for link, _ in items],))
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO article_chunks (article_id, chunk_index, source, image_id, content,
                                               embedding, embedding_model)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                [(ids[link], i, c.source, c.image_id, c.content, _vec(c.embedding),
                  model or settings.embedding_model) for link, chunks in items for i, c in enumerate(chunks)],
            )


# ---------------------------------------------------------------- [5단계] 검색

@dataclass
class ChunkHit:
    content: str
    source: str
    similarity: float   # 코사인 유사도 (1 = 같음)
    article: Article    # 출처 표시용 (본문·임베딩은 비워서 돌려준다)


def search_chunks(
    topic_id: int,
    query_embedding: list[float],
    k: int = 5,
    period_days: Optional[int] = None,
    now: Optional[datetime] = None,
    statuses: Iterable[str] = ("관련",),
) -> list[ChunkHit]:
    """SQL 조건(topic, 판정, 발행일 기간) + 벡터 유사도로 청크 top-k를 찾는다."""
    since, until = _window(period_days, now)
    with connect() as conn:
        rows = conn.execute(
            """SELECT c.content, c.source, 1 - (c.embedding <=> %(q)s) AS similarity,
                      a.title, a.link, a.press, a.pub_date
               FROM article_chunks c
               JOIN articles a ON a.id = c.article_id
               JOIN article_topics at ON at.article_id = a.id
               WHERE at.topic_id = %(topic_id)s AND at.status = ANY(%(statuses)s)
                 AND c.embedding IS NOT NULL
                 AND (%(since)s::timestamptz IS NULL
                      OR a.pub_date BETWEEN %(since)s::timestamptz AND %(until)s::timestamptz)
               ORDER BY c.embedding <=> %(q)s
               LIMIT %(k)s""",
            {"q": _vec(query_embedding), "topic_id": topic_id, "statuses": list(statuses),
             "since": since, "until": until, "k": k},
        ).fetchall()
    return [
        ChunkHit(
            content=r["content"], source=r["source"], similarity=float(r["similarity"]),
            article=Article(title=r["title"], link=r["link"], press=r["press"], pub_date=r["pub_date"]),
        )
        for r in rows
    ]
