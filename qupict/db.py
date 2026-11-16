"""
DB 저장·조회 (Google Cloud Firestore + 벡터 검색)

파이프라인 코드는 기사 id 대신 link(언론사 원문 주소)로 기사를 가리킨다.
기사 문서 id는 link의 해시(article_key)라서 link만 알면 바로 문서를 찾는다. 키는 이 파일 안에서만 다룬다.

컬렉션 (예전 PostgreSQL 테이블 → Firestore)
  query_senses/{hash(검색어, 이름)}     [0단계] 검색어별 의미 후보 (같은 검색어 + 이름은 문서 하나)
  topics/{정수 id}                       수집 대상 = 검색어 + 고른 의미. 번호는 counters/topics에서 발급
  topic_keys/{hash(검색어, 의미 이름)}   (검색어, 의미 이름) → topic id. 같은 topic이 두 번 생기지 않게 막는다
  articles/{article_key}                 [1단계] 기사. 이미지는 문서 안 맵 images.{이미지 키}
  article_topics/{topic_id}_{article_key} [2·3단계] topic별 판정 (조인 대신 link · 발행일 · 청크 유무를 복사해 둔다)
  chunks/{article_key}_{번호}            [4단계] 청크 + 임베딩. search_keys = ["topic_id:판정", ...] (대표 기사로 판정된 topic만)

[5단계] 검색은 chunks에서 search_keys(topic · 판정) + 발행일 기간으로 거른 뒤 find_nearest(코사인)로 찾는다.
판정이 바뀌면(save_judgments 등) 그 기사 청크의 search_keys도 같이 바꾼다 (_sync_search_keys).

단계별로 쓰는 함수
  초기화       init_db()  (검색에 필요한 복합 인덱스 · 벡터 인덱스 만들기)
  [0단계]      get_sense_candidates / save_sense_candidates / record_sense_use
               save_topic / get_topic / list_topics / mark_topic_collected
  [1단계]      link_known_articles  → DB에 이미 있는 기사는 topic에 연결만 하고, 새로 크롤링할 것만 돌려준다
               save_articles
  [2·3단계]    get_topic_articles / save_article_embeddings
               reset_topic_filter / save_duplicates / save_judgments / copy_duplicate_judgments / save_labels
  [4단계]      get_images_to_describe / save_image_descriptions / described_image_rows / processed_image_counts
               get_articles_without_chunks / save_chunks_many
  [5단계]      search_chunks / searchable_stats
  현황 · 평가  count_pending / topic_status_rows / representative_count / label_rows / labeled_links
               duplicate_groups / articles_by_links / evidence_lookup / ping

기간은 topic에 저장하지 않는다. 조회할 때 period_days를 주면 기준 시각(now, 기본 = 지금 = 질문한 시각)에서
period_days일 전 ~ now 사이에 발행된(pub_date) 기사만 돌려준다.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Optional

import numpy as np
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from qupict.config import settings
from qupict.schemas import Article, ArticleImage, NewsMeta, SelectedSense, Sense, SenseCandidates

EMBEDDING_DIM = 1536  # text-embedding-3-small. 바꾸면 init_db의 벡터 인덱스 차원도 같이 바뀐다 (Firestore 최대 2048)

STATUSES = ("관련", "무관", "보류")

SENSES, TOPICS, TOPIC_KEYS, COUNTERS = "query_senses", "topics", "topic_keys", "counters"
ARTICLES, ARTICLE_TOPICS, CHUNKS = "articles", "article_topics", "chunks"

WRITE_BATCH = 200   # 한 번에 보내는 쓰기 수 (Firestore 한도 500 · 요청 10MiB. 청크 문서는 임베딩 때문에 약 15KB)
IN_LIMIT = 30       # Firestore 'in' · 'array_contains_any' 조건 값 개수 한도
DISTANCE_FIELD = "vector_distance"


# ---------------------------------------------------------------- 연결

_client_obj = None
_client_lock = threading.Lock()


def _client():
    """Firestore 클라이언트 (프로그램에서 하나를 같이 쓴다. 스레드에서 같이 써도 안전).
    인증: Cloud Run은 붙어 있는 서비스 계정, 로컬은 GOOGLE_APPLICATION_CREDENTIALS(서비스 계정 키 파일)."""
    global _client_obj
    with _client_lock:
        if _client_obj is None:
            if not settings.has_db:
                raise EnvironmentError("FIRESTORE_PROJECT가 설정되지 않았습니다. .env 파일을 확인하세요 (.env.example 참고).")
            _client_obj = firestore.Client(project=settings.firestore_project, database=settings.firestore_database)
        return _client_obj


def _transact(fn: Callable):
    """fn(transaction)을 트랜잭션으로 실행한다 (충돌하면 Firestore가 다시 실행). 읽기를 모두 한 뒤에 써야 한다."""
    return firestore.transactional(fn)(_client().transaction())


def warm_up() -> None:
    """클라이언트를 만들고 인증 토큰을 미리 받아 둔다 (뒤에서 진행되고 바로 돌아온다). 웹 서버를 켤 때 부른다."""
    if settings.has_db:
        threading.Thread(target=_safe_ping, daemon=True).start()


def _safe_ping() -> None:
    try:
        ping()
    except Exception as e:
        print(f"  [경고] Firestore 연결 확인 실패 ({type(e).__name__}: {e})")


def ping() -> None:
    """DB 왕복 1번 (문서 하나 읽기). 연결 · 인증 확인용."""
    _client().collection(COUNTERS).document("topics").get()


def close_pool() -> None:
    """예전(PostgreSQL 연결 풀) 호환용. Firestore는 닫을 연결이 없다."""


# ---------------------------------------------------------------- 공통 도구

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()[:24]


def article_key(link: str) -> str:
    """기사 문서 id (link의 해시)."""
    return _hash(link)


def _image_key(url: str) -> str:
    return _hash(url)[:12]


def _sense_id(query: str, name: str) -> str:
    return _hash(query, name)


def _at_id(topic_id: int, akey: str) -> str:
    return f"{topic_id}_{akey}"


def _chunk_id(akey: str, index: int) -> str:
    return f"{akey}_{index:03d}"


def _vec(embedding: Optional[Iterable[float]]) -> Optional[list[float]]:
    if embedding is None:
        return None
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.shape != (EMBEDDING_DIM,):
        raise ValueError(f"임베딩 차원이 {arr.shape}입니다. DB는 {EMBEDDING_DIM}차원으로 되어 있습니다.")
    return arr.tolist()


def _window(period_days: Optional[int], now: Optional[datetime]) -> tuple[Optional[datetime], Optional[datetime]]:
    """발행일 기간 [now - period_days, now]. period_days가 없으면 조건 없음."""
    if period_days is None:
        return None, None
    now = now or datetime.now(timezone.utc)
    return now - timedelta(days=period_days), now


def _col(name: str):
    return _client().collection(name)


def _ref(collection: str, doc_id: str):
    return _col(collection).document(doc_id)


def _chunks_of(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _get_all(collection: str, ids: Iterable[str], fields: Optional[list[str]] = None) -> dict[str, dict]:
    """문서 여러 개를 한 번에 읽는다. 있는 문서만 {id: 내용}."""
    refs = [_ref(collection, i) for i in dict.fromkeys(ids)]
    out: dict[str, dict] = {}
    for part in _chunks_of(refs, 300):
        for snap in _client().get_all(part, field_paths=fields):
            if snap.exists:
                out[snap.id] = snap.to_dict()
    return out


def _query(collection: str, *filters: tuple, fields: Optional[list[str]] = None):
    q = _col(collection)
    for field, op, value in filters:
        q = q.where(filter=FieldFilter(field, op, value))
    if fields is not None:
        q = q.select(fields)
    return q


def _stream(collection: str, *filters: tuple, fields: Optional[list[str]] = None) -> list[tuple[str, dict]]:
    return [(s.id, s.to_dict()) for s in _query(collection, *filters, fields=fields).stream()]


def _stream_in(collection: str, field: str, values: Iterable, *filters: tuple,
               fields: Optional[list[str]] = None) -> list[tuple[str, dict]]:
    """field in values (30개씩 나눠서 묻는다)."""
    out = []
    for part in _chunks_of(list(dict.fromkeys(values)), IN_LIMIT):
        out += _stream(collection, (field, "in", part), *filters, fields=fields)
    return out


class _Writes:
    """쓰기를 모았다가 WRITE_BATCH개씩 묶어 보낸다 (묶음 하나는 한꺼번에 반영되거나 하나도 안 된다)."""

    def __init__(self) -> None:
        self.ops: list[tuple] = []

    def set(self, collection: str, doc_id: str, data: dict, merge: bool = False) -> None:
        self.ops.append(("set", collection, doc_id, data, merge))

    def update(self, collection: str, doc_id: str, data: dict) -> None:
        self.ops.append(("update", collection, doc_id, data, False))

    def delete(self, collection: str, doc_id: str) -> None:
        self.ops.append(("delete", collection, doc_id, None, False))

    def commit(self) -> int:
        for part in _chunks_of(self.ops, WRITE_BATCH):
            batch = _client().batch()
            for kind, collection, doc_id, data, merge in part:
                ref = _ref(collection, doc_id)
                if kind == "set":
                    batch.set(ref, data, merge=merge)
                elif kind == "update":
                    batch.update(ref, data)
                else:
                    batch.delete(ref)
            batch.commit()
        return len(self.ops)


def _sort_newest(rows: list, date_of: Callable, tie: Callable) -> list:
    """발행일 최신순, 발행일 없는 것은 맨 뒤 (예전 ORDER BY pub_date DESC NULLS LAST)."""
    return sorted(rows, key=lambda r: (date_of(r) is None, -(date_of(r).timestamp() if date_of(r) else 0), tie(r)))


# ---------------------------------------------------------------- [0단계] 의미 후보

def _new_sense(query: str, s: Sense, rank: Optional[int], model: str, includes: Iterable[str] = ()) -> dict:
    return {"query": query, "rank": rank, "name": s.name, "definition": s.definition, "aliases": list(s.aliases),
            "related_terms": list(s.related_terms), "is_custom": s.is_custom, "includes": list(includes),
            "use_count": 0, "last_used_at": None, "model": model, "created_at": _now()}


def get_sense_candidates(query: str) -> Optional[SenseCandidates]:
    """저장된 의미 후보를 돌려준다: LLM 후보 + 예전에 "기타"로 직접 입력해 저장된 의미.
    많이 고른 순(use_count) → LLM 순위 → 직접 입력(먼저 저장한 순). 저장된 게 없으면 None."""
    rows = [d for _, d in _stream(SENSES, ("query", "==", query))]
    if not rows:
        return None
    far = datetime.max.replace(tzinfo=timezone.utc)
    rows.sort(key=lambda r: (-r.get("use_count", 0), r.get("is_custom", False), r.get("rank") is None,
                             r.get("rank") or 0, r.get("created_at") or far, r["name"]))
    senses = [Sense(name=r["name"], definition=r.get("definition", ""), aliases=r.get("aliases") or [],
                    related_terms=r.get("related_terms") or [], is_custom=r.get("is_custom", False)) for r in rows]
    model = next((r.get("model", "") for r in rows if not r.get("is_custom")), "")
    return SenseCandidates(
        query=query, senses=senses, model=model,
        custom_includes={r["name"]: list(r.get("includes") or []) for r in rows if r.get("is_custom")},
        use_counts={r["name"]: r.get("use_count", 0) for r in rows},
    )


def save_sense_candidates(candidates: SenseCandidates) -> None:
    """LLM 의미 후보를 저장한다. 같은 (검색어, 이름)이 있으면 내용을 갱신한다 (고른 횟수는 그대로)."""
    ranked = [(rank, s) for rank, s in enumerate(candidates.senses, start=1) if not s.is_custom]
    existing = _get_all(SENSES, [_sense_id(candidates.query, s.name) for _, s in ranked], fields=["name"])
    w = _Writes()
    for rank, s in ranked:
        doc_id = _sense_id(candidates.query, s.name)
        data = _new_sense(candidates.query, s, rank, candidates.model)
        if doc_id in existing:    # 고른 횟수 · 포함 후보 · 만든 시각은 그대로 둔다
            for key in ("includes", "use_count", "last_used_at", "created_at"):
                data.pop(key)
        w.set(SENSES, doc_id, data, merge=True)
    w.commit()


def record_sense_use(query: str, sense: Sense, includes: Iterable[str] = ()) -> None:
    """사용자가 이 의미를 골랐다고 기록한다 (use_count + 1).
    직접 입력한 의미는 처음이면 저장한다 → 다음부터 이 검색어의 후보로 나온다 (LLM 다시 안 부름).
    같은 이름의 LLM 후보가 있으면 그 내용은 덮어쓰지 않고 고른 횟수만 올린다."""
    ref = _ref(SENSES, _sense_id(query, sense.name))
    includes = list(includes)

    def work(tx) -> None:
        snap = ref.get(transaction=tx)
        doc = snap.to_dict() if snap.exists else None
        if sense.is_custom:
            if doc is None:
                doc = _new_sense(query, sense, None, "", includes)
            elif doc.get("is_custom"):
                doc.update(definition=sense.definition, aliases=list(sense.aliases),
                           related_terms=list(sense.related_terms), includes=includes)
        if doc is None:          # 저장된 적 없는 LLM 후보 (예전 UPDATE가 아무 행도 못 찾은 경우와 같음)
            return
        doc["use_count"] = doc.get("use_count", 0) + 1
        doc["last_used_at"] = _now()
        tx.set(ref, doc)

    _transact(work)


# ---------------------------------------------------------------- topics

@dataclass
class TopicRecord:
    id: int
    selected: SelectedSense
    last_collected_at: Optional[datetime]


def save_topic(selected: SelectedSense) -> int:
    """고른 의미를 topic으로 저장하고 id를 돌려준다. 같은 (검색어, 의미 이름)이 있으면 그 topic을 재사용한다."""
    s = selected.sense
    sense_ref = _ref(SENSES, _sense_id(selected.query, s.name))
    sense_doc = sense_ref.get(field_paths=["name"])
    if not sense_doc.exists and s.is_custom:
        sense_ref.set(_new_sense(selected.query, s, None, ""))
    sense_id = sense_ref.id if (sense_doc.exists or s.is_custom) else None
    data = {"query": selected.query, "sense_id": sense_id, "sense_name": s.name, "definition": s.definition,
            "aliases": list(s.aliases), "related_terms": list(s.related_terms), "is_custom": s.is_custom,
            "excluded_terms": list(selected.excluded_terms),
            "other_senses": [o.model_dump() for o in selected.other_senses], "updated_at": _now()}
    key_ref = _ref(TOPIC_KEYS, _hash(selected.query, s.name))
    counter_ref = _ref(COUNTERS, "topics")

    def work(tx) -> int:
        key = key_ref.get(transaction=tx)
        if key.exists:
            topic_id = key.to_dict()["id"]
            tx.set(_ref(TOPICS, str(topic_id)), data, merge=True)
            return topic_id
        counter = counter_ref.get(transaction=tx)
        topic_id = (counter.to_dict() or {}).get("next", 1) if counter.exists else 1
        tx.set(counter_ref, {"next": topic_id + 1})
        tx.set(key_ref, {"id": topic_id, "query": selected.query, "sense_name": s.name})
        tx.set(_ref(TOPICS, str(topic_id)),
               {**data, "id": topic_id, "last_collected_at": None, "created_at": _now()})
        return topic_id

    return _transact(work)


def _topic_record(d: dict) -> TopicRecord:
    selected = SelectedSense(
        query=d["query"],
        sense=Sense(name=d["sense_name"], definition=d.get("definition", ""), aliases=d.get("aliases") or [],
                    related_terms=d.get("related_terms") or [], is_custom=d.get("is_custom", False)),
        excluded_terms=d.get("excluded_terms") or [],
        other_senses=[Sense(**o) for o in d.get("other_senses") or []],
    )
    return TopicRecord(id=int(d["id"]), selected=selected, last_collected_at=d.get("last_collected_at"))


def get_topic(query: str, sense_name: str) -> Optional[TopicRecord]:
    rows = _stream(TOPICS, ("query", "==", query), ("sense_name", "==", sense_name))
    return _topic_record(rows[0][1]) if rows else None


def list_topics(query: Optional[str] = None) -> list[TopicRecord]:
    """저장된 topic 전체 (검색어 → 의미 이름 순). query를 주면 그 검색어만."""
    rows = _stream(TOPICS, ("query", "==", query)) if query is not None else _stream(TOPICS)
    return sorted((_topic_record(d) for _, d in rows), key=lambda r: (r.selected.query, r.selected.sense.name))


def mark_topic_collected(topic_id: int, at: Optional[datetime] = None) -> None:
    _ref(TOPICS, str(topic_id)).update({"last_collected_at": at or _now()})


# ---------------------------------------------------------------- [1단계] 기사

def _new_link(topic_id: int, akey: str, link: str, pub_date: Optional[datetime], search_term: str,
              chunked: bool) -> dict:
    """article_topics 새 문서. 판정 칸은 처음부터 None으로 넣어 둔다 (== None 조건으로 찾을 수 있게)."""
    return {"topic_id": topic_id, "article_key": akey, "link": link, "pub_date": pub_date,
            "search_term": search_term or "", "duplicate_of": None, "status": None, "stage": None,
            "reason": None, "judged_model": None, "judged_at": None, "label": None, "labeled_at": None,
            "chunked": chunked, "created_at": _now()}


def link_known_articles(topic_id: int, metas: list[NewsMeta]) -> list[NewsMeta]:
    """
    검색 결과 중 DB에 이미 있는 기사는 이 topic에 연결만 하고(크롤링 생략),
    DB에 없는 기사만 돌려준다. 돌려받은 것만 crawl_articles → save_articles 하면 된다.
    """
    keys = {m.link: article_key(m.link) for m in metas}
    known = _get_all(ARTICLES, keys.values(), fields=["pub_date", "chunk_count"])
    linked = _get_all(ARTICLE_TOPICS, [_at_id(topic_id, k) for k in keys.values() if k in known], fields=["topic_id"])
    w = _Writes()
    for m in metas:
        akey = keys[m.link]
        at_id = _at_id(topic_id, akey)
        if akey in known and at_id not in linked:
            art = known[akey]
            w.set(ARTICLE_TOPICS, at_id, _new_link(topic_id, akey, m.link, art.get("pub_date"), m.search_term,
                                                   (art.get("chunk_count") or 0) > 0))
            linked[at_id] = {}
    w.commit()
    return [m for m in metas if keys[m.link] not in known]


def save_articles(topic_id: int, articles: list[Article]) -> dict[str, str]:
    """기사·이미지를 저장하고 topic에 연결한다. 이미 있는 기사는 본문 등을 갱신한다. {link: article_key}를 돌려준다.

    언론사 · 발행일 · 임베딩은 새 값이 있을 때만 바꾼다 (예전 COALESCE). 이미지 설명(llm_description)은 남긴다.
    이미 이 topic에 연결된 기사는 연결(판정 등)을 그대로 둔다 (예전 ON CONFLICT DO NOTHING).
    """
    if not articles:
        return {}
    keys = {a.link: article_key(a.link) for a in articles}
    existing = _get_all(ARTICLES, keys.values(), fields=["chunk_count"])
    linked = _get_all(ARTICLE_TOPICS, [_at_id(topic_id, k) for k in keys.values()], fields=["topic_id"])
    w = _Writes()
    for a in articles:
        akey = keys[a.link]
        data: dict = {"link": a.link, "title": a.title, "description": a.description, "content": a.content,
                      "collected_at": a.collected_at or _now()}
        if a.press is not None:
            data["press"] = a.press
        if a.pub_date is not None:
            data["pub_date"] = a.pub_date
        if a.embedding is not None:
            data["embedding"] = Vector(_vec(a.embedding))
            data["embedding_model"] = settings.embedding_model
        if a.images:   # 빈 맵을 merge로 쓰면 기존 이미지(설명 포함)가 지워지므로 이미지가 있을 때만
            data["images"] = {_image_key(img.url): {"position": i, "url": img.url, "caption": img.caption}
                              for i, img in enumerate(a.images)}
            data["image_bases"] = firestore.ArrayUnion(sorted({img.url.split("?")[0] for img in a.images}))
        if akey not in existing:
            data.setdefault("press", None)
            data.setdefault("pub_date", None)
            data["chunk_count"] = 0
            existing[akey] = {"chunk_count": 0}
        w.set(ARTICLES, akey, data, merge=True)
        at_id = _at_id(topic_id, akey)
        if at_id not in linked:
            w.set(ARTICLE_TOPICS, at_id, _new_link(topic_id, akey, a.link, a.pub_date, a.search_term,
                                                   (existing[akey].get("chunk_count") or 0) > 0))
            linked[at_id] = {}
        elif a.pub_date is not None:
            w.set(ARTICLE_TOPICS, at_id, {"pub_date": a.pub_date}, merge=True)
    w.commit()
    return keys


def _topic_links(topic_id: int, period_days: Optional[int] = None, now: Optional[datetime] = None,
                 fields: Optional[list[str]] = None) -> list[dict]:
    """topic에 연결된 article_topics 문서. period_days가 있으면 발행일이 기간 안인 것만 (발행일 없는 기사 제외)."""
    filters: list[tuple] = [("topic_id", "==", topic_id)]
    since, until = _window(period_days, now)
    if since is not None:
        filters += [("pub_date", ">=", since), ("pub_date", "<=", until)]
    return [d for _, d in _stream(ARTICLE_TOPICS, *filters, fields=fields)]


def _to_article(d: dict, search_term: str = "", with_body: bool = True) -> Article:
    images = sorted((d.get("images") or {}).values(), key=lambda i: i.get("position", 0))
    emb = d.get("embedding")
    return Article(
        title=d.get("title", ""), link=d["link"], press=d.get("press"), pub_date=d.get("pub_date"),
        content=d.get("content", "") if with_body else "", description=d.get("description", "") if with_body else "",
        search_term=search_term, collected_at=d.get("collected_at"),
        images=[ArticleImage(url=i["url"], caption=i.get("caption", "")) for i in images] if with_body else [],
        embedding=list(emb) if (with_body and emb is not None) else None,
    )


def _articles_for(links: list[dict]) -> list[Article]:
    """article_topics 문서들 → 기사 (본문 · 임베딩 · 이미지 포함), 발행일 최신순."""
    docs = _get_all(ARTICLES, [d["article_key"] for d in links])
    out = [_to_article(docs[d["article_key"]], d.get("search_term", "")) for d in links if d["article_key"] in docs]
    return _sort_newest(out, lambda a: a.pub_date, lambda a: a.link)


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
      statuses           : 예) ["관련"] — 해당 판정만 (중복 기사 제외)
      include_duplicates : False면 중복(duplicate_of 있음)으로 표시된 기사 제외
      unjudged_only      : 아직 중복 표시도 판정도 없는 기사만 (새로 들어온 기사 처리용)
    """
    wanted = set(statuses) if statuses is not None else None
    links = [d for d in _topic_links(topic_id, period_days, now)
             if (wanted is None or (d.get("status") in wanted and d.get("duplicate_of") is None))
             and (include_duplicates or d.get("duplicate_of") is None)
             and (not unjudged_only or (d.get("duplicate_of") is None and d.get("status") is None))]
    return _articles_for(links)


def save_article_embeddings(articles: list[Article], model: Optional[str] = None) -> None:
    """기사 전체 임베딩을 저장한다 (다음 실행 때 다시 계산하지 않도록)."""
    w = _Writes()
    for a in articles:
        if a.embedding is not None:
            w.update(ARTICLES, article_key(a.link),
                     {"embedding": Vector(_vec(a.embedding)), "embedding_model": model or settings.embedding_model})
    w.commit()


# ---------------------------------------------------------------- [2·3단계] 판정

_JUDGMENT_FIELDS = ("status", "stage", "reason", "judged_model", "judged_at")


def _sync_search_keys(akeys: Iterable[str]) -> None:
    """기사들의 청크에 search_keys(검색 대상 topic · 판정)를 다시 적는다. 판정 · 중복 표시를 바꾼 뒤 부른다.
    search_keys = 이 기사가 대표 기사(중복 아님)로 판정된 topic마다 "topic_id:판정"."""
    akeys = list(dict.fromkeys(akeys))
    if not akeys:
        return
    counts = {k: d.get("chunk_count") or 0 for k, d in _get_all(ARTICLES, akeys, fields=["chunk_count"]).items()}
    with_chunks = [k for k in akeys if counts.get(k)]
    if not with_chunks:
        return
    keys = _search_keys(with_chunks)
    w = _Writes()
    for akey in with_chunks:
        for i in range(counts[akey]):
            w.update(CHUNKS, _chunk_id(akey, i), {"search_keys": keys.get(akey, [])})
    w.commit()


def _search_keys(akeys: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for _, d in _stream_in(ARTICLE_TOPICS, "article_key", akeys,
                           fields=["article_key", "topic_id", "status", "duplicate_of"]):
        if d.get("duplicate_of") is None and d.get("status"):
            out.setdefault(d["article_key"], []).append(f"{d['topic_id']}:{d['status']}")
    return {k: sorted(v) for k, v in out.items()}


def reset_topic_filter(topic_id: int) -> None:
    """규칙을 바꿔 필터링을 처음부터 다시 돌릴 때: 중복 표시와 판정을 지운다 (정답 라벨은 남긴다)."""
    links = _topic_links(topic_id, fields=["article_key"])
    w = _Writes()
    for d in links:
        w.update(ARTICLE_TOPICS, _at_id(topic_id, d["article_key"]),
                 {"duplicate_of": None, **{f: None for f in _JUDGMENT_FIELDS}})
    w.commit()
    _sync_search_keys(d["article_key"] for d in links)


def save_duplicates(topic_id: int, duplicates: dict[str, str]) -> None:
    """2단계 결과 저장. duplicates = {중복 기사 link: 대표 기사 link}. 판정은 copy_duplicate_judgments가 채운다."""
    pairs = [(article_key(dup), article_key(rep)) for dup, rep in duplicates.items() if dup != rep]
    linked = _get_all(ARTICLE_TOPICS, [_at_id(topic_id, d) for d, _ in pairs], fields=["topic_id"])
    reps = _get_all(ARTICLES, [r for _, r in pairs], fields=["chunk_count"])
    w = _Writes()
    changed = []
    for dup, rep in pairs:
        if _at_id(topic_id, dup) in linked and rep in reps:
            w.update(ARTICLE_TOPICS, _at_id(topic_id, dup), {"duplicate_of": rep})
            changed.append(dup)
    w.commit()
    _sync_search_keys(changed)


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
        if j.stage not in ("rule", "llm"):
            raise ValueError(f"판정 단계는 'rule' 또는 'llm'이어야 합니다: {j.stage!r}")
    linked = _get_all(ARTICLE_TOPICS, [_at_id(topic_id, article_key(j.link)) for j in judgments], fields=["topic_id"])
    w = _Writes()
    changed = []
    for j in judgments:
        akey = article_key(j.link)
        if _at_id(topic_id, akey) in linked:
            w.update(ARTICLE_TOPICS, _at_id(topic_id, akey), {"status": j.status, "stage": j.stage, "reason": j.reason,
                                                              "judged_model": j.model, "judged_at": _now()})
            changed.append(akey)
    w.commit()
    _sync_search_keys(changed)


def copy_duplicate_judgments(topic_id: int) -> int:
    """중복 기사에 대표 기사의 판정(관련/무관)을 복사한다. 판정이 끝난 뒤 부른다. 바꾼 문서 수를 돌려준다.
    (중복 기사는 검색 대상이 아니라서 청크 search_keys는 바뀌지 않는다)"""
    links = {d["article_key"]: d for d in _topic_links(
        topic_id, fields=["article_key", "duplicate_of", "status", "stage", "judged_model"])}
    w = _Writes()
    for akey, d in links.items():
        rep = links.get(d.get("duplicate_of")) if d.get("duplicate_of") else None
        if rep and rep.get("status") is not None and d.get("status") != rep["status"]:
            w.update(ARTICLE_TOPICS, _at_id(topic_id, akey),
                     {"status": rep["status"], "stage": rep.get("stage"), "judged_model": rep.get("judged_model"),
                      "judged_at": _now(), "reason": "중복: 대표 기사 판정을 따름"})
    return w.commit()


def save_labels(topic_id: int, labels: dict[str, str]) -> None:
    """사람이 붙인 정답 라벨 저장. labels = {link: '관련' | '무관' | '보류'}"""
    for v in labels.values():
        if v not in STATUSES:
            raise ValueError(f"라벨 값은 {STATUSES} 중 하나여야 합니다: {v!r}")
    linked = _get_all(ARTICLE_TOPICS, [_at_id(topic_id, article_key(k)) for k in labels], fields=["topic_id"])
    w = _Writes()
    for link, v in labels.items():
        at_id = _at_id(topic_id, article_key(link))
        if at_id in linked:
            w.update(ARTICLE_TOPICS, at_id, {"label": v, "labeled_at": _now()})
    w.commit()


def labeled_links(topic_id: int) -> set[str]:
    """사람이 정답 라벨을 붙인 기사 link (중복으로 바꾸지 않게 지킬 기사)."""
    return {d["link"] for d in _topic_links(topic_id, fields=["link", "label"]) if d.get("label") is not None}


# ---------------------------------------------------------------- [4단계] 이미지 설명 · 청크

@dataclass
class ImageToDescribe:
    image_id: str               # "기사키/이미지키"
    article_link: str
    url: str
    caption: str


def _representatives(topic_id: int, statuses: Iterable[str], fields: Optional[list[str]] = None) -> list[dict]:
    """topic에서 이 판정을 받은 대표 기사(중복 아님)의 article_topics 문서."""
    wanted = set(statuses)
    base = ["article_key", "link", "status", "duplicate_of"]
    return [d for d in _topic_links(topic_id, fields=None if fields is None else [*base, *fields])
            if d.get("status") in wanted and d.get("duplicate_of") is None]


def get_images_to_describe(topic_id: int, statuses: Iterable[str] = ("관련",)) -> list[ImageToDescribe]:
    """판정을 통과한 기사 중 아직 LLM 설명이 없는 이미지 (기사 link 순 → 기사 안 순서)."""
    reps = _representatives(topic_id, statuses, fields=[])
    docs = _get_all(ARTICLES, [d["article_key"] for d in reps], fields=["link", "images"])
    out = []
    for akey, d in sorted(docs.items(), key=lambda kv: kv[1]["link"]):
        for ikey, img in sorted((d.get("images") or {}).items(), key=lambda kv: kv[1].get("position", 0)):
            if img.get("llm_description") is None:
                out.append(ImageToDescribe(f"{akey}/{ikey}", d["link"], img["url"], img.get("caption", "")))
    return out


def save_image_description(image_id: str, description: str, model: Optional[str] = None) -> None:
    save_image_descriptions([(image_id, description)], model)


def save_image_descriptions(items: list[tuple[str, str]], model: Optional[str] = None) -> None:
    """여러 이미지의 LLM 설명을 한 번에 저장한다. items = [(image_id, 설명)]"""
    if not items:
        return
    w = _Writes()
    now = _now()
    for image_id, description in items:
        akey, ikey = image_id.split("/", 1)
        w.update(ARTICLES, akey, {f"images.{ikey}.llm_description": description,
                                  f"images.{ikey}.described_model": model or settings.chat_model,
                                  f"images.{ikey}.described_at": now})
    w.commit()


def _images_with_text(article_links: list[str]) -> list[dict]:
    """기사들의 이미지 중 글(설명 또는 OCR 글자, 빈 글자 포함)이 저장된 것. 기사는 받은 순서, 기사 안은 순서대로."""
    docs = _get_all(ARTICLES, [article_key(link) for link in article_links], fields=["link", "images"])
    rows = []
    for link in dict.fromkeys(article_links):
        akey = article_key(link)
        d = docs.get(akey)
        if d is None:
            continue
        for ikey, img in sorted((d.get("images") or {}).items(), key=lambda kv: kv[1].get("position", 0)):
            if img.get("llm_description") is not None:
                rows.append({"link": d["link"], "id": f"{akey}/{ikey}", "caption": img.get("caption", ""),
                             "llm_description": img["llm_description"], "described_model": img.get("described_model")})
    return rows


def described_image_rows(article_links: list[str]) -> list[dict]:
    """이미 글이 있는 이미지 행: {link, id, caption, llm_description, described_model} (indexing.described_images용)."""
    return _images_with_text(article_links) if article_links else []


def processed_image_counts(article_links: list[str]) -> dict[str, int]:
    """기사별로 이미 처리한(글이 비어 있어도 저장된) 이미지 수."""
    out: dict[str, int] = {}
    for r in described_image_rows(article_links):
        out[r["link"]] = out.get(r["link"], 0) + 1
    return out


def get_articles_without_chunks(topic_id: int, statuses: Iterable[str] = ("관련",)) -> list[Article]:
    """판정을 통과했는데 아직 청크가 없는 기사 (4단계 대상)."""
    return _articles_for([d for d in _representatives(topic_id, statuses) if not d.get("chunked")])


@dataclass
class Chunk:
    content: str
    embedding: list[float]
    source: str = "body"            # 'body' | 'image'
    image_id: Optional[str] = None  # source = 'image'일 때 이미지 id ("기사키/이미지키")


def save_chunks(article_link: str, chunks: list[Chunk], model: Optional[str] = None) -> None:
    """기사 하나의 청크를 통째로 교체 저장한다 (청크 규칙을 바꿔 다시 돌려도 안전)."""
    save_chunks_many([(article_link, chunks)], model)


def save_chunks_many(items: list[tuple[str, list[Chunk]]], model: Optional[str] = None) -> None:
    """여러 기사의 청크를 한 번에 통째로 교체 저장한다. items = [(기사 link, 청크 목록)]

    청크 문서에는 기사 제목 · 언론사 · 발행일(출처 표시와 기간 조건용)과 search_keys(검색 대상 topic · 판정)를
    복사해 둔다. 예전보다 청크가 줄었으면 남는 청크 문서는 지운다. 기사의 chunk_count와 topic 연결의 chunked도 바꾼다.
    DB에 없는 기사가 있으면 아무것도 저장하지 않고 ValueError를 낸다.
    """
    if not items:
        return
    keys = {link: article_key(link) for link, _ in items}
    docs = _get_all(ARTICLES, keys.values(), fields=["link", "title", "press", "pub_date", "chunk_count"])
    missing = [link for link, _ in items if keys[link] not in docs]
    if missing:
        raise ValueError(f"DB에 없는 기사입니다: {missing[0]}" + (f" 외 {len(missing) - 1}건" if len(missing) > 1 else ""))
    search_keys = _search_keys(list(keys.values()))
    links_of: dict[str, list[str]] = {}
    for at_id, d in _stream_in(ARTICLE_TOPICS, "article_key", keys.values(), fields=["article_key"]):
        links_of.setdefault(d["article_key"], []).append(at_id)
    model = model or settings.embedding_model
    now = _now()
    w = _Writes()
    for link, chunks in items:
        akey = keys[link]
        art = docs[akey]
        for i, c in enumerate(chunks):
            w.set(CHUNKS, _chunk_id(akey, i), {
                "article_key": akey, "link": link, "title": art.get("title", ""), "press": art.get("press"),
                "pub_date": art.get("pub_date"), "chunk_index": i, "source": c.source, "image_id": c.image_id,
                "content": c.content, "embedding": Vector(_vec(c.embedding)), "embedding_model": model,
                "search_keys": search_keys.get(akey, []), "created_at": now})
        for i in range(len(chunks), art.get("chunk_count") or 0):
            w.delete(CHUNKS, _chunk_id(akey, i))
        w.update(ARTICLES, akey, {"chunk_count": len(chunks)})
        art["chunk_count"] = len(chunks)
        for at_id in links_of.get(akey, []):
            w.update(ARTICLE_TOPICS, at_id, {"chunked": bool(chunks)})
    w.commit()


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
    """topic · 판정 · 발행일 기간으로 거른 청크에서 질문 임베딩과 가까운 top-k (Firestore find_nearest, 코사인)."""
    since, until = _window(period_days, now)
    filters: list[tuple] = [("search_keys", "array_contains_any", [f"{topic_id}:{s}" for s in statuses])]
    if since is not None:
        filters += [("pub_date", ">=", since), ("pub_date", "<=", until)]
    q = _query(CHUNKS, *filters).find_nearest(
        "embedding", Vector(_vec(query_embedding)), limit=min(k, 1000),
        distance_measure=DistanceMeasure.COSINE, distance_result_field=DISTANCE_FIELD)
    hits = []
    for snap in q.stream():
        d = snap.to_dict()
        hits.append(ChunkHit(
            content=d["content"], source=d.get("source", "body"), similarity=1.0 - float(d[DISTANCE_FIELD]),
            article=Article(title=d.get("title", ""), link=d["link"], press=d.get("press"), pub_date=d.get("pub_date")),
        ))
    return sorted(hits, key=lambda h: -h.similarity)


def _search_chunk_query(topic_id: int, statuses: Iterable[str], fields: Optional[list[str]] = None):
    """검색 대상 청크 (search_chunks와 같은 topic · 판정 조건, 발행일 기간은 없음)."""
    return _query(CHUNKS, ("search_keys", "array_contains_any", [f"{topic_id}:{s}" for s in statuses]),
                  fields=fields)


def count_search_chunks(topic_id: int, statuses: Iterable[str] = ("관련", "보류")) -> int:
    """검색 대상 청크 수 (집계 쿼리라 문서를 읽지 않는다). BM25 인덱스를 다시 만들지 정할 때 쓴다."""
    result = _search_chunk_query(topic_id, statuses).count().get()
    return int(result[0][0].value)


def search_chunk_rows(topic_id: int, statuses: Iterable[str] = ("관련", "보류")) -> list[ChunkHit]:
    """검색 대상 청크 전부 (임베딩 없이). 하이브리드 검색의 BM25 인덱스를 만들 때 쓴다 (qupict/bm25.py).
    similarity는 0 (키워드 검색에는 유사도가 없다)."""
    fields = ["link", "title", "press", "pub_date", "content", "source", "chunk_index"]
    rows = sorted((d for _, d in _stream_query(_search_chunk_query(topic_id, statuses, fields))),
                  key=lambda d: (d["link"], d.get("chunk_index", 0)))       # 순서를 고정해 결과가 매번 같게
    return [ChunkHit(content=d["content"], source=d.get("source", "body"), similarity=0.0,
                     article=Article(title=d.get("title", ""), link=d["link"], press=d.get("press"),
                                     pub_date=d.get("pub_date")))
            for d in rows]


def _stream_query(q) -> list[tuple[str, dict]]:
    return [(s.id, s.to_dict()) for s in q.stream()]


def searchable_stats(topic_id: int, period_days: Optional[int] = None,
                     statuses: Iterable[str] = ("관련", "보류")) -> tuple[int, Optional[datetime]]:
    """질문할 때 검색되는 기사(search_chunks와 같은 조건: topic · 판정 · 발행일 기간 · 청크 있음)의
    (기사 수, 가장 오래된 발행일). 없으면 (0, None)."""
    wanted = set(statuses)
    dates = [d.get("pub_date") for d in _topic_links(
                 topic_id, period_days, fields=["status", "duplicate_of", "chunked", "pub_date"])
             if d.get("status") in wanted and d.get("duplicate_of") is None and d.get("chunked")]
    present = [x for x in dates if x is not None]
    return len(dates), (min(present) if present else None)


# ---------------------------------------------------------------- 현황 · 라벨링 · 평가용 조회

def count_pending(topic_id: int, statuses: Iterable[str] = ("관련", "보류")) -> tuple[int, int]:
    """(판정이 비어 있는 기사 수, 질문에 쓰는 판정인데 청크가 없는 대표 기사 수)."""
    wanted = set(statuses)
    links = _topic_links(topic_id, fields=["status", "duplicate_of", "chunked"])
    unjudged = sum(1 for d in links if d.get("status") is None)
    unindexed = sum(1 for d in links if d.get("duplicate_of") is None and d.get("status") in wanted
                    and not d.get("chunked"))
    return unjudged, unindexed


def representative_count(topic_id: int) -> int:
    """중복이 아닌 기사 수 (= 라벨 붙일 후보)."""
    return sum(1 for d in _topic_links(topic_id, fields=["duplicate_of"]) if d.get("duplicate_of") is None)


def topic_status_rows() -> list[dict]:
    """topic별 진행 현황 (pipeline.TopicStatus의 칸과 같은 이름). 판정 칸은 대표 기사만 센다."""
    fields = ["topic_id", "duplicate_of", "status", "stage", "label", "chunked"]
    by_topic: dict[int, list[dict]] = {}
    for _, d in _stream(ARTICLE_TOPICS, fields=fields):
        by_topic.setdefault(d["topic_id"], []).append(d)
    rows = []
    for t in list_topics():
        links = by_topic.get(t.id, [])
        reps = [d for d in links if d.get("duplicate_of") is None]
        judged = lambda stage, status: sum(1 for d in reps if d.get("stage") == stage and d.get("status") == status)
        rows.append({
            "topic_id": t.id, "query": t.selected.query, "sense_name": t.selected.sense.name,
            "articles": len(links), "duplicates": len(links) - len(reps),
            "unjudged": sum(1 for d in reps if d.get("status") is None),
            "rule_irrelevant": judged("rule", "무관"), "llm_relevant": judged("llm", "관련"),
            "llm_irrelevant": judged("llm", "무관"), "llm_hold": judged("llm", "보류"),
            "labeled": sum(1 for d in reps if d.get("label") is not None), "to_label": len(reps),
            "chunked": sum(1 for d in links if d.get("chunked")),
        })
    return rows


def label_rows(topic_id: int, include_labeled: bool = False, limit: Optional[int] = None) -> list[dict]:
    """정답 라벨을 붙일 대표 기사 (발행일 최신순): {link, title, press, pub_date, search_term, description, content}"""
    links = [d for d in _topic_links(topic_id) if d.get("duplicate_of") is None
             and (include_labeled or d.get("label") is None)]
    articles = _articles_for(links)
    if limit is not None:
        articles = articles[:limit]
    return [{"link": a.link, "title": a.title, "press": a.press, "pub_date": a.pub_date, "search_term": a.search_term,
             "description": a.description, "content": a.content} for a in articles]


@dataclass
class DuplicateGroup:
    representative: Article     # 대표 기사 (본문 포함, 임베딩 · 이미지 없음)
    duplicates: int             # 묶인 중복 기사 수
    members: list[Article]      # 묶인 기사들 (제목 · 본문 · link · 언론사 · 발행일)


def duplicate_groups(topic_id: int, period_days: Optional[int] = None, limit: int = 5,
                     statuses: Iterable[str] = ("관련", "보류")) -> list[DuplicateGroup]:
    """topic의 대표 기사 중 중복이 많이 묶인 순으로 limit건 (중복이 없는 기사는 빠진다).
    period_days: 대표 기사 발행일이 최근 며칠 이내인 것만. 같은 개수면 발행일 최신순."""
    wanted = set(statuses)
    since, until = _window(period_days, None)
    links = _topic_links(topic_id, fields=["article_key", "duplicate_of", "status", "pub_date"])
    members: dict[str, list[str]] = {}
    for d in links:
        if d.get("duplicate_of"):
            members.setdefault(d["duplicate_of"], []).append(d["article_key"])
    reps = [d for d in links if d.get("duplicate_of") is None and d.get("status") in wanted
            and d["article_key"] in members
            and (since is None or (d.get("pub_date") is not None and since <= d["pub_date"] <= until))]
    reps = sorted(_sort_newest(reps, lambda d: d.get("pub_date"), lambda d: d["article_key"]),
                  key=lambda d: -len(members[d["article_key"]]))[:limit]
    fields = ["link", "title", "press", "pub_date", "description", "content"]
    docs = _get_all(ARTICLES, [k for d in reps for k in (d["article_key"], *members[d["article_key"]])], fields=fields)
    return [DuplicateGroup(
        representative=_to_article(docs[d["article_key"]]),
        duplicates=len(members[d["article_key"]]),
        members=[_to_article(docs[m]) for m in members[d["article_key"]] if m in docs],
    ) for d in reps if d["article_key"] in docs]


def articles_by_links(links: list[str]) -> list[dict]:
    """link로 기사 제목 · 본문 ({link, title, content}). 실험 C의 근거 글에 쓴다."""
    docs = _get_all(ARTICLES, [article_key(link) for link in links], fields=["link", "title", "content"])
    return [docs[article_key(link)] for link in dict.fromkeys(links) if article_key(link) in docs]


def evidence_lookup(query: str, links: list[str]) -> tuple[list[str], list[tuple], dict[str, str]]:
    """실험용 정답 근거 확인 (experiments/qa_set.db_lookup).
    돌려주는 값: (그 검색어의 topic 의미 이름들, [(기사 link, 의미 이름, 판정, 대표 기사 link)], {이미지 주소: 기사 link})
    근거로 기사 대신 기사 속 이미지 주소를 적은 경우 그 이미지가 실린 기사를 찾는다 (주소 뒤 ?type=... 은 무시)."""
    topics = list_topics(query)
    senses = [t.selected.sense.name for t in sorted(topics, key=lambda t: t.id)]
    if not links:
        return senses, [], {}
    bases = list(dict.fromkeys(link.split("?")[0] for link in links))
    via: dict[str, str] = {}
    for part in _chunks_of(bases, IN_LIMIT):
        for _, d in _stream(ARTICLES, ("image_bases", "array_contains_any", part), fields=["link", "image_bases"]):
            for base in d.get("image_bases") or []:
                if base in part:
                    via.setdefault(base, d["link"])
    image_of = {via[link.split("?")[0]]: link for link in links if link.split("?")[0] in via}
    sense_of = {t.id: t.selected.sense.name for t in topics}
    targets = list(dict.fromkeys([*links, *image_of]))
    found = [d for _, d in _stream_in(ARTICLE_TOPICS, "article_key", [article_key(x) for x in targets])
             if d["topic_id"] in sense_of]
    reps = _get_all(ARTICLES, [d["duplicate_of"] for d in found if d.get("duplicate_of")], fields=["link"])
    rows = [(d["link"], sense_of[d["topic_id"]], d.get("status"),
             reps[d["duplicate_of"]]["link"] if d.get("duplicate_of") in reps else None) for d in found]
    return senses, rows, image_of


# ---------------------------------------------------------------- 초기화 (인덱스)

# 검색에 필요한 복합 인덱스 (나머지 조건은 Firestore가 자동으로 만드는 한 필드 인덱스로 된다)
#   (컬렉션, [(필드, "ASC" | "CONTAINS")], 벡터 인덱스 여부)
INDEXES = [
    (ARTICLE_TOPICS, [("topic_id", "ASC"), ("pub_date", "ASC")], False),     # topic + 발행일 기간
    (CHUNKS, [("search_keys", "CONTAINS")], True),                           # 벡터 검색 (기간 없음)
    (CHUNKS, [("search_keys", "CONTAINS"), ("pub_date", "ASC")], True),      # 벡터 검색 + 발행일 기간
]


def init_db(wait: bool = False) -> list[str]:
    """검색에 필요한 복합 인덱스 · 벡터 인덱스를 만든다 (이미 있으면 건너뜀). 여러 번 실행해도 안전하다.
    인덱스를 처음 만들면 Firestore가 몇 분 동안 만든다 (그동안 검색은 '인덱스 필요' 오류).
    필요한 권한: Cloud Datastore 색인 관리자 (roles/datastore.indexAdmin). 각 인덱스의 상태 글을 돌려준다."""
    from google.api_core.exceptions import AlreadyExists
    from google.cloud import firestore_admin_v1 as fa

    admin = fa.FirestoreAdminClient()
    Field = fa.Index.IndexField
    db_path = f"projects/{settings.firestore_project}/databases/{settings.firestore_database}"
    ops, out = [], []
    for collection, fields, vector in INDEXES:
        index_fields = [Field(field_path=f, array_config=Field.ArrayConfig.CONTAINS) if kind == "CONTAINS"
                        else Field(field_path=f, order=Field.Order.ASCENDING) for f, kind in fields]
        if vector:
            index_fields.append(Field(field_path="embedding", vector_config=Field.VectorConfig(
                dimension=EMBEDDING_DIM, flat=Field.VectorConfig.FlatIndex())))
        name = f"{collection}({', '.join(f for f, _ in fields)}{', embedding 벡터' if vector else ''})"
        try:
            ops.append(admin.create_index(parent=f"{db_path}/collectionGroups/{collection}",
                                          index=fa.Index(query_scope=fa.Index.QueryScope.COLLECTION,
                                                         fields=index_fields)))
            out.append(f"만들기 시작: {name}")
        except AlreadyExists:
            out.append(f"이미 있음: {name}")
    if wait:
        for op in ops:
            op.result(timeout=1800)
    return out
