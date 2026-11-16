"""
예전 DB(Supabase PostgreSQL + pgvector) → Firestore로 데이터 옮기기. 한 번만 쓰는 스크립트.

    python tools/migrate_supabase_to_firestore.py            옮기기 (Firestore가 비어 있어야 함)
    python tools/migrate_supabase_to_firestore.py --force    Firestore에 이미 있는 문서를 덮어쓰며 다시 옮기기
    python tools/migrate_supabase_to_firestore.py --check    옮기지 않고 개수만 비교

필요한 것
  - .env: DATABASE_URL (예전 Supabase 주소, 읽기만 함), FIRESTORE_PROJECT
  - 예전 DB 패키지 (requirements.txt에서 뺐으므로 따로): pip install "psycopg[binary]==3.2.3" pgvector==0.3.6

옮기는 내용 (임베딩은 그대로 복사 → 다시 계산하지 않음, LLM 비용 0)
  query_senses → query_senses      topics → topics (+ topic_keys, counters/topics). topic id는 그대로
  articles + article_images → articles (이미지는 문서 안 images 맵)
  article_topics → article_topics (duplicate_of는 대표 기사의 article_key로, chunked 채움)
  article_chunks → chunks (image_id는 "기사키/이미지키"로, search_keys · 출처 정보 채움)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qupict.cli import setup_console  # noqa: E402
from qupict import db  # noqa: E402
from google.cloud.firestore_v1.vector import Vector  # noqa: E402

setup_console()

COLLECTIONS = [db.SENSES, db.TOPICS, db.TOPIC_KEYS, db.ARTICLES, db.ARTICLE_TOPICS, db.CHUNKS]
TABLES = ["query_senses", "topics", "articles", "article_images", "article_topics", "article_chunks"]


def old_connect():
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg.rows import dict_row

    url = os.getenv("DATABASE_URL", "")
    if not url:
        sys.exit("DATABASE_URL(예전 Supabase 주소)이 .env에 없습니다.")
    conn = psycopg.connect(url, prepare_threshold=None, connect_timeout=15, row_factory=dict_row)
    register_vector(conn)
    return conn


def read_old() -> dict[str, list[dict]]:
    """예전 DB 6개 테이블을 같은 시점(스냅샷)으로 읽는다. 읽기만 한다."""
    with old_connect() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        data = {t: conn.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in TABLES}
        conn.rollback()
    return data


def firestore_counts() -> dict[str, int]:
    return {c: int(db._col(c).count().get()[0][0].value) for c in COLLECTIONS}


def build(data: dict[str, list[dict]]) -> list[tuple[str, str, dict]]:
    """예전 행 → Firestore 문서 (컬렉션, 문서 id, 내용)."""
    docs: list[tuple[str, str, dict]] = []
    sense_doc = {}     # 예전 query_senses.id → 새 문서 id
    for r in data["query_senses"]:
        doc_id = db._sense_id(r["query"], r["name"])
        sense_doc[r["id"]] = doc_id
        docs.append((db.SENSES, doc_id, {
            "query": r["query"], "rank": r["rank"], "name": r["name"], "definition": r["definition"],
            "aliases": list(r["aliases"]), "related_terms": list(r["related_terms"]), "is_custom": r["is_custom"],
            "includes": list(r["includes"]), "use_count": r["use_count"], "last_used_at": r["last_used_at"],
            "model": r["model"], "created_at": r["created_at"]}))

    for r in data["topics"]:
        docs.append((db.TOPICS, str(r["id"]), {
            "id": r["id"], "query": r["query"], "sense_id": sense_doc.get(r["sense_id"]), "sense_name": r["sense_name"],
            "definition": r["definition"], "aliases": list(r["aliases"]), "related_terms": list(r["related_terms"]),
            "is_custom": r["is_custom"], "excluded_terms": list(r["excluded_terms"]), "other_senses": r["other_senses"],
            "last_collected_at": r["last_collected_at"], "created_at": r["created_at"], "updated_at": r["updated_at"]}))
        docs.append((db.TOPIC_KEYS, db._hash(r["query"], r["sense_name"]),
                     {"id": r["id"], "query": r["query"], "sense_name": r["sense_name"]}))
    next_id = max((r["id"] for r in data["topics"]), default=0) + 1
    docs.append((db.COUNTERS, "topics", {"next": next_id}))

    akey = {r["id"]: db.article_key(r["link"]) for r in data["articles"]}
    articles = {r["id"]: r for r in data["articles"]}
    chunk_count: dict[int, int] = {}
    for c in data["article_chunks"]:
        chunk_count[c["article_id"]] = chunk_count.get(c["article_id"], 0) + 1
    images: dict[int, dict] = {}
    image_ref = {}     # 예전 article_images.id → "기사키/이미지키"
    for i in data["article_images"]:
        ikey = db._image_key(i["url"])
        image_ref[i["id"]] = f"{akey[i['article_id']]}/{ikey}"
        img = {"position": i["position"], "url": i["url"], "caption": i["caption"]}
        if i["llm_description"] is not None:
            img.update(llm_description=i["llm_description"], described_model=i["described_model"],
                       described_at=i["described_at"])
        images.setdefault(i["article_id"], {})[ikey] = img

    for r in data["articles"]:
        doc = {"link": r["link"], "title": r["title"], "press": r["press"], "pub_date": r["pub_date"],
               "description": r["description"], "content": r["content"], "collected_at": r["collected_at"],
               "chunk_count": chunk_count.get(r["id"], 0)}
        if r["embedding"] is not None:
            doc["embedding"] = Vector(r["embedding"].tolist())
            doc["embedding_model"] = r["embedding_model"]
        if r["id"] in images:
            doc["images"] = images[r["id"]]
            doc["image_bases"] = sorted({img["url"].split("?")[0] for img in images[r["id"]].values()})
        docs.append((db.ARTICLES, akey[r["id"]], doc))

    search_keys: dict[int, list[str]] = {}
    for r in data["article_topics"]:
        if r["duplicate_of"] is None and r["status"]:
            search_keys.setdefault(r["article_id"], []).append(f"{r['topic_id']}:{r['status']}")
        a = articles[r["article_id"]]
        docs.append((db.ARTICLE_TOPICS, db._at_id(r["topic_id"], akey[r["article_id"]]), {
            "topic_id": r["topic_id"], "article_key": akey[r["article_id"]], "link": a["link"], "pub_date": a["pub_date"],
            "search_term": r["search_term"], "duplicate_of": akey.get(r["duplicate_of"]) if r["duplicate_of"] else None,
            "status": r["status"], "stage": r["stage"], "reason": r["reason"], "judged_model": r["judged_model"],
            "judged_at": r["judged_at"], "label": r["label"], "labeled_at": r["labeled_at"],
            "chunked": chunk_count.get(r["article_id"], 0) > 0, "created_at": r["created_at"]}))

    for c in data["article_chunks"]:
        a = articles[c["article_id"]]
        docs.append((db.CHUNKS, db._chunk_id(akey[c["article_id"]], c["chunk_index"]), {
            "article_key": akey[c["article_id"]], "link": a["link"], "title": a["title"], "press": a["press"],
            "pub_date": a["pub_date"], "chunk_index": c["chunk_index"], "source": c["source"],
            "image_id": image_ref.get(c["image_id"]) if c["image_id"] else None, "content": c["content"],
            "embedding": Vector(c["embedding"].tolist()), "embedding_model": c["embedding_model"],
            "search_keys": sorted(search_keys.get(c["article_id"], [])), "created_at": c["created_at"]}))
    return docs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Firestore에 문서가 있어도 덮어쓰며 옮긴다")
    p.add_argument("--check", action="store_true", help="옮기지 않고 개수만 비교")
    args = p.parse_args()

    start = time.perf_counter()
    data = read_old()
    print("예전 DB: " + ", ".join(f"{t} {len(rows)}" for t, rows in data.items())
          + f"  ({time.perf_counter() - start:.1f}초)")
    before = firestore_counts()
    print("Firestore 지금: " + ", ".join(f"{c} {n}" for c, n in before.items()))
    if args.check:
        return
    if any(before.values()) and not args.force:
        sys.exit("Firestore에 이미 문서가 있습니다. 덮어쓰며 다시 옮기려면 --force")

    docs = build(data)
    start = time.perf_counter()
    w = db._Writes()
    for collection, doc_id, doc in docs:
        w.set(collection, doc_id, doc)
    w.commit()
    print(f"쓰기 {len(docs)}건 ({time.perf_counter() - start:.1f}초)")
    after = firestore_counts()
    print("Firestore 이후: " + ", ".join(f"{c} {n}" for c, n in after.items()))
    expected = {db.SENSES: len(data["query_senses"]), db.TOPICS: len(data["topics"]),
                db.TOPIC_KEYS: len(data["topics"]), db.ARTICLES: len(data["articles"]),
                db.ARTICLE_TOPICS: len(data["article_topics"]), db.CHUNKS: len(data["article_chunks"])}
    bad = {c: (after[c], n) for c, n in expected.items() if after[c] < n}
    print("개수 확인: " + ("모두 맞음" if not bad else f"모자람 {bad}"))


if __name__ == "__main__":
    main()
