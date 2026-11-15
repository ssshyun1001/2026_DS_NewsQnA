"""
DB 커넥션 풀 · 한 번에 저장하기 — 예전(한 줄씩)과 DB에 보내는 내용이 같은지 (가짜 연결)

실제 DB에 연결하지 않는다. 실행: python tests/run_offline.py
"""
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from qupict import db  # noqa: E402
from qupict.schemas import Article, ArticleImage  # noqa: E402

norm = lambda sql: " ".join(sql.split())


class FakeCursor:
    """execute / executemany로 보낸 (SQL, 값)을 기록한다. RETURNING id는 1000부터 차례로 준다."""
    def __init__(self, log, ids):
        self.log, self.ids, self._rows = log, ids, []

    def __enter__(self): return self
    def __exit__(self, *a): pass

    def execute(self, sql, params=None):
        self.log.append((norm(sql), params))
        if "SELECT id, link FROM articles" in sql:
            self._rows = [{"id": self.ids[l], "link": l} for l in params[0] if l in self.ids]
        elif "RETURNING id" in sql:
            self.ids.setdefault(params[0], 1000 + len(self.ids)); self._rows = [{"id": self.ids[params[0]]}]
        return self

    def executemany(self, sql, rows, returning=False):
        rows = list(rows)
        for r in rows:
            self.log.append((norm(sql), r))
        if returning:
            for r in rows:
                self.ids.setdefault(r[0], 1000 + len(self.ids))
            self._rows = [{"id": self.ids[r[0]]} for r in rows]

    def fetchone(self): return self._rows.pop(0)
    def fetchall(self): rows, self._rows = self._rows, []; return rows
    def nextset(self): return True if self._rows else None


class FakeConn(FakeCursor):
    def cursor(self): return FakeCursor(self.log, self.ids)


def run_with_fake(fn):
    log, ids, opened = [], {}, []

    @contextmanager
    def fake_connect():
        opened.append(1)
        yield FakeConn(log, ids)
    real = db.connect
    db.connect = fake_connect
    try:
        fn()
    finally:
        db.connect = real
    return log, len(opened)


now = datetime(2026, 11, 10, tzinfo=timezone.utc)
arts = [Article(title=f"t{i}", link=f"https://a/{i}", content=f"본문 {i}", collected_at=now, search_term="강남 예능",
                images=[ArticleImage(url=f"https://img/{i}/{k}", caption=f"c{k}") for k in range(i % 3)])
        for i in range(5)]

# 1) save_articles: 예전(기사마다 3번) 방식과 DB에 보내는 행이 같다
def old_save_articles(topic_id, articles):
    with db.connect() as conn:
        for a in articles:
            row = conn.execute(OLD_INSERT, (a.link, a.title, a.press, a.pub_date, a.description, a.content,
                                            a.collected_at, db._vec(a.embedding), None)).fetchone()
            with conn.cursor() as cur:
                cur.executemany(IMG_SQL, [(row["id"], i, img.url, img.caption) for i, img in enumerate(a.images)])
            conn.execute(TOPIC_SQL, (row["id"], topic_id, a.search_term))

new_log, new_conns = run_with_fake(lambda: db.save_articles(7, arts))
OLD_INSERT = next(sql for sql, _ in new_log if sql.startswith("INSERT INTO articles"))
IMG_SQL = next(sql for sql, _ in new_log if sql.startswith("INSERT INTO article_images"))
TOPIC_SQL = next(sql for sql, _ in new_log if sql.startswith("INSERT INTO article_topics"))
old_log, old_conns = run_with_fake(lambda: old_save_articles(7, arts))
by_table = lambda log: {t: [p for s, p in log if s.startswith(f"INSERT INTO {t} ")]
                        for t in ("articles", "article_images", "article_topics")}
assert by_table(new_log) == by_table(old_log), (by_table(new_log), by_table(old_log))
assert len(by_table(new_log)["article_images"]) == 0 + 1 + 2 + 0 + 1
assert new_conns == old_conns == 1
assert run_with_fake(lambda: db.save_articles(7, []))[1] == 0        # 빈 목록이면 연결도 안 함
print("1 ok: save_articles (한 번에 보내도 행이 예전과 같음)")

# 2) save_chunks_many: 기사별 save_chunks를 여러 번 한 것과 같은 청크가 들어간다, 연결은 1번
chunks = {f"https://a/{i}": [db.Chunk(f"청크 {i}-{k}", [0.1 * k] * db.EMBEDDING_DIM) for k in range(i + 1)] for i in range(3)}
known = {link: 1000 + i for i, link in enumerate(chunks)}          # 가짜 DB에 이미 있는 기사 id


def run_with_known(fn):
    """기사 id가 미리 있는 가짜 DB로 fn을 실행하고 (보낸 기록, 연결 횟수)를 돌려준다."""
    log, opened = [], []

    @contextmanager
    def fake_connect():
        opened.append(1)
        yield FakeConn(log, dict(known))
    real = db.connect
    db.connect = fake_connect
    try:
        fn()
    finally:
        db.connect = real
    return log, len(opened)


def old_way():                                    # 예전: 기사마다 save_chunks 한 번씩
    for link, cs in chunks.items():
        db.save_chunks(link, cs)

new_log, new_conns = run_with_known(lambda: db.save_chunks_many(list(chunks.items())))
old_log, old_conns = run_with_known(old_way)
inserted = lambda log: [(r[0], r[1], r[2], r[3], r[4], list(r[5]), r[6])
                        for s, r in log if s.startswith("INSERT INTO article_chunks")]
assert inserted(new_log) == inserted(old_log) and len(inserted(new_log)) == 1 + 2 + 3
deleted = lambda log: sorted(i for s, p in log if s.startswith("DELETE FROM article_chunks") for i in p[0])
assert deleted(new_log) == deleted(old_log) == [1000, 1001, 1002]
assert (new_conns, old_conns) == (1, 3)                             # 연결 3번 → 1번
try:
    run_with_known(lambda: db.save_chunks_many([("https://a/0", []), ("https://없는기사", [])]))
    raise AssertionError("예외 없음")
except ValueError as e:
    assert "없는기사" in str(e)
print("2 ok: save_chunks_many (기사별로 따로 저장한 것과 같음, 없는 기사는 저장 안 함)")

# 3) 커넥션 풀: 한 번만 만들고, 최근에 쓴 연결은 확인을 건너뛰고, 오래 쉰 연결은 확인한다
created = []
class FakePool:
    def __init__(self, url, **kw): created.append(kw); self.closed = False
    @contextmanager
    def connection(self): yield "conn"
    def close(self, timeout=None): self.closed = True
db.ConnectionPool, db.settings, db._pool = FakePool, SimpleNamespace(database_url="postgresql://가짜", chat_model="m",
                                                                   embedding_model="e"), None
for _ in range(3):
    with db.connect() as c:
        assert c == "conn"
assert len(created) == 1 and created[0]["kwargs"]["prepare_threshold"] is None and created[0]["max_size"] == db.POOL_MAX_SIZE
pinged = []
class C:
    def execute(self, sql): pinged.append(sql)
conn = C(); db._check(conn); assert pinged == ["SELECT 1"]          # 한 번도 안 쓴 연결 → 확인
db._reset(conn); db._check(conn); assert pinged == ["SELECT 1"]     # 방금 돌아온 연결 → 확인 생략
conn._qupict_last_used = time.monotonic() - db.POOL_CHECK_AFTER - 1
db._check(conn); assert pinged == ["SELECT 1", "SELECT 1"]          # 오래 쉰 연결 → 확인
pool = db._pool; db.close_pool(); assert pool.closed and db._pool is None
db.settings = SimpleNamespace(database_url="")
try:
    with db.connect():
        pass
    raise AssertionError("예외 없음")
except EnvironmentError:
    pass
print("3 ok: 커넥션 풀 (한 번 생성, 연결 확인 규칙, 닫기, 주소 없음)")
print("ALL PASS")
