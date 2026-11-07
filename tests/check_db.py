"""
DB 점검 스크립트 — DB 연결, 테이블, 저장·조회 함수(qupict/db.py)가 제대로 동작하는지 확인한다.

실행 (프로젝트 폴더에서):
    python tests/check_db.py

  1) 연결·확장·테이블 확인
  2) 가짜 데이터로 0~5단계 저장·조회를 한 바퀴 돌려 결과를 확인
  3) 끝나면 가짜 데이터를 지운다 (실패해도 지움). 실제 데이터는 건드리지 않는다.
LLM·네이버 API는 부르지 않는다 (크레딧 0).
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from qupict import db  # noqa: E402
from qupict.schemas import Article, ArticleImage, NewsMeta, SelectedSense, Sense, SenseCandidates  # noqa: E402

TEST_QUERY = "__DB점검__"
TEST_LINK = "https://dbcheck.invalid/"
TABLES = ["query_senses", "topics", "articles", "article_images", "article_topics", "article_chunks"]

_passed = 0


def check(name: str, ok: bool, detail: object = "") -> None:
    global _passed
    if not ok:
        raise AssertionError(f"{name} 실패 {detail}")
    _passed += 1
    print(f"  ✅ {name}")


def fake_embedding(seed: int) -> list[float]:
    rnd = random.Random(seed)
    return [rnd.random() for _ in range(db.EMBEDDING_DIM)]


def check_connection() -> None:
    print("[1] 연결·확장·테이블")
    db.init_db()
    with db.connect() as conn:
        ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        tables = {r["tablename"] for r in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'").fetchall()}
        counts = {t: conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"] for t in TABLES if t in tables}
    check("DB 연결 (DATABASE_URL)", True)
    check(f"pgvector 확장 (버전 {ext['extversion'] if ext else '없음'})", ext is not None)
    missing = [t for t in TABLES if t not in tables]
    check("테이블 6개", not missing, f"없는 테이블: {missing}")
    print("  현재 저장된 행 수:", counts)


def link(i: int) -> str:
    return f"{TEST_LINK}{i}"


def check_pipeline() -> None:
    now = datetime.now(timezone.utc)

    print("[2] 0단계: 의미 후보 캐싱 · topic")
    check("처음 보는 검색어는 캐시 없음", db.get_sense_candidates(TEST_QUERY) is None)
    a = Sense(name="의미A", definition="정의A", aliases=["별칭A"], related_terms=["연관A"])
    b = Sense(name="의미B", definition="정의B", related_terms=["연관B"])
    db.save_sense_candidates(SenseCandidates(query=TEST_QUERY, senses=[a, b], model="test"))
    cached = db.get_sense_candidates(TEST_QUERY)
    check("저장한 후보를 순서대로 다시 읽음", [s.name for s in cached.senses] == ["의미A", "의미B"])
    selected = SelectedSense(query=TEST_QUERY, sense=a, excluded_terms=["연관B"], other_senses=[b])
    topic_id = db.save_topic(selected)
    check("같은 (검색어, 의미)는 같은 topic 재사용", db.save_topic(selected) == topic_id)
    check("topic에서 SelectedSense 복원", db.get_topic(TEST_QUERY, "의미A").selected.other_senses[0].name == "의미B")

    print("[3] 1단계: 기사 저장 · 재수집 시 새 기사만")
    articles = [
        Article(title=f"기사{i}", link=link(i), pub_date=now - timedelta(days=days), content="본문",
                search_term=TEST_QUERY, embedding=fake_embedding(i),
                images=[ArticleImage(url=f"{TEST_LINK}img{i}.jpg", caption="사진 설명")])
        for i, days in [(1, 1), (2, 3), (3, 40)]
    ]
    db.save_articles(topic_id, articles)
    metas = [NewsMeta(title="x", link=link(i), search_term=TEST_QUERY) for i in (1, 2, 4)]
    new = db.link_known_articles(topic_id, metas)
    check("DB에 있는 기사(1,2)는 빼고 새 기사(4)만 크롤링 대상", [m.link for m in new] == [link(4)], new)
    check("topic 기사 3건", len(db.get_topic_articles(topic_id)) == 3)
    check("발행일 기준 최근 30일 → 2건", len(db.get_topic_articles(topic_id, period_days=30)) == 2)
    check("임베딩 저장·복원", db.get_topic_articles(topic_id)[0].embedding is not None)

    print("[4] 2·3단계: 중복 · 판정 · 정답 라벨")
    db.save_duplicates(topic_id, {link(2): link(1)})
    db.save_judgments(topic_id, [
        db.Judgment(link(1), "관련", "llm", "근거 한 줄", "test"),
        db.Judgment(link(3), "무관", "rule", "배제어 포함"),
    ])
    db.save_labels(topic_id, {link(1): "관련"})
    check("'관련' 판정만 조회", [x.link for x in db.get_topic_articles(topic_id, statuses=["관련"])] == [link(1)])
    check("중복 기사 제외 조회", len(db.get_topic_articles(topic_id, include_duplicates=False)) == 2)
    try:
        db.save_judgments(topic_id, [db.Judgment(link(1), "중복", "rule")])
        check("잘못된 판정 값 거부", False)
    except ValueError:
        check("잘못된 판정 값 거부", True)

    print("[5] 4단계: 이미지 설명 · 청크")
    images = db.get_images_to_describe(topic_id)
    check("'관련' 기사의 설명 없는 이미지 1장", len(images) == 1, images)
    db.save_image_description(images[0].image_id, "LLM 이미지 설명", "test")
    todo = db.get_articles_without_chunks(topic_id)
    check("청크 없는 '관련' 기사 1건", [x.link for x in todo] == [link(1)])
    chunks = [db.Chunk("본문 청크", fake_embedding(10)),
              db.Chunk("LLM 이미지 설명", fake_embedding(11), "image", images[0].image_id)]
    db.save_chunks(link(1), chunks)
    db.save_chunks(link(1), chunks)  # 다시 저장해도 교체만 된다
    check("청크 다시 저장 시 교체 (중복 안 생김)", not db.get_articles_without_chunks(topic_id))

    print("[6] 5단계: 벡터 검색")
    hits = db.search_chunks(topic_id, fake_embedding(10), k=2, period_days=7)
    check("가장 비슷한 청크가 1위 (유사도 1.0)",
          hits and hits[0].content == "본문 청크" and abs(hits[0].similarity - 1) < 1e-5, hits)
    check("질문 시점이 30일 전이면 그 뒤 기사는 안 나옴",
          db.search_chunks(topic_id, fake_embedding(10), period_days=7, now=now - timedelta(days=30)) == [])

    db.reset_topic_filter(topic_id)
    check("필터 초기화 후 전부 미판정", len(db.get_topic_articles(topic_id, unjudged_only=True)) == 3)


def cleanup() -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM articles WHERE link LIKE %s", (TEST_LINK + "%",))
        conn.execute("DELETE FROM topics WHERE query = %s", (TEST_QUERY,))
        conn.execute("DELETE FROM query_senses WHERE query = %s", (TEST_QUERY,))


def main() -> None:
    try:
        check_connection()
    except Exception as e:
        print(f"\n❌ DB에 연결하지 못했습니다: {e}")
        print("   - Docker Desktop이 켜져 있고 `docker compose up -d`를 했는지")
        print("   - .env의 DATABASE_URL 비밀번호가 POSTGRES_PASSWORD와 같은지 확인하세요.")
        sys.exit(1)
    try:
        check_pipeline()
    except Exception as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    finally:
        cleanup()
    print(f"\n🎉 전부 통과 ({_passed}개). 점검용 데이터는 지웠습니다.")


if __name__ == "__main__":
    main()
