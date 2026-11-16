"""
DB(qupict/db.py, Firestore) 저장·조회 — 가짜 Firestore(메모리)로 0~5단계를 한 바퀴 돌린다

실제 Firestore에 연결하지 않는다. 실행: python tests/run_offline.py
(실제 DB 점검은 python tests/check_db.py)
"""
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from qupict import db  # noqa: E402
from qupict.schemas import Article, ArticleImage, NewsMeta, SelectedSense, Sense, SenseCandidates  # noqa: E402
import fake_firestore  # noqa: E402

real_client = db._client
fake = fake_firestore.install(db)
now = datetime.now(timezone.utc)
link = lambda i: f"https://news.example/{i}"


def vec(seed: int) -> list[float]:
    return [((seed * 7919 + i * 104729) % 1000) / 1000 - 0.5 for i in range(db.EMBEDDING_DIM)]


# 1) 의미 후보: 없음 → 저장 → 순서 · 고른 횟수 · 직접 입력
assert db.get_sense_candidates("강남") is None
a, b = Sense(name="방송인 강남", definition="가수", aliases=["Kangnam"]), Sense(name="지역 강남", definition="강남구")
db.save_sense_candidates(SenseCandidates(query="강남", senses=[a, b], model="m"))
assert [s.name for s in db.get_sense_candidates("강남").senses] == ["방송인 강남", "지역 강남"]   # LLM 순위
db.record_sense_use("강남", b)
cached = db.get_sense_candidates("강남")
assert [s.name for s in cached.senses] == ["지역 강남", "방송인 강남"] and cached.use_counts["지역 강남"] == 1
custom = Sense(name="강남 맛집", definition="직접 입력", is_custom=True)
db.record_sense_use("강남", custom, ["지역 강남"])
db.record_sense_use("강남", Sense(name="방송인 강남", definition="덮어쓰면 안 됨", is_custom=True))   # 같은 이름 LLM 후보
cached = db.get_sense_candidates("강남")
assert cached.custom_includes == {"강남 맛집": ["지역 강남"]}
assert next(s for s in cached.senses if s.name == "방송인 강남").definition == "가수"      # LLM 후보 내용 유지
db.save_sense_candidates(SenseCandidates(query="강남", senses=[a, b], model="m2"))          # 다시 저장해도 횟수 유지
assert db.get_sense_candidates("강남").use_counts["지역 강남"] == 1
print("1 ok: 의미 후보 (저장 · 순서 · 고른 횟수 · 직접 입력)")

# 2) topic: 같은 (검색어, 의미)는 같은 id, 새 topic은 다음 번호
sel = SelectedSense(query="강남", sense=a, excluded_terms=["강남구"], other_senses=[b])
tid = db.save_topic(sel)
assert db.save_topic(sel) == tid == 1 and db.save_topic(SelectedSense(query="강남", sense=b)) == 2
rec = db.get_topic("강남", "방송인 강남")
assert rec.id == tid and rec.selected.other_senses[0].name == "지역 강남" and rec.selected.excluded_terms == ["강남구"]
assert [r.id for r in db.list_topics()] == [1, 2] and [r.id for r in db.list_topics("없음")] == []
db.mark_topic_collected(tid)
assert db.get_topic("강남", "방송인 강남").last_collected_at is not None
print("2 ok: topic (재사용 · 번호 · 복원)")

# 3) 기사: 저장 · 이미 있는 기사 연결 · 기간 · 갱신 규칙
arts = [Article(title=f"제목{i}", link=link(i), press="뉴스1" if i == 1 else None, content=f"본문 {i}",
                pub_date=now - timedelta(days=d), search_term="강남 예능",
                images=[ArticleImage(url=f"https://img/{i}/{k}.jpg?type=w", caption=f"캡션{k}") for k in range(n)],
                embedding=vec(i) if i == 1 else None)
        for i, d, n in [(1, 1, 2), (2, 2, 1), (3, 60, 0)]]
keys = db.save_articles(tid, arts)
assert keys[link(1)] == db.article_key(link(1))
metas = [NewsMeta(title="x", link=link(i), search_term="강남") for i in (1, 4)]
assert [m.link for m in db.link_known_articles(2, metas)] == [link(4)]          # 1은 이미 있음 → topic 2에 연결만
assert len(db.get_topic_articles(tid)) == 3 and len(db.get_topic_articles(2)) == 1
assert [x.link for x in db.get_topic_articles(tid, period_days=30)] == [link(1), link(2)]    # 최신순, 60일 전 제외
got = db.get_topic_articles(tid)[0]
assert got.embedding == db._vec(vec(1)) and [i.caption for i in got.images] == ["캡션0", "캡션1"]
assert got.search_term == "강남 예능" and got.press == "뉴스1"
db.save_articles(tid, [arts[0].model_copy(update={"press": None, "embedding": None, "content": "새 본문"})])
got = db.get_topic_articles(tid)[0]
assert got.press == "뉴스1" and got.embedding is not None and got.content == "새 본문"      # 빈 값은 덮어쓰지 않음
db.save_article_embeddings([arts[1].model_copy(update={"embedding": vec(2)})])
assert db.get_topic_articles(tid)[1].embedding == db._vec(vec(2))
assert db.save_articles(tid, []) == {}
print("3 ok: 기사 (저장 · 연결 · 기간 · 빈 값 유지 · 임베딩)")

# 4) 판정: 중복 · 판정 · 라벨 · 중복 기사에 판정 복사
db.save_duplicates(tid, {link(2): link(1), link(9): link(1)})                   # 없는 기사는 건너뜀
db.save_judgments(tid, [db.Judgment(link(1), "관련", "llm", "근거", "m"), db.Judgment(link(3), "무관", "rule")])
for bad in (db.Judgment(link(1), "중복", "rule"), db.Judgment(link(1), "관련", "사람")):
    try:
        db.save_judgments(tid, [bad])
        raise AssertionError("예외 없음")
    except ValueError:
        pass
db.save_labels(tid, {link(1): "관련"})
assert [x.link for x in db.get_topic_articles(tid, statuses=["관련"])] == [link(1)]
assert len(db.get_topic_articles(tid, include_duplicates=False)) == 2
assert db.copy_duplicate_judgments(tid) == 1 and db.copy_duplicate_judgments(tid) == 0
assert db.count_pending(tid) == (0, 1)                 # 미판정 0, 관련인데 청크 없는 대표 1
assert db.labeled_links(tid) == {link(1)} and db.representative_count(tid) == 2
print("4 ok: 판정 (중복 · 판정 · 잘못된 값 · 라벨 · 판정 복사 · 남은 일)")

# 5) 이미지 설명
todo = db.get_images_to_describe(tid)
assert [(i.article_link, i.caption) for i in todo] == [(link(1), "캡션0"), (link(1), "캡션1")]   # 관련 · 대표 기사만
db.save_image_descriptions([(todo[0].image_id, "LLM 설명"), (todo[1].image_id, "")], "vlm")
assert db.get_images_to_describe(tid) == []
assert [(r["id"], r["llm_description"]) for r in db.described_image_rows([link(1)])] == [
    (todo[0].image_id, "LLM 설명"), (todo[1].image_id, "")]
assert db.processed_image_counts([link(1), link(2)]) == {link(1): 2}
db.save_articles(tid, [arts[0]])                                                # 다시 저장해도 설명은 남는다
assert len(db.described_image_rows([link(1)])) == 2
print("5 ok: 이미지 설명 (대상 · 저장 · 다시 저장해도 유지)")

# 6) 청크: 교체 저장 · 남는 청크 삭제 · 검색 · 판정 바뀌면 검색 대상도 바뀜
assert [a.link for a in db.get_articles_without_chunks(tid)] == [link(1)]
db.save_chunks(link(1), [db.Chunk("본문 청크", vec(10)), db.Chunk("둘째", vec(11)),
                         db.Chunk("이미지", vec(12), "image", todo[0].image_id)])
db.save_chunks(link(1), [db.Chunk("본문 청크", vec(10)), db.Chunk("이미지", vec(12), "image", todo[0].image_id)])
assert sorted(i for c, i in fake.docs if c == "chunks") == [f"{db.article_key(link(1))}_000", f"{db.article_key(link(1))}_001"]
assert db.get_articles_without_chunks(tid) == [] and db.count_pending(tid) == (0, 0)
hits = db.search_chunks(tid, vec(10), k=2, period_days=7, statuses=("관련", "보류"))
assert [h.content for h in hits] == ["본문 청크", "이미지"] and abs(hits[0].similarity - 1) < 1e-6
assert hits[0].article.title == "제목1" and hits[0].article.press == "뉴스1"
assert db.search_chunks(tid, vec(10), period_days=7, now=now - timedelta(days=30)) == []      # 기간 밖
assert db.search_chunks(2, vec(10)) == []                  # topic 2에서는 아직 판정 없음
assert db.searchable_stats(tid, 30) == (1, arts[0].pub_date)
db.save_judgments(tid, [db.Judgment(link(1), "무관", "llm")])
assert db.search_chunks(tid, vec(10), statuses=("관련", "보류")) == []      # 판정이 바뀌면 바로 검색에서 빠진다
db.save_judgments(2, [db.Judgment(link(1), "보류", "llm")])
assert len(db.search_chunks(2, vec(10), statuses=("관련", "보류"))) == 2     # 다른 topic 판정은 따로
db.save_judgments(tid, [db.Judgment(link(1), "관련", "llm")])
try:
    db.save_chunks_many([(link(2), []), ("https://없는기사", [])])
    raise AssertionError("예외 없음")
except ValueError as e:
    assert "없는기사" in str(e)
print("6 ok: 청크 (교체 · 삭제 · 검색 · 기간 · topic별 판정 · 없는 기사)")

# 7) 현황 · 평가용 조회
groups = db.duplicate_groups(tid, period_days=30)
assert [(g.representative.link, g.duplicates, [m.link for m in g.members]) for g in groups] == [(link(1), 1, [link(2)])]
rows = {r["topic_id"]: r for r in db.topic_status_rows()}
assert (rows[tid]["articles"], rows[tid]["duplicates"], rows[tid]["llm_relevant"], rows[tid]["rule_irrelevant"],
        rows[tid]["labeled"], rows[tid]["chunked"]) == (3, 1, 1, 1, 1, 1)
assert rows[2]["articles"] == 1 and rows[2]["chunked"] == 1                     # 같은 기사의 청크가 topic 2에도 보인다
assert [r["link"] for r in db.label_rows(tid)] == [link(3)] and len(db.label_rows(tid, include_labeled=True)) == 2
assert [d["title"] for d in db.articles_by_links([link(2), link(1)])] == ["제목2", "제목1"]
senses, found, image_of = db.evidence_lookup("강남", [link(2), "https://img/1/0.jpg?type=other"])
assert senses == ["방송인 강남", "지역 강남"] and image_of == {link(1): "https://img/1/0.jpg?type=other"}
assert sorted(found) == sorted([(link(2), "방송인 강남", "관련", link(1)), (link(1), "방송인 강남", "관련", None),
                                (link(1), "지역 강남", "보류", None)])
print("7 ok: 현황 · 라벨 · 많이 보도된 기사 · 평가용 조회")

# 8) 필터 초기화: 판정 · 중복이 지워지고 검색 대상에서도 빠진다 (라벨은 남음)
db.reset_topic_filter(tid)
assert len(db.get_topic_articles(tid, unjudged_only=True)) == 3 and db.labeled_links(tid) == {link(1)}
assert db.search_chunks(tid, vec(10), statuses=("관련", "보류")) == []
assert len(db.search_chunks(2, vec(10), statuses=("관련", "보류"))) == 2
print("8 ok: 필터 초기화")

# 9) 쓰기 묶음: WRITE_BATCH개씩 나눠 보낸다, DB 설정이 없으면 알려 준다
fake.commits.clear()
w = db._Writes()
for i in range(db.WRITE_BATCH * 2 + 5):
    w.set("tmp", str(i), {"i": i})
assert w.commit() == db.WRITE_BATCH * 2 + 5 and fake.commits == [db.WRITE_BATCH, db.WRITE_BATCH, 5]
db._client, db.settings, db._client_obj = real_client, SimpleNamespace(has_db=False), None
try:
    db.ping()
    raise AssertionError("예외 없음")
except EnvironmentError:
    pass
print("9 ok: 쓰기 묶음 · 설정 없음")
print("ALL PASS")
