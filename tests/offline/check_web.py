"""
웹 서비스(web.py) — 화면 → 의미 후보 → 준비(백그라운드) → 진행 로그 → 질문 흐름이 run.py 단계 함수와 맞물리는지

단계 함수(수집·필터·인덱싱·검색·답변)는 가짜로 바꾼다. LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0).
실행: python tests/run_offline.py
"""
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    from fastapi.testclient import TestClient

    import web
    from qupict import collect, db, filtering, indexing, qa, retrieval, sense
    from qupict.collect import CollectResult
    from qupict.filtering import FilterOutcome
    from qupict.indexing import IndexResult
    from qupict.schemas import Answer, Article, Sense, SenseCandidates

    cands = SenseCandidates(query="강남", senses=[Sense(name="방송인 강남", definition="가수 겸 방송인"),
                                                Sense(name="지역 강남", definition="서울 강남구")])
    sense.load_sense_candidates = lambda q: (cands, "DB 캐시")
    web.validate_settings = lambda: None
    web.settings = SimpleNamespace(database_url="postgresql://가짜", chat_model="가짜 모델")

    calls = {}

    crawled = {"n": 7}                                # 새로 크롤링한 기사 수 (두 번째 준비에서는 0)

    async def fake_collect(selected, period, max_articles):
        calls["collect"] = (selected.sense.name, period, max_articles)
        await asyncio.sleep(0.3)                      # 수집하는 동안 화면이 질문을 받을 수 있는지 보려고
        return CollectResult(topic_id=19, searched=10, reused=10 - crawled["n"], crawled=crawled["n"],
                             crawl_failed=0, images=3)

    async def fake_filter(topic_id, selected):
        calls["filter"] = calls.get("filter", 0) + 1
        return FilterOutcome()

    async def fake_index(topic_id, describe_images=True, ocr_images=False):
        calls.setdefault("index", []).append((describe_images, ocr_images))
        if ocr_images:
            time.sleep(0.3)                           # 이미지 글자 읽기는 준비 완료 뒤 백그라운드에서 돈다
            return IndexResult(articles=1, image_chunks=2, images_described=2, images_no_text=3)
        return IndexResult(articles=3, body_chunks=9)

    import run
    run.image_text_mode = lambda: "ocr"               # EasyOCR 설치 여부와 상관없이 OCR 모드로
    waited = []
    run.wait_after_prepare = lambda log=print, wrote=True: waited.append(wrote)   # 실제로 3초 기다리지 않게
    # 남은 일 세기 · 저장된 topic 찾기 · 질문 가능 기사 세기도 가짜 DB로 (처음에는 처음 검색하는 대상)
    from qupict import pipeline
    pending = {"now": pipeline.PendingWork(unjudged=7, unindexed=0)}
    pipeline.count_pending = lambda topic_id, statuses=("관련", "보류"): pending["now"]
    known = {"topic": None}
    db.get_topic = lambda query, sense_name: known["topic"]

    collect.collect_to_db, filtering.filter_topic, indexing.index_topic = fake_collect, fake_filter, fake_index
    db.get_topic_articles = lambda *a, **k: [Article(title="t", link="L")] * 4
    def fake_search(tid, q, sel, period_days=None, history=""):
        calls["search"], calls["search_history"] = (tid, q), history
        return []
    retrieval.search_topic = fake_search
    from datetime import datetime as _dt, timezone as _tz
    retrieval.oldest_searchable_date = lambda tid, days=None: _dt(2026, 10, 14, 3, tzinfo=_tz.utc)
    retrieval.searchable_stats = lambda tid, days=None: (6, _dt(2026, 10, 20, tzinfo=_tz.utc))
    def fake_answer(q, hits, on_token=None, subject="", period="", history=""):
        calls["subject"], calls["period"], calls["history"] = subject, period, history
        if q == "터짐":
            raise RuntimeError("LLM 오류")
        pieces = ["강남은 ", "예능에 나왔다 ", "(기사 1)."]
        for p in pieces if on_token else []:
            on_token(p)
        return Answer(question=q, answer="".join(pieces),
                      sources=[Article(title="기사", link="https://a", press="연합뉴스")], source_kinds=["본문"], cited=[1])
    qa.generate_answer_from_chunks = fake_answer

    # 많이 보도된 기사: DB 조회 · LLM 요약을 가짜로 (web.py는 hot.top_duplicated_db → hot.summarize 순서로 부른다)
    from qupict import hot

    def fake_top(topic_id, period_days=None):
        calls["hot"] = (topic_id, period_days)
        return [hot.HotArticle(title="강남 새 예능 출연", link="https://hot/1", press="연합뉴스",
                               pub_date=_dt(2026, 11, 1, tzinfo=_tz.utc), description="요약문", content="본문", duplicates=7),
                hot.HotArticle(title="강남 귀화 소식", link="https://hot/2", press=None, pub_date=None,
                               description="", content="강남이 귀화했다. 뒷이야기.", duplicates=2)]

    def fake_summarize(items, terms=()):
        time.sleep(0.2)                               # 한 줄 요약은 준비 완료 뒤에 돈다
        calls["hot_terms"] = list(terms)
        items[0].summary = "강남이 새 예능에 나온다."
        return items
    hot.top_duplicated_db, hot.summarize = fake_top, fake_summarize

    client = TestClient(web.app)
    assert client.get("/").status_code == 200
    # 웹 화면은 첫 질문을 고정하지 않는다 (터미널 run.py만 qa.first_question으로 요약부터 답한다)
    page = client.get("/").text
    assert "enqueueQuestion(`" not in page and "전체적인 요약을 제공해줘" not in page
    assert 'id="c-hot"' in page and page.index('id="c-job"') < page.index('id="c-hot"') < page.index('id="c-qa"')
    assert qa.first_question(" 강남 ") == "강남에 대해 수집된 기사들의 전체적인 요약을 제공해줘."
    assert client.get("/api/info").json() == {"has_db": True, "chat_model": "가짜 모델"}
    r = client.post("/api/senses", json={"query": "강남"}).json()
    assert [s["name"] for s in r["senses"]] == ["방송인 강남", "지역 강남"]
    print("1 ok: 화면 · 의미 후보")

    job = client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 0,
                                            "max_articles": 20}).json()
    for _ in range(100):
        status = client.get(f"/api/jobs/{job['job_id']}").json()
        if status["status"] != "running":
            break
        time.sleep(0.05)
    assert status["status"] == "done", status
    assert calls["collect"] == ("방송인 강남", 30, 20)
    assert waited == [True]                           # 새로 저장한 것이 있으면 준비 완료 전에 한 번 기다린다
    assert status["early"] == 0 and calls["filter"] == 1  # 처음 검색하는 대상: 저장된 기사로 먼저 답하지 않음
    assert run.PREPARE_SETTLE_SECONDS == 3
    assert calls["index"][0] == (False, False)        # 준비 단계: 본문만 (이미지는 준비 완료 뒤)
    assert status["images"] == "running", status      # 준비 완료 직후: 이미지 글자 읽기가 뒤에서 도는 중
    assert status["hot"] == "running" and status["hot_articles"] == [], status   # 많이 보도된 기사도 준비 완료 뒤에
    for _ in range(100):
        later = client.get(f"/api/jobs/{job['job_id']}").json()
        if later["hot"] != "running":
            break
        time.sleep(0.05)
    assert later["hot"] == "done" and calls["hot"] == (19, 30), (later, calls)
    assert calls["hot_terms"] == ["강남"]               # 한 줄 요약은 검색어가 든 문장끼리 비교한다
    assert later["hot_articles"] == [
        {"title": "강남 새 예능 출연", "link": "https://hot/1", "press": "연합뉴스", "pub_date": "2026-11-01",
         "duplicates": 7, "summary": "강남이 새 예능에 나온다."},
        {"title": "강남 귀화 소식", "link": "https://hot/2", "press": "알 수 없음", "pub_date": "",
         "duplicates": 2, "summary": ""}], later["hot_articles"]
    for _ in range(100):
        later = client.get(f"/api/jobs/{job['job_id']}").json()
        if later["images"] != "running":
            break
        time.sleep(0.05)
    assert later["images"] == "done" and calls["index"][1] == (False, True), (later, calls)
    assert any("[이미지] 이미지 속 글자 읽기(OCR) 끝" in line for line in later["logs"]), later["logs"]
    logs = "\n".join(status["logs"])
    assert "[1단계] 수집" in logs and "[4단계] 인덱싱" in logs and "질문에 쓸 수 있는 기사(관련·보류): 4건" in logs, logs
    assert status["logs"][1].startswith("[  0.0초] [1단계] 검색어") and status["elapsed"] > 0   # 줄마다 걸린 시간
    assert f"준비 완료! 이제 질문할 수 있습니다. (총 {status['elapsed']:.1f}초)" in logs, logs
    print("2 ok: 준비 (run.prepare_topic이 log로 진행 상황을 넘김)")

    ans = client.post("/api/ask", json={"session_id": status["session_id"], "question": "강남 근황?"}).json()
    assert calls["search"] == (19, "강남 근황?") and ans["sources"][0]["press"] == "연합뉴스", ans
    assert calls["subject"].startswith("강남 (방송인 강남"), calls["subject"]    # 고른 의미가 답변의 검색 대상으로
    assert calls["period"].startswith("2026-10-14 ~ "), calls["period"]         # 가장 오래된 기사 날짜 ~ 오늘
    assert client.post("/api/ask", json={"session_id": "없음", "question": "q"}).status_code == 404
    assert calls["history"] == qa.NO_HISTORY and calls["search_history"] == ""     # 첫 질문: 이전 대화 없음
    client.post("/api/ask", json={"session_id": status["session_id"], "question": "그 사람 직업은?"})
    assert calls["history"].startswith("사용자: 강남 근황?\n답변: 강남은 예능에 나왔다") and "(기사" not in calls["history"]
    assert calls["search_history"] == calls["history"]                             # 검색(HyDE 맥락)에도 같은 대화
    print("3 ok: 질문 · 답변 · 이전 대화 기억")

    # 4) 스트리밍: 글 조각이 차례로 오고, 마지막(done)은 /api/ask와 같은 내용
    import json
    res = client.post("/api/ask/stream", json={"session_id": status["session_id"], "question": "강남 근황?"})
    assert res.status_code == 200 and res.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in res.text.splitlines() if line.strip()]
    assert [e["text"] for e in events if e["type"] == "token"] == ["강남은 ", "예능에 나왔다 ", "(기사 1)."]
    done = events[-1]
    assert done["type"] == "done" and {k: v for k, v in done.items() if k != "type"} == ans, (done, ans)
    err = [json.loads(x) for x in client.post("/api/ask/stream", json={"session_id": status["session_id"],
                                                                       "question": "터짐"}).text.splitlines()]
    assert err == [{"type": "error", "message": "LLM 오류"}]
    assert client.post("/api/ask/stream", json={"session_id": "없음", "question": "q"}).status_code == 404
    print("4 ok: 답변 스트리밍 (조각 → 완성본, 오류, 없는 세션)")

    # 5) 예전에 준비한 대상: 수집이 끝나기 전에 DB에 저장된 기사로 먼저 질문을 받고, 새로 할 일이 없으면 건너뛴다
    known["topic"] = db.TopicRecord(id=19, selected=None, last_collected_at=None)
    pending["now"] = pipeline.PendingWork(unjudged=0, unindexed=0)
    crawled["n"], waited[:], calls["filter"], calls["index"] = 0, [], 0, []
    job2 = client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 0}).json()
    for _ in range(100):
        early = client.get(f"/api/jobs/{job2['job_id']}").json()
        if early["session_id"] or early["status"] != "running":
            break
        time.sleep(0.02)
    assert early["status"] == "running" and early["session_id"] and early["early"] == 6, early   # 수집 중인데 질문 가능
    ans2 = client.post("/api/ask", json={"session_id": early["session_id"], "question": "근황?"}).json()
    assert calls["search"] == (19, "근황?") and calls["period"].startswith("2026-10-20 ~ "), calls
    for _ in range(100):
        final = client.get(f"/api/jobs/{job2['job_id']}").json()
        if final["status"] != "running":
            break
        time.sleep(0.05)
    assert final["status"] == "done" and final["session_id"] == early["session_id"]          # 같은 세션을 계속 쓴다
    assert calls["filter"] == 0 and calls["index"] == [] and waited == [False], calls        # 판정·인덱싱·대기 건너뜀
    logs2 = "\n".join(final["logs"])
    assert "DB에 저장된 기사 6건으로 지금 바로 질문할 수 있습니다" in logs2 and "새로 판정할 기사 없음" in logs2, logs2
    print("5 ok: 저장된 기사로 먼저 답하기 · 할 일 없는 단계 건너뛰기")
    print("ALL PASS")


if __name__ == "__main__":
    main()
