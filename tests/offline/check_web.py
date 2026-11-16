"""
웹 서비스(web.py) — 화면 → 의미 후보 → 준비(백그라운드) → 진행 로그 → 질문 흐름이 run.py 단계 함수와 맞물리는지

단계 함수(수집·필터·인덱싱·검색·답변)는 가짜로 바꾼다. LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0).
실행: python tests/run_offline.py
"""
import asyncio
import sys
import threading
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
    remembered = []
    sense.remember_selection = lambda c, sel: remembered.append(sel.sense.name)   # 실제 DB에 기록하지 않게
    threads = []
    _make_thread = web._remember_thread
    web._remember_thread = lambda fn, *a: threads.append(_make_thread(fn, *a)) or threads[-1]
    web.validate_settings = lambda: None
    web.settings = SimpleNamespace(has_db=True, chat_model="가짜 모델")

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
                               pub_date=_dt(2026, 11, 1, tzinfo=_tz.utc), description="요약문", content="본문", duplicates=7,
                               similar=[hot.SimilarArticle("강남, 예능 또 출연", "https://dup/1", "뉴스1",
                                                           _dt(2026, 11, 2, tzinfo=_tz.utc)),
                                        hot.SimilarArticle("강남 예능 출연 화제", "https://dup/2", None, None)]),
                hot.HotArticle(title="강남 귀화 소식", link="https://hot/2", press=None, pub_date=None,
                               description="", content="강남이 귀화했다. 뒷이야기.", duplicates=2)]

    def fake_summarize(items):
        time.sleep(0.2)                               # LLM 요약은 준비 완료 뒤에 돈다
        items[0].summary = "강남이 새 예능에 나온다."
        return items
    hot.top_duplicated_db, hot.summarize = fake_top, fake_summarize

    client = TestClient(web.app)
    assert client.get("/").status_code == 200
    # 웹 화면은 첫 질문을 고정하지 않는다 (터미널 run.py만 qa.first_question으로 요약부터 답한다)
    page = client.get("/").text
    # 화면은 세 파일: 뼈대(index.html) · 모양(style.css) · 동작(app.js). 둘 다 /static/에서 받을 수 있어야 한다
    assert '/static/style.css' in page and '/static/app.js' in page
    css, js = client.get("/static/style.css"), client.get("/static/app.js")
    assert css.status_code == js.status_code == 200 and ".hot" in css.text and "function poll" in js.text
    assert "enqueueQuestion(`" not in js.text and "전체적인 요약을 제공해줘" not in page + js.text
    assert 'id="c-hot"' in page and page.index('id="c-job"') < page.index('id="c-hot"') < page.index('id="c-qa"')
    assert qa.first_question(" 강남 ") == "강남에 대해 수집된 기사들의 전체적인 요약을 제공해줘."
    assert client.get("/api/info").json() == {"has_db": True, "chat_model": "가짜 모델"}
    r = client.post("/api/senses", json={"query": "강남"}).json()
    assert [s["name"] for s in r["senses"]] == ["방송인 강남", "지역 강남"]
    assert r["senses"][0]["is_custom"] is False and r["senses"][0]["use_count"] == 0
    # 이름으로 고르기: 후보 순서가 바뀌어도(많이 고른 순) 화면에서 본 의미를 고른다
    assert web._select(web.PrepareRequest(query="강남", period_days=30, sense_index=0,
                                          sense_name="지역 강남")).sense.name == "지역 강남"
    try:
        web._select(web.PrepareRequest(query="강남", period_days=30, sense_name="없는 의미"))
        raise AssertionError("없는 이름이면 400")
    except web.HTTPException as e:
        assert e.status_code == 400
    # 기록은 뒤에서: 기록이 느려도 고르기는 바로 끝난다
    for t in threads: t.join(2)
    assert remembered == ["지역 강남"]                # 없는 이름(400)은 기록하지 않음
    slow = threading.Event()
    sense.remember_selection = lambda c, sel: (slow.wait(2), remembered.append(sel.sense.name))
    t0 = time.perf_counter()
    web._select(web.PrepareRequest(query="강남", period_days=30, sense_name="방송인 강남"))
    assert time.perf_counter() - t0 < 0.5 and remembered == ["지역 강남"]   # 기록을 기다리지 않음
    slow.set(); threads[-1].join(2)
    assert remembered == ["지역 강남", "방송인 강남"]
    sense.remember_selection = lambda c, sel: remembered.append(sel.sense.name)
    for t in threads: t.join(2)
    remembered.clear()
    print("1 ok: 화면 · 의미 후보")

    job = client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 0,
                                            "max_articles": 20}).json()
    for _ in range(100):
        status = client.get(f"/api/jobs/{job['job_id']}").json()
        if status["status"] != "running":
            break
        time.sleep(0.05)
    assert status["status"] == "done", status
    for t in threads: t.join(2)
    assert calls["collect"] == ("방송인 강남", 30, 20)
    assert remembered == ["방송인 강남"]               # 고른 의미는 DB에 기록 (고른 횟수 · 직접 입력 저장)
    assert waited == [True]                           # 새로 저장한 것이 있으면 준비 완료 전에 한 번 기다린다
    assert "early" not in status and calls["filter"] == 1   # 저장된 기사로 먼저 답하는 기능은 없앴다
    assert run.PREPARE_SETTLE_SECONDS == 3
    assert calls["index"][0] == (False, False)        # 준비 단계: 본문만 (이미지는 준비 완료 뒤)
    assert status["images"] == "running", status      # 준비 완료 직후: 이미지 글자 읽기가 뒤에서 도는 중
    assert status["session_id"] is None, status        # 이미지 글자 읽기까지 끝나야 질문을 받는다 (아직 답하지 않음)
    assert status["hot"] == "running" and status["hot_articles"] == [], status   # 많이 보도된 기사도 준비 완료 뒤에
    for _ in range(100):
        later = client.get(f"/api/jobs/{job['job_id']}").json()
        if later["hot"] != "running":
            break
        time.sleep(0.05)
    assert later["hot"] == "done" and calls["hot"] == (19, 30), (later, calls)
    assert later["hot_articles"] == [
        {"title": "강남 새 예능 출연", "link": "https://hot/1", "press": "연합뉴스", "pub_date": "2026-11-01",
         "duplicates": 7, "summary": "강남이 새 예능에 나온다.",
         "similar": [{"title": "강남, 예능 또 출연", "link": "https://dup/1", "press": "뉴스1", "pub_date": "2026-11-02"},
                     {"title": "강남 예능 출연 화제", "link": "https://dup/2", "press": "알 수 없음", "pub_date": ""}]},
        {"title": "강남 귀화 소식", "link": "https://hot/2", "press": "알 수 없음", "pub_date": "",
         "duplicates": 2, "summary": "", "similar": []}], later["hot_articles"]
    for _ in range(100):
        later = client.get(f"/api/jobs/{job['job_id']}").json()
        if later["images"] != "running":
            break
        time.sleep(0.05)
    assert later["images"] == "done" and calls["index"][1] == (False, True), (later, calls)
    assert later["session_id"], later                                      # 이미지까지 끝난 뒤에 세션이 열린다
    ocr_end = next(i for i, line in enumerate(later["logs"]) if "[이미지] 이미지 속 글자 읽기(OCR) 끝" in line)
    asked = next(i for i, line in enumerate(later["logs"]) if line.endswith("이제 질문할 수 있습니다."))
    assert ocr_end < asked, later["logs"]                                  # "이제 질문할 수 있습니다"는 OCR 끝 다음 줄
    assert any("OCR)를 뒤에서 진행합니다. 끝나면 질문에 답합니다." in line for line in later["logs"]), later["logs"]
    sid = later["session_id"]
    logs = "\n".join(status["logs"])
    assert "[1단계] 수집" in logs and "[4단계] 인덱싱" in logs and "질문에 쓸 수 있는 기사(관련·보류): 4건" in logs, logs
    assert status["logs"][1].startswith("[  0.0초] [1단계] 검색어") and status["elapsed"] > 0   # 줄마다 걸린 시간
    assert f"준비 완료! (총 {status['elapsed']:.1f}초) 이미지 속 글자 읽기가 끝나면 질문에 답합니다." in logs, logs
    print("2 ok: 준비 (run.prepare_topic이 log로 진행 상황을 넘김) · 이미지 글자 읽기 뒤에 질문 열림")

    ans = client.post("/api/ask", json={"session_id": sid, "question": "강남 근황?"}).json()
    assert calls["search"] == (19, "강남 근황?") and ans["sources"][0]["press"] == "연합뉴스", ans
    assert calls["subject"].startswith("강남 (방송인 강남"), calls["subject"]    # 고른 의미가 답변의 검색 대상으로
    assert calls["period"].startswith("2026-10-14 ~ "), calls["period"]         # 가장 오래된 기사 날짜 ~ 오늘
    assert client.post("/api/ask", json={"session_id": "없음", "question": "q"}).status_code == 404
    assert calls["history"] == qa.NO_HISTORY and calls["search_history"] == ""     # 첫 질문: 이전 대화 없음
    client.post("/api/ask", json={"session_id": sid, "question": "그 사람 직업은?"})
    assert calls["history"].startswith("사용자: 강남 근황?\n답변: 강남은 예능에 나왔다") and "(기사" not in calls["history"]
    assert calls["search_history"] == calls["history"]                             # 검색(HyDE 맥락)에도 같은 대화
    print("3 ok: 질문 · 답변 · 이전 대화 기억")

    # 4) 스트리밍: 글 조각이 차례로 오고, 마지막(done)은 /api/ask와 같은 내용
    import json
    res = client.post("/api/ask/stream", json={"session_id": sid, "question": "강남 근황?"})
    assert res.status_code == 200 and res.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in res.text.splitlines() if line.strip()]
    assert [e["text"] for e in events if e["type"] == "token"] == ["강남은 ", "예능에 나왔다 ", "(기사 1)."]
    done = events[-1]
    assert done["type"] == "done" and {k: v for k, v in done.items() if k != "type"} == ans, (done, ans)
    err = [json.loads(x) for x in client.post("/api/ask/stream", json={"session_id": sid,
                                                                       "question": "터짐"}).text.splitlines()]
    assert err == [{"type": "error", "message": "LLM 오류"}]
    assert client.post("/api/ask/stream", json={"session_id": "없음", "question": "q"}).status_code == 404
    print("4 ok: 답변 스트리밍 (조각 → 완성본, 오류, 없는 세션)")

    # 5) 예전에 준비한 대상이어도: 준비(수집 · 필터링 · 인덱싱)가 끝나기 전에는 질문을 받지 않는다.
    #    새로 할 일이 없으면 판정 · 인덱싱 · 대기는 건너뛴다
    known["topic"] = db.TopicRecord(id=19, selected=None, last_collected_at=None)
    pending["now"] = pipeline.PendingWork(unjudged=0, unindexed=0)
    crawled["n"], waited[:], calls["filter"], calls["index"] = 0, [], 0, []
    job2 = client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 0}).json()
    seen_before = []                  # 이미지 글자 읽기까지 끝나기 전에 본 (상태, 세션)
    for _ in range(200):
        final = client.get(f"/api/jobs/{job2['job_id']}").json()
        if final["status"] == "error" or final["images"] == "done":
            break
        seen_before.append((final["status"], final["session_id"]))
        time.sleep(0.02)
    assert all(sid_ is None for _, sid_ in seen_before), seen_before   # 준비 중 · 이미지 읽는 중에는 세션이 없다
    assert ("done", None) in seen_before, seen_before                   # 준비 완료 뒤에도 이미지가 끝날 때까지 기다림
    assert final["status"] == "done" and final["session_id"], final     # 모두 끝나야 질문을 받는다
    ans2 = client.post("/api/ask", json={"session_id": final["session_id"], "question": "근황?"}).json()
    assert calls["search"] == (19, "근황?") and calls["period"].startswith("2026-10-14 ~ "), calls
    # 판정 · 본문 인덱싱 · 대기는 건너뜀. 이미지 글자 읽기(OCR)만 돈다 (질문은 그게 끝난 뒤에 열리므로 기록에 남는다)
    assert calls["filter"] == 0 and calls["index"] == [(False, True)] and waited == [False], calls
    logs2 = "\n".join(final["logs"])
    assert "DB에 저장된 기사" not in logs2 and "새로 판정할 기사 없음" in logs2, logs2
    print("5 ok: 예전에 준비한 대상도 3번 과정(이미지까지)이 끝난 뒤에 답하기 · 할 일 없는 단계 건너뛰기")

    def wait_job(job_id, until, tries=300):
        for _ in range(tries):
            j = client.get(f"/api/jobs/{job_id}").json()
            if until(j):
                return j
            time.sleep(0.02)
        raise AssertionError(f"기다리던 상태가 오지 않음: {j}")

    # 6) 질문은 Top 5 정리를 기다리지 않는다 (Top 5는 따로 돈다. 여러 사용자가 몰리면 Top 5 요약이 10초 넘게 걸림)
    gate = threading.Event()
    _summarize = hot.summarize
    hot.summarize = lambda items: (gate.wait(5), _summarize(items))[1]   # Top 5가 끝나지 않고 붙잡혀 있게
    job3 = client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 0}).json()
    opened = wait_job(job3["job_id"], lambda j: j["session_id"] or j["status"] == "error")
    assert opened["session_id"] and opened["hot"] == "running", opened     # Top 5가 아직인데 질문은 열림
    gate.set()
    assert wait_job(job3["job_id"], lambda j: j["hot"] != "running")["hot"] == "done"
    hot.summarize = _summarize
    print("6 ok: 질문은 Top 5 정리를 기다리지 않음")

    # 7) 같은 검색어 · 의미를 사용자 셋이 동시에 준비하면: 한 명만 수집 · 판정 · 인덱싱하고,
    #    나머지는 기다렸다가 그 결과를 그대로 쓴다 (다시 수집하지 않음). 의미가 다르면(다른 topic) 서로 기다리지 않고 동시에 돈다
    active = {"now": 0, "max": 0, "calls": 0}
    _collect = collect.collect_to_db

    async def counting_collect(*a, **k):
        active["now"] += 1
        active["calls"] += 1
        active["max"] = max(active["max"], active["now"])
        try:
            return await _collect(*a, **k)
        finally:
            active["now"] -= 1

    collect.collect_to_db = counting_collect
    same = [client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 0}).json()
            for _ in range(3)]
    done = [wait_job(j["job_id"], lambda s: s["session_id"] or s["status"] == "error") for j in same]
    assert all(d["status"] == "done" for d in done), done
    assert active["calls"] == 1 and active["max"] == 1, active              # 수집은 한 번뿐
    has = lambda d, text: any(text in line for line in d["logs"])
    reused = [d for d in done if has(d, "다른 사용자가 방금 준비를 마쳐서, 그 결과를 그대로 씁니다")]
    assert len(reused) == 2 and all(has(d, "끝나기를 기다립니다") for d in reused), [d["logs"] for d in done]
    assert all(not has(d, "[1단계] 수집") for d in reused)                  # 기다린 두 명은 수집 단계가 없음
    sids = {d["session_id"] for d in done}
    assert len(sids) == 3                                                   # 질문 세션(대화 기록)은 사람마다 따로
    # 짧은 기간 · 적은 기사 수로 준비한 것을 기다렸으면 그대로 쓰지 않고 직접 준비한다
    web._recent_prepared[web._topic_key(SimpleNamespace(query="강남", sense=SimpleNamespace(name="방송인 강남")))] = \
        web._Prepared(topic_id=19, period_days=7, max_articles=20)
    lock = web._topic_lock(SimpleNamespace(query="강남", sense=SimpleNamespace(name="방송인 강남")))
    lock.acquire()                                                          # 누가 준비 중인 것처럼
    longer = client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 0}).json()
    time.sleep(0.1)
    lock.release()
    longer_done = wait_job(longer["job_id"], lambda s: s["session_id"] or s["status"] == "error")
    assert has(longer_done, "끝나기를 기다립니다") and has(longer_done, "[1단계] 수집"), longer_done["logs"]
    assert active["calls"] == 2, active
    active.update(now=0, max=0)
    other = [client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": i}).json()
             for i in (0, 1)]
    for j in other:
        wait_job(j["job_id"], lambda s: s["session_id"] or s["status"] == "error")
    assert active["max"] == 2, active                                       # 다른 의미는 동시에 돈다
    collect.collect_to_db = _collect
    print("7 ok: 같은 검색어 · 의미는 한 명만 준비하고 나머지는 결과를 같이 씀, 다른 의미는 동시에")

    # 8) 처음 보는 검색어를 셋이 동시에 찾아도 의미 후보는 한 번만 만든다 (LLM은 부를 때마다 이름이 조금씩 달라서,
    #    따로 만들면 같은 대상이 다른 이름으로 DB에 따로 저장 · 준비됐다)
    made = []

    def slow_new_candidates(q):
        made.append(q)
        time.sleep(0.3)                                  # LLM이 후보를 만드는 동안
        return SenseCandidates(query=q, senses=[Sense(name=f"기업 MS {len(made)}", definition="d")]), "LLM → DB 저장"

    sense.load_sense_candidates = slow_new_candidates
    got = []
    users = [threading.Thread(target=lambda: got.append(client.post("/api/senses", json={"query": "MS"}).json()))
             for _ in range(3)]
    for t in users: t.start()
    for t in users: t.join()
    assert made == ["MS"], made                                             # 후보는 한 번만 만듦
    assert {g["senses"][0]["name"] for g in got} == {"기업 MS 1"}, got      # 셋 다 같은 이름
    assert sorted(g["source"] for g in got).count("다른 사용자가 방금 만든 후보") == 2, got
    client.post("/api/senses", json={"query": "MS"})                        # 나중에(기다리지 않고) 찾으면 평소처럼 다시 읽는다
    assert len(made) == 2
    print("8 ok: 처음 보는 검색어를 동시에 찾아도 의미 후보는 한 번만 만든다")
    print("ALL PASS")


if __name__ == "__main__":
    main()
