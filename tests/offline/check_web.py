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
    found = {"hits": []}                               # 검색 결과 (먼저 답하는 중에 근거가 없으면 기다리는지 보려고 바꾼다)
    def fake_search(tid, q, sel, period_days=None, history=""):
        calls["search"], calls["search_history"] = (tid, q), history
        return found["hits"]
    retrieval.search_topic = fake_search
    from datetime import datetime as _dt, timezone as _tz
    retrieval.oldest_searchable_date = lambda tid, days=None: _dt(2026, 10, 14, 3, tzinfo=_tz.utc)
    retrieval.searchable_stats = lambda tid, days=None: (6, _dt(2026, 10, 20, tzinfo=_tz.utc))
    def fake_answer(q, hits, on_token=None, subject="", period="", history=""):
        calls["subject"], calls["period"], calls["history"] = subject, period, history
        if q == "터짐":
            raise RuntimeError("LLM 오류")
        calls["answers"] = calls.get("answers", 0) + 1
        if q == "모르는 것?":                          # 검색은 됐지만 기사에 답할 내용이 없음 → 기사를 인용하지 않은 답
            return Answer(question=q, answer="기사에서 찾지 못했습니다.", sources=[Article(title="기사", link="https://a")],
                          source_kinds=["본문"], cited=[])
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
    # 화면 파일은 쓸 때마다 새 버전인지 확인하게 (예전 app.js를 계속 쓰면 배포한 수정이 안 보인다)
    for path in ("/", "/static/app.js", "/static/style.css", "/static/QuPicT_logo.svg"):
        assert client.get(path).headers.get("cache-control") == "no-cache", path
    assert "cache-control" not in client.get("/api/info").headers
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
    # 처음 찾는 대상(DB에 저장된 기사 없음): 먼저 답하지 않고, 새 기사를 저장한 뒤에 답한다
    assert status["db_articles"] == 0 and status["basis"] is None and calls["filter"] == 1, status
    assert any("DB에 저장된 관련 기사가 없어서, 새 기사를 모아 저장한 뒤에 답합니다." in x for x in status["logs"])
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
    assert later["basis"]["stage"] == "fresh" and later["basis"]["note"].endswith("에 새로 모은 기사까지 참고한 답변입니다."), later
    ocr_end = next(i for i, line in enumerate(later["logs"]) if "[이미지] 이미지 속 글자 읽기(OCR) 끝" in line)
    asked = next(i for i, line in enumerate(later["logs"]) if line.endswith("이제 질문할 수 있습니다."))
    assert ocr_end < asked, later["logs"]                                  # "이제 질문할 수 있습니다"는 OCR 끝 다음 줄
    assert any("OCR)를 뒤에서 진행합니다. 끝나면 질문에 답합니다." in line for line in later["logs"]), later["logs"]
    sid = later["session_id"]
    logs = "\n".join(status["logs"])
    assert "[1단계] 수집" in logs and "[4단계] 인덱싱" in logs and "질문에 쓸 수 있는 기사(관련·보류): 4건" in logs, logs
    # 줄마다 걸린 시간 (DB 확인 줄은 준비와 동시에 돌아서 순서가 바뀔 수 있다)
    assert any(x.startswith("[  0.0초] [1단계] 검색어") for x in status["logs"][1:3]) and status["elapsed"] > 0, status
    assert f"준비 완료! (총 {status['elapsed']:.1f}초) 이미지 속 글자 읽기가 끝나면 질문에 답합니다." in logs, logs
    print("2 ok: 준비 (run.prepare_topic이 log로 진행 상황을 넘김) · 이미지 글자 읽기 뒤에 질문 열림")

    ans = client.post("/api/ask", json={"session_id": sid, "question": "강남 근황?"}).json()
    assert calls["search"] == (19, "강남 근황?") and ans["sources"][0]["press"] == "연합뉴스", ans
    assert ans["basis"] == later["basis"] and "wait" not in ans, ans    # 새 기사까지 들어간 뒤: 근거가 없어도 답한다
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
    assert events[0] == {"type": "basis", **ans["basis"]}, events[0]        # 맨 처음: 어느 때 기사를 참고하는지
    assert [e["text"] for e in events if e["type"] == "token"] == ["강남은 ", "예능에 나왔다 ", "(기사 1)."]
    done = events[-1]
    assert done["type"] == "done" and {k: v for k, v in done.items() if k != "type"} == ans, (done, ans)
    err = [json.loads(x) for x in client.post("/api/ask/stream", json={"session_id": sid,
                                                                       "question": "터짐"}).text.splitlines()]
    assert err == [{"type": "basis", **ans["basis"]}, {"type": "error", "message": "LLM 오류"}]
    assert client.post("/api/ask/stream", json={"session_id": "없음", "question": "q"}).status_code == 404
    print("4 ok: 답변 스트리밍 (조각 → 완성본, 오류, 없는 세션)")

    def wait_job(job_id, until, tries=300):
        for _ in range(tries):
            j = client.get(f"/api/jobs/{job_id}").json()
            if until(j):
                return j
            time.sleep(0.02)
        raise AssertionError(f"기다리던 상태가 오지 않음: {j}")

    # 3번 과정이 모두 끝나 새 기사까지 들어갔다 (또는 실패)
    fresh_or_error = lambda j: (j["basis"] or {}).get("stage") == "fresh" or j["status"] == "error"

    # 5) 예전에 준비해서 DB에 기사가 있는 대상: 준비(수집 · 필터링 · 인덱싱)가 끝나기 전에 그 기사로 먼저 답한다 (교수님 피드백).
    #    답변마다 어느 때 저장된 기사를 참고했는지 함께 보낸다. 먼저 답하는 중에 근거가 DB에 없는 질문은 답하지 않고
    #    새 기사를 저장한 뒤에 답한다. 준비가 끝나면 같은 세션(대화)에서 새 기사까지 참고한다.
    #    새로 할 일이 없으면 판정 · 인덱싱 · 대기는 건너뛴다
    from datetime import datetime as _dt2, timezone as _tz2
    known["topic"] = db.TopicRecord(id=19, selected=None,
                                    last_collected_at=_dt2(2026, 11, 16, 6, 20, tzinfo=_tz2.utc))   # 한국 시간 15:20
    pending["now"] = pipeline.PendingWork(unjudged=0, unindexed=0)
    crawled["n"], waited[:], calls["filter"], calls["index"] = 0, [], 0, []
    gate5 = threading.Event()                      # 수집을 붙잡아 두고 "준비 중"에 질문해 본다
    _collect5 = collect.collect_to_db

    async def held_collect5(*a, **k):
        await asyncio.to_thread(gate5.wait, 5)
        return await _collect5(*a, **k)

    collect.collect_to_db = held_collect5
    job2 = client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 0}).json()
    early = wait_job(job2["job_id"], lambda j: j["session_id"] or j["status"] == "error")
    assert early["status"] == "running" and early["db_articles"] == 6, early     # 수집이 끝나기 전에 질문을 받는다
    assert early["basis"] == {"stage": "db", "note": "11월 16일 15:20까지 수집해 DB에 저장해 둔 기사 6건을 참고한 답변입니다. "
                                                     "새 기사는 아직 모으는 중입니다."}, early["basis"]
    assert any("DB에 저장된 기사 6건 (마지막 수집 11월 16일 15:20)으로 지금 먼저 답합니다." in x for x in early["logs"])
    sid2 = early["session_id"]
    # 근거가 DB에 없는 질문: 답하지 않는다 (LLM을 부르지 않고, 대화 기록에도 남기지 않음)
    calls.pop("subject", None)
    held = client.post("/api/ask", json={"session_id": sid2, "question": "새 소식?"}).json()
    assert held == {"question": "새 소식?", "wait": True, "message": web.WAIT_MESSAGE, "basis": early["basis"]}, held
    assert "subject" not in calls and web._sessions[sid2].turns == []
    held_s = [json.loads(x) for x in client.post("/api/ask/stream", json={"session_id": sid2, "question": "새 소식?"}).text.splitlines()]
    assert held_s == [{"type": "basis", **early["basis"]}, {"type": "wait", "message": web.WAIT_MESSAGE}], held_s
    # 근거가 있으면 저장된 기사로 바로 답한다. 기간은 DB에 있던 기사 기준 (searchable_stats의 가장 오래된 날)
    found["hits"] = ["청크"]
    first = client.post("/api/ask", json={"session_id": sid2, "question": "근황?"}).json()
    assert first["basis"] == early["basis"] and first["answer"].startswith("강남은"), first
    assert calls["search"] == (19, "근황?") and calls["period"].startswith("2026-10-20 ~ "), calls
    # 먼저 답하는 중에는 인용 여부를 본 뒤에 보내므로 글자가 한 번에 온다 (완성된 답변은 같음)
    early_s = [json.loads(x) for x in client.post("/api/ask/stream", json={"session_id": sid2, "question": "근황?"}).text.splitlines()]
    assert [e["type"] for e in early_s] == ["basis", "token", "done"] and early_s[1]["text"] == first["answer"], early_s
    # 검색은 됐지만 답변이 기사를 인용하지 않음 = DB에 답할 내용 없음 → 보여 주지 않고 새 기사 저장 뒤에 답한다
    n_turns = len(web._sessions[sid2].turns)
    unknown = client.post("/api/ask", json={"session_id": sid2, "question": "모르는 것?"}).json()
    assert unknown["wait"] is True and len(web._sessions[sid2].turns) == n_turns, unknown
    found["hits"] = []
    gate5.set()                                   # 준비를 마저 하게 둔다
    final = wait_job(job2["job_id"], fresh_or_error)
    collect.collect_to_db = _collect5
    assert final["status"] == "done" and final["session_id"] == sid2, final       # 같은 세션 (대화가 이어진다)
    assert final["basis"]["stage"] == "fresh" and final["images"] == "done", final  # 이미지 글자 읽기까지 끝난 뒤
    ans2 = client.post("/api/ask", json={"session_id": sid2, "question": "새 소식?"}).json()
    assert "wait" not in ans2 and ans2["basis"] == final["basis"], ans2          # 이제는 근거가 없어도 답한다
    assert calls["search"] == (19, "새 소식?") and calls["period"].startswith("2026-10-14 ~ "), calls   # 새 기간
    assert "사용자: 근황?" in calls["history"] and "모르는 것?" not in calls["history"], calls["history"]   # 먼저 답한 대화를 기억
    after = client.post("/api/ask", json={"session_id": sid2, "question": "모르는 것?"}).json()
    assert "wait" not in after and after["answer"] == "기사에서 찾지 못했습니다.", after     # 새 기사까지 본 뒤에는 그대로 답한다
    # 판정 · 본문 인덱싱 · 대기는 건너뜀. 이미지 글자 읽기(OCR)만 돈다
    assert calls["filter"] == 0 and calls["index"] == [(False, True)] and waited == [False], calls
    logs2 = "\n".join(final["logs"])
    assert "새로 판정할 기사 없음" in logs2 and "이미지 속 글자 읽기가 끝나면 새 기사까지 넣어 답합니다." in logs2, logs2
    assert logs2.endswith("새 기사까지 모두 저장했습니다. 이제부터 답변은 새로 모은 기사까지 참고합니다."), logs2
    print("5 ok: DB에 기사가 있으면 먼저 답하기 (참고 시점 표시 · 근거 없으면 저장 뒤에) · 끝나면 같은 대화에서 새 기사까지")

    # 6) 질문은 Top 5 정리를 기다리지 않는다 (Top 5는 따로 돈다. 여러 사용자가 몰리면 Top 5 요약이 10초 넘게 걸림)
    gate = threading.Event()
    _summarize = hot.summarize
    hot.summarize = lambda items: (gate.wait(5), _summarize(items))[1]   # Top 5가 끝나지 않고 붙잡혀 있게
    job3 = client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 0}).json()
    opened = wait_job(job3["job_id"], fresh_or_error)
    assert opened["session_id"] and opened["hot"] == "running", opened     # Top 5가 아직인데 새 기사 기준으로 답함
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
    done = [wait_job(j["job_id"], fresh_or_error) for j in same]
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
    longer_done = wait_job(longer["job_id"], fresh_or_error)
    assert has(longer_done, "끝나기를 기다립니다") and has(longer_done, "[1단계] 수집"), longer_done["logs"]
    assert active["calls"] == 2, active
    active.update(now=0, max=0)
    other = [client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": i}).json()
             for i in (0, 1)]
    for j in other:
        wait_job(j["job_id"], fresh_or_error)
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

    # 9) 페이지를 나가면(/api/leave) 그 사용자의 작업 · 질문 세션을 서버 메모리에서 지운다. DB 쪽 함수는 부르지 않는다
    sense.load_sense_candidates = lambda q: (cands, "DB 캐시")
    j = client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 0}).json()
    s = wait_job(j["job_id"], lambda x: x["session_id"] or x["status"] == "error")
    assert s["session_id"] in web._sessions and j["job_id"] in web._jobs
    assert client.post("/api/leave", json={"job_id": j["job_id"], "session_id": s["session_id"]}).json() == {"ok": True}
    assert j["job_id"] not in web._jobs and s["session_id"] not in web._sessions
    assert client.get(f"/api/jobs/{j['job_id']}").status_code == 404
    assert client.post("/api/ask", json={"session_id": s["session_id"], "question": "q"}).status_code == 404
    assert client.post("/api/leave", json={"job_id": "없음"}).status_code == 200      # 이미 지운 것 · 없는 것도 괜찮음
    # 준비 중에 나가면: 준비는 끝까지 돌지만(다른 사람이 같이 쓸 수 있음) 질문 세션은 만들지 않는다
    gate2 = threading.Event()
    _collect2 = collect.collect_to_db

    async def held_collect(*a, **k):
        await asyncio.to_thread(gate2.wait, 5)
        return await _collect2(*a, **k)

    collect.collect_to_db = held_collect
    known["topic"] = None                          # DB에 기사가 없는 대상 (있으면 준비 중에 먼저 세션이 열린다)
    before = set(web._sessions)
    j2 = client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 1}).json()
    job_obj = web._jobs[j2["job_id"]]
    client.post("/api/leave", json={"job_id": j2["job_id"]})
    gate2.set()
    for _ in range(300):
        if job_obj.status != "running" and job_obj.images != "running":
            break
        time.sleep(0.02)
    time.sleep(0.1)
    assert job_obj.status == "done" and job_obj.session_id is None, job_obj   # 준비는 끝났지만 세션 없음
    assert set(web._sessions) == before
    collect.collect_to_db = _collect2
    known["topic"] = db.TopicRecord(id=19, selected=None, last_collected_at=None)
    # 신호 없이 사라진 사용자: 1시간 넘게 안 쓴 작업 · 세션은 _sweep이 지운다. 최근에 쓴 것은 남긴다
    j3 = client.post("/api/prepare", json={"query": "강남", "period_days": 30, "sense_index": 0}).json()
    s3 = wait_job(j3["job_id"], lambda x: x["session_id"] or x["status"] == "error")
    client.post("/api/ask", json={"session_id": s3["session_id"], "question": "q"})          # 세션은 방금 씀
    web._sweep(now=time.time() + web.IDLE_SECONDS / 2)
    assert j3["job_id"] in web._jobs and s3["session_id"] in web._sessions                # 30분: 그대로
    web._jobs[j3["job_id"]].last_seen -= web.IDLE_SECONDS                                  # 작업 확인만 오래 멈춤
    web._sweep()
    assert j3["job_id"] not in web._jobs and s3["session_id"] in web._sessions            # 작업만 지우고 대화는 계속
    web._sweep(now=time.time() + web.IDLE_SECONDS + 1)
    assert s3["session_id"] not in web._sessions                                           # 1시간 넘게 질문 없음 → 지움
    print("9 ok: 나가면 작업 · 대화를 서버에서 지움 (준비 중에 나가도), 오래 안 쓴 것은 자동 정리")
    print("ALL PASS")


if __name__ == "__main__":
    main()
