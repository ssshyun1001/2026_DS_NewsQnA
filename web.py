"""
QuPicT 웹 서비스 — run.py의 질문 서비스를 브라우저 화면으로 쓴다

    python web.py                 http://127.0.0.1:8000 에서 실행
    python web.py --port 8080     포트 바꾸기
    python web.py --host 0.0.0.0  같은 네트워크의 다른 컴퓨터에서도 접속 (팀원 시연용)

화면 흐름 (run.py와 같은 단계, 실제 일은 qupict/ 단계 파일이 한다)
    1) 검색어 + 기간 입력          → POST /api/senses        [0단계] 의미 후보 (DB 캐시 또는 LLM)
    2) 의미 고르기 (또는 직접 입력) → POST /api/prepare       [1~4단계] 수집·필터·인덱싱 (백그라운드 작업)
    3) 진행 상황 보기               → GET  /api/jobs/{id}     단계별 로그를 몇 초마다 확인
       많이 보도된 기사 Top 5       → (같은 응답의 hot_articles) 준비 완료 뒤, 중복이 많이 묶인 기사 한 줄 요약 · 링크
    4) 질문하기                     → POST /api/ask/stream    [5단계] 검색 · 답변 (만들어지는 대로 한 줄씩 보냄)
                                       POST /api/ask           같은 답변을 다 만든 뒤 한 번에 (예전 방식)

준비하는 동안에도 질문 칸이 열려 있다. 미리 적은 질문은 화면이 기억해 두었다가 준비(이미지 글자 읽기까지)가 끝나면 차례로 보낸다.

여러 사용자가 동시에 쓸 때
    - 준비 작업은 사용자마다 따로 백그라운드 스레드에서 돈다.
    - 같은 검색어 · 의미는 한 번에 한 작업만 준비한다 (_topic_lock). 두 번째 사용자는 기다렸다가 저장된 결과를 그대로 쓴다.
    - Top 5 정리는 따로 돌아서 질문을 막지 않는다. Kiwi(형태소 분석기)는 서버를 켤 때 한 번만 만든다.
    - LLM · 임베딩 요청은 서버 전체 동시 상한(qupict.llm.llm_slot) 안에서 보낸다.

세션(준비한 뉴스)은 서버 메모리에만 있다 → 서버를 끄면 사라진다. 사용자가 페이지를 나가거나(/api/leave)
1시간 안 쓰면(_sweep) 그 사용자의 작업 · 대화 기록도 지운다. DB 버전은 DB에 쌓인 것을 다시 쓰므로
같은 검색어를 다시 준비하면 이미 한 일은 건너뛰어 빠르다. 세션은 앞서 한 질문·답변을 기억해서
"그 사람 직업은?"처럼 앞 답변을 가리키는 질문도 알아듣는다 (qa.history_text).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qupict.cli import setup_console
from qupict.config import settings, validate_settings
from qupict.schemas import Answer, SelectedSense, SenseCandidates

setup_console()

WEB_DIR = Path(__file__).parent / "web"     # 화면 파일: index.html(뼈대) · style.css(모양) · app.js(동작)

app = FastAPI(title="QuPicT 뉴스 Q&A")
# web/ 폴더의 파일을 /static/… 주소로 내보낸다 (index.html이 /static/style.css · /static/app.js를 불러 쓴다)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.middleware("http")
async def _always_check_for_new_screen(request, call_next):
    """화면 파일(/, /static/…)은 브라우저가 쓸 때마다 새 버전이 있는지 서버에 물어보게 한다 (Cache-Control: no-cache).
    없으면 브라우저가 예전 app.js · style.css · 로고를 한동안 그대로 써서, 배포한 수정이 바로 보이지 않았다.
    바뀐 게 없으면 서버는 "그대로"(304)만 답하므로 파일을 다시 받지 않는다 → 느려지지 않는다."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ---------------------------------------------------------------------------
# 서버 메모리 상태: 의미 후보 · 준비 작업 · 질문 세션
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """[1~4단계] 준비 작업 하나 (백그라운드 스레드에서 돈다)."""
    label: str
    status: str = "running"            # running / done / error
    logs: list[str] = field(default_factory=list)
    session_id: Optional[str] = None
    error: str = ""
    images: str = ""                   # 준비 완료 뒤 이미지 → 글(OCR·VLM): "" 안 함 / running / done
    hot: str = ""                      # 준비 완료 뒤 많이 보도된 기사 Top 5: "" 아직 / running / done / error
    hot_articles: list[dict] = field(default_factory=list)   # 중복이 많이 묶인 대표 기사 (한 줄 요약 · 링크)
    elapsed: float = 0.0               # 준비에 걸린 시간(초). 준비 완료 때 채운다
    last_seen: float = field(default_factory=time.time)      # 화면이 마지막으로 확인한 시각 (오래되면 지운다, _sweep)
    left: bool = False                 # 사용자가 페이지를 나갔다 → 준비가 끝나도 질문 세션을 만들지 않는다


@dataclass
class Session:
    """질문에 답할 수 있는 뉴스. 질문이 오면 ask(질문, on_token)으로 답한다.
    answer_fn(질문, on_token, 이전 대화 글): on_token을 주면 답변 글 조각이 만들어질 때마다 불린다 (스트리밍).
    turns: 이 세션에서 한 질문·답변. 다음 질문의 [이전 대화]로 넘겨 "그 사람"처럼 앞 답변을 가리키는 질문도 알아듣게 한다."""
    label: str
    mode: str                          # "DB" / "메모리"
    answer_fn: Callable[[str, Optional[Callable[[str], None]], str], Answer]
    turns: list[tuple[str, str]] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)      # 마지막으로 질문한 시각 (오래되면 지운다, _sweep)

    def ask(self, question: str, on_token: Optional[Callable[[str], None]] = None) -> Answer:
        from qupict.qa import history_text

        with _lock:
            self.last_used = time.time()
            history = history_text(self.turns)
        answer = self.answer_fn(question, on_token, history)
        with _lock:
            self.turns.append((question, answer.answer))
        return answer


_candidates: dict[str, SenseCandidates] = {}   # 검색어 → 의미 후보 (선택할 때 다시 쓴다)
_jobs: dict[str, Job] = {}
_sessions: dict[str, Session] = {}
_lock = threading.Lock()
# (검색어, 의미 이름) → 잠금. 같은 대상(DB의 같은 topic)은 한 번에 한 작업만 수집 · 판정 · 인덱싱한다.
# 두 사용자가 동시에 같은 검색어 · 의미를 준비하면 두 번째는 첫 번째가 끝나기를 기다렸다가 이어서 하고,
# 그때는 이미 DB에 저장된 기사 · 판정 · 청크를 그대로 써서 거의 바로 끝난다 (같은 기사를 두 번 판정 · 임베딩하지 않음).
_topic_locks: dict[tuple[str, str], threading.Lock] = {}


@dataclass
class _Prepared:
    """같은 검색어 · 의미를 마지막으로 준비한 결과. 그 준비를 기다린 사용자가 다시 수집하지 않고 그대로 쓴다."""
    topic_id: int
    period_days: int
    max_articles: int


_recent_prepared: dict[tuple[str, str], _Prepared] = {}


def _topic_key(selected: SelectedSense) -> tuple[str, str]:
    return (selected.query, selected.sense.name)


def _topic_lock(selected: SelectedSense) -> threading.Lock:
    with _lock:
        return _topic_locks.setdefault(_topic_key(selected), threading.Lock())


_query_locks: dict[str, threading.Lock] = {}     # 검색어 → 잠금 (의미 후보를 한 번에 한 명만 만들게)


def _query_lock(query: str) -> threading.Lock:
    with _lock:
        return _query_locks.setdefault(query, threading.Lock())


# ---------------------------------------------------------------------------
# 사용자가 나가면 서버 메모리에서 지운다
#   작업(진행 로그 · Top 5)과 질문 세션(대화 기록)은 사용자마다 서버 메모리에 있다. 지우지 않으면 사용자가 늘수록 쌓인다.
#   1) 페이지를 나가거나 새로 검색하면 화면(app.js)이 /api/leave를 보낸다 → 그 사용자의 작업 · 세션을 바로 지운다
#   2) 그 신호가 못 오는 경우(브라우저 강제 종료 · 휴대폰 앱 전환 등)를 위해, 오래 안 쓴 것은 _sweep이 지운다
#   DB에 모은 기사 · 판정 · 청크는 지우지 않는다 (다음 사람이 같은 검색어를 준비할 때 다시 써서 빠르다)
# ---------------------------------------------------------------------------

IDLE_SECONDS = 60 * 60      # 이만큼(1시간) 화면이 확인하지 않은 작업 · 질문하지 않은 세션은 지운다


def _forget(job_id: str = "", session_id: str = "") -> None:
    """작업과 그 작업의 질문 세션을 서버 메모리에서 지운다. 준비가 아직 돌고 있으면 끝나도 세션을 만들지 않는다
    (준비 자체는 끝까지 돈다: DB에 저장되는 결과는 같은 검색어를 찾는 다른 사람도 쓰고, 기다리는 사람이 있을 수 있다)."""
    with _lock:
        job = _jobs.pop(job_id, None) if job_id else None
        if job is not None:
            job.left = True
            if job.session_id:
                _sessions.pop(job.session_id, None)
        if session_id:
            _sessions.pop(session_id, None)


def _sweep(now: Optional[float] = None) -> None:
    """오래 안 쓴 작업 · 세션을 지운다 (새 요청이 올 때 같이 부른다. 따로 도는 스레드 없음)."""
    now = time.time() if now is None else now
    with _lock:
        # 작업과 세션은 따로 본다: 준비가 끝나면 화면이 작업 확인을 멈추지만, 사용자는 세션으로 계속 질문할 수 있다
        for jid in [jid for jid, j in _jobs.items() if now - j.last_seen > IDLE_SECONDS]:
            _jobs.pop(jid).left = True
        for sid in [sid for sid, s in _sessions.items() if now - s.last_used > IDLE_SECONDS]:
            _sessions.pop(sid)


# ---------------------------------------------------------------------------
# 요청 · 응답 형식
# ---------------------------------------------------------------------------

class SensesRequest(BaseModel):
    query: str = Field(min_length=1)


class PrepareRequest(BaseModel):
    query: str = Field(min_length=1)
    period_days: int = Field(ge=1, le=365)
    sense_index: Optional[int] = None       # 후보 번호 (0부터). 없으면 custom_description을 쓴다
    sense_name: str = ""                    # 후보 이름 (있으면 번호 대신 이것으로 찾는다: 후보 순서는 고른 횟수에 따라 바뀔 수 있음)
    custom_description: str = ""            # "기타 (직접 입력)" 설명
    max_articles: int = Field(default=100, ge=1, le=500)
    memory: bool = False                    # DB 없이 메모리(FAISS) 버전


class LeaveRequest(BaseModel):
    job_id: str = ""
    session_id: str = ""


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/info")
def info() -> dict:
    return {"has_db": settings.has_db, "chat_model": settings.chat_model}


# ---------------------------------------------------------------------------
# [0단계] 의미 후보
# ---------------------------------------------------------------------------

@app.post("/api/senses")
def senses(req: SensesRequest) -> dict:
    from qupict.sense import load_sense_candidates

    query = req.query.strip()
    # 같은 검색어의 의미 후보는 한 번에 한 명만 만든다 (_query_lock).
    # 처음 보는 검색어를 여러 명이 동시에 찾으면 각자 LLM에 물어 이름이 조금씩 다른 후보("기업 마이크로소프트" ·
    # "기업 Microsoft")가 생기고, 같은 대상이 DB에 따로 저장 · 준비됐다. 기다린 사람은 방금 만든 후보를 그대로 쓴다.
    query_lock = _query_lock(query)
    waited = not query_lock.acquire(blocking=False)
    if waited:
        query_lock.acquire()
    try:
        with _lock:
            just_made = _candidates.get(query) if waited else None
        if just_made is not None:
            candidates, source = just_made, "다른 사용자가 방금 만든 후보"
        else:
            candidates, source = load_sense_candidates(query)
            with _lock:
                _candidates[query] = candidates
    finally:
        query_lock.release()
    return {
        "query": query,
        "source": source,
        "senses": [{"name": s.name, "definition": s.definition,
                    "aliases": s.aliases, "related_terms": s.related_terms,
                    "is_custom": s.is_custom, "use_count": candidates.use_counts.get(s.name, 0)}
                   for s in candidates.senses],
    }


def _select(req: PrepareRequest) -> SelectedSense:
    """고른 의미를 만들고 DB에 기록한다 (고른 횟수, 새 직접 입력 의미 저장 → 다음부터 후보로 나온다).
    기록은 뒤에서 한다: 기록이 끝나기를 기다리지 않고 바로 준비 작업을 시작한다 (DB 왕복 약 0.1~0.2초 절약).
    기록이 실패해도 경고만 출력하고 작업에는 영향이 없다."""
    from qupict import sense

    query = req.query.strip()
    with _lock:
        candidates = _candidates.get(query)
    if candidates is None:                  # 서버를 다시 켠 경우 등: 후보를 다시 가져온다
        candidates, _ = sense.load_sense_candidates(query)
    selected = _choose(req, candidates)
    _remember_thread(sense.remember_selection, candidates, selected).start()
    return selected


def _remember_thread(fn, *args) -> threading.Thread:
    return threading.Thread(target=fn, args=args, daemon=True)


def _choose(req: PrepareRequest, candidates) -> SelectedSense:
    from qupict.sense import select_described_sense, select_sense

    if req.sense_name:
        names = [s.name for s in candidates.senses]
        if req.sense_name not in names:
            raise HTTPException(400, "의미 후보가 바뀌었습니다. 검색어를 다시 입력해 주세요.")
        return select_sense(candidates, names.index(req.sense_name))
    if req.sense_index is not None:
        if not 0 <= req.sense_index < len(candidates.senses):
            raise HTTPException(400, "의미 후보 번호가 맞지 않습니다. 검색어를 다시 입력해 주세요.")
        return select_sense(candidates, req.sense_index)
    desc = req.custom_description.strip()
    if not desc:
        raise HTTPException(400, "의미를 고르거나, 찾고 싶은 대상을 직접 설명해 주세요.")
    return select_described_sense(candidates, desc)


# ---------------------------------------------------------------------------
# [1~4단계] 수집 · 필터 · 인덱싱 (백그라운드)
# ---------------------------------------------------------------------------

def _run_job(job_id: str, selected: SelectedSense, req: PrepareRequest, memory: bool) -> None:
    from qupict import hot

    job = _jobs[job_id]
    start = time.perf_counter()

    def log(line: str) -> None:        # 줄마다 시작부터 걸린 시간을 붙인다 → 화면에서 단계별 속도가 보인다
        job.logs.append(f"[{time.perf_counter() - start:5.1f}초] {line}")

    def open_session(answer_fn) -> None:
        """질문을 받기 시작한다. 3번 카드의 과정(수집 · 필터링 · 인덱싱 · 이미지 글자 읽기)이 모두 끝난 뒤에 한 번만 부른다.
        많이 보도된 기사 Top 5는 기다리지 않는다 (따로 돈다).
        그 전에 화면에서 적은 질문은 app.js가 기다렸다가 이 세션이 열리면 차례로 보낸다."""
        session_id = uuid.uuid4().hex
        with _lock:
            if job.left:                   # 준비하는 동안 사용자가 나갔다 → 세션을 만들지 않는다 (_forget)
                return
            _sessions[session_id] = Session(job.label, "메모리" if memory else "DB", answer_fn)
            job.session_id = session_id

    try:
        if memory:
            from qupict.qa import answer_period, answer_subject, generate_answer, oldest_date
            from qupict.retrieval import retrieve_relevant_articles
            from run import collect_news

            news = asyncio.run(collect_news(selected, req.period_days, req.max_articles, log=log))
            if news is None:
                raise RuntimeError("질문에 쓸 기사가 없습니다. 다른 검색어·의미·기간으로 다시 시도해 주세요.")
            oldest = oldest_date(news.articles)
            find_hot = lambda: hot.top_duplicated(news.articles, news.duplicates, pool=news.pool)

            def answer_fn(question: str, on_token=None, history: str = "") -> Answer:
                relevant = retrieve_relevant_articles(news.store, news.articles, question, topic=news.topic)
                return generate_answer(question, relevant, on_token=on_token, subject=answer_subject(selected),
                                       period=answer_period(oldest), history=history)
        else:
            from qupict.qa import NO_HISTORY, answer_period, answer_subject, generate_answer_from_chunks
            from qupict.retrieval import search_topic, warm_up_search
            from run import oldest_searchable_or_none, prepare_topic, wait_after_prepare

            # 수집 · 필터링 · 인덱싱을 모두 끝낸 뒤에야 질문을 받는다 (예전에 준비한 검색어여도 먼저 답하지 않는다)
            prepared = asyncio.run(prepare_topic(selected, req.period_days, req.max_articles, log=log))
            topic_id = prepared.topic_id
            # "준비 완료" 전에 잠깐 기다린다 (run.PREPARE_SETTLE_SECONDS). 새로 저장한 것이 없으면 기다리지 않는다
            wait_after_prepare(log=log, wrote=prepared.wrote)
            # 하이브리드 검색의 키워드(BM25) 인덱스를 미리 만든다 (첫 질문이 인덱스 만드는 시간만큼 늦어지지 않게)
            threading.Thread(target=warm_up_search, args=(topic_id,), daemon=True).start()
            # 수집 · 필터링 · 인덱싱을 모두 끝낸 뒤에야 질문을 받는다 (예전에 준비한 검색어여도 먼저 답하지 않는다).
            # 같은 검색어 · 의미를 다른 사용자가 준비하는 중이면 (_topic_lock):
            #   끝나기를 기다렸다가 그 결과를 그대로 쓴다 → 같은 기사를 다시 수집 · 판정 · 임베딩하지 않고, 거의 같이 끝난다.
            #   (다시 수집하면 그새 네이버 결과가 조금 바뀌어 몇 건을 또 판정하느라 10초 넘게 더 걸렸다. 측정: 3명 동시 25 · 38 · 40초)
            #   단, 그 준비가 이번 요청보다 짧은 기간 · 적은 기사 수였으면 이어서 직접 준비한다.
            key = _topic_key(selected)
            topic_lock = _topic_lock(selected)
            waited = not topic_lock.acquire(blocking=False)
            if waited:
                log("같은 검색어 · 의미를 다른 사용자가 준비하고 있어서, 끝나기를 기다립니다.")
                topic_lock.acquire()
            try:
                recent = _recent_prepared.get(key) if waited else None
                if recent and recent.period_days >= req.period_days and recent.max_articles >= req.max_articles:
                    topic_id = recent.topic_id
                    log("다른 사용자가 방금 준비를 마쳐서, 그 결과를 그대로 씁니다 (수집 · 판정 · 인덱싱 건너뜀).")
                else:
                    prepared = asyncio.run(prepare_topic(selected, req.period_days, req.max_articles, log=log))
                    # "준비 완료" 전에 잠깐 기다린다 (run.PREPARE_SETTLE_SECONDS). 새로 저장한 것이 없으면 기다리지 않는다
                    wait_after_prepare(log=log, wrote=prepared.wrote)
                    topic_id = prepared.topic_id
                    _recent_prepared[key] = _Prepared(topic_id, req.period_days, req.max_articles)
            finally:
                topic_lock.release()
            # 하이브리드 검색의 키워드(BM25) 인덱스를 미리 만든다 (첫 질문이 인덱스 만드는 시간만큼 늦어지지 않게).
            # 다른 사용자의 준비 결과를 그대로 쓴 경우에도 부른다 (bm25.topic_index가 같은 topic의 인덱스를 다시 쓴다)
            threading.Thread(target=warm_up_search, args=(topic_id,), daemon=True).start()
            oldest = oldest_searchable_or_none(topic_id, req.period_days)    # 답변에 적는 기사 기간의 시작
            find_hot = lambda: hot.top_duplicated_db(topic_id, req.period_days)

            def answer_fn(question: str, on_token=None, history: str = NO_HISTORY) -> Answer:
                hits = search_topic(topic_id, question, selected, period_days=req.period_days,
                                    history="" if history == NO_HISTORY else history)
                return generate_answer_from_chunks(question, hits, on_token=on_token, subject=answer_subject(selected),
                                                   period=answer_period(oldest), history=history)

        if not memory:
            from run import image_text_mode
            job.images = "running" if image_text_mode() else ""    # done보다 먼저: 화면이 계속 확인하게
        job.hot = "running"                                        # 이것도 done보다 먼저 (같은 이유)
        job.elapsed = time.perf_counter() - start
        job.status = "done"                                        # 수집 · 필터링 · 인덱싱 끝 (질문은 아직)
        log(f"준비 완료! (총 {job.elapsed:.1f}초)"
            + (" 이미지 속 글자 읽기가 끝나면 질문에 답합니다." if job.images == "running" else ""))
    except Exception as e:
        traceback.print_exc()
        job.error = str(e) or type(e).__name__
        job.status = "error"
        log(f"[오류] {job.error}")
        return

    # 준비 완료 뒤: 많이 보도된 기사 Top 5 (중복이 많이 묶인 대표 기사 + 한 줄 요약. LLM 없이 묶음이 공통으로 쓴 문장을 고른다).
    # 다른 스레드에서 따로 돌린다 → 질문은 Top 5를 기다리지 않는다.
    # (Top 5 요약은 형태소 분석이라 CPU를 많이 써서, 여러 사용자가 몰리면 10초 넘게 걸린다. 측정: 3명 동시 약 12초)
    # 실패해도 질문 답변에는 영향이 없다.
    def make_hot() -> None:
        try:
            job.hot_articles = [_hot_payload(h) for h in hot.summarize(find_hot())]
            job.hot = "done"
        except Exception as e:
            traceback.print_exc()
            log(f"[많이 보도된 기사] 실패해서 건너뜁니다 ({type(e).__name__}: {e})")
            job.hot = "error"

    threading.Thread(target=make_hot, daemon=True).start()

    # 준비 완료 뒤: 이미지 → 글(OCR·VLM)을 이 스레드에서 이어서 한다.
    # 3번 카드의 과정(이미지 글자 읽기까지)이 모두 끝나야 질문에 답한다 → 첫 답변부터 이미지 속 내용도 검색된다.
    # 실패해도 질문은 받는다 (이미지 없이 답함).
    try:
        if job.images == "running":
            from run import add_image_text
            try:
                asyncio.run(add_image_text(topic_id, log=log, note="끝나면 질문에 답합니다."))
            except Exception as e:
                traceback.print_exc()
                log(f"[이미지] 실패해서 건너뜁니다 ({type(e).__name__}: {e})")
    finally:
        open_session(answer_fn)            # images를 done으로 바꾸기 전에: 화면이 확인을 멈추기 전에 세션이 보이게
        log("이제 질문할 수 있습니다.")
        if job.images == "running":
            job.images = "done"


def _hot_payload(h) -> dict:
    """많이 보도된 기사 하나 → 화면에 보낼 내용 (qupict.hot.HotArticle)
    similar: 묶인 비슷한 기사 목록 (화면에서 "비슷한 기사 N건"을 누르면 펼쳐진다)"""
    day = lambda d: d.strftime("%Y-%m-%d") if d else ""
    return {"title": h.title, "link": h.link, "press": h.press or "알 수 없음",
            "pub_date": day(h.pub_date), "duplicates": h.duplicates, "summary": h.summary,
            "similar": [{"title": s.title, "link": s.link, "press": s.press or "알 수 없음",
                         "pub_date": day(s.pub_date)} for s in h.similar]}


@app.post("/api/prepare")
def prepare(req: PrepareRequest) -> dict:
    try:
        validate_settings()
    except EnvironmentError as e:
        raise HTTPException(500, str(e))
    selected = _select(req)
    memory = req.memory or not settings.has_db
    label = f"{selected.query} ({selected.sense.name})"
    job_id = uuid.uuid4().hex
    job = Job(label=label)
    job.logs.append(f"[0단계] 선택한 의미 → {selected.sense.name}: {selected.sense.definition}")
    if memory and not req.memory:
        job.logs.append("FIRESTORE_PROJECT가 없어 DB 없이(메모리 버전으로) 실행합니다.")
    _sweep()                                   # 새 작업을 받을 때 오래 안 쓴 작업 · 세션을 같이 정리한다
    with _lock:
        _jobs[job_id] = job
    threading.Thread(target=_run_job, args=(job_id, selected, req, memory), daemon=True).start()
    return {"job_id": job_id, "label": label}


@app.post("/api/leave")
def leave(req: LeaveRequest) -> dict:
    """사용자가 페이지를 나갔거나 새로 검색한다 → 그 사용자의 작업 · 질문 세션을 서버 메모리에서 지운다 (_forget).
    화면(app.js)이 navigator.sendBeacon으로 보낸다: 페이지가 닫히는 중에도 전송되는 방법이다."""
    _forget(req.job_id, req.session_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다 (오래 쓰지 않았거나 서버가 다시 켜졌습니다. 처음부터 다시 해 주세요).")
    job.last_seen = time.time()                # 화면이 아직 보고 있다 (_sweep이 지우지 않게)
    return {"status": job.status, "logs": job.logs, "session_id": job.session_id,
            "error": job.error, "label": job.label, "images": job.images,
            "hot": job.hot, "hot_articles": job.hot_articles, "elapsed": round(job.elapsed, 1)}


# ---------------------------------------------------------------------------
# [5단계] 질문 · 답변
# ---------------------------------------------------------------------------

def _get_session(session_id: str) -> Session:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "세션이 없습니다 (오래 쓰지 않았거나 서버가 다시 켜졌습니다. 뉴스를 다시 준비해 주세요).")
    return session


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    """답변을 다 만든 뒤 한 번에 돌려준다 (예전 방식. 화면은 /api/ask/stream을 쓴다)."""
    answer = _get_session(req.session_id).ask(req.question.strip(), None)
    return _answer_payload(answer)


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """답변을 만들어지는 대로 보낸다. 한 줄에 JSON 하나 (NDJSON):

        {"type": "token", "text": "강남은 "}            답변 글 조각 (여러 번)
        {"type": "done", "answer": ..., "sources": ...}  마지막: /api/ask와 같은 내용 (완성된 답변 + 출처)
        {"type": "error", "message": "..."}              실패

    완성된 답변 글은 /api/ask와 같다. 사용자는 첫 글자를 훨씬 빨리 본다.
    """
    session = _get_session(req.session_id)
    question = req.question.strip()
    events: queue.Queue = queue.Queue()

    def work() -> None:          # 답변 만들기는 별도 스레드에서, 조각이 생길 때마다 events에 넣는다
        try:
            answer = session.ask(question, lambda text: events.put({"type": "token", "text": text}))
            events.put({"type": "done", **_answer_payload(answer)})
        except Exception as e:
            traceback.print_exc()
            events.put({"type": "error", "message": str(e) or type(e).__name__})

    def lines() -> Iterator[str]:
        while True:
            event = events.get()
            yield json.dumps(event, ensure_ascii=False) + "\n"
            if event["type"] != "token":
                return

    threading.Thread(target=work, daemon=True).start()
    return StreamingResponse(lines(), media_type="application/x-ndjson")


def _answer_payload(answer: Answer) -> dict:
    """답변 → 화면에 보낼 내용. 답변에서 실제로 인용한 기사만 출처로 보여 준다
    (번호는 답변 속 "(기사 N)"과 같게 유지) — run.print_answer와 같은 규칙"""
    shown = answer.cited or list(range(1, len(answer.sources) + 1))
    sources = []
    for i in shown:
        if not 1 <= i <= len(answer.sources):
            continue
        src = answer.sources[i - 1]
        sources.append({
            "n": i, "title": src.title, "press": src.press or "알 수 없음", "link": src.link,
            "pub_date": src.pub_date.strftime("%Y-%m-%d") if src.pub_date else "",
            "kind": answer.source_kinds[i - 1] if i <= len(answer.source_kinds) else "",
        })
    return {
        "question": answer.question,
        "answer": answer.answer,
        "sources": sources,
        "unused": len(answer.sources) - len(sources),
    }


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(prog="python web.py", description="QuPicT 웹 서비스")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    print(f"브라우저에서 http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port} 을 여세요.")
    from qupict import collect, db, filtering
    db.warm_up()                # DB 클라이언트 · 인증 토큰을 서버를 켤 때 미리 준비해 둔다 → 첫 요청이 빨라짐
    collect.warm_up_parser()    # 크롤링 해석용 프로세스(띄우는 데 약 2초)도 미리 띄워 둔다
    filtering.warm_up_kiwi()    # 형태소 분석기 Kiwi(약 500MB, 몇 초)도 뒤에서 미리 만든다 → 첫 사용자가 기다리지 않고, 한 번만 만든다
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
