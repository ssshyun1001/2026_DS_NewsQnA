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

준비하는 동안에도 질문 칸이 열려 있다. 미리 적은 질문은 화면이 기억해 두었다가 준비가 끝나면 차례로 보낸다.
예전에 준비한 검색어·의미(DB 버전)는 준비가 끝나기를 기다리지 않고, DB에 저장된 기사로 바로 답하기 시작한다.

세션(준비한 뉴스)은 서버 메모리에만 있다 → 서버를 끄면 사라진다. DB 버전은 DB에 쌓인 것을 다시 쓰므로
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
from pydantic import BaseModel, Field

from qupict.cli import setup_console
from qupict.config import settings, validate_settings
from qupict.schemas import Answer, SelectedSense, SenseCandidates

setup_console()

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="QuPicT 뉴스 Q&A")


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
    early: int = 0                     # 준비가 끝나기 전에 DB에 저장된 기사로 먼저 답할 때, 그 기사 수 (0 = 아님)
    elapsed: float = 0.0               # 준비에 걸린 시간(초). 준비 완료 때 채운다


@dataclass
class Session:
    """질문에 답할 수 있는 뉴스. 질문이 오면 ask(질문, on_token)으로 답한다.
    answer_fn(질문, on_token, 이전 대화 글): on_token을 주면 답변 글 조각이 만들어질 때마다 불린다 (스트리밍).
    turns: 이 세션에서 한 질문·답변. 다음 질문의 [이전 대화]로 넘겨 "그 사람"처럼 앞 답변을 가리키는 질문도 알아듣게 한다."""
    label: str
    mode: str                          # "DB" / "메모리"
    answer_fn: Callable[[str, Optional[Callable[[str], None]], str], Answer]
    turns: list[tuple[str, str]] = field(default_factory=list)

    def ask(self, question: str, on_token: Optional[Callable[[str], None]] = None) -> Answer:
        from qupict.qa import history_text

        with _lock:
            history = history_text(self.turns)
        answer = self.answer_fn(question, on_token, history)
        with _lock:
            self.turns.append((question, answer.answer))
        return answer


_candidates: dict[str, SenseCandidates] = {}   # 검색어 → 의미 후보 (선택할 때 다시 쓴다)
_jobs: dict[str, Job] = {}
_sessions: dict[str, Session] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 요청 · 응답 형식
# ---------------------------------------------------------------------------

class SensesRequest(BaseModel):
    query: str = Field(min_length=1)


class PrepareRequest(BaseModel):
    query: str = Field(min_length=1)
    period_days: int = Field(ge=1, le=365)
    sense_index: Optional[int] = None       # 후보 번호 (0부터). 없으면 custom_description을 쓴다
    custom_description: str = ""            # "기타 (직접 입력)" 설명
    max_articles: int = Field(default=100, ge=1, le=500)
    memory: bool = False                    # DB 없이 메모리(FAISS) 버전


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
    return {"has_db": bool(settings.database_url), "chat_model": settings.chat_model}


# ---------------------------------------------------------------------------
# [0단계] 의미 후보
# ---------------------------------------------------------------------------

@app.post("/api/senses")
def senses(req: SensesRequest) -> dict:
    from qupict.sense import load_sense_candidates

    query = req.query.strip()
    candidates, source = load_sense_candidates(query)
    with _lock:
        _candidates[query] = candidates
    return {
        "query": query,
        "source": source,
        "senses": [{"name": s.name, "definition": s.definition,
                    "aliases": s.aliases, "related_terms": s.related_terms}
                   for s in candidates.senses],
    }


def _select(req: PrepareRequest) -> SelectedSense:
    from qupict.sense import build_custom_sense, load_sense_candidates, select_custom_sense, select_sense

    query = req.query.strip()
    with _lock:
        candidates = _candidates.get(query)
    if candidates is None:                  # 서버를 다시 켠 경우 등: 후보를 다시 가져온다
        candidates, _ = load_sense_candidates(query)
    if req.sense_index is not None:
        if not 0 <= req.sense_index < len(candidates.senses):
            raise HTTPException(400, "의미 후보 번호가 맞지 않습니다. 검색어를 다시 입력해 주세요.")
        return select_sense(candidates, req.sense_index)
    desc = req.custom_description.strip()
    if not desc:
        raise HTTPException(400, "의미를 고르거나, 찾고 싶은 대상을 직접 설명해 주세요.")
    return select_custom_sense(candidates, build_custom_sense(query, desc))


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
        """질문을 받기 시작한다 (한 작업에 한 번만. 저장된 기사로 먼저 열었으면 그 세션을 계속 쓴다)."""
        with _lock:
            if job.session_id is None:
                session_id = uuid.uuid4().hex
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
            find_hot = lambda: hot.top_duplicated(news.articles, news.duplicates)

            def answer_fn(question: str, on_token=None, history: str = "") -> Answer:
                relevant = retrieve_relevant_articles(news.store, news.articles, question, topic=news.topic)
                return generate_answer(question, relevant, on_token=on_token, subject=answer_subject(selected),
                                       period=answer_period(oldest), history=history)
        else:
            from qupict.qa import NO_HISTORY, answer_period, answer_subject, generate_answer_from_chunks
            from qupict.retrieval import search_topic
            from run import oldest_searchable_or_none, prepare_topic, wait_after_prepare

            # 답변이 쓰는 topic · 기사 기간. 저장된 기사로 먼저 답하는 동안에도 같은 함수를 쓰고, 준비가 끝나면 값만 바꾼다
            # (검색은 질문할 때마다 DB에서 하므로, 준비 중에 새로 저장된 기사도 저장되는 대로 답변에 들어간다)
            current = {"topic_id": None, "oldest": None}

            def answer_fn(question: str, on_token=None, history: str = NO_HISTORY) -> Answer:
                hits = search_topic(current["topic_id"], question, selected, period_days=req.period_days,
                                    history="" if history == NO_HISTORY else history)
                return generate_answer_from_chunks(question, hits, on_token=on_token, subject=answer_subject(selected),
                                                   period=answer_period(current["oldest"]), history=history)

            # 예전에 준비한 적 있는 검색어·의미면, 수집·판정을 기다리지 않고 DB에 저장된 기사로 바로 답하기 시작한다.
            # (DB에 쌓아 두고 다시 쓰는 것이 이 서비스가 DB를 쓰는 이유: 같은 검색어는 기다리지 않는다)
            early = threading.Thread(target=_open_from_db, daemon=True,
                                     args=(job, selected, req.period_days, current, answer_fn, open_session, log))
            early.start()
            prepared = asyncio.run(prepare_topic(selected, req.period_days, req.max_articles, log=log))
            early.join()
            topic_id = prepared.topic_id
            # "준비 완료" 전에 잠깐 기다린다 (run.PREPARE_SETTLE_SECONDS). 새로 저장한 것이 없으면 기다리지 않는다
            wait_after_prepare(log=log, wrote=prepared.wrote)
            current.update(topic_id=topic_id, oldest=oldest_searchable_or_none(topic_id, req.period_days))
            find_hot = lambda: hot.top_duplicated_db(topic_id, req.period_days)

        open_session(answer_fn)
        if not memory:
            from run import image_text_mode
            job.images = "running" if image_text_mode() else ""    # done보다 먼저: 화면이 계속 확인하게
        job.hot = "running"                                        # 이것도 done보다 먼저 (같은 이유)
        job.elapsed = time.perf_counter() - start
        job.status = "done"
        log(f"준비 완료! 이제 질문할 수 있습니다. (총 {job.elapsed:.1f}초)")
    except Exception as e:
        traceback.print_exc()
        job.error = str(e) or type(e).__name__
        job.status = "error"
        log(f"[오류] {job.error}")
        return

    # 준비 완료 뒤: 많이 보도된 기사 Top 5 (중복이 많이 묶인 대표 기사 + 한 줄 요약, LLM 한 번).
    # 질문은 그동안에도 받는다. 실패해도 질문 답변에는 영향이 없다.
    try:
        job.hot_articles = [_hot_payload(h) for h in hot.summarize(find_hot())]
        job.hot = "done"
    except Exception as e:
        traceback.print_exc()
        log(f"[많이 보도된 기사] 실패해서 건너뜁니다 ({type(e).__name__}: {e})")
        job.hot = "error"

    # 준비 완료 뒤: 이미지 → 글(OCR·VLM)을 이 스레드에서 이어서 한다. 질문은 그동안에도 받는다.
    # 끝나면 그 뒤 질문부터 이미지 속 내용도 검색된다. 실패해도 질문 답변에는 영향이 없다.
    if job.images == "running":
        from run import add_image_text
        try:
            asyncio.run(add_image_text(topic_id, log=log))
        except Exception as e:
            traceback.print_exc()
            log(f"[이미지] 실패해서 건너뜁니다 ({type(e).__name__}: {e})")
        job.images = "done"


def _open_from_db(job: Job, selected: SelectedSense, period_days: int, current: dict,
                  answer_fn, open_session, log) -> None:
    """[DB 버전] 준비를 시작하는 동시에 돈다. 이 검색어·의미를 예전에 준비해서 질문에 쓸 수 있는 기사
    (판정 통과 + 청크 있음 + 기간 안)가 DB에 있으면, 준비가 끝나기 전에 그 기사로 먼저 질문을 받는다.
    처음 검색하는 대상이거나 조회가 실패하면 아무것도 하지 않는다 (준비가 끝나면 평소처럼 열린다)."""
    from qupict import db
    from qupict.retrieval import searchable_stats

    try:
        record = db.get_topic(selected.query, selected.sense.name)
        if record is None:
            return
        count, oldest = searchable_stats(record.id, period_days)
        if not count:
            return
        current.update(topic_id=record.id, oldest=oldest)
        job.early = count
        open_session(answer_fn)
        log(f"DB에 저장된 기사 {count}건으로 지금 바로 질문할 수 있습니다 "
            f"(수집·판정을 기다리지 않음. 새 기사는 저장되는 대로 답변에 들어갑니다)")
    except Exception as e:
        traceback.print_exc()
        log(f"[DB 재사용] 저장된 기사 확인에 실패해서 준비가 끝난 뒤 답합니다 ({type(e).__name__})")


def _hot_payload(h) -> dict:
    """많이 보도된 기사 하나 → 화면에 보낼 내용 (qupict.hot.HotArticle)"""
    return {"title": h.title, "link": h.link, "press": h.press or "알 수 없음",
            "pub_date": h.pub_date.strftime("%Y-%m-%d") if h.pub_date else "",
            "duplicates": h.duplicates, "summary": h.summary}


@app.post("/api/prepare")
def prepare(req: PrepareRequest) -> dict:
    try:
        validate_settings()
    except EnvironmentError as e:
        raise HTTPException(500, str(e))
    selected = _select(req)
    memory = req.memory or not settings.database_url
    label = f"{selected.query} ({selected.sense.name})"
    job_id = uuid.uuid4().hex
    job = Job(label=label)
    job.logs.append(f"[0단계] 선택한 의미 → {selected.sense.name}: {selected.sense.definition}")
    if memory and not req.memory:
        job.logs.append("DATABASE_URL이 없어 DB 없이(메모리 버전으로) 실행합니다.")
    _jobs[job_id] = job
    threading.Thread(target=_run_job, args=(job_id, selected, req, memory), daemon=True).start()
    return {"job_id": job_id, "label": label}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다 (서버를 다시 켰다면 처음부터 다시 해 주세요).")
    return {"status": job.status, "logs": job.logs, "session_id": job.session_id,
            "error": job.error, "label": job.label, "images": job.images,
            "hot": job.hot, "hot_articles": job.hot_articles, "early": job.early, "elapsed": round(job.elapsed, 1)}


# ---------------------------------------------------------------------------
# [5단계] 질문 · 답변
# ---------------------------------------------------------------------------

def _get_session(session_id: str) -> Session:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "세션이 없습니다 (서버를 다시 켰다면 뉴스를 다시 준비해 주세요).")
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
    from qupict import collect, db
    db.warm_up()                # DB 연결(첫 연결 약 1.4초)을 서버를 켤 때 미리 열어 둔다 → 첫 요청이 빨라짐
    collect.warm_up_parser()    # 크롤링 해석용 프로세스(띄우는 데 약 2초)도 미리 띄워 둔다
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
