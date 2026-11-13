"""
6. RAG & LLM 답변 생성

1순위 근거는 항상 [참고 기사]다. 답변은 하나의 자연스러운 글로 작성하되,
참고 기사로 뒷받침되는 문장에는 (기사 N) 형태로 인라인 인용을 붙인다.
참고 기사에 없는 내용은 모델의 일반 지식으로 자연스럽게 보충하되,
실제 기사에 없는 내용에 (기사 N) 표시를 지어내 붙이는 것은 금지한다.
관련 뉴스를 아예 찾지 못한 경우에는 답변 끝에 그 사실만 짧게 덧붙인다.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from qupict import timing
from qupict.llm import get_chat_model
from qupict.schemas import Answer, Article

NO_ARTICLE_CONTEXT = (
    "(이번 질문과 관련성 있는 기사를 수집된 뉴스에서 찾지 못했습니다. "
    "참고할 기사가 없습니다.)"
)

_SYSTEM_PROMPT = """\
당신은 신뢰할 수 있는 뉴스 기반 답변 에이전트입니다.

답변 작성 방식:
1. 질문에 대해 하나의 자연스럽고 완결된 답변을 작성하세요. 별도 섹션이나 헤더로
   나누지 말고, 보통 사람이 설명하듯 이어지는 문단으로 쓰세요.
2. [참고 기사]에 질문과 관련된 구체적 사실(최신 발표, 수치, 동향 등)이 있다면
   그 내용을 답변 문장에 자연스럽게 녹이고, 문장 끝에 "(기사 N)" 형태로
   출처를 표시하세요.
   예: "제논은 최근 실무형 업무 자동화를 강조한 'Gen AI 2.0' 개념을 제시했다(기사 1)."
3. [참고 기사]에 없는 내용은 당신의 일반 지식으로 자연스럽게 보충해도 됩니다.
   단, 이 경우 (기사 N) 표시를 절대 붙이지 마세요 — 기사에 없는 내용에
   출처 표시를 붙이는 것은 사실을 지어내는 것과 같습니다.
4. 질문과 관련된 뉴스 내용이 [참고 기사]에 전혀 없거나, 컨텍스트에
   "관련성 있는 기사를 찾지 못했습니다"라고 표시된 경우:
   먼저 일반 지식만으로 답변을 완결한 뒤, 답변 맨 마지막에 짧게 한 문장을 덧붙이세요.
   - [참고 가능한 기사 기간]에 날짜가 있으면, 그 시작일과 끝일을 "2026년 10월 14일"처럼 써서
     "다만 2026년 10월 14일부터 2026년 11월 13일까지의 기사 중에서는 참고할 수 있는 기사를 찾지 못했습니다."
     라고 덧붙이세요. 이때 "최신 뉴스는 찾지 못했습니다", "제공된 기사로 확인할 수 없습니다"처럼
     기간이 없는 표현은 답변 어디에도 쓰지 마세요. 기사를 못 찾았다는 말은 이 기간 문장 한 번만 하세요.
   - 날짜가 없으면 "다만 이와 관련된 최신 뉴스는 찾지 못했습니다." 라고 덧붙이세요.
   큰 섹션 표시나 경고 문구, 굵은 헤더는 쓰지 마세요.
5. 일반 지식 중 확실하지 않은 내용은 추측하지 말고 "정확히 확인되지 않았습니다"
   등으로 솔직하게 표현하세요.
6. [매우 중요] 특정 인물의 이름, 직함, 소속, 구체적 수치, 날짜 등 "고유하게 확정되는
   사실"은 [참고 기사]에 명시되어 있지 않으면 절대로 지어내지 마세요. 그럴듯하게
   들리는 답을 만들어내는 것보다, "누구인지/무엇인지 확인되지 않았습니다"라고
   솔직히 답하는 것이 항상 낫습니다. 특히 인명은 비슷한 발음이나 흔한 이름을
   추측해서 답하지 마세요.
7. 사용자가 질문에 특정 사실을 전제하거나 주장하는 경우(예: "OO 아니야?", "OO 맞지?"),
   그 주장에 무조건 동의하지 마세요. [참고 기사]에서 실제로 확인되는 경우에만
   동의하고, 확인되지 않으면 "확인되지 않았습니다"라고 답한 뒤 4번 규칙의 기간 문장으로 끝내세요.
   직전에 다른 답을 했더라도, 사용자가 정정을 요구했다고 해서 근거 없이 말을
   바꾸지 마세요.
8. 답변은 한국어로, 간결하고 자연스럽게 작성하세요.
9. [참고 기사]에서 "[기사 이미지]"로 시작하는 부분은 기사 본문이 아니라 기사에 실린 사진을 글로 옮긴
   설명입니다. 이 내용을 근거로 쓸 때는 "(기사 N, 이미지)"로 표시하세요.
10. 사용자는 [검색 대상]에 대해 묻고 있습니다. 질문에 주어가 없거나 "그", "그 사람", "여기"처럼
    가리키는 말만 있으면, 되묻지 말고 아래 순서로 대상을 정해 바로 답하세요.
    - [이전 대화]에 그 말이 가리키는 대상이 분명히 있으면 그 대상
      (예: 앞 답변이 "강남의 아내는 이상화입니다"이고 질문이 "그 사람 직업은?"이면 이상화)
    - 그렇지 않으면 [검색 대상]
    [이전 대화]는 무엇을 가리키는지 알아내는 데만 쓰세요. 사실의 근거는 [참고 기사]이고,
    [이전 대화]의 내용에는 (기사 N) 표시를 붙이지 마세요.
11. 답변 첫 문장은 질문에 대한 직접적인 답으로 시작하고, 질문이 가리키는 대상의 이름을 주어로 넣으세요.
    대상을 부를 때는 괄호 앞의 검색어를 쓰세요. 배경 설명은 그 뒤에 이어 쓰세요.
    예: 검색 대상이 "강남 (방송인 강남)"이고 질문이 "아내가 누구야?"이면
        "강남의 아내는 ○○입니다(기사 N). …" 처럼 답하세요.
    단, 6번 규칙은 그대로 지키세요: [참고 기사]에 없는 이름·수치는 지어내지 말고
    "강남의 아내가 누구인지는 확인되지 않았습니다"처럼 답한 뒤 4번 규칙의 기간 문장으로 끝내세요.
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "[검색 대상] {subject}\n[이전 대화]\n{history}\n[참고 가능한 기사 기간] {period}\n질문: {question}"
                  "\n\n[참고 기사]\n{context}"),
    ]
)

NO_SUBJECT = "(지정 없음)"
NO_PERIOD = "(알 수 없음)"
NO_HISTORY = "(없음)"

# [이전 대화]에 넣는 범위. 바로 앞 질문을 가리키는 말("그 사람", "거기")을 알아듣는 데는 최근 몇 번이면 충분하다.
# 답변 전체를 넣으면 프롬프트가 길어지므로 앞부분만 (첫 문장에 답이 있다: 11번 규칙).
HISTORY_TURNS = 2
HISTORY_ANSWER_CHARS = 300
_CITE_MARK_RE = re.compile(r"\s*\(기사\s*\d+[^)]*\)")


def history_text(turns: list[tuple[str, str]]) -> str:
    """앞서 한 질문·답변 [(질문, 답변), …] → 답변 프롬프트의 [이전 대화] 글 (최근 HISTORY_TURNS개).
    답변의 "(기사 N)" 표시는 뺀다: 번호는 그 답변의 출처 번호라 이번 질문의 [참고 기사] 번호와 다르다.
    LLM을 따로 부르지 않는다 (앞 대화를 요약하지 않고 그대로 앞부분만 넣는다)."""
    lines = []
    for question, answer in turns[-HISTORY_TURNS:]:
        text = " ".join(_CITE_MARK_RE.sub("", answer).split())
        if len(text) > HISTORY_ANSWER_CHARS:
            text = text[:HISTORY_ANSWER_CHARS - 1].rstrip() + "…"
        lines.append(f"사용자: {question.strip()}\n답변: {text}")
    return "\n".join(lines) or NO_HISTORY

# 뉴스 준비가 끝나면 자동으로 하는 첫 질문 (웹 화면 · 터미널 모두). 사용자 질문은 그 뒤에 답한다.
# web/index.html에도 같은 문장이 있다 (tests/offline/check_web.py가 둘이 같은지 확인한다).
FIRST_QUESTION_TEMPLATE = "{query}에 대해 수집된 기사들의 전체적인 요약을 제공해줘."


def first_question(query: str) -> str:
    """첫 질문(고정). 예: "강남에 대해 수집된 기사들의 전체적인 요약을 제공해줘." """
    return FIRST_QUESTION_TEMPLATE.format(query=query.strip())
_KST = timezone(timedelta(hours=9))


def answer_period(oldest: datetime | None, today: datetime | None = None) -> str:
    """답변 프롬프트의 [참고 가능한 기사 기간] 글. 예: "2026-10-14 ~ 2026-11-13"
    oldest: 질문에 쓸 수 있는 기사 중 가장 오래된 기사의 발행일. 끝은 오늘(질문한 날). 날짜는 한국 시간 기준.
    oldest가 없으면(쓸 수 있는 기사가 없거나 발행일을 모름) NO_PERIOD."""
    if oldest is None:
        return NO_PERIOD
    today = today or datetime.now(timezone.utc)
    start = oldest.astimezone(_KST) if oldest.tzinfo else oldest
    return f"{start:%Y-%m-%d} ~ {today.astimezone(_KST):%Y-%m-%d}"


def oldest_date(articles: list[Article]) -> datetime | None:
    """기사 목록에서 가장 오래된 발행일 (메모리 버전에서 answer_period에 넘길 값). 발행일이 없으면 None."""
    dates = [a.pub_date for a in articles if a.pub_date]
    return min(dates) if dates else None


def answer_subject(selected) -> str:
    """답변 프롬프트의 [검색 대상] 글. 예: "강남 (방송인 강남: 가수 겸 방송인, M.I.B 출신)"
    selected: 사용자가 고른 의미(SelectedSense). 없으면 NO_SUBJECT."""
    if selected is None:
        return NO_SUBJECT
    s = selected.sense
    return f"{selected.query} ({s.name}: {s.definition})" if s.definition else f"{selected.query} ({s.name})"


def _build_chain():
    llm = get_chat_model(temperature=0.1)
    return _prompt | llm | StrOutputParser()


_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = _build_chain()
    return _chain


def generate_answer(question: str, articles: list[Article],
                    on_token: Callable[[str], None] | None = None, subject: str = NO_SUBJECT,
                    period: str = NO_PERIOD, history: str = NO_HISTORY) -> Answer:
    """
    질문 + 관련 기사 목록을 받아 답변을 생성한다. (메모리 버전)

    관련 기사가 없어도 답변을 포기하지 않는다. 이 경우 컨텍스트에 "관련 기사 없음"을
    명시해 LLM이 일반 지식으로 답변을 완결한 뒤, 마지막에 그 사실을 짧게 덧붙이도록
    유도한다.
    on_token: 주면 답변을 스트리밍으로 받으며 조각마다 부른다 (generate_answer_from_chunks와 같음)
    subject: 사용자가 검색한 대상 (answer_subject로 만든 글). 질문에 주어가 없으면 이 대상에 대한 질문으로 본다
    period: 질문에 쓸 수 있는 기사의 기간 (answer_period로 만든 글). 관련 기사를 못 찾았을 때 이 기간을 밝힌다
    history: 앞서 한 질문·답변 (history_text로 만든 글). "그 사람"처럼 앞 답변을 가리키는 질문을 알아듣는 데 쓴다
    """
    if articles:
        context = "\n\n".join(
            article.to_context_block(i + 1) for i, article in enumerate(articles)
        )
    else:
        context = NO_ARTICLE_CONTEXT

    with timing.timed("답변 생성(LLM)") as t:
        answer_text = _run_answer_chain(question, context, on_token, subject, period, history)
        t.note = f"입력 근거 {len(context)}자 → 답변 {len(answer_text)}자"

    return Answer(question=question, answer=answer_text, sources=articles)


def build_chunk_context(hits: list) -> tuple[str, list[Article], list[str]]:
    """검색된 청크(db.ChunkHit)를 기사별로 묶어 [참고 기사] 글을 만든다.

    같은 기사의 청크는 한 번호(기사 N) 아래에 모은다 → 답변의 (기사 N)과 출처 목록 번호가 맞는다.
    돌려주는 값: (참고 기사 글, 출처 기사 목록, 기사별 근거 종류)
    """
    groups: dict[str, list] = {}
    for hit in hits:                                   # 유사도 높은 기사부터 번호를 붙인다
        groups.setdefault(hit.article.link, []).append(hit)

    blocks, sources, kinds = [], [], []
    for n, chunk_hits in enumerate(groups.values(), start=1):
        a = chunk_hits[0].article
        date = a.pub_date.strftime("%Y-%m-%d") if a.pub_date else "알 수 없음"
        body = "\n".join(f"- {h.content}" for h in chunk_hits)
        blocks.append(f"[기사 {n}]\n제목: {a.title}\n언론사: {a.press or '알 수 없음'}\n발행일: {date}\n"
                      f"URL: {a.link}\n관련 내용:\n{body}")
        sources.append(a)
        has_body = any(h.source == "body" for h in chunk_hits)
        has_image = any(h.source == "image" for h in chunk_hits)
        kinds.append("본문 + 이미지 설명" if has_body and has_image else "이미지 설명" if has_image else "본문")
    return "\n\n".join(blocks), sources, kinds


def _run_answer_chain(question: str, context: str, on_token: Callable[[str], None] | None,
                      subject: str = NO_SUBJECT, period: str = NO_PERIOD, history: str = NO_HISTORY) -> str:
    """답변 LLM 호출. on_token을 주면 스트리밍: 글자가 만들어지는 대로 on_token(조각)을 부른다.

    스트리밍은 "다 만들어진 뒤 한 번에" 대신 "만들어지는 대로" 받는 것뿐이라 완성된 답변 글은 같다.
    사용자는 첫 글자를 훨씬 빨리 본다 (전체 시간은 거의 같음).
    스트리밍 연결이 첫 조각 전에 실패하면 예전처럼 한 번에 받는다.
    """
    chain = _get_chain()
    inputs = {"question": question, "context": context, "subject": subject or NO_SUBJECT,
              "period": period or NO_PERIOD, "history": history or NO_HISTORY}
    if on_token is None:
        return chain.invoke(inputs)
    parts: list[str] = []
    start = time.perf_counter()
    try:
        for piece in chain.stream(inputs):
            if not parts:
                timing.add("첫 글자까지", time.perf_counter() - start)
            parts.append(piece)
            on_token(piece)
    except Exception:
        if parts:
            raise
        text = chain.invoke(inputs)
        on_token(text)
        return text
    return "".join(parts)


def generate_answer_from_chunks(question: str, hits: list,
                                on_token: Callable[[str], None] | None = None,
                                subject: str = NO_SUBJECT, period: str = NO_PERIOD,
                                history: str = NO_HISTORY) -> Answer:
    """[DB 버전] 검색된 청크로 답변을 만든다. 출처에 근거 종류(본문 / 이미지 설명)를 함께 돌려준다.
    on_token: 주면 답변을 스트리밍으로 받으며 조각마다 부른다 (run.py가 화면에 바로 찍는 데 씀)
    subject: 사용자가 검색한 대상 (answer_subject로 만든 글). 질문에 주어가 없으면 이 대상에 대한 질문으로 본다
    period: 질문에 쓸 수 있는 기사의 기간 (answer_period로 만든 글). 관련 기사를 못 찾았을 때 이 기간을 밝힌다
    history: 앞서 한 질문·답변 (history_text로 만든 글). 없으면 NO_HISTORY"""
    context, sources, kinds = build_chunk_context(hits) if hits else (NO_ARTICLE_CONTEXT, [], [])
    with timing.timed("답변 생성(LLM)") as t:
        answer_text = _run_answer_chain(question, context, on_token, subject, period, history)
        t.note = f"입력 근거 {len(context)}자 → 답변 {len(answer_text)}자"
    return Answer(question=question, answer=answer_text, sources=sources, source_kinds=kinds,
                  cited=cited_numbers(answer_text, len(sources)))


_CITE_RE = re.compile(r"기사\s*(\d+(?:\s*[,，]\s*(?:기사\s*)?\d+)*)")


def cited_numbers(answer_text: str, n_sources: int) -> list[int]:
    """답변에서 인용한 기사 번호. "(기사 1, 기사 2)", "(기사 1, 2)", "(기사 3, 이미지)" 모두 찾는다."""
    found: set[int] = set()
    for group in _CITE_RE.findall(answer_text):
        found.update(int(n) for n in re.findall(r"\d+", group))
    return sorted(n for n in found if 1 <= n <= n_sources)