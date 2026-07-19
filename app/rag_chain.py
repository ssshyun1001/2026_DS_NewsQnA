"""
6. RAG & LLM 답변 생성

1순위 근거는 항상 [참고 기사]다. 답변은 하나의 자연스러운 글로 작성하되,
참고 기사로 뒷받침되는 문장에는 (기사 N) 형태로 인라인 인용을 붙인다.
참고 기사에 없는 내용은 모델의 일반 지식으로 자연스럽게 보충하되,
실제 기사에 없는 내용에 (기사 N) 표시를 지어내 붙이는 것은 금지한다.
관련 뉴스를 아예 찾지 못한 경우에는 답변 끝에 그 사실만 짧게 덧붙인다.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from app.schemas import Answer, Article

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
   먼저 일반 지식만으로 답변을 완결한 뒤, 답변 맨 마지막에 짧게 한 문장으로
   "다만 이와 관련된 최신 뉴스는 찾지 못했습니다." 라고 덧붙이세요.
   큰 섹션 표시나 경고 문구, 굵은 헤더는 쓰지 마세요.
5. 일반 지식 중 확실하지 않은 내용은 추측하지 말고 "정확히 확인되지 않았습니다"
   등으로 솔직하게 표현하세요.
6. 답변은 한국어로, 간결하고 자연스럽게 작성하세요.
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "질문: {question}\n\n[참고 기사]\n{context}"),
    ]
)


def _build_chain():
    llm = ChatOpenAI(
        model=settings.chat_model, api_key=settings.openai_api_key, temperature=0.1
    )
    return _prompt | llm | StrOutputParser()


_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = _build_chain()
    return _chain


def generate_answer(question: str, articles: list[Article]) -> Answer:
    """
    질문 + 관련 기사 목록을 받아 답변을 생성한다.

    관련 기사가 없어도 답변을 포기하지 않는다. 이 경우 컨텍스트에 "관련 기사 없음"을
    명시해 LLM이 일반 지식으로 답변을 완결한 뒤, 마지막에 그 사실을 짧게 덧붙이도록
    유도한다.
    """
    if articles:
        context = "\n\n".join(
            article.to_context_block(i + 1) for i, article in enumerate(articles)
        )
    else:
        context = (
            "(이번 질문과 관련성 있는 기사를 수집된 뉴스에서 찾지 못했습니다. "
            "참고할 기사가 없습니다.)"
        )

    answer_text = _get_chain().invoke({"question": question, "context": context})

    return Answer(question=question, answer=answer_text, sources=articles)