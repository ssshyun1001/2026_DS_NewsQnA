"""
6. RAG & LLM 답변 생성

1순위 근거는 항상 [참고 기사]다. 다만 참고 기사만으로 답할 수 없는 부분이 있으면,
그 사실을 먼저 명확히 밝힌 뒤 모델의 일반 지식으로 보충 답변한다. 기사 근거와
일반 지식 부분이 절대 섞이지 않고 명확히 구분되도록 프롬프트로 강제한다.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from app.schemas import Answer, Article

_SYSTEM_PROMPT = """\
당신은 신뢰할 수 있는 뉴스 기반 답변 에이전트입니다.

답변 규칙 (반드시 순서대로 따르세요):
1. 가장 먼저 [참고 기사] 안의 내용만으로 답변 가능한 부분을 찾아 답변하세요.
   이 부분은 반드시 기사 내용에 근거해야 하며, 어떤 기사(번호)를 근거로 했는지
   문장 중간에 자연스럽게 밝히세요. 예: "(기사 1)"
2. 질문의 일부 또는 전체에 대한 답이 [참고 기사] 안에 없다면:
   a. 먼저 그 사실을 명확히 밝히세요. 예: "제공된 뉴스 기사에는 관련 내용이 없습니다."
   b. 그 다음, 당신이 알고 있는 일반 지식으로 보충 설명을 이어가세요.
   c. 보충 설명 앞에는 반드시 아래 표시를 붙여 기사 근거와 구분하세요:
      "[일반 지식 기반 답변 — 아래 내용은 수집된 뉴스 기사가 아닌 모델의 사전 지식에 근거합니다]"
3. 기사 근거 부분과 일반 지식 부분이 한 문장 안에 섞이지 않도록 문단을 분리하세요.
4. 참고 기사가 아예 없는 경우(컨텍스트에 "관련성 있는 기사를 찾지 못했습니다"라고 표시된 경우)에도
   2번 규칙을 그대로 적용해 "관련 뉴스를 찾지 못했다"는 사실을 먼저 밝힌 뒤 일반 지식으로 답하세요.
5. 답변은 한국어로, 간결하고 사실 위주로 작성하세요. 일반 지식 부분도 확실하지 않은 내용은
   추측하지 말고 "정확히 확인되지 않았습니다" 등으로 솔직하게 표현하세요.
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
    명시해 LLM이 그 사실을 먼저 밝히고, 이어서 일반 지식으로 답하도록 유도한다.
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