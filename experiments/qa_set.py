"""
실험 B·C용 질문 세트 — "질문 + 정답 + 정답 근거 기사"

파일: experiments/qa_set.yaml (git에 올려서 팀이 같이 쓴다)

  - id: 강남-001
    query: 강남                      # topic = 검색어 + 의미 이름 (DB topics와 같게)
    sense: 방송인 강남
    question: 강남이 괌에서 무슨 일을 했어?
    answer: 괌 동굴에서 물에 빠진 관광객을 헤엄쳐 가서 구했다
    evidence: [https://...]          # 정답이 들어 있는 기사 link (1개 이상)

만들기: python run.py make-qa [--target 강남] [--suggest]
  topic의 '관련' 기사를 하나씩 보여 주고, 그 기사로 답할 수 있는 질문과 정답을 입력한다.
  --suggest를 주면 LLM이 질문·정답 초안을 먼저 써 준다 (사람이 확인하고 Enter로 받거나 고쳐 쓴다).
  최근 사건을 묻는 질문을 넣으면 "LLM 단독 답변의 한계"(실험 C)가 잘 드러난다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

QA_PATH = Path(__file__).resolve().parent / "qa_set.yaml"


@dataclass
class QAItem:
    id: str
    query: str
    sense: str
    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)


def load_qa(path: Path | None = None, query: str | None = None) -> list[QAItem]:
    # 기본 경로는 부를 때 정한다 (정의할 때 정하면 테스트에서 경로를 바꿔도 실제 파일을 건드리게 됨)
    path = path or QA_PATH
    if not path.exists():
        return []
    items = [QAItem(**x) for x in (yaml.safe_load(path.read_text(encoding="utf-8")) or [])]
    return [x for x in items if query is None or x.query == query]


def save_qa(items: list[QAItem], path: Path | None = None) -> None:
    path = path or QA_PATH
    header = "# 실험 B·C 질문 세트 (만들기: python run.py make-qa). 형식은 experiments/qa_set.py 참고\n"
    body = yaml.safe_dump([asdict(x) for x in items], allow_unicode=True, sort_keys=False, width=1000)
    path.write_text(header + body, encoding="utf-8")


def next_id(items: list[QAItem], query: str) -> str:
    nums = [int(x.id.rsplit("-", 1)[-1]) for x in items if x.query == query and x.id.rsplit("-", 1)[-1].isdigit()]
    return f"{query}-{max(nums, default=0) + 1:03d}"


# ---------------------------------------------------------------------------
# LLM 초안 (--suggest)
# ---------------------------------------------------------------------------

class _QADraft(BaseModel):
    question: str = Field(description="이 기사만 읽으면 답할 수 있는, 사용자가 실제로 물어볼 법한 질문 한 문장")
    answer: str = Field(description="기사에 나온 사실로만 쓴 짧은 정답 (한 문장)")


_DRAFT_PROMPT = """\
너는 뉴스 Q&A 시스템을 평가할 질문을 만든다.
아래 기사를 읽고, 이 기사에 답이 있는 질문 하나와 정답을 써라.

- 질문은 사용자가 실제로 물어볼 법한 자연스러운 한국어로 쓴다. 기사 제목을 그대로 베끼지 마라.
- 정답은 기사에 나온 사실만으로 한 문장으로 쓴다. 숫자·이름·날짜처럼 확인 가능한 내용이면 더 좋다.
- 찾는 대상: {target}
"""


def draft_qa(title: str, content: str, target: str) -> tuple[str, str] | None:
    from qupict.llm import get_chat_model
    try:
        llm = get_chat_model(temperature=0).with_structured_output(_QADraft)
        d: _QADraft = llm.invoke([("system", _DRAFT_PROMPT.format(target=target)),
                                  ("human", f"제목: {title}\n본문: {content[:2500]}")])
        return d.question.strip(), d.answer.strip()
    except Exception as e:
        print(f"  [경고] 초안 만들기 실패 ({type(e).__name__})")
        return None


# ---------------------------------------------------------------------------
# 만들기 (터미널)
# ---------------------------------------------------------------------------

def make_qa(topic_query: str, sense_name: str, articles: list, suggest: bool = False,
            read=input) -> int:
    """'관련' 기사를 하나씩 보여 주고 질문·정답을 받아 qa_set.yaml에 추가한다. 추가한 개수를 돌려준다."""
    items = load_qa()
    used = {link for x in items for link in x.evidence}
    todo = [a for a in articles if a.link not in used]
    print(f"\n[{topic_query} → {sense_name}] 질문을 만들 기사 {len(todo)}건 (이미 질문이 있는 기사는 뺌)")
    print("  Enter: 초안 그대로 저장 / e: 직접 쓰기 / s: 건너뛰기 / q: 끝내기" if suggest else
          "  질문을 입력하세요 (빈 칸 Enter: 건너뛰기 / q: 끝내기)")
    added = 0
    for i, a in enumerate(todo, 1):
        print("\n" + "-" * 70)
        print(f"[{i}/{len(todo)}] {a.title}\n  본문: {a.content[:700]}{'…' if len(a.content) > 700 else ''}")
        question = answer = ""
        if suggest:
            draft = draft_qa(a.title, a.content, f"{topic_query} ({sense_name})")
            if draft:
                print(f"  초안 질문: {draft[0]}\n  초안 정답: {draft[1]}")
            key = read("선택> ").strip().lower()
            if key == "q":
                break
            if key == "s" or (key == "" and not draft):
                continue
            if key == "" and draft:
                question, answer = draft
        if not question:
            question = read("질문> ").strip()
            if question.lower() == "q":
                break
            if not question:
                continue
            answer = read("정답> ").strip()
            if not answer:
                print("  정답이 비어 있어 저장하지 않았습니다.")
                continue
        items.append(QAItem(next_id(items, topic_query), topic_query, sense_name, question, answer, [a.link]))
        save_qa(items)                        # 하나 만들 때마다 바로 저장 (중간에 끊겨도 남게)
        added += 1
    return added
