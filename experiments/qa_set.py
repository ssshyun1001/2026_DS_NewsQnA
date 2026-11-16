"""
실험 B·C용 질문 세트 (골든셋) — "질문 + 정답 + 정답 근거 기사"

파일: experiments/qa_set.csv (git에 올려서 팀이 같이 쓴다. 엑셀로 열고 고쳐도 된다)
실험 B · C · 유사도 확인은 이 CSV를 바로 읽는다. 다른 CSV로 돌리려면 --qa 파일.csv

  열 (첫 줄 = 열 이름. 한글 이름도 됨: 검색어 / 의미 / 질문 / 정답 / 근거 기사)
    query    필수  검색어                                   예: 강남
    question 필수  질문                                     예: 강남이 부른 드라마 OST 곡 제목이 뭐야?
    sense    선택  의미 이름 (DB topic과 같게)             예: 방송인 강남
                   비우면 읽을 때 DB에서 채운다: 그 검색어의 topic이 하나면 그것,
                   여러 개면 근거 기사가 '관련'으로 들어 있는 topic
    answer   선택  정답 (없으면 실험 C는 근거 기사만 보고 채점)
    evidence 선택  정답 근거 기사 link. 여러 개면 ; 로 구분 (없으면 실험 B · 유사도 확인에서 빠짐)
    id       선택  비우면 "검색어-001"처럼 붙인다

  읽을 때 DB를 보고(읽기만) 확인하는 것
    - 근거 기사가 중복 기사면 대표 기사 link도 근거로 넣는다 (검색 · 답변에는 대표 기사만 쓰인다)
    - 근거 기사가 그 topic에 없거나 '무관' 판정이면 [확인]으로 알려 준다 (검색에 안 나옴)
    - 의미를 정할 수 없거나 topic이 없는 줄은 [건너뜀]으로 알려 주고 뺀다

만들기: python evaluate.py make-qa [--target 강남] [--suggest]
  topic의 '관련' 기사를 하나씩 보여 주고, 그 기사로 답할 수 있는 질문과 정답을 입력한다 (CSV 끝에 추가).
  --suggest를 주면 LLM이 질문·정답 초안을 먼저 써 준다 (사람이 확인하고 Enter로 받거나 고쳐 쓴다).
  최근 사건을 묻는 질문을 넣으면 "LLM 단독 답변의 한계"(실험 C)가 잘 드러난다.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

QA_PATH = Path(__file__).resolve().parent / "qa_set.csv"
CSV_FIELDS = ["id", "query", "sense", "question", "answer", "evidence"]


@dataclass
class QAItem:
    id: str
    query: str
    sense: str
    question: str
    answer: str = ""                                    # 비어 있으면 실험 C는 근거 기사만 보고 채점
    evidence: list[str] = field(default_factory=list)   # 비어 있으면 실험 B · 유사도 확인에서 빠짐


def load_qa(path: Path | None = None, query: str | None = None, resolve: bool = True) -> list[QAItem]:
    """질문 세트 CSV를 읽는다.
    resolve=True: 빈 의미 채우기 · 대표 기사 link 넣기 · 확인할 것 출력 (DB 읽기). False: 파일에 적힌 그대로."""
    # 기본 경로는 부를 때 정한다 (정의할 때 정하면 테스트에서 경로를 바꿔도 실제 파일을 건드리게 됨)
    path = path or QA_PATH
    if not path.exists():
        return []
    rows = read_csv_rows(path)
    if resolve:
        # 실험은 질문 세트를 여러 번 읽는다 (main에서 확인, run에서 다시). DB 확인은 파일이 그대로면 한 번만 한다
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, query, id(db_lookup))
        if key not in _resolved:
            items, report = resolve_rows(rows, only_query=query)   # 검색어를 골랐으면 그 검색어 줄만 DB로 확인
            _resolved[key] = items
            report.print_once()
        items = list(_resolved[key])
    else:
        items = []
        for r in rows:
            items.append(QAItem(r["id"] or next_id(items, r["query"]), r["query"], r["sense"], r["question"],
                                r["answer"], split_links(r["evidence"])))
    return [x for x in items if query is None or x.query == query]


def save_qa(items: list[QAItem], path: Path | None = None) -> None:
    path = path or QA_PATH
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for x in items:
        writer.writerow({"id": x.id, "query": x.query, "sense": x.sense, "question": x.question,
                         "answer": x.answer, "evidence": "; ".join(x.evidence)})
    path.write_text(buf.getvalue(), encoding="utf-8-sig")    # utf-8-sig: 엑셀에서 한글이 안 깨지게


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
    from qupict.llm import get_structured_model
    try:
        llm = get_structured_model(_QADraft)
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
    """'관련' 기사를 하나씩 보여 주고 질문·정답을 받아 qa_set.csv에 추가한다. 추가한 개수를 돌려준다."""
    items = load_qa(resolve=False)         # 파일에 적힌 그대로 (DB로 채운 값을 덮어쓰지 않게)
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


# ---------------------------------------------------------------------------
# CSV 읽기 · DB로 의미 채우기
# ---------------------------------------------------------------------------

CSV_COLUMNS = {
    "id": ["id", "번호"],
    "query": ["query", "검색어"],
    "sense": ["sense", "의미", "의미이름", "sensename"],
    "question": ["question", "질문"],
    "answer": ["answer", "정답", "답"],
    "evidence": ["evidence", "근거", "근거기사", "링크", "link", "links"],
}
JUDGED_FOR_ANSWER = ("관련", "보류")   # 질문할 때 검색하는 판정 (이 밖의 근거 기사는 검색에 안 나온다)


@dataclass
class EvidenceRow:
    """근거 기사 link가 어느 topic에 어떤 판정으로 들어 있는지 (DB)."""
    link: str
    sense: str
    status: str | None
    representative: str | None     # 중복 기사면 대표 기사 link (검색에는 대표 기사만 나온다)
    via: str | None = None         # CSV에 기사 대신 이미지 주소를 적었으면 그 주소 (link = 그 이미지가 실린 기사)


_printed: set[str] = set()
_resolved: dict[tuple, list[QAItem]] = {}   # load_qa 결과 (파일 · 수정 시각 · 검색어별)


@dataclass
class ResolveReport:
    skipped: list[str] = field(default_factory=list)    # 뺀 줄과 이유
    warnings: list[str] = field(default_factory=list)   # 넣었지만 확인할 것

    def print_once(self) -> None:
        """실험이 질문 세트를 여러 번 읽어도 같은 안내는 한 번만 출력한다."""
        for tag, lines in (("건너뜀", self.skipped), ("확인", self.warnings)):
            for line in lines:
                if line not in _printed:
                    _printed.add(line)
                    print(f"  [질문 세트 {tag}] {line}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """CSV → [{id, query, sense, question, answer, evidence}] (열 이름은 CSV_COLUMNS로 맞춘다). 엑셀 CSV(cp949)도 읽는다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"{path}: UTF-8 · CP949로 읽을 수 없습니다.")
    reader = csv.DictReader(io.StringIO(text))
    alias = {a: key for key, names in CSV_COLUMNS.items() for a in names}
    mapping = {col: alias.get(re.sub(r"[\s_]", "", (col or "").lower())) for col in reader.fieldnames or []}
    missing = [k for k in ("query", "question") if k not in mapping.values()]
    if missing:
        raise ValueError(f"{path}: {', '.join(missing)} 열이 없습니다. 지금 열: {reader.fieldnames}")
    rows = []
    for raw in reader:
        row = {k: "" for k in CSV_COLUMNS}
        for col, key in mapping.items():
            if key and raw.get(col):
                row[key] = raw[col].strip()
        if any(row.values()):
            rows.append(row)
    return rows


def split_links(text: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in re.split(r"[;\n|,\s]+", text or "") if x.strip()))


def _normalize(text: str) -> str:
    return re.sub(r"[\s()\[\]·,./-]", "", text).lower()


def db_lookup(query: str, links: list[str]) -> tuple[list[str], list[EvidenceRow]]:
    """(그 검색어의 topic 의미 이름들, 근거 기사가 들어 있는 topic · 판정). DB는 읽기만 한다."""
    from qupict import db
    # 기사 대신 기사 속 이미지 주소를 적은 근거는 db.evidence_lookup이 그 이미지가 실린 기사로 바꿔 준다 (image_of)
    senses, rows, image_of = db.evidence_lookup(query, links)
    if not links:
        return senses, []
    found = [EvidenceRow(link, sense_name, status, rep, image_of.get(link)) for link, sense_name, status, rep in rows]
    # 이미지가 실린 기사가 이 검색어 topic에 없어도 기사 link로는 바꿔 둔다 (경고는 resolve_rows가 낸다)
    in_topic = {e.link for e in found}
    found += [EvidenceRow(art, "", None, None, img) for art, img in image_of.items() if art not in in_topic]
    return senses, found


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[\s()\[\]·,./-]+", text.lower()) if w}


def _closest_names(given: str, senses: list[str]) -> list[str]:
    """이름에 같은 단어가 가장 많은 의미들 (하나도 안 겹치면 [])."""
    scores = {s: len(_words(given) & _words(s)) for s in senses}
    best = max(scores.values(), default=0)
    return [s for s, n in scores.items() if n == best] if best else []


def _resolve_sense(query: str, given: str, senses: list[str], evidence: list[EvidenceRow]) -> tuple[str | None, str]:
    """(의미 이름, 이유). 의미를 정했으면 이유는 확인할 점(없으면 ""), 못 정했으면 건너뛰는 이유."""
    if not senses:
        return None, f"DB에 '{query}' topic이 없습니다 (먼저 python admin.py collect {query})"
    if given:
        for s in senses:
            if s == given or _normalize(s) == _normalize(given):
                return s, ""
    # sense가 비었거나 DB 의미 이름과 다르면 → 이름에 같은 단어가 가장 많은 의미 (예: '기업 애플' ↔ DB '기업 Apple')
    # → 근거 기사가 '관련' · '보류'로 든 topic → 하나뿐인 topic 순으로 정한다
    hit = _closest_names(given, senses) if given else []
    if len(hit) != 1:
        hit = list(dict.fromkeys(e.sense for e in evidence if e.sense and e.status in JUDGED_FOR_ANSWER))
    sense = hit[0] if len(hit) == 1 else senses[0] if len(senses) == 1 else None
    if sense is None:
        if given:
            return None, f"'{query}'에 의미 '{given}'이 없습니다 (DB 의미: {' / '.join(senses)})"
        return None, f"'{query}'은 의미가 여러 개라 sense 열을 채워 주세요 ({' / '.join(senses)})"
    return sense, f"의미 '{given}'이 DB에 없어 '{sense}'로 봤습니다" if given else ""


def resolve_rows(rows: list[dict[str, str]], lookup=None,
                 only_query: str | None = None) -> tuple[list[QAItem], ResolveReport]:
    """CSV 줄 → QAItem. 빈 의미를 DB로 채우고, 중복 기사인 근거에는 대표 기사 link를 더한다.
    lookup: (검색어, 근거 link들) → (의미 이름들, EvidenceRow들). 기본 = db_lookup (DB 읽기)"""
    lookup = lookup or db_lookup
    report = ResolveReport()
    items: list[QAItem] = []
    # DB는 검색어마다 한 번만 읽는다 (그 검색어 모든 줄의 근거 link를 한꺼번에). 줄마다 읽으면 연결만 수십 초~수 분
    by_query: dict[str, list[str]] = {}
    for row in rows:
        if row["query"] and only_query in (None, row["query"]):
            by_query.setdefault(row["query"], []).extend(split_links(row["evidence"]))
    cache: dict[str, tuple[list[str], list[EvidenceRow]]] = {}

    for line, row in enumerate(rows, start=2):                          # 1번 줄은 열 이름
        query, question = row["query"], row["question"]
        if only_query is not None and query != only_query:
            continue
        if not query or not question:
            report.skipped.append(f"{line}번 줄: 검색어나 질문이 비어 있습니다")
            continue
        links = split_links(row["evidence"])
        if query not in cache:
            cache[query] = lookup(query, list(dict.fromkeys(by_query[query])))
        senses, all_found = cache[query]
        found = [e for e in all_found if e.link in links or e.via in links]     # 이 줄의 근거만
        sense, why = _resolve_sense(query, row["sense"], senses, found)
        if sense is None:
            report.skipped.append(f"{line}번 줄: {why}")
            continue
        if why:
            report.warnings.append(f"{line}번 줄: {why}")

        # 이미지 주소 근거 → 그 이미지가 실린 기사 link
        article_of = {e.via: e.link for e in found if e.via}
        links = list(dict.fromkeys(article_of.get(link, link) for link in links))
        evidence = list(links)
        mine = {e.link: e for e in found if e.sense == sense}
        for link in links:
            e = mine.get(link)
            if e is None:
                report.warnings.append(f"{line}번 줄: 근거 기사가 '{query} → {sense}'에 없습니다 (검색에 안 나옴): {link}")
            elif e.status not in JUDGED_FOR_ANSWER:
                report.warnings.append(f"{line}번 줄: 근거 기사 판정이 '{e.status or '미판정'}'이라 검색에 안 나옵니다: {link}")
            if e and e.representative and e.representative not in evidence:
                evidence.append(e.representative)
        if not row["answer"] and not evidence:
            report.warnings.append(f"{line}번 줄: 정답 · 근거 기사가 모두 없어 실험 B · C에서 빠집니다")
        items.append(QAItem(row["id"] or next_id(items, query), query, sense, question, row["answer"], evidence))
    return items, report
