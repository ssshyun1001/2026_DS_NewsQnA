"""
DB 파이프라인 묶음 — 0~5단계를 topic(검색어 × 고른 의미) 단위로 이어서 실행한다

admin.py(서비스 DB 관리)와 evaluate.py(평가)의 명령들이 이 파일의 함수를 부른다.
(서비스 run.py는 이 파일을 거치지 않고 단계 함수를 직접 부른다)

  choose_topic(query)        검색어 → LLM 의미 후보에서 고르기 → topic 저장   [0단계, 새 대상 추가]
  stored_topics(query)       DB에 저장된 topic 목록 (일괄 명령의 대상)
  resolve_topic(record)      저장된 topic의 기준을 LLM 의미 후보로 다시 맞춘다 [0단계, LLM 비용은 처음 한 번]
  collect_target / filter / index                                           [1·2·3·4단계, 각 단계 함수 호출]
  topic_status()             topic별 진행 현황 (수집·중복·판정·라벨·청크 수)
  articles_to_label(...)     사람이 정답 라벨을 붙일 기사 목록

의미·별칭·연관어는 LLM이 만든 후보(DB의 query_senses에 캐시)만 쓴다. 사람이 적은 값으로 덮어쓰지 않는다.
각 단계는 "아직 안 한 것만" 처리하므로, 같은 명령을 여러 번 실행해도 안전하고 비용이 다시 들지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from qupict import db
from qupict.collect import CollectResult, collect_to_db
from qupict.filtering import FilterOutcome, dedup_topic, filter_topic
from qupict.indexing import IndexResult, index_topic
from qupict.schemas import SelectedSense
from qupict.sense import choose_sense_interactively, load_sense_candidates, select_sense_by_name

DEFAULT_PERIOD_DAYS = 30
LABEL_PER_TARGET = 150          # 대상당 정답 라벨 수: 중복 제거 후 최신 대표 기사 150건
COLLECT_MAX_PER_TARGET = 300    # 대상당 수집 상한: 본문 실패·중복을 빼고도 대표 기사가 150건 남게
                                # (강남 30일 검색 후보 약 295건 기준. 모자라면 collect-all이 알려 준다)


# ---------------------------------------------------------------------------
# 대상 = DB에 저장된 topic
# ---------------------------------------------------------------------------

def choose_topic(query: str) -> tuple[int, SelectedSense]:
    """새 대상 추가: LLM 의미 후보(DB 캐시)에서 터미널로 고르고 topic으로 저장한다. (topic id, 고른 의미)"""
    candidates, source = load_sense_candidates(query)
    print(f"[의미 후보] {source}")
    selected = choose_sense_interactively(candidates)
    return db.save_topic(selected), selected


def stored_topics(query: str | None = None) -> list[db.TopicRecord]:
    """DB에 저장된 topic 목록. query를 주면 그 검색어만 (예: "눈" → 눈-날씨, 눈-신체)."""
    with db.connect() as conn:
        rows = conn.execute("SELECT query, sense_name FROM topics ORDER BY query, sense_name").fetchall()
    records = [db.get_topic(r["query"], r["sense_name"]) for r in rows if query is None or r["query"] == query]
    if query and not records:
        raise ValueError(f"DB에 '{query}' 검색어의 topic이 없습니다. 먼저 python admin.py collect {query} 로 추가하세요.")
    return records


def topic_label(record: db.TopicRecord) -> str:
    return f"{record.selected.query} → {record.selected.sense.name}"


def resolve_topic(record: db.TopicRecord) -> tuple[int, SelectedSense]:
    """저장된 topic의 기준(정의·별칭·연관어·배제어)을 LLM 의미 후보 값으로 다시 맞춰 저장한다. (topic id, 고른 의미)

    예전에 사람이 적은 값(지금은 지운 targets.yaml)이 DB topic에 남아 있어도 여기서 LLM 값으로 바뀐다.
    후보에서 같은 이름을 못 찾으면(직접 입력한 의미 등) 저장된 기준을 그대로 쓴다.
    """
    candidates, _ = load_sense_candidates(record.selected.query)
    selected = select_sense_by_name(candidates, record.selected.sense.name)
    if selected is None or selected.sense.name != record.selected.sense.name:
        return record.id, record.selected
    return db.save_topic(selected), selected


# ---------------------------------------------------------------------------
# 단계 실행 (topic 하나)
# ---------------------------------------------------------------------------

@dataclass
class CollectTargetResult:
    """collect-all 한 대상의 결과: 수집 + 중복 제거 후 대표 기사 수 (라벨 150건이 되는지 바로 보이게)"""
    collected: CollectResult
    duplicates: int         # 이번에 새로 중복으로 표시한 기사
    representatives: int    # 중복이 아닌 기사 = 라벨 붙일 후보
    max_articles: int       # 이번 수집 상한

    @property
    def topic_id(self) -> int:
        return self.collected.topic_id

    def summary(self) -> str:
        text = (f"{self.collected.summary()} → 중복 {self.duplicates}건 표시, "
                f"대표 기사 {self.representatives}건")
        if self.representatives < LABEL_PER_TARGET:
            # 상한까지 찼으면 검색 결과는 더 있다 → 상한을 올린다. 상한보다 적게 찾았으면 기간 안 기사가 부족하다
            hint = "--max를 늘려" if self.collected.searched >= self.max_articles else "--days를 늘려"
            text += (f"  ⚠ 라벨 목표 {LABEL_PER_TARGET}건보다 {LABEL_PER_TARGET - self.representatives}건 적음"
                     f" ({hint} 다시 수집하면 이어서 채움)")
        return text


async def collect_target(selected: SelectedSense, period_days: int = DEFAULT_PERIOD_DAYS,
                         max_articles: int = COLLECT_MAX_PER_TARGET) -> CollectTargetResult:
    """수집 → 중복 제거 (임베딩만, LLM 판정 없음). 라벨 붙일 대표 기사가 몇 건인지 같이 돌려준다."""
    collected = await collect_to_db(selected, period_days, max_articles)
    duplicates = await dedup_topic(collected.topic_id)
    return CollectTargetResult(collected, len(duplicates), count_representatives(collected.topic_id), max_articles)


async def filter_target(record: db.TopicRecord, redo: bool = False) -> FilterOutcome:
    topic_id, selected = resolve_topic(record)
    return await filter_topic(topic_id, selected, redo=redo)


async def index_target(record: db.TopicRecord, describe_images: bool = True) -> IndexResult:
    return await index_topic(record.id, describe_images=describe_images)


# ---------------------------------------------------------------------------
# 현황 · 라벨링용 조회
#   db.py에 없는 읽기 전용 조회라 여기에 두었다 (DB 담당이 db.py로 옮겨도 된다)
# ---------------------------------------------------------------------------

@dataclass
class TopicStatus:
    topic_id: int
    query: str
    sense_name: str
    articles: int          # 이 topic에 연결된 기사
    duplicates: int        # 중복 표시
    unjudged: int          # 중복도 판정도 아직 없음
    rule_irrelevant: int   # 규칙으로 무관
    llm_relevant: int
    llm_irrelevant: int
    llm_hold: int
    labeled: int           # 사람이 정답 라벨을 붙인 기사 (중복 제외)
    to_label: int          # 라벨을 붙일 기사 (중복 제외)
    chunked: int           # 청크가 있는 기사


def topic_status() -> list[TopicStatus]:
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT t.id, t.query, t.sense_name,
                      count(at.article_id) AS articles,
                      count(at.duplicate_of) AS duplicates,
                      count(*) FILTER (WHERE at.article_id IS NOT NULL AND at.duplicate_of IS NULL
                                       AND at.status IS NULL) AS unjudged,
                      count(*) FILTER (WHERE at.stage = 'rule' AND at.status = '무관') AS rule_irrelevant,
                      count(*) FILTER (WHERE at.stage = 'llm' AND at.status = '관련') AS llm_relevant,
                      count(*) FILTER (WHERE at.stage = 'llm' AND at.status = '무관') AS llm_irrelevant,
                      count(*) FILTER (WHERE at.stage = 'llm' AND at.status = '보류') AS llm_hold,
                      count(*) FILTER (WHERE at.duplicate_of IS NULL AND at.label IS NOT NULL) AS labeled,
                      count(*) FILTER (WHERE at.article_id IS NOT NULL AND at.duplicate_of IS NULL) AS to_label,
                      count(*) FILTER (WHERE EXISTS (SELECT 1 FROM article_chunks c
                                                     WHERE c.article_id = at.article_id)) AS chunked
               FROM topics t LEFT JOIN article_topics at ON at.topic_id = t.id
               GROUP BY t.id ORDER BY t.query, t.sense_name"""
        ).fetchall()
    return [TopicStatus(r["id"], r["query"], r["sense_name"], *(r[k] for k in (
        "articles", "duplicates", "unjudged", "rule_irrelevant", "llm_relevant", "llm_irrelevant",
        "llm_hold", "labeled", "to_label", "chunked"))) for r in rows]


@dataclass
class LabelItem:
    link: str
    title: str
    press: str | None
    pub_date: datetime | None
    search_term: str
    description: str
    content: str


def count_representatives(topic_id: int) -> int:
    """중복이 아닌 기사 수 (= 라벨 붙일 후보)."""
    with db.connect() as conn:
        r = conn.execute("SELECT count(*) AS n FROM article_topics WHERE topic_id = %s AND duplicate_of IS NULL",
                         (topic_id,)).fetchone()
    return r["n"]


def articles_to_label(topic_id: int, include_labeled: bool = False,
                      limit: int | None = None) -> list[LabelItem]:
    """정답 라벨을 붙일 기사 (중복으로 표시된 기사는 대표 기사만 보면 되므로 뺀다). 발행일 최신순.
    limit: 최신 기사부터 이 개수만 (라벨 150건 목표에서 남은 개수)"""
    with db.connect() as conn:
        rows = conn.execute(
            f"""SELECT a.link, a.title, a.press, a.pub_date, at.search_term, a.description, a.content
                FROM article_topics at JOIN articles a ON a.id = at.article_id
                WHERE at.topic_id = %s AND at.duplicate_of IS NULL
                      {'' if include_labeled else 'AND at.label IS NULL'}
                ORDER BY a.pub_date DESC NULLS LAST, a.id
                {'LIMIT %s' if limit is not None else ''}""",
            (topic_id, limit) if limit is not None else (topic_id,),
        ).fetchall()
    return [LabelItem(r["link"], r["title"], r["press"], r["pub_date"], r["search_term"],
                      r["description"], r["content"]) for r in rows]
