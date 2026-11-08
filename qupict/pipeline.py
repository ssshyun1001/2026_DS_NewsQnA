"""
DB 파이프라인 묶음 — 0~5단계를 topic(검색어 × 고른 의미) 단위로 이어서 실행한다

run.py의 명령들이 이 파일의 함수를 부른다.

  resolve_target(target)     targets.yaml 항목 → (topic id, 고른 의미)   [0단계, LLM 비용은 처음 한 번]
  collect_target / filter / index                                         [1·2·3·4단계, 각 단계 함수 호출]
  topic_status()             topic별 진행 현황 (수집·중복·판정·라벨·청크 수)
  articles_to_label(...)     사람이 정답 라벨을 붙일 기사 목록

각 단계는 "아직 안 한 것만" 처리하므로, 같은 명령을 여러 번 실행해도 안전하고 비용이 다시 들지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from qupict import db
from qupict.collect import CollectResult, collect_to_db
from qupict.config import load_targets
from qupict.filtering import FilterOutcome, dedup_topic, filter_topic
from qupict.indexing import IndexResult, index_topic
from qupict.schemas import SelectedSense
from qupict.sense import load_sense_candidates, select_for_target, select_sense_by_name

DEFAULT_PERIOD_DAYS = 30
LABEL_PER_TARGET = 150          # 대상당 정답 라벨 수: 중복 제거 후 최신 대표 기사 150건
COLLECT_MAX_PER_TARGET = 300    # 대상당 수집 상한: 본문 실패·중복을 빼고도 대표 기사가 150건 남게
                                # (강남 30일 검색 후보 약 295건 기준. 모자라면 collect-all이 알려 준다)


# ---------------------------------------------------------------------------
# 대상(targets.yaml) → topic
# ---------------------------------------------------------------------------

def pick_targets(query: str | None = None) -> list[dict]:
    """targets.yaml의 대상 목록. query를 주면 그 검색어만 (예: "눈" → 눈-날씨, 눈-신체)."""
    targets = [t for t in load_targets() if not str(t["sense"]).startswith("TODO")]
    if query:
        targets = [t for t in targets if t["query"] == query]
        if not targets:
            raise ValueError(f"targets.yaml에 '{query}' 검색어가 없습니다.")
    return targets


def target_label(target: dict) -> str:
    return f"{target['query']} → {target['sense']}"


def resolve_target(target: dict) -> tuple[int, SelectedSense]:
    """targets.yaml 항목으로 의미를 고르고 topic으로 저장한다. (topic id, 고른 의미)

    - 의미 후보는 DB 캐시에서 가져온다 (처음 한 번만 LLM)
    - 직접 입력 의미(예: 베라 → 배스킨라빈스)는 이전에 만든 topic이 있으면 그 의미를 그대로 쓴다 (LLM 다시 안 부름)
    - targets.yaml에 사람이 적은 정의·별칭·연관어는 매번 반영된다
    """
    candidates, _ = load_sense_candidates(target["query"])
    known_custom = None
    if select_sense_by_name(candidates, target["sense"]) is None:
        record = db.get_topic(target["query"], str(target["sense"]).strip())
        if record is not None:
            known_custom = record.selected.sense
    selected = select_for_target(candidates, target, known_custom=known_custom)
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


async def collect_target(target: dict, period_days: int = DEFAULT_PERIOD_DAYS,
                         max_articles: int = COLLECT_MAX_PER_TARGET) -> CollectTargetResult:
    """수집 → 중복 제거 (임베딩만, LLM 판정 없음). 라벨 붙일 대표 기사가 몇 건인지 같이 돌려준다."""
    _, selected = resolve_target(target)
    collected = await collect_to_db(selected, period_days, max_articles)
    duplicates = await dedup_topic(collected.topic_id)
    return CollectTargetResult(collected, len(duplicates), count_representatives(collected.topic_id), max_articles)


async def filter_target(target: dict, redo: bool = False) -> FilterOutcome:
    topic_id, selected = resolve_target(target)
    return await filter_topic(topic_id, selected, redo=redo)


async def index_target(target: dict, describe_images: bool = True) -> IndexResult:
    topic_id, _ = resolve_target(target)
    return await index_topic(topic_id, describe_images=describe_images)


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
