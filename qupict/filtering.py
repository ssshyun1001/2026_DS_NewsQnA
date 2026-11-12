"""
[2·3단계] 필터링 — 중복 제거 → 1차 규칙 필터 → 2차 LLM 검증

0단계에서 고른 의미(SelectedSense)를 기준으로, 수집한 기사가 그 대상을 실제로 다루는지 판정한다.

  [2단계] 중복 제거      같은 사안을 다룬 비슷한 기사는 대표 기사 1건만 남긴다
                         기준 DEDUP_BY (쉼표로 여러 개):
                           title    제목 임베딩 유사도                  (DEDUP_TITLE_THRESHOLD)
                           content  본문 앞 N문장 임베딩 유사도          (DEDUP_CONTENT_THRESHOLD, N = DEDUP_LEAD_SENTENCES)
                           noun     제목의 드문 고유명사 겹침 (TF-IDF)   (DEDUP_NOUN_THRESHOLD)
                         여러 기준을 묶는 방법 DEDUP_RULE: "and"(모두 넘어야 중복) | "or"(하나만 넘어도)
  [2단계] 1차 규칙 필터  분명히 무관한 기사만 규칙으로 걸러 LLM 호출을 줄인다      → stage = "rule"
  [3단계] 2차 LLM 검증   나머지를 LLM이 관련 / 무관으로 판정하고 근거 한 줄을 남긴다 → stage = "llm"

  filter_articles(articles, selected)   메모리 버전 (run.py가 사용)
  filter_topic(topic_id, selected)      DB 버전 (DB의 기사를 판정하고 결과를 article_topics에 저장)
  rule_check(article, selected)         규칙 점수만 따로 (실험 A의 "방법 1-c"로도 씀)

## 1차 규칙 필터 (기획안 방법 1-c)

  점수 = 연관어 점수 − 배제어 점수     (단어마다 제목에 있으면 2점, 본문에만 있으면 1점)
    연관어 = 고른 의미의 related_terms                 예: 방송인 강남 → 이상화, 예능, 유튜브
    배제어 = 고르지 않은 의미의 연관어 (excluded_terms) 예: 지역 강남 → 강남역, 재건축, 압구정

  무관 (rule)  기사에 검색어·별칭이 아예 없다          예: "서울여대" 검색에 걸린 성신여대 기사
  무관 (rule)  점수 ≤ RULE_REJECT_SCORE (기본 -2)      다른 의미의 단어가 분명히 더 많다
  나머지       → LLM으로 넘긴다

  규칙은 "분명히 아닌 것"만 거른다. 관련 판정은 규칙으로 하지 않는다 (단어만으로는 단순 언급과
  실제 주제를 구분할 수 없어서). 기준값은 정답 라벨이 생기면 실험으로 조정한다.

## 2차 LLM 검증

  LLM에게 고른 의미(정의·별칭·연관어)와 고르지 않은 의미들을 함께 보여 주고 대조하게 한다.
    관련  찾는 대상이 기사의 주요 소재다
    무관  다른 의미의 기사이거나, 찾는 대상이 스쳐 지나가듯 언급만 된다
    보류 없음: 애매해도 관련 / 무관 중 더 가까운 쪽으로 반드시 판정한다 (예전에는 보류가 있었지만 한 번도 나오지 않았다)
  LLM이 답하지 못한 기사는 판정하지 않고 남겨 둔다 (DB 버전은 다음 실행 때 다시 시도).
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from qupict import timing
from qupict.config import settings
from qupict.concurrency import run_parallel
from qupict.llm import aembed_in_batches, get_embeddings, get_structured_model
from qupict.schemas import Article, SelectedSense

RULE_REJECT_SCORE = -2          # 규칙 점수가 이 값 이하면 '무관' (정답 라벨이 생기면 실험으로 조정)
LLM_BATCH_SIZE = 8              # LLM 한 번에 판정할 기사 수
# 동시에 보내는 LLM 요청 수 (게이트웨이 한도: 60초 120회). 4 → 8: 기사 100건이면 요청 13번 안팎이라
# 8개씩 보내도 1분에 수십 번 수준. 배치 나누는 방식은 그대로라 각 요청의 입력(= 판정 결과)은 예전과 같다.
LLM_CONCURRENCY = 8
LLM_BODY_CHARS = 800            # LLM에 보여 줄 본문 앞부분 길이
EMBED_CHARS = 2000              # 기사 임베딩에 쓰는 본문 길이 (임베딩 입력 한도 안쪽, 비교에는 충분)
DEDUP_CRITERIA = ("title", "content", "noun")
DEDUP_RULES = ("and", "or")

STATUSES = ("관련", "무관", "보류")


@dataclass
class Verdict:
    """기사 하나의 판정. (DB의 db.Judgment와 같은 내용)"""
    link: str
    status: str                  # '관련' | '무관' | '보류'
    stage: str                   # 'rule' | 'llm'
    reason: str = ""
    model: Optional[str] = None  # LLM 판정이면 모델명


@dataclass
class FilterOutcome:
    """필터링 결과 한 묶음."""
    duplicates: dict[str, str] = field(default_factory=dict)  # {중복 기사 link: 대표 기사 link}
    verdicts: list[Verdict] = field(default_factory=list)
    unjudged: list[str] = field(default_factory=list)          # LLM이 답하지 못한 기사 link

    def links_with(self, *statuses: str) -> set[str]:
        return {v.link for v in self.verdicts if v.status in statuses}

    def summary(self) -> str:
        count = lambda stage, status: sum(1 for v in self.verdicts if v.stage == stage and v.status == status)
        return (f"중복 {len(self.duplicates)}건 | 규칙 무관 {count('rule', '무관')}건 | "
                f"LLM 관련 {count('llm', '관련')} · 무관 {count('llm', '무관')} · 보류 {count('llm', '보류')}건"
                + (f" | 판정 못 함 {len(self.unjudged)}건" if self.unjudged else ""))


# ---------------------------------------------------------------------------
# [2단계] 중복 제거
# ---------------------------------------------------------------------------

async def embed_articles(articles: list[Article]) -> list[Article]:
    """기사 임베딩을 채운다. 이미 있는 기사(DB에 저장된 것 포함)는 다시 계산하지 않는다."""
    to_embed = [a for a in articles if a.embedding is None]
    if to_embed:
        texts = [f"{a.title}\n{a.content[:EMBED_CHARS]}" for a in to_embed]
        vectors = await aembed_in_batches(get_embeddings(), texts)   # 100건씩 나눈 요청을 동시에
        for article, vec in zip(to_embed, vectors):
            article.embedding = vec
    return articles


_SENTENCE_END = re.compile(r"(?<=[.!?。])\s+|(?<=[.!?。][\"'”’)\]])\s+|\n+")
# 문장 끝: . ! ? 뒤(닫는 따옴표·괄호 하나까지 포함)의 공백, 또는 줄바꿈. "3.5%"처럼 공백 없는 점은 안 나눈다


def lead_text(article: Article, n: int | None = None) -> str:
    """본문 앞 n문장 (중복 제거 content 기준). 문장 끝(. ! ?) 뒤 공백이나 줄바꿈으로 나눈다.
    본문이 비어 있으면 제목을 쓴다 (빈 문자열은 임베딩할 수 없으므로)."""
    n = settings.dedup_lead_sentences if n is None else n
    sentences = [x.strip() for x in _SENTENCE_END.split(article.content or "") if x and x.strip()]
    return " ".join(sentences[:n]) or article.title


async def embed_leads(articles: list[Article]) -> list[Article]:
    """본문 앞 n문장으로 임베딩을 채운다 (중복 제거용). DB에 저장하지 않으므로 실행할 때마다 계산한다."""
    to_embed = [a for a in articles if a.lead_embedding is None]
    if to_embed:
        vectors = await aembed_in_batches(get_embeddings(), [lead_text(a) for a in to_embed])
        for article, vec in zip(to_embed, vectors):
            article.lead_embedding = vec
    return articles


async def embed_titles(articles: list[Article]) -> list[Article]:
    """제목만으로 임베딩을 채운다 (중복 제거용). DB에 저장하지 않으므로 실행할 때마다 계산한다 (제목이라 저렴)."""
    to_embed = [a for a in articles if a.title_embedding is None]
    if to_embed:
        vectors = await aembed_in_batches(get_embeddings(), [a.title for a in to_embed])
        for article, vec in zip(to_embed, vectors):
            article.title_embedding = vec
    return articles


def _dedup_criteria(by: str | None = None) -> list[str]:
    """DEDUP_BY → 비교할 기준 목록. "title,content,noun"처럼 쉼표로 여러 개, "both" = title,content, "all" = 셋 다."""
    raw = (by or settings.dedup_by).strip().lower()
    alias = {"both": "title,content", "all": ",".join(DEDUP_CRITERIA)}
    criteria = [c.strip() for c in alias.get(raw, raw).split(",") if c.strip()]
    if not criteria or any(c not in DEDUP_CRITERIA for c in criteria):
        raise ValueError(f"DEDUP_BY는 {DEDUP_CRITERIA} 중 하나 이상(쉼표로 구분) 또는 both/all이어야 합니다: {raw!r}")
    return list(dict.fromkeys(criteria))


def _dedup_rule(rule: str | None = None) -> str:
    rule = (rule or settings.dedup_rule).strip().lower()
    if rule not in DEDUP_RULES:
        raise ValueError(f"DEDUP_RULE은 {DEDUP_RULES} 중 하나여야 합니다: {rule!r}")
    return rule


_EMBEDDING_OF = {
    "title": lambda a: a.title_embedding,    # 제목만
    "content": lambda a: a.lead_embedding,   # 본문 앞 n문장
}


async def embed_for_dedup(articles: list[Article], by: str | None = None) -> list[Article]:
    """find_duplicates에 필요한 임베딩을 채운다 (title: 제목 / content: 본문 앞 n문장, noun은 임베딩 불필요)."""
    criteria = _dedup_criteria(by)
    jobs = []
    if "title" in criteria:
        jobs.append(embed_titles(articles))
    if "content" in criteria:
        jobs.append(embed_leads(articles))
    await asyncio.gather(*jobs)   # 제목 · 본문 앞 문장 임베딩 요청을 동시에
    return articles


# --- 고유명사 겹침 (noun 기준) ---------------------------------------------------
#  기사 제목에서 고유명사(NNP)·영문 단어(SL)를 뽑아 TF-IDF로 가중치를 매기고 코사인 유사도를 잰다.
#  IDF는 이번에 비교하는 기사 묶음 안에서 계산한다:
#    거의 모든 기사에 나오는 단어(검색어 "강남" 등)  → 가중치 낮음
#    몇몇 기사에만 나오는 드문 고유명사(인물·장소·행사명) → 가중치 높음
#  → 드문 고유명사를 제목에 같이 쓰는 기사끼리 유사도가 높아진다 (같은 사건을 다룬 기사).
#  제목에 고유명사가 하나도 없으면 이 기준에서는 어떤 기사와도 비슷하지 않다 (유사도 0).

_NOUN_TAGS = ("NNP", "SL")
_kiwi = None


def extract_nouns(text: str) -> list[str]:
    """고유명사(NNP)와 영문 단어(SL, 소문자로)를 등장 순서대로 (중복 포함). 한 글자는 뺀다."""
    global _kiwi
    if _kiwi is None:
        try:
            from kiwipiepy import Kiwi
        except ImportError as e:
            raise ImportError("고유명사 기준(noun) 중복 제거에는 kiwipiepy가 필요합니다: pip install kiwipiepy") from e
        _kiwi = Kiwi()
    return [t.form.lower() if t.tag == "SL" else t.form
            for t in _kiwi.tokenize(text or "") if t.tag in _NOUN_TAGS and len(t.form) >= 2]


def noun_text(article: Article) -> str:
    """고유명사를 뽑을 텍스트: 제목만 (본문에 스쳐 지나가듯 나온 이름은 섞지 않는다)."""
    return article.title


def noun_tfidf(articles: list[Article]) -> dict[str, np.ndarray]:
    """기사 묶음의 고유명사 TF-IDF 벡터 {link: 벡터}.
    tf = 1 + log(등장 횟수), idf = log((1 + 기사 수) / (1 + 그 단어가 나온 기사 수)) + 1"""
    tokens = {a.link: extract_nouns(noun_text(a)) for a in articles}
    vocab = {w: i for i, w in enumerate(dict.fromkeys(w for ws in tokens.values() for w in ws))}
    df = np.zeros(len(vocab))
    for ws in tokens.values():
        for w in set(ws):
            df[vocab[w]] += 1
    idf = np.log((1 + len(tokens)) / (1 + df)) + 1
    vectors = {}
    for link, ws in tokens.items():
        v = np.zeros(len(vocab))
        for w in ws:
            v[vocab[w]] += 1
        nz = v > 0
        v[nz] = (1 + np.log(v[nz])) * idf[nz]
        vectors[link] = v
    return vectors


def _unit(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    return vectors / norm


def find_duplicates(
    articles: list[Article],
    existing: list[Article] = (),
    keep: set[str] | frozenset[str] = frozenset(),
    by: str | None = None,
    title_threshold: float | None = None,
    content_threshold: float | None = None,
    noun_threshold: float | None = None,
    rule: str | None = None,
) -> dict[str, str]:
    """유사도가 임계값 이상인 기사를 중복으로 묶는다. {중복 기사 link: 대표 기사 link}

    - by: 비교 기준 (쉼표로 여러 개). 비우면 settings.dedup_by.
        title    제목 임베딩
        content  본문 앞 n문장 임베딩
        noun     제목 고유명사 TF-IDF (드문 고유명사를 같이 쓰는 기사일수록 비슷함, 임베딩 불필요)
        both = title,content / all = title,content,noun
      임베딩은 미리 채워 둬야 한다 (embed_for_dedup). 필요한 임베딩이 없는 기사는 비교에서 빠진다.
    - *_threshold: 기준별 임계값. 비우면 settings 값.
    - rule: 기준이 여러 개일 때 "and"(모두 임계값 이상이어야 중복) | "or"(하나라도 이상이면 중복).
      비슷한 대표 기사가 여럿이면 (기준별 유사도 평균이) 가장 높은 대표 기사에 묶는다.

    - 대표 기사는 본문이 가장 긴 기사 (정보가 가장 많다고 본다)
    - existing: 이미 판정을 마친 대표 기사들 (DB 버전에서 새 기사만 처리할 때).
      이 기사들은 항상 대표로 남는다. 새 기사가 이 중 하나와 비슷하면 새 기사가 중복이 된다.
      (이미 판정한 기사를 중복으로 바꾸면 판정을 다시 해야 하므로)
    - keep: articles 중 절대 중복으로 바꾸면 안 되는 기사 link (사람이 이미 정답 라벨을 붙인 기사).
      중복이 되면 실험 A에서 빠져서 라벨이 버려지므로, 이 기사들은 항상 대표로 남긴다.
    """
    criteria, rule = _dedup_criteria(by), _dedup_rule(rule)
    thresholds = np.array([[{
        "title": settings.dedup_title_threshold if title_threshold is None else title_threshold,
        "content": settings.dedup_content_threshold if content_threshold is None else content_threshold,
        "noun": settings.dedup_noun_threshold if noun_threshold is None else noun_threshold,
    }[c]] for c in criteria])
    has_vectors = lambda a: all(_EMBEDDING_OF[c](a) is not None for c in criteria if c in _EMBEDDING_OF)
    existing = [a for a in existing if has_vectors(a)]
    articles = [a for a in articles if has_vectors(a)]

    # 기준별 단위 벡터 {link: 벡터}를 미리 만든다 (코사인 유사도 = 단위 벡터 내적)
    pool = existing + articles
    unit: dict[str, dict[str, np.ndarray]] = {}
    for c in criteria:
        raw = noun_tfidf(pool) if c == "noun" else {a.link: np.asarray(_EMBEDDING_OF[c](a), dtype=float) for a in pool}
        links = list(raw)
        mat = _unit(np.array([raw[l] for l in links])) if links else np.zeros((0, 0))
        unit[c] = dict(zip(links, mat))

    reps: list[Article] = list(existing)
    duplicates: dict[str, str] = {}
    # 라벨 붙은 기사를 먼저 대표로 세우고, 나머지는 긴 기사부터 대표 후보로 본다
    # → 비슷한 기사 무리에서 (라벨 붙은 기사가 없으면) 가장 긴 기사가 대표가 된다
    ordered = sorted(articles, key=lambda a: (a.link not in keep, -len(a.content)))
    for article in ordered:
        if reps and article.link not in keep:
            sims = np.array([   # [기준 수, 대표 수]
                np.array([unit[c][r.link] for r in reps]) @ unit[c][article.link] for c in criteria
            ])
            passed = sims >= thresholds
            matched = passed.all(axis=0) if rule == "and" else passed.any(axis=0)
            if matched.any():
                score = np.where(matched, sims.mean(axis=0), -np.inf)
                duplicates[article.link] = reps[int(np.argmax(score))].link
                continue
        reps.append(article)
    return duplicates


# ---------------------------------------------------------------------------
# [2단계] 1차 규칙 필터 (방법 1-c)
# ---------------------------------------------------------------------------

def _normalize(text: str | None) -> str:
    """대소문자·공백·특수문자를 없앤다. "Apple Inc." → "appleinc", "M.I.B" → "mib" """
    return re.sub(r"[^0-9a-z가-힣]", "", (text or "").lower())


@dataclass
class RuleResult:
    score: int
    mentioned: bool              # 검색어나 별칭이 기사에 있는가
    related_hits: list[str]      # 기사에서 찾은 연관어
    excluded_hits: list[str]     # 기사에서 찾은 배제어


def _term_score(terms: list[str], title: str, body: str) -> tuple[int, list[str]]:
    score, hits = 0, []
    for term in terms:
        t = _normalize(term)
        if not t:
            continue
        if t in title:
            score += 2
            hits.append(term)
        elif t in body:
            score += 1
            hits.append(term)
    return score, hits


def rule_check(article: Article, selected: SelectedSense) -> RuleResult:
    """규칙 점수 = 연관어 점수 − 배제어 점수 (제목 2점, 본문 1점)."""
    title, body = _normalize(article.title), _normalize(article.content)
    mentioned = any(_normalize(t) and _normalize(t) in title + body for t in selected.match_terms)
    plus, related_hits = _term_score(selected.sense.related_terms, title, body)
    minus, excluded_hits = _term_score(selected.excluded_terms, title, body)
    return RuleResult(plus - minus, mentioned, related_hits, excluded_hits)


def rule_verdict(article: Article, selected: SelectedSense) -> Verdict | None:
    """분명히 무관하면 Verdict('무관', 'rule'), 아니면 None (→ LLM으로)."""
    r = rule_check(article, selected)
    if not r.mentioned:
        return Verdict(article.link, "무관", "rule", f"검색어·별칭({', '.join(selected.match_terms)})이 기사에 없음")
    if r.score <= RULE_REJECT_SCORE:
        return Verdict(article.link, "무관", "rule",
                       f"다른 의미 단어가 더 많음 (점수 {r.score}: {', '.join(r.excluded_hits[:4])})")
    return None


# ---------------------------------------------------------------------------
# [3단계] 2차 LLM 검증
# ---------------------------------------------------------------------------

class _LLMVerdict(BaseModel):
    # 필드 순서가 중요하다: LLM은 이 순서대로 답을 쓴다. 판정(status)을 먼저 쓰게 하면
    # 근거와 판정이 엇갈리는 경우가 있었다 (실제: 근거는 "지역 의미로 쓰임, 무관 쪽"인데 판정은 '관련').
    # 근거를 먼저 쓰고 그에 맞춰 판정하게 한다.
    index: int = Field(description="기사 번호")
    reason: str = Field(description="판단 근거 한 줄 (40자 이내, 근거가 된 단어나 내용)")
    # 보류는 받지 않는다: 관련 / 무관 중 하나로 반드시 판정하게 한다 (실제로도 보류는 한 번도 나오지 않았다).
    # DB·검색 쪽의 '보류'(STATUSES 등)는 예전 판정 · 사람 라벨과 맞추려고 그대로 둔다.
    status: Literal["관련", "무관"] = Field(description="위 근거에 따른 판정")


class _LLMVerdictBatch(BaseModel):
    verdicts: list[_LLMVerdict]


_VERIFY_SYSTEM_PROMPT = """\
너는 뉴스 기사가 사용자가 고른 대상을 실제로 다루는지 판정한다.

[찾는 대상]
검색어: {query}
의미: {name} — {definition}
다른 표기: {aliases}
관련 단어: {related}

[헷갈리기 쉬운 다른 의미 — 이 의미의 기사는 '무관']
{others}

판정 기준:
- 관련: 찾는 대상이 기사의 주요 소재이거나, 기사에서 의미 있게 다뤄진다.
- 무관: 다른 의미의 기사이거나, 찾는 대상이 한두 줄 스쳐 지나가듯 언급만 된다.

규칙:
- 모든 기사를 반드시 '관련' 또는 '무관' 중 하나로 판정하라. 판단이 애매해도 둘 중 더 가까운 쪽을 골라라.
- 검색어 글자가 들어 있다는 이유만으로 '관련'이라고 하지 마라. 어떤 의미로 쓰였는지 봐라.
- reason은 한국어 한 줄(40자 이내)로, 판단 근거가 된 단어나 내용을 적어라.
- 입력된 모든 기사 번호에 대해 하나씩 답하라.
"""


def _verify_prompt(selected: SelectedSense) -> str:
    s = selected.sense
    others = "\n".join(f"- {o.name}: {o.definition} (관련 단어: {', '.join(o.related_terms[:6])})"
                       for o in selected.other_senses) or "- (없음)"
    return _VERIFY_SYSTEM_PROMPT.format(
        query=selected.query, name=s.name, definition=s.definition,
        aliases=", ".join(s.aliases) or "(없음)", related=", ".join(s.related_terms) or "(없음)",
        others=others,
    )


def _articles_block(articles: list[Article]) -> str:
    return "\n\n".join(
        f"[{i}] 제목: {a.title}\n요약: {a.description}\n본문: {a.content[:LLM_BODY_CHARS]}"
        for i, a in enumerate(articles)
    )


async def llm_verify(articles: list[Article], selected: SelectedSense) -> tuple[list[Verdict], list[str]]:
    """LLM으로 관련 / 무관을 판정한다 (보류 없음). (판정 목록, 판정 못 한 기사 link 목록)"""
    if not articles:
        return [], []
    llm = get_structured_model(_LLMVerdictBatch)
    system = _verify_prompt(selected)
    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

    async def judge(batch: list[Article]) -> list[Verdict]:
        async with semaphore:
            try:
                result: _LLMVerdictBatch = await llm.ainvoke(
                    [("system", system), ("human", f"아래 기사들을 판정하라.\n\n{_articles_block(batch)}")]
                )
            except Exception as e:
                print(f"  [경고] LLM 판정 실패 ({type(e).__name__}) — 기사 {len(batch)}건은 판정하지 않음")
                return []
        verdicts, seen = [], set()
        for v in result.verdicts:
            if 0 <= v.index < len(batch) and v.index not in seen:   # 없는 번호·중복 답은 무시
                seen.add(v.index)
                verdicts.append(Verdict(batch[v.index].link, v.status, "llm",
                                        v.reason.strip()[:200], settings.chat_model))
        return verdicts

    batches = [articles[i:i + LLM_BATCH_SIZE] for i in range(0, len(articles), LLM_BATCH_SIZE)]
    verdicts = [v for vs in await asyncio.gather(*(judge(b) for b in batches)) for v in vs]
    judged = {v.link for v in verdicts}
    return verdicts, [a.link for a in articles if a.link not in judged]


# ---------------------------------------------------------------------------
# 전체 흐름
# ---------------------------------------------------------------------------

async def filter_articles(
    articles: list[Article],
    selected: SelectedSense,
    existing: list[Article] = (),
    keep: set[str] | frozenset[str] = frozenset(),
) -> FilterOutcome:
    """[메모리 버전] 중복 제거 → 규칙 → LLM. 기사 목록은 바꾸지 않고 판정 결과만 돌려준다.
    keep: 중복으로 바꾸지 않을 기사 link (정답 라벨 붙은 기사, find_duplicates 참고)"""
    with timing.timed("새 기사 임베딩") as t:
        t.note = f"{sum(a.embedding is None for a in articles)}건 계산"
        await embed_articles(articles)
    with timing.timed("중복 제거용 임베딩(제목 · 본문 앞 문장)") as t:
        t.note = f"기준 {','.join(_dedup_criteria())}"
        await embed_for_dedup(list(articles) + list(existing))
    with timing.timed("중복 비교") as t:
        duplicates = find_duplicates(articles, existing, keep=keep)
        t.note = f"기사 {len(articles)}건 vs 대표 {len(existing)}건 → 중복 {len(duplicates)}건"
    candidates = [a for a in articles if a.link not in duplicates]

    rule_verdicts, to_llm = [], []
    with timing.timed("1차 규칙 필터") as t:
        for a in candidates:
            v = rule_verdict(a, selected)
            if v:
                rule_verdicts.append(v)
            else:
                to_llm.append(a)
        t.note = f"무관 {len(rule_verdicts)}건, LLM으로 {len(to_llm)}건"

    with timing.timed("2차 LLM 판정") as t:
        llm_verdicts, unjudged = await llm_verify(to_llm, selected)
        batches = -(-len(to_llm) // LLM_BATCH_SIZE)
        t.note = (f"{len(to_llm)}건, 요청 {batches}번 (배치 {LLM_BATCH_SIZE}, 동시 {LLM_CONCURRENCY})"
                  f"{f', 실패 {len(unjudged)}건' if unjudged else ''}")
    return FilterOutcome(duplicates, rule_verdicts + llm_verdicts, unjudged)


def _labeled_links(topic_id: int) -> set[str]:
    """사람이 정답 라벨을 붙인 기사 link (중복으로 바꾸지 않게 지킬 기사)."""
    from qupict import db
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT a.link FROM article_topics at JOIN articles a ON a.id = at.article_id
               WHERE at.topic_id = %s AND at.label IS NOT NULL""",
            (topic_id,),
        ).fetchall()
    return {r["link"] for r in rows}


async def dedup_topic(topic_id: int) -> dict[str, str]:
    """[DB 버전, 2단계만] 판정 없이 중복 표시만 한다. 라벨 붙이기 전에 쓴다 (python evaluate.py label).

    - LLM을 부르지 않는다. 기사 임베딩만 계산한다 (이미 DB에 있으면 다시 계산 안 함, 크레딧 거의 0).
    - 아직 판정하지 않은 기사끼리, 그리고 이미 판정한 대표 기사와 비교한다 (filter_topic과 같은 기준).
    - 대표 기사는 판정하지 않은 채로 남는다 → 나중에 filter-all이 대표 기사만 규칙·LLM으로 판정한다.
    - 여러 번 실행해도 결과가 같다. 라벨 붙은 기사는 중복으로 바뀌지 않는다.
    돌려주는 값: 이번에 새로 중복으로 표시한 {중복 기사 link: 대표 기사 link}
    """
    from qupict import db

    new = db.get_topic_articles(topic_id, unjudged_only=True)
    if not new:
        return {}
    existing = db.get_topic_articles(topic_id, statuses=STATUSES, include_duplicates=False)
    missing = [a for a in new + existing if a.embedding is None]
    await embed_articles(new + existing)
    await embed_for_dedup(new + existing)
    duplicates = find_duplicates(new, existing, keep=_labeled_links(topic_id))
    db.save_article_embeddings(missing)
    db.save_duplicates(topic_id, duplicates)
    return duplicates


async def filter_topic(
    topic_id: int,
    selected: SelectedSense,
    period_days: int | None = None,
    redo: bool = False,
) -> FilterOutcome:
    """[DB 버전] topic의 아직 판정하지 않은 기사를 필터링하고 결과를 article_topics에 저장한다.

    - 새로 들어온 기사만 처리한다. 이미 판정한 대표 기사와 비슷한 새 기사는 중복으로 표시한다.
    - 기사 임베딩은 DB에 저장해 두고 다시 계산하지 않는다.
    - redo=True: 규칙·프롬프트를 바꿔서 처음부터 다시 판정할 때 (정답 라벨은 지워지지 않는다)
    """
    from qupict import db

    with timing.timed("판정할 기사 불러오기(DB)") as t:
        if redo:
            await asyncio.to_thread(db.reset_topic_filter, topic_id)
        # 세 조회는 서로 상관없어서 동시에 한다 (DB 연결 1.6초씩 3번 → 한 번 정도). 읽는 내용은 예전과 같다
        new, existing, keep = await asyncio.to_thread(
            run_parallel,
            lambda: db.get_topic_articles(topic_id, period_days=period_days, unjudged_only=True),
            lambda: db.get_topic_articles(topic_id, statuses=STATUSES, include_duplicates=False),
            lambda: _labeled_links(topic_id),
        )
        t.note = f"새 기사 {len(new)}건"
        if not new:
            return FilterOutcome()
        t.note += f", 판정된 대표 {len(existing)}건"

    missing = [a for a in new + existing if a.embedding is None]   # 임베딩을 새로 계산할 기사
    with timing.timed("기사 임베딩(기존 · 새, 동시)") as t:
        t.note = f"{len(missing)}건 계산"
        # 예전처럼 기존 기사 / 새 기사를 따로 요청하되(같은 요청 묶음 → 같은 임베딩 값) 두 요청을 동시에 보낸다.
        # 한 요청으로 합치면 묶음이 달라져 값이 아주 조금 달라질 수 있다 (indexing.build_article_chunks 참고)
        await asyncio.gather(embed_articles(existing), embed_articles(new))
    outcome = await filter_articles(new, selected, existing, keep=keep)
    with timing.timed("판정 저장(DB, 동시)"):
        # 세 저장은 서로 다른 칸·행을 고친다 (기사 임베딩 / 중복 기사 행 / 판정한 대표 기사 행) → 동시에 해도 결과가 같다
        await asyncio.to_thread(
            run_parallel,
            lambda: db.save_article_embeddings(missing),
            lambda: db.save_duplicates(topic_id, outcome.duplicates),
            lambda: db.save_judgments(topic_id, [db.Judgment(v.link, v.status, v.stage, v.reason, v.model)
                                                 for v in outcome.verdicts]),
        )
    return outcome
