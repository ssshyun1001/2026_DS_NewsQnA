"""
많이 보도된 기사 — 수집한 기사 중 중복(비슷한 기사)이 가장 많이 묶인 대표 기사 Top N + 한 줄 요약

2단계 중복 제거에서 "비슷한 기사"는 대표 기사 하나에 묶인다 (article_topics.duplicate_of / FilterOutcome.duplicates).
묶인 기사가 많다 = 여러 언론사가 같은 내용을 보도했다 = 그 기간의 주요 이슈로 본다.

    top_duplicated_db(topic_id, period_days)      DB 버전: article_topics에서 센다
    top_duplicated(articles, duplicates, pool)    메모리 버전: {중복 link: 대표 link}에서 센다
    summarize(hot, terms)                         기사마다 한 줄 요약을 채운다 (LLM 없이, 아래)

## 한 줄 요약: 묶음이 공통으로 쓴 문장 고르기 (LLM 0번)

여러 언론사가 같은 사건을 쓰면, 핵심 사실(누가 · 무엇을)에 쓰는 단어는 거의 모든 기사에 나오고
각 언론사가 덧붙인 배경 · 해설은 그 기사에만 나온다. 그래서 대표 기사에서 검색어가 들어간 문장 중
"같은 묶음의 다른 기사들도 쓴 단어"를 가장 많이 담은 문장을 고른다.

    1) 후보 문장   대표 기사 본문 전체에서 검색어(또는 별칭)가 들어간 문장 (Kiwi 문장 분리).
                   기사 첫 문장에 검색 대상이 없는 경우가 많아서다 (예: 강남 1위 기사 첫 문장 "지상파 예능 팀과
                   개인 유튜버가 …"). 저장된 본문에 문단 구분이 없어서 문단 대신 문장 단위로 찾는다.
                   검색어가 든 문장이 없으면 본문 앞 CANDIDATE_SENTENCES문장 (예전 방식).
                   기자명 · 매체 머리말을 떼고, "~다."로 끝나는 문장만 남긴다.
                   너무 짧거나 긴 문장, 연락처 · 저작권 · 방송 대본 문장은 뺀다
    2) 단어        명사를 뽑는다 (filtering.extract_nouns, 중복 제거와 같은 Kiwi). 고유명사 · 영문은 1, 일반명사는 0.5
    3) 공통도      그 단어가 묶음의 다른 기사(제목 + 검색어가 든 문장) 중 몇 %에 나오나
                   → 각 언론사가 검색 대상에 대해 쓴 부분끼리 비교한다
    4) 흔함 빼기   Top N 묶음 전체에 다 나오는 단어(검색어 "애플" 등)는 IDF로 낮춘다 → 묶음마다 다른 사건을 가리키는 단어가 남는다
    5) 점수        Σ(품사 가중치 × 공통도 × IDF) / (문장의 단어 수)^0.3 — 긴 문장이 단어 수로만 이기지 않게

결과는 기사에 실제로 있는 문장 그대로라서 지어낸 내용이 없고, 같은 입력이면 항상 같은 문장이 나온다.
고를 문장이 없으면 요약을 비워 둔다 (화면은 제목 · 링크만).

웹 화면(web.py)의 "3 수집·필터링"과 "4 질문하기" 사이에 보여 준다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Iterable, Optional

from qupict.schemas import Article

TOP_N = 5
ANSWER_STATUSES = ("관련", "보류")     # 질문 답변에 쓰는 기사와 같은 판정만 (다른 의미의 기사는 보여 주지 않는다)
SUMMARY_MAX_CHARS = 120                # 한 줄 요약 길이 상한 (화면에서 한두 줄)

BODY_SCAN_CHARS = 4000                 # 검색어가 들어간 문장을 찾을 본문 범위 (기사 대부분이 이 안에 끝난다)
MEMBER_TERM_SENTENCES = 5              # 묶음의 다른 기사는 제목 + 검색어가 든 문장 몇 개로 단어를 센다
# 검색어가 든 문장이 없을 때(또는 검색어를 안 줬을 때)의 예전 방식
CANDIDATE_SENTENCES = 5                # 대표 기사 본문 앞 몇 문장에서 고를지 (뉴스는 핵심을 앞에 쓴다)
MEMBER_LEAD_SENTENCES = 3              # 묶음의 다른 기사는 제목 + 앞 몇 문장으로 단어를 센다
MIN_SENTENCE_CHARS = 20                # 이보다 짧은 문장은 후보에서 뺀다 ("세 기업의 전략은 다르다." 같은 연결 문장)
# 이보다 긴 문장도 뺀다: 부제목 여러 줄이 마침표 없이 첫 문장에 붙은 덩어리 (실제 192자: "변곡점 맞는 서울집값
# 강남권 집값 급격히 떨어지며 … 세부담에 집값 더 조정될 수도 …"). 단어가 많아 점수로는 걸러지지 않는다.
# 실제 기사 첫 문장은 대부분 120자 안팎이고 긴 것은 160자 정도 (측정 2026-11-14, 애플 · 강남 · 당근 · 서울여대
# Top 5 후보 문장. 150자로 두었더니 "조승래 … 의원이 오는 11일 … 세미나를 연다."가 빠지고 덜 중요한 문장이 뽑혔다)
MAX_SENTENCE_CHARS = 180
WORD_TAGS = ("NNP", "SL", "NNG")       # 뽑을 품사: 고유명사 · 영문 · 일반명사
TAG_WEIGHT = {"NNG": 0.5}              # 일반명사는 절반만 (고유명사가 사건을 더 잘 가리킨다). 나머지는 1
# 점수를 (문장의 단어 수)^LENGTH_PENALTY로 나눈다. 0이면 단어가 많은 긴 문장이, 0.5(√)면 짧은 문장이 이긴다.
# 측정(당근 1위 묶음): √이면 "거래완료 건수는 약 2.7배 증가했다."(주어 없음),
#   0.3이면 "모델 Y의 문의자 수는 지난해 같은 기간보다 88%, 개인 매물 등록 수는 약 2.9배 늘었다."
LENGTH_PENALTY = 0.3
# 후보 문장(검색어가 든 문장)이 본문에서 하나 뒤로 갈 때마다 점수에 곱하는 값. 뉴스는 핵심을 앞에 쓴다.
# 측정(2026-11-14, 5개 topic Top 5): 1(위치 무시)이면 모든 언론사가 그대로 옮긴 보도자료 인용문
#   ("당근 관계자는 …"), 참석자 명단 같은 뒷부분이 공통도로 이겨 리드를 밀어냈다. 0.85면 리드로 돌아오고
#   강남 묶음에서 좋아진 결과(검색 대상이 주어인 문장)는 그대로. 0.7은 서울여대 2위가 인용문으로 바뀌었다.
POSITION_DECAY = 0.85


@dataclass
class HotArticle:
    title: str
    link: str
    press: Optional[str]
    pub_date: Optional[datetime]
    description: str        # 네이버 검색 API가 준 요약문 (한 줄 요약에는 쓰지 않는다: summarize 참고)
    content: str
    duplicates: int         # 이 기사에 묶인 비슷한 기사 수 (대표 기사 자신은 빼고)
    summary: str = ""       # 한 줄 요약 (summarize가 채운다)
    # 묶인 비슷한 기사들의 (제목, 본문) — 한 줄 요약에서 "다른 언론사도 쓴 단어"를 셀 때 쓴다
    members: list[tuple[str, str]] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# 중복이 많이 묶인 대표 기사 찾기
# ---------------------------------------------------------------------------

def top_duplicated_db(topic_id: int, period_days: Optional[int] = None, limit: int = TOP_N,
                      statuses: Iterable[str] = ANSWER_STATUSES) -> list[HotArticle]:
    """[DB 버전] topic의 대표 기사 중 중복이 많이 묶인 순으로 limit건. 중복이 하나도 없는 기사는 빠진다.
    period_days: 대표 기사 발행일이 최근 며칠 이내인 것만 (질문 검색과 같은 기간).
    묶인 기사들의 제목 · 본문(members)도 함께 가져온다 (한 줄 요약용, 같은 연결에서 쿼리 한 번 더)."""
    from qupict import db

    params: dict = {"topic_id": topic_id, "statuses": list(statuses), "limit": limit}
    period = ""
    if period_days is not None:
        now = datetime.now(timezone.utc)
        period = "AND a.pub_date BETWEEN %(since)s AND %(until)s"
        params.update(since=now - timedelta(days=period_days), until=now)
    with db.connect() as conn:
        rows = conn.execute(
            f"""SELECT a.id, a.title, a.link, a.press, a.pub_date, a.description, a.content,
                       count(*) AS duplicates
                FROM article_topics at
                JOIN articles a ON a.id = at.article_id
                JOIN article_topics d ON d.topic_id = at.topic_id AND d.duplicate_of = at.article_id
                WHERE at.topic_id = %(topic_id)s AND at.duplicate_of IS NULL
                  AND at.status = ANY(%(statuses)s) {period}
                GROUP BY a.id
                ORDER BY duplicates DESC, a.pub_date DESC NULLS LAST, a.id
                LIMIT %(limit)s""",
            params,
        ).fetchall()
        members: dict[int, list[tuple[str, str]]] = {}
        if rows:
            for m in conn.execute(
                """SELECT at.duplicate_of AS rep, a.title, a.content FROM article_topics at
                   JOIN articles a ON a.id = at.article_id
                   WHERE at.topic_id = %s AND at.duplicate_of = ANY(%s)""",
                (topic_id, [r["id"] for r in rows]),
            ).fetchall():
                members.setdefault(m["rep"], []).append((m["title"], m["content"] or ""))
    return [HotArticle(title=r["title"], link=r["link"], press=r["press"], pub_date=r["pub_date"],
                       description=r["description"] or "", content=r["content"] or "",
                       duplicates=r["duplicates"], members=members.get(r["id"], [])) for r in rows]


def top_duplicated(articles: list[Article], duplicates: dict[str, str], limit: int = TOP_N,
                   pool: Iterable[Article] = ()) -> list[HotArticle]:
    """[메모리 버전] articles(질문에 쓰는 대표 기사) 중 중복이 많이 묶인 순으로 limit건.
    duplicates = {중복 기사 link: 대표 기사 link} (FilterOutcome.duplicates).
    pool: 중복 기사까지 포함한 수집 기사 전체 — 묶인 기사들의 제목 · 본문(members)을 여기서 찾는다."""
    counts = Counter(duplicates.values())
    by_link = {a.link: a for a in pool}
    members: dict[str, list[tuple[str, str]]] = {}
    for dup, rep in duplicates.items():
        if dup in by_link:
            members.setdefault(rep, []).append((by_link[dup].title, by_link[dup].content))
    hot = [a for a in articles if counts.get(a.link)]
    hot.sort(key=lambda a: (-counts[a.link], -(a.pub_date.timestamp() if a.pub_date else 0)))
    return [HotArticle(title=a.title, link=a.link, press=a.press, pub_date=a.pub_date,
                       description=a.description, content=a.content, duplicates=counts[a.link],
                       members=members.get(a.link, []))
            for a in hot[:limit]]


# ---------------------------------------------------------------------------
# 한 줄 요약: 묶음이 공통으로 쓴 문장 고르기 (LLM 없이)
# ---------------------------------------------------------------------------

# 기사 앞 머리말: "(서울=뉴스1) 이창규 기자 =", "[이데일리 김현아 기자]", "홍길동 기자 =", "공공뉴스=정진영 기자" 등.
# 머리말 앞에 붙은 부제목도 함께 떼어 내도록, 문장 안에서 가장 마지막 머리말 뒤만 남긴다.
# 본문 속 "…라고 기자에게 말했다"를 머리말로 잘못 자르지 않게, 모든 형식에 "=" 또는 "[ ]"가 있어야 한다.
_BYLINE_RE = re.compile(r"^.*(?:\[[^\]]*(?:기자|특파원)[^\]]*\]"
                        r"|\([^)]*=[^)]*\)\s*(?:[가-힣]{2,4}\s*(?:기자|특파원))?\s*=?"
                        r"|[가-힣]{2,4}\s*(?:기자|특파원)\s*="
                        r"|\S+=\s*[가-힣]{2,4}\s*(?:기자|특파원))\s*")
# 후보에서 뺄 문장: 기자 연락처 · 저작권 · 사진 설명 · 방송 대본 ("◆ 김우성 : …", "앵커 : …")
#   □ ☎ "* 아래 텍스트는 실제 방송 내용과 차이가 있을 수 있으니 …": 라디오 대본 머리말 (방송일시 · 진행 · 전화 목록)
_SKIP_RE = re.compile(r"@|무단\s*전재|재배포|저작권|ⓒ|©|▶|◆|■|□|☎|사진\s*=|\[사진\]|=\s*연합뉴스\)"
                      r"|방송\s*내용과\s*차이|^[가-힣A-Za-z]{1,6}\s*:")
# 본문 맨 앞에 붙은 사진 설명: "▲ 당근 CI © 당근 지역생활 커뮤니티 당근은 …" → "© 출처" 뒤부터 문장으로 본다.
#   예전에는 이 문장을 "©"(저작권) 때문에 통째로 뺐는데, 하필 그 문장이 기사의 리드(가장 중요한 첫 문장)였다
#   (측정 2026-11-14, 당근 1위 묶음: 리드 점수 5.60으로 가장 높았는데 빠지고 세 번째 문장이 뽑힘).
#   "ⓒ 뉴스1 무단전재 …" 같은 저작권 줄은 잘라도 "무단 전재"가 남아 _SKIP_RE로 그대로 빠진다.
_CAPTION_RE = re.compile(r"^.*[©ⓒ]\s*\S+\s+")
# 뉴스 문장의 끝: "~다." (뒤에 닫는 따옴표 · 괄호가 붙을 수 있음). 이렇게 끝나지 않는 조각은 후보에서 뺀다
#   (예: "YTN 라디오(FM 94.5) [YTN 해!", "사진=서울여대 … 총장.", 표 · 목록 줄)
_DECLARATIVE_END_RE = re.compile(r"다\.[\"'”’)\]]*$")
# Kiwi가 한 문장으로 묶은 것 안에 "~다. "가 또 있으면 거기서 한 번 더 나눈다.
#   예: "…세미나를 연다. 「K-컬처 수출 50조 달성을 위한 정부의 역할」" → 리드 "…세미나를 연다."가 따로 후보가 된다
#   (측정 2026-11-14, 서울여대 3위 묶음: 리드가 꺾쇠 제목과 묶여 "~다."로 끝나지 않아 빠지고 "조 의원이 주최하고 …"가 뽑힘)
_SENTENCE_END_SPLIT_RE = re.compile(r"(?<=다\.)\s+|(?<=다\.[\"'”’)\]])\s+")
# 문장을 나눌 본문 앞부분 길이 (앞 몇 문장만 쓰므로 전체를 나누지 않는다. 문장 하나는 길어야 200자 안팎)
SPLIT_CHARS_PER_SENTENCE = 250


@lru_cache(maxsize=1024)
def _split(head: str) -> tuple[str, ...]:
    """Kiwi 문장 분리기로 나눈다: 따옴표 안의 "!" "?"("'장한, 나!' 특집으로 꾸며졌다.")에서
    끊지 않는다 (마침표 기준으로 나누면 "…누가 밥 살까?"처럼 문장 중간이 잘렸다).
    같은 기사를 대표 · 묶음 · IDF 계산에서 여러 번 나누므로 결과를 기억해 둔다."""
    from qupict.filtering import get_kiwi

    out: list[str] = []
    for s in get_kiwi().split_into_sents(head):
        out += [part.strip() for part in _SENTENCE_END_SPLIT_RE.split(s.text) if part.strip()]
    return tuple(out)


def _sentences(text: str, n: int) -> list[str]:
    """본문 앞 n문장."""
    if not (text or "").strip():
        return []
    return list(_split(text[:n * SPLIT_CHARS_PER_SENTENCE + 300]))[:n]      # 머리말 · 부제목 몫으로 조금 더


@lru_cache(maxsize=256)
def _single_token(term: str) -> bool:
    """Kiwi가 검색어를 한 단어로 읽나 ("애플" O, "애플컴퓨터"는 "애플" + "컴퓨터"로 나뉨 → X)."""
    from qupict.filtering import get_kiwi

    return len(get_kiwi().tokenize(term)) == 1


def _has_term(sentence: str, terms: Iterable[str]) -> bool:
    """문장에 검색어(또는 별칭)가 독립된 단어로 들어 있나. 영문 별칭("Apple")은 대소문자를 가리지 않는다.

    글자만 들어 있는 것으로는 부족하다: 과일 사과의 "애플파이" · "골든피치애플", 기업 Apple 기사 속
    메타 제품 설명의 "애플워치"처럼 검색어가 다른 단어의 일부인 문장이 뽑혔다 (측정 2026-11-14).
    Kiwi로 나눠서 검색어가 한 단어로 나오고, 바로 뒤에(띄어쓰기 없이) 명사가 붙지 않은 경우만 인정한다.
      O "애플의" "강남과" "'당근'에" "서울여대 총장"    X "애플파이"(애플+파이) "애플워치" "강남구청"
    Kiwi가 한 단어로 읽지 못하는 검색어("애플컴퓨터")는 글자가 들어 있으면 인정한다."""
    from qupict.filtering import get_kiwi

    lowered = sentence.lower()
    present = [t for t in terms if t and t.lower() in lowered]
    if not present:
        return False
    if any(not _single_token(t) for t in present):
        return True
    wanted = {t.lower() for t in present}
    tokens = get_kiwi().tokenize(sentence)
    for i, tok in enumerate(tokens):
        if tok.form.lower() not in wanted:
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt is None or nxt.start != tok.start + tok.len or not nxt.tag.startswith("NN"):
            return True
    return False


def term_sentences(text: str, terms: Iterable[str], limit: Optional[int] = None) -> list[str]:
    """본문(앞 BODY_SCAN_CHARS자)에서 검색어(또는 별칭)가 들어간 문장만, 본문 순서대로 최대 limit개.
    저장된 본문에는 문단 구분(줄바꿈)이 없어서(크롤링할 때 사라짐) 문단 대신 문장 단위로 찾는다."""
    terms = tuple(t for t in terms if t)
    if not terms or not (text or "").strip():
        return []
    found = _term_sentences(text[:BODY_SCAN_CHARS], terms)
    return list(found if limit is None else found[:limit])


@lru_cache(maxsize=1024)
def _term_sentences(head: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    """term_sentences의 실제 계산. 같은 기사를 IDF 계산 · 묶음 비교에서 다시 보므로 결과를 기억해 둔다."""
    return tuple(s for s in _split(head) if _has_term(s, terms))


def _clean(sentence: str) -> str:
    """앞에 붙은 사진 설명("… © 출처")과 기자 머리말을 떼고 공백을 정리한다."""
    return " ".join(_BYLINE_RE.sub("", _CAPTION_RE.sub("", sentence)).split())


def _usable(sentences: Iterable[str]) -> list[str]:
    """머리말을 떼고, "~다."로 끝나는 문장만 남긴다.
    연락처 · 저작권 · 방송 대본 문장과 너무 짧거나 긴 문장은 뺀다. 순서는 그대로."""
    out = []
    for s in sentences:
        s = _clean(s)
        if (MIN_SENTENCE_CHARS <= len(s) <= MAX_SENTENCE_CHARS and _DECLARATIVE_END_RE.search(s)
                and not _SKIP_RE.search(s)):
            out.append(s)
    return out


def candidate_sentences(content: str, terms: Iterable[str] = ()) -> list[str]:
    """한 줄 요약 후보.
    terms(검색어 · 별칭)를 주면 본문 전체에서 검색어가 들어간 문장 중 쓸 수 있는 것.
    없거나(terms를 안 줌 · 검색어가 든 문장이 다 빠짐) 하면 예전처럼 본문 앞 CANDIDATE_SENTENCES문장에서 고른다."""
    terms = list(terms)
    return (_usable(term_sentences(content, terms)) if terms else []) or \
        _usable(_sentences(content, CANDIDATE_SENTENCES))


def _words(text: str) -> dict[str, float]:
    """명사 → 품사 가중치 (같은 단어는 한 번만, 영문은 소문자로, 한 글자는 뺀다).
    filtering.extract_nouns와 같은 Kiwi · 같은 규칙에 일반명사(NNG)만 더한 것."""
    from qupict.filtering import get_kiwi

    weights: dict[str, float] = {}
    for t in get_kiwi().tokenize(text or ""):
        if t.tag in WORD_TAGS and len(t.form) >= 2:
            weights.setdefault(t.form.lower() if t.tag == "SL" else t.form, TAG_WEIGHT.get(t.tag, 1.0))
    return weights


def _member_doc(title: str, content: str, terms: Iterable[str] = ()) -> set[str]:
    """묶인 기사 하나에서 비교할 단어: 제목 + 검색어가 들어간 문장(앞에서 MEMBER_TERM_SENTENCES개).
    검색어가 든 문장이 없거나 terms를 안 주면 제목 + 앞 MEMBER_LEAD_SENTENCES문장 (예전 방식)."""
    body = term_sentences(content, terms, MEMBER_TERM_SENTENCES) or _sentences(content, MEMBER_LEAD_SENTENCES)
    return set(_words(title + "\n" + " ".join(body)))


def pick_common_sentence(h: HotArticle, idf: dict[str, float], terms: Iterable[str] = ()) -> str:
    """대표 기사의 후보 문장 중, 묶인 다른 기사들도 쓴 단어를 가장 많이 담은 문장. 후보가 없으면 ""."""
    terms = list(terms)
    candidates = candidate_sentences(h.content, terms)
    if not candidates:
        return ""
    member_docs = [_member_doc(t, c, terms) for t, c in h.members] or [_member_doc(h.title, "")]
    share = Counter(w for doc in member_docs for w in doc)          # 단어 → 그 단어를 쓴 기사 수

    def score(sentence: str) -> float:
        words = _words(sentence)
        if not words:
            return 0.0
        total = sum(weight * share[w] / len(member_docs) * idf.get(w, 1.0) for w, weight in words.items())
        return total / len(words) ** LENGTH_PENALTY

    # 뉴스는 중요한 내용을 앞에 쓴다: 후보 순서(본문 순서)대로 POSITION_DECAY를 곱해 앞 문장을 조금 더 쳐준다.
    # 점수가 같으면 앞 문장 (max는 처음 나온 것을 고른다)
    best = max(range(len(candidates)), key=lambda i: score(candidates[i]) * POSITION_DECAY ** i)
    return candidates[best]


def _one_line(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= SUMMARY_MAX_CHARS else text[:SUMMARY_MAX_CHARS - 1].rstrip() + "…"


def summarize(hot: list[HotArticle], terms: Iterable[str] = ()) -> list[HotArticle]:
    """기사마다 한 줄 요약(summary)을 채운다. LLM을 부르지 않는다 (위 설명).
    terms: 검색어와 별칭 (SelectedSense.match_terms). 주면 본문 앞부분 대신 검색어가 들어간 문장끼리 비교한다.
    IDF는 Top N 묶음 전체(대표 + 묶인 기사)를 한 문서 집합으로 보고 계산한다:
    idf = log((1 + 문서 수) / (1 + 그 단어가 나온 문서 수)) + 1 (filtering.noun_tfidf와 같은 식).
    고를 문장이 없으면(방송 대본 · 표처럼 "~다."로 끝나는 문장이 없는 기사) summary를 비워 둔다 →
    화면은 제목 · 링크만 보여 준다. 네이버 요약문은 "…"로 이어 붙인 조각이라 대신 쓰지 않는다."""
    if not hot:
        return hot
    terms = list(terms)
    docs = [_member_doc(t, c, terms) for h in hot for t, c in [(h.title, h.content), *h.members]]
    df = Counter(w for doc in docs for w in doc)
    idf = {w: math.log((1 + len(docs)) / (1 + n)) + 1 for w, n in df.items()}
    for h in hot:
        h.summary = _one_line(pick_common_sentence(h, idf, terms))
    return hot
