"""
많이 보도된 기사 Top 5 (qupict/hot.py) — 한 줄 요약을 LLM 없이 "묶음이 공통으로 쓴 문장"으로 고르는지

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0, Kiwi 형태소 분석기만 씀). 실행: python tests/run_offline.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import qupict.llm as llm
from qupict import hot
from qupict.schemas import Article


def _no_llm(*a, **k):
    raise AssertionError("한 줄 요약에서 LLM을 불렀습니다")


llm.get_chat_model = llm.get_structured_model = llm.get_embeddings = _no_llm   # 부르면 바로 실패

# 1) 후보 문장: 머리말 · 사진 설명을 떼고, "~다."로 끝나는 문장만, 본문 순서 그대로
c = hot.candidate_sentences
assert c("(서울=뉴스1) 이창규 기자 = 애플이 첫 폴더블폰 아이폰 듀오를 공개했다.") == ["애플이 첫 폴더블폰 아이폰 듀오를 공개했다."]
assert c("[이데일리 김현아 기자] 애플이 첫 폴더블폰 아이폰 듀오를 공개했다.") == ["애플이 첫 폴더블폰 아이폰 듀오를 공개했다."]
assert c("공공뉴스=정진영 기자 애플이 첫 폴더블폰 아이폰 듀오를 공개했다.") == ["애플이 첫 폴더블폰 아이폰 듀오를 공개했다."]
# 본문 속 "기자에게"는 머리말이 아니다 (머리말은 = 또는 [ ]가 있어야 함)
assert c("팀 쿡은 새 제품이 곧 나온다고 기자에게 말했다.") == ["팀 쿡은 새 제품이 곧 나온다고 기자에게 말했다."]
# 사진 설명 "… © 출처" 뒤부터 문장 (예전에는 ©때문에 리드 문장이 통째로 빠졌다)
assert c("▲ 당근 CI © 당근 지역생활 커뮤니티 당근은 중고 전기차 거래가 54% 늘었다고 밝혔다.") == \
    ["지역생활 커뮤니티 당근은 중고 전기차 거래가 54% 늘었다고 밝혔다."]
# 리드 뒤에 꺾쇠 제목이 붙어 한 덩어리가 돼도 "~다. "에서 나눠 리드가 후보가 된다
lead = "조승래 의원이 오는 11일 게임산업 수출을 논의하는 세미나를 연다."
assert c(f"{lead} 「K-컬처 수출 50조 달성을 위한 정부의 역할」 세미나는 국회의원회관에서 열린다.")[0] == lead
# 따옴표 안의 "!"에서는 끊지 않는다
quoted = "지난 11일 방송된 라디오스타는 '장한, 나!' 특집으로 꾸며져 큰 관심을 모았다."
assert c(quoted) == [quoted]
# 빼는 것: 연락처 · 저작권 · 방송 대본 머리말 · 너무 짧은 문장 · "~다."로 안 끝나는 조각
body = ("YTN 라디오(FM 94.5) [YTN 해! 봅시다] □ 방송일시 : 2026년 11월 12일 "
        "* 아래 텍스트는 실제 방송 내용과 차이가 있을 수 있으니 방송으로 확인하시기 바랍니다. "
        "짧은 문장이다. 홍길동 기자 hong@news.com 으로 제보를 받는다. ⓒ 뉴스1 무단전재 및 재배포 금지를 알린다.")
assert c(body) == [], c(body)
assert c("") == [] and c("   ") == []
long_sentence = "가" * (hot.MAX_SENTENCE_CHARS + 1) + "다."
assert c(long_sentence) == []                                        # 부제목이 붙은 너무 긴 덩어리
print("1 ok: 후보 문장 (머리말 · 사진 설명 · 문장 나누기 · 빼는 문장)")

# 2) 묶음이 공통으로 쓴 문장: 다른 언론사들도 쓴 단어를 많이 담은 문장이 뽑힌다
duo = hot.HotArticle(
    title="애플 아이폰 듀오 화면 결함 논란", link="A", press="A일보", pub_date=None, description="", duplicates=3,
    content=("애플은 올해 연말 쇼핑 시즌을 앞두고 매장 직원을 늘렸다. "
             "애플의 첫 폴더블폰 아이폰 듀오가 화면 결함 논란으로 리콜 요구에 직면했다. "
             "업계 관계자는 시장 상황을 지켜봐야 한다고 말했다."),
    members=[("아이폰 듀오 화면 결함…리콜 요구 커져", "애플 아이폰 듀오의 화면 결함으로 리콜 요구가 커지고 있다."),
             ("애플 폴더블폰 아이폰 듀오 결함 논란", "아이폰 듀오 화면에 줄이 생기는 결함이 보고됐다."),
             ("아이폰 듀오 리콜하나", "소비자들이 아이폰 듀오 리콜을 요구하고 있다.")])
cook = hot.HotArticle(
    title="트럼프 젠슨 황 후임으로 팀 쿡 언급", link="B", press="B신문", pub_date=None, description="", duplicates=2,
    content=("애플은 올해 연말 쇼핑 시즌을 앞두고 매장 직원을 늘렸다. "
             "트럼프 대통령이 젠슨 황의 후임으로 애플 전 CEO 팀 쿡을 언급했다."),
    members=[("트럼프 \"젠슨 황 대신할 사람은 팀 쿡\"", "트럼프 대통령은 엔비디아 젠슨 황 후임으로 팀 쿡을 꼽았다."),
             ("팀 쿡, 엔비디아 가나", "트럼프가 팀 쿡을 젠슨 황 후임으로 거론했다.")])
first = [h.summary for h in hot.summarize([duo, cook])]
assert first == ["애플의 첫 폴더블폰 아이폰 듀오가 화면 결함 논란으로 리콜 요구에 직면했다.",
                 "트럼프 대통령이 젠슨 황의 후임으로 애플 전 CEO 팀 쿡을 언급했다."], first
# 같은 입력이면 항상 같은 문장 (LLM처럼 실행마다 바뀌지 않는다), 그리고 기사 속 문장 그대로
assert [h.summary for h in hot.summarize([duo, cook])] == first
assert all(s in h.content for s, h in zip(first, [duo, cook]))
print("2 ok: 묶음 공통 문장 (흔한 문장 대신 사건 문장, 항상 같은 결과, 기사 속 문장 그대로)")

# 3) 고를 문장이 없으면 요약을 비운다 (화면은 제목 · 링크만). 묶인 기사가 없으면 대표 기사 제목으로 센다
radio = hot.HotArticle(title="방송 대본", link="R", press=None, pub_date=None, description="네이버 요약문…",
                       content="YTN 라디오(FM 94.5) [YTN 해! 봅시다] □ 진행 : 김우성 PD", duplicates=1)
alone = hot.HotArticle(title="당근 중고 전기차 거래 54% 늘어", link="S", press=None, pub_date=None, description="",
                       content="날씨가 맑았다고 전했다. 당근은 중고 전기차 거래가 54% 늘었다고 밝혔다.", duplicates=1)
out = hot.summarize([radio, alone])
assert out[0].summary == "" and out[1].summary == "당근은 중고 전기차 거래가 54% 늘었다고 밝혔다.", [h.summary for h in out]
assert hot.summarize([]) == []
long_lead = "당근은 " + "아주 " * 60 + "늘었다고 밝혔다."
one_line = hot._one_line(long_lead)
assert len(one_line) == hot.SUMMARY_MAX_CHARS and one_line.endswith("…")
print("3 ok: 요약 없음 · 묶인 기사 없음 · 길이 상한")

# 4) 메모리 버전: 중복이 많은 순, 묶인 기사들의 제목 · 본문은 pool(중복 포함 전체)에서 찾는다
d = lambda day: datetime(2026, 11, day, tzinfo=timezone.utc)
reps = [Article(title="대표1", link="r1", content="본문1", pub_date=d(1)),
        Article(title="대표2", link="r2", content="본문2", pub_date=d(5)),
        Article(title="대표3", link="r3", content="본문3", pub_date=d(9))]
pool = reps + [Article(title=f"중복{i}", link=f"d{i}", content=f"중복 본문{i}") for i in range(5)]
dups = {"d0": "r1", "d1": "r1", "d2": "r2", "d3": "r2", "d4": "r3", "없는기사": "r3"}
top = hot.top_duplicated(reps, dups, limit=2, pool=pool)
assert [(h.link, h.duplicates) for h in top] == [("r3", 2), ("r2", 2)]   # 같은 수면 최신 기사 먼저, 상위 2건
assert top[0].members == [("중복4", "중복 본문4")]                         # pool에 없는 기사는 본문을 모르니 뺀다
assert top[1].members == [("중복2", "중복 본문2"), ("중복3", "중복 본문3")]
assert hot.top_duplicated(reps, {}) == []                                 # 중복이 없으면 보여 줄 기사 없음
print("4 ok: 메모리 버전 순위 · 묶인 기사")
print("ALL PASS")
