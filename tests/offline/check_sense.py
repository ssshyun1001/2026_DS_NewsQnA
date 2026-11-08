"""
0단계: 의미 후보 정리·선택·targets.yaml 덮어쓰기

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import sys, types
from unittest.mock import MagicMock
# langchain 계열만 가짜로 (pydantic은 진짜)
for name in ["langchain_core", "langchain_core.output_parsers", "langchain_core.prompts", "langchain_openai"]:
    m = types.ModuleType(name); sys.modules[name] = m
sys.modules["langchain_core.output_parsers"].StrOutputParser = MagicMock()
sys.modules["langchain_core.prompts"].ChatPromptTemplate = MagicMock()
sys.modules["langchain_openai"].ChatOpenAI = MagicMock(); sys.modules["langchain_openai"].OpenAIEmbeddings = MagicMock()
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from qupict import sense
from qupict.schemas import Sense

def fake_llm(senses=None, exc=None):
    llm = MagicMock()
    if exc: llm.invoke.side_effect = exc
    else: llm.invoke.return_value = sense._SenseList(senses=senses or [])
    return llm

# 1) 애플: 공백 정리, 검색어 제거, 겹치는 연관어 제거, 중복 이름 제거, TOP3 제한
sense._candidates_llm = fake_llm([
    Sense(name=" 기업 Apple ", definition="아이폰을 만드는 미국 IT 기업", aliases=["Apple", "애플", "Apple"], related_terms=["아이폰", "맥북", "가격", "애플", "팀 쿡"]),
    Sense(name="과일 사과", definition="과일", aliases=["사과"], related_terms=["과수원", "농가", "가격", "수확"]),
    Sense(name="과일 사과", definition="중복", related_terms=["x"]),
    Sense(name="애플 TV", definition="d", related_terms=["드라마"]),
    Sense(name="넷째", definition="d", related_terms=["y"]),
])
c = sense.generate_sense_candidates("애플")
assert [s.name for s in c.senses] == ["기업 Apple", "과일 사과", "애플 TV"], c.senses
apple = c.senses[0]
assert apple.aliases == ["Apple"], apple.aliases
assert apple.related_terms == ["아이폰", "맥북", "팀 쿡"], apple.related_terms  # '가격'(공유), '애플'(검색어) 제거
assert c.needs_selection
sel = sense.select_sense(c, 0)
assert sel.excluded_terms == ["과수원", "농가", "수확", "드라마"], sel.excluded_terms
assert sel.match_terms == ["애플", "Apple"]
assert [s.name for s in sel.other_senses] == ["과일 사과", "애플 TV"]
print("1 ok: 애플 후보 정리 + 배제어", sel.excluded_terms)

# 2) 이름으로 찾기 (targets.yaml)
assert sense.select_sense_by_name(c, "기업Apple").sense.name == "기업 Apple"   # 공백 무시
assert sense.select_sense_by_name(c, "사과").sense.name == "과일 사과"          # 별칭
assert sense.select_sense_by_name(c, "Apple TV") is None                       # 없음
assert sense.select_sense_by_name(c, "") is None
# 부분 일치가 여러 개면 고르지 않음
sense._candidates_llm = fake_llm([Sense(name="지역 강남", definition="d", related_terms=["강남역"]), Sense(name="방송인 강남", definition="d", related_terms=["예능"])])
g = sense.generate_sense_candidates("강남")
assert sense.select_sense_by_name(g, "방송인 강남").sense.name == "방송인 강남"
assert sense.select_sense_by_name(g, "강남") is None
print("2 ok: 이름으로 선택")

# 3) 후보 1개 → 선택 불필요, 배제어 없음
sense._candidates_llm = fake_llm([Sense(name="서울여자대학교", definition="d", aliases=["서울여자대학교", "SWU"], related_terms=["캠퍼스"])])
one = sense.generate_sense_candidates("서울여대")
assert not one.needs_selection
s1 = sense.select_sense(one, 0)
assert s1.excluded_terms == [] and s1.match_terms == ["서울여대", "서울여자대학교", "SWU"]
print("3 ok: 후보 1개")

# 4) LLM 실패 → 빈 후보, 범위 밖 번호 → IndexError
sense._candidates_llm = fake_llm(exc=RuntimeError("api down"))
empty = sense.generate_sense_candidates("후드")
assert empty.senses == [] and not empty.needs_selection
try: sense.select_sense(empty, 0); raise SystemExit("IndexError 기대")
except IndexError: pass
print("4 ok: LLM 실패 처리")

# 5) 직접 입력: LLM 성공 / 실패, 모든 후보가 배제어로
sense._custom_llm = fake_llm()
sense._custom_llm.invoke.return_value = Sense(name="래퍼 후드", definition="", aliases=["후드"], related_terms=["앨범", "공연"])
cs = sense.build_custom_sense("후드", "힙합 래퍼 후드")
assert cs.is_custom and cs.definition == "힙합 래퍼 후드" and cs.aliases == []
sense._custom_llm = fake_llm(exc=RuntimeError("x"))
cs2 = sense.build_custom_sense("후드", "힙합 래퍼 후드")
assert cs2.is_custom and cs2.name == "힙합 래퍼 후드" and cs2.related_terms == []
sel_c = sense.select_custom_sense(c, cs)
assert sel_c.sense.is_custom and len(sel_c.other_senses) == 3
assert "아이폰" in sel_c.excluded_terms and "앨범" not in sel_c.excluded_terms
print("5 ok: 직접 입력")

# 6) DB 저장용 직렬화 (친구가 쓰는 형태)
d = sel.model_dump(); assert set(d) == {"query", "sense", "excluded_terms", "other_senses"}
print("6 ok: model_dump keys", list(d))

# 7) 새 정리 규칙: 한자 제거, 검색어 일부 제거, 연관어 없는 후보 제거
sense._candidates_llm = fake_llm([
    Sense(name="대학 서울여자대학교", definition="d", aliases=["서울여자대학교", "SWU", "漢城"], related_terms=["노원구", "여대", "서울", "총장", "出荷"]),
    Sense(name="인물 서울여대(확실하지 않음)", definition="?", related_terms=[]),
])
sw = sense.generate_sense_candidates("서울여대")
assert [x.name for x in sw.senses] == ["대학 서울여자대학교"], sw.senses     # 연관어 없는 후보 제거
assert sw.senses[0].related_terms == ["노원구", "총장"], sw.senses[0].related_terms  # 여대·서울(검색어 일부), 出荷(한자) 제거
assert sw.senses[0].aliases == ["서울여자대학교", "SWU"]                     # 漢城 제거
sense._candidates_llm = fake_llm([Sense(name="인물 베라(확실하지 않음)", definition="?", related_terms=[])])
assert sense.generate_sense_candidates("베라").senses == []                   # → 직접 입력 경로
print("7 ok: 새 정리 규칙")

# 8) select_for_target: 매칭 / 직접 입력 / 별칭 덮어쓰기
sense._candidates_llm = fake_llm([
    Sense(name="지역 강남", definition="d", aliases=["강남구"], related_terms=["강남역", "재건축"]),
    Sense(name="방송인 강남", definition="d", aliases=["조남식"], related_terms=["예능", "유튜브"]),
])
g = sense.generate_sense_candidates("강남")
t1 = sense.select_for_target(g, {"query": "강남", "sense": "방송인 강남", "aliases": []})
assert t1.sense.name == "방송인 강남" and t1.sense.aliases == [] and t1.match_terms == ["강남"]
assert t1.excluded_terms == ["강남역", "재건축"]
t2 = sense.select_for_target(g, {"query": "강남", "sense": "방송인 강남"})
assert t2.sense.aliases == ["조남식"]                                          # 덮어쓰기 없으면 LLM 값 유지
t3 = sense.select_for_target(g, {"query": "강남", "sense": "방송인 강남", "related_terms": ["예능", "MIB"]})
assert t3.sense.related_terms == ["예능", "MIB"] and t3.sense.aliases == ["조남식"]

sense._candidates_llm = fake_llm([])
b = sense.generate_sense_candidates("베라")
sense._custom_llm = fake_llm(); seen = {}
def _custom_invoke(msgs): seen["human"] = msgs[1][1]; return Sense(name="배스킨라빈스", definition="아이스크림", aliases=["베라킹"], related_terms=["아이스크림"])
sense._custom_llm.invoke.side_effect = _custom_invoke
t4 = sense.select_for_target(b, {"query": "베라", "sense": "배스킨라빈스", "description": "아이스크림 브랜드 배스킨라빈스", "aliases": ["배스킨라빈스", "배라"]})
assert "아이스크림 브랜드 배스킨라빈스" in seen["human"]                          # description이 LLM에 전달됨
assert t4.sense.is_custom and t4.sense.aliases == ["배스킨라빈스", "배라"]         # 사람이 적은 별칭으로 교체
assert t4.match_terms == ["베라", "배스킨라빈스", "배라"]
t5 = sense.select_for_target(g, {"query": "강남", "sense": "방송인 강남", "definition": " 가수 겸 방송인 강남 "})
assert t5.sense.definition == "가수 겸 방송인 강남" and t5.sense.aliases == ["조남식"]
print("8 ok: select_for_target")

# 9) 직접 입력이 LLM 후보와 같은 의미면 그 후보를 배제어에 쓰지 않음
cands = sense.SenseCandidates(query="마스크", senses=[
    Sense(name="방역 마스크", definition="d", related_terms=["KF94", "미세먼지", "방역", "황사"]),
    Sense(name="가면 마스크", definition="d", related_terms=["가면무도회", "할로윈"]),
])
custom = Sense(name="보건용 마스크", definition="d", related_terms=["KF94", "미세먼지", "감염병"], is_custom=True)
m = sense.select_custom_sense(cands, custom)
assert [x.name for x in m.other_senses] == ["가면 마스크"], m.other_senses
assert m.excluded_terms == ["가면무도회", "할로윈"], m.excluded_terms   # 방역·황사는 배제어가 아님
# 겹치는 연관어가 1개뿐이면 다른 의미로 유지
custom2 = Sense(name="x", definition="d", related_terms=["KF94", "기타"], is_custom=True)
assert len(sense.select_custom_sense(cands, custom2).other_senses) == 2
print("9 ok: 같은 의미 후보는 배제어에서 제외")
print("ALL PASS")
