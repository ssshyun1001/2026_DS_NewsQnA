"""
old/ 스크립트 공용 도우미 — 지금 구조(qupict/)에 맞춰 고친 예전 평가·라벨링 스크립트들이 같이 쓴다

예전에는 "검색어(topic) + 검증 키워드(validation_terms)"로 대상을 정했다.
지금은 0단계에서 고른 의미(SelectedSense)가 그 역할을 한다. 의미는 LLM 의미 후보(DB 캐시 또는 LLM)에서 정한다.

  1) 의미 이름이 주어지고 후보에 있으면 → 그 후보
  2) 없으면 → 의미 후보를 보여 주고 터미널에서 고른다

실행은 프로젝트 맨 위 폴더에서: python old/스크립트이름.py ...
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:            # old/ 안에서 실행해도 qupict를 찾을 수 있게
    sys.path.insert(0, str(ROOT))

from qupict.cli import setup_console                     # noqa: E402

setup_console()                           # Windows: 한글 깨짐 방지 + asyncio 루프 설정 (WinError 10054 로그 방지)

from qupict.config import settings                       # noqa: E402
from qupict.schemas import SelectedSense                 # noqa: E402


def get_db_topic(query: str, sense_name: str | None):
    """DB에 저장된 topic (없거나 DB를 못 쓰면 None). 반환값: db.TopicRecord (id, selected)"""
    if not settings.database_url or not sense_name:
        return None
    try:
        from qupict import db
        return db.get_topic(query, sense_name)
    except Exception as e:
        print(f"  [경고] DB를 읽지 못했습니다 ({type(e).__name__}). DB 없이 진행합니다.")
        return None


def pick_sense(query: str, sense_name: str | None = None) -> SelectedSense:
    """검색어(+의미 이름)로 고른 의미를 정한다. LLM 의미 후보만 쓴다. (위 1~2 순서)"""
    from qupict.sense import choose_sense_interactively, load_sense_candidates, select_sense_by_name

    candidates, source = load_sense_candidates(query)
    if sense_name:
        selected = select_sense_by_name(candidates, sense_name)
        if selected is not None:
            print(f"[의미] 의미 후보 사용 ({source}) → {selected.sense.name}")
            return selected
        print(f"[의미] '{sense_name}'이 의미 후보에 없어 후보에서 고릅니다 ({source})")
    selected = choose_sense_interactively(candidates)
    print(f"[의미] 선택 → {selected.sense.name}: {selected.sense.definition}")
    return selected


def describe(selected: SelectedSense) -> str:
    """출력용 한 줄: 예) 강남 → 방송인 강남 (배제어: 강남역, 재건축)"""
    excluded = ", ".join(selected.excluded_terms) or "없음"
    return f"{selected.query} → {selected.sense.name} (배제어: {excluded})"
