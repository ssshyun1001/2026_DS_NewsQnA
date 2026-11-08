"""
old/ 스크립트 공용 도우미 — 지금 구조(qupict/)에 맞춰 고친 예전 평가·라벨링 스크립트들이 같이 쓴다

예전에는 "검색어(topic) + 검증 키워드(validation_terms)"로 대상을 정했다.
지금은 0단계에서 고른 의미(SelectedSense)가 그 역할을 한다. 의미는 아래 순서로 정한다.

  1) DB에 같은 topic(검색어 × 의미 이름)이 있으면 → 저장된 기준 (사람이 고친 정의·별칭 포함)
  2) targets.yaml에 같은 항목이 있으면 → 의미 후보(DB 캐시 또는 LLM) + targets.yaml에 적은 값
  3) 둘 다 없으면 → 의미 후보를 보여 주고 터미널에서 고른다

실행은 프로젝트 맨 위 폴더에서: python old/스크립트이름.py ...
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:            # old/ 안에서 실행해도 qupict를 찾을 수 있게
    sys.path.insert(0, str(ROOT))

if sys.platform.startswith("win"):        # Windows 콘솔에서 한글이 깨지지 않게
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
    """검색어(+의미 이름)로 고른 의미를 정한다. (위 1~3 순서)"""
    from qupict.config import load_targets
    from qupict.sense import choose_sense_interactively, load_sense_candidates, select_for_target

    record = get_db_topic(query, sense_name)
    if record is not None:
        print(f"[의미] DB에 저장된 기준 사용 → {record.selected.sense.name}")
        return record.selected

    candidates, source = load_sense_candidates(query)
    target = next((t for t in load_targets()
                   if t["query"] == query and str(t["sense"]).strip() == (sense_name or "").strip()), None)
    if target is not None:
        selected = select_for_target(candidates, target)
        print(f"[의미] targets.yaml 항목 사용 ({source}) → {selected.sense.name}")
        return selected

    if sense_name:
        print(f"[의미] '{sense_name}'이 DB·targets.yaml에 없어 후보에서 고릅니다 ({source})")
    selected = choose_sense_interactively(candidates)
    print(f"[의미] 선택 → {selected.sense.name}: {selected.sense.definition}")
    return selected


def describe(selected: SelectedSense) -> str:
    """출력용 한 줄: 예) 강남 → 방송인 강남 (배제어: 강남역, 재건축)"""
    excluded = ", ".join(selected.excluded_terms) or "없음"
    return f"{selected.query} → {selected.sense.name} (배제어: {excluded})"
