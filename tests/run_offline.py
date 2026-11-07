"""
오프라인 테스트 전체 실행 — LLM·네이버·DB를 부르지 않는다 (크레딧 0, DB 없이도 됨)

    python tests/run_offline.py

각 파일은 함수를 가짜로 바꿔 끼우는 방식이라, 서로 영향을 주지 않도록 파일마다 따로 실행한다.
(실제 DB 점검은 python tests/check_db.py)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "offline"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔에서 한글이 깨지지 않게
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    failed = []
    for path in sorted(HERE.glob("check_*.py")):
        result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", env=env)
        ok = result.returncode == 0
        last = (result.stdout.strip().splitlines() or [""])[-1]
        print(f"{'통과' if ok else '실패'}  {path.name:32} {last if ok else ''}")
        if not ok:
            failed.append(path.name)
            print("      " + "\n      ".join((result.stderr or result.stdout).strip().splitlines()[-8:]))
    print(f"\n{len(failed)}개 실패" if failed else "\n모두 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
