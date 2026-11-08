"""
터미널 입력 도우미 — 실행 파일 3개(run.py · admin.py · evaluate.py)가 같이 쓴다

  setup_console()      Windows 콘솔에서 한글이 깨지지 않게 (실행 파일 맨 위에서 한 번)
  read_line(prompt)    빈 입력·한글 조합 중 깨진 입력을 다시 받는다
  read_period_days()   기간(일)을 1 이상의 정수로 받는다
  choose_number(prompt, count)   1~count 중 하나를 받아 0부터 시작하는 번호로 돌려준다
"""
from __future__ import annotations

import sys

_INCOMPLETE_JAMO = set(
    "ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
    "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
)


def setup_console() -> None:
    if sys.platform.startswith("win"):
        try:
            sys.stdin.reconfigure(encoding="utf-8")
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _looks_corrupted(text: str) -> bool:
    return bool(text) and text[-1] in _INCOMPLETE_JAMO


def read_line(prompt: str) -> str:
    while True:
        text = input(prompt).strip()
        if not text:
            print("입력이 비어 있습니다. 다시 입력해주세요.")
            continue
        if _looks_corrupted(text):
            print(
                f"입력이 깨진 것 같습니다 ('{text}'). "
                "한/영 전환 직후거나 IME 조합 중 Enter가 눌렸을 수 있어요. 다시 입력해주세요."
            )
            continue
        return text


def read_period_days() -> int:
    while True:
        raw = input("기간을 숫자로 입력하세요 (최근 며칠, 예: 7): ").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("기간은 1 이상의 정수로 입력해주세요 (예: 7).")


def choose_number(prompt: str, count: int) -> int:
    """1~count 중 하나를 입력받아 0부터 시작하는 번호로 돌려준다."""
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        print(f"1~{count} 사이 번호를 입력하세요.")
