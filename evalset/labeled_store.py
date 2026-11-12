"""
labeled_store.py — 정답 라벨 JSON을 여러 파일로 나눠 저장·읽기

파일 하나가 너무 커지지 않게 evalset/labeled_set_1.json, labeled_set_2.json, ... 로 나눈다.

  - 나누는 단위는 검색어다. 한 검색어(의미가 여러 개여도)의 라벨은 모두 같은 파일에 들어간다.
  - 이미 있는 검색어는 그 검색어가 들어 있는 파일에 이어 쓴다.
  - 새 검색어는 마지막 파일에 넣는다. 단, 마지막 파일에 검색어 하나 분량
    (LABEL_PER_TARGET건 × 기사 한 건 평균 크기)을 더하면 MAX_FILE_BYTES를 넘으면 다음 번호 파일을 새로 만든다.
  - 읽을 때는 labeled_set_*.json 을 번호 순서대로 모두 읽어 하나의 목록으로 합친다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LABELED_DIR = Path(__file__).resolve().parent
PREFIX = "labeled_set_"
MAX_FILE_BYTES = 3_500_000           # 파일 하나 최대 크기 (대략). 검색어 하나는 쪼개지 않는다
DEFAULT_ENTRY_BYTES = 7_000          # 기존 라벨이 없을 때 쓰는 기사 한 건 크기 추정값

_PART_RE = re.compile(rf"^{PREFIX}(\d+)\.json$")


def part_paths(directory: Path = LABELED_DIR) -> list[Path]:
    """labeled_set_N.json 을 N 순서대로."""
    parts = [(int(m.group(1)), p) for p in directory.glob(f"{PREFIX}*.json") if (m := _PART_RE.match(p.name))]
    return [p for _, p in sorted(parts)]


def part_path(directory: Path, number: int) -> Path:
    return directory / f"{PREFIX}{number}.json"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []


def load_entries(path: Path = LABELED_DIR) -> list[dict]:
    """폴더면 labeled_set_*.json 전부(번호 순)를 합쳐서, 파일이면 그 파일 하나만 읽는다."""
    if path.is_dir():
        return [e for p in part_paths(path) for e in _read(p)]
    return _read(path)


class LabeledStore:
    """라벨 파일들을 메모리에 들고 있다가, 라벨 하나 추가할 때마다 그 라벨이 들어간 파일만 다시 저장한다."""

    def __init__(self, directory: Path = LABELED_DIR, max_bytes: int = MAX_FILE_BYTES,
                 per_query: int = 150) -> None:
        self.directory = directory
        self.max_bytes = max_bytes
        self.per_query = per_query                  # 검색어 하나 분량 추정에 쓰는 라벨 수
        self.files: dict[Path, list[dict]] = {p: _read(p) for p in part_paths(directory)}

    @property
    def entries(self) -> list[dict]:
        return [e for es in self.files.values() for e in es]

    def path_for(self, query: str) -> Path:
        """이 검색어의 라벨을 쓸 파일."""
        for path, entries in self.files.items():
            if any(e.get("query") == query for e in entries):
                return path
        if not self.files:
            return part_path(self.directory, 1)
        last = next(reversed(self.files))
        if self.files[last] and self._size(last) + self._query_estimate() > self.max_bytes:
            return part_path(self.directory, self._number(last) + 1)
        return last

    def add(self, entry: dict) -> Path:
        path = self.path_for(entry["query"])
        self.files.setdefault(path, []).append(entry)
        self._save(path)
        return path

    # -- 내부 -----------------------------------------------------------------

    def _size(self, path: Path) -> int:
        return path.stat().st_size if path.exists() else 0

    def _query_estimate(self) -> int:
        sizes = [self._size(p) for p in self.files]
        count = sum(len(es) for es in self.files.values())
        per_entry = sum(sizes) // count if count else DEFAULT_ENTRY_BYTES
        return per_entry * self.per_query

    @staticmethod
    def _number(path: Path) -> int:
        return int(_PART_RE.match(path.name).group(1))

    def _save(self, path: Path) -> None:
        """임시 파일에 다 쓴 뒤 바꿔치기한다 (쓰는 도중 꺼져도 기존 파일이 깨지지 않게)."""
        tmp = path.with_suffix(".json.tmp")
        tmp.write_bytes(json.dumps(self.files[path], ensure_ascii=False, indent=2).encode("utf-8"))   # OS 상관없이 LF
        tmp.replace(path)
