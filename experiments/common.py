"""
실험 공통 도구 — 지표 계산, 결과 저장, 표 출력

결과는 experiments/results/에 "실험이름_날짜시각.json"과 ".csv"로 저장한다.
(검색 결과는 날짜마다 달라지므로 언제 돌린 결과인지 파일 이름에 남긴다)
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class BinaryScores:
    """관련(positive) / 무관(negative) 이진 분류 지표."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted and not actual:
            self.fp += 1
        elif not predicted and actual:
            self.fn += 1
        else:
            self.tn += 1

    def merge(self, other: "BinaryScores") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def irrelevant_inflow(self) -> float:
        """무관 기사 유입률: 통과시킨 기사 중 정답이 무관인 비율 (= 1 - precision)"""
        return self.fp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    def as_row(self) -> dict:
        return {"n": self.n, "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "precision": round(self.precision, 3), "recall": round(self.recall, 3),
                "f1": round(self.f1, 3), "accuracy": round(self.accuracy, 3),
                "irrelevant_inflow": round(self.irrelevant_inflow, 3)}


def save_results(name: str, rows: list[dict], meta: dict | None = None) -> tuple[Path, Path]:
    """결과를 JSON(설정 포함)과 CSV(표)로 저장한다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = RESULTS_DIR / f"{name}_{stamp}.json"
    csv_path = RESULTS_DIR / f"{name}_{stamp}.csv"
    payload = {"experiment": name, "run_at": datetime.now().isoformat(timespec="seconds"),
               "meta": meta or {}, "rows": rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    if rows:
        columns: list[str] = []                 # 행마다 칸이 다를 수 있어서(요약 행 / 질문별 행) 전부 모은다
        for r in rows:
            columns += [k for k in r if k not in columns]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀에서 한글이 안 깨지게
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def _jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def print_table(rows: list[dict], columns: list[str]) -> None:
    """간단한 표 출력 (한글 폭은 대충 맞춘다)."""
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))
