"""실험 B 복제본: 기본 벡터, HyDE, Hybrid, HyDE + Hybrid 비교."""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter, time_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import print_table, save_results
from experiments.exp_b_retrieval import CHUNKS_PER_SEARCH, RANK_DEPTH, article_ranking_db, score
from experiments.qa_set import load_qa
from qupict import db
from qupict.filtering import get_kiwi
from qupict.retrieval import SEARCH_STATUSES

RESULTS_DIR = Path(__file__).resolve().parent / "results_hyde_hybrid"
VARIANTS = {"base": (False, False), "hyde": (True, False),
            "hybrid": (False, True), "hyde_hybrid": (True, True)}
RRF_K = 60


def tokens(text: str) -> list[str]:
    return [
        token.form.casefold()
        for token in get_kiwi().tokenize(text)
        if token.tag.startswith(("N", "V", "SL", "SN", "XR"))
    ]


class BM25Index:
    """벡터 검색과 같은 주제·판정의 청크로 BM25 인덱스를 구성한다."""

    def __init__(self, topic_id: int):
        query = db._query(
            db.CHUNKS, ("search_keys", "array_contains_any", [f"{topic_id}:{s}" for s in SEARCH_STATUSES])
        )
        self.documents = []
        frequencies = Counter()
        total_length = 0
        for snap in query.select(["link", "title", "content"]).stream():
            row = snap.to_dict()
            terms = tokens(f"{row.get('title', '')} {row['content']}")
            counts = Counter(terms)
            self.documents.append((row["link"], counts, len(terms)))
            frequencies.update(counts.keys())
            total_length += len(terms)
        count = len(self.documents)
        self.average_length = total_length / count if total_length else 1.0
        self.idf = {
            term: math.log1p((count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in frequencies.items()
        }

    def rank(self, question: str) -> list[str]:
        terms = set(tokens(question)).intersection(self.idf)
        best = {}
        for link, counts, length in self.documents:
            norm = 1.2 * (0.25 + 0.75 * length / self.average_length)
            score = sum(
                self.idf[term] * count * 2.2 / (count + norm)
                for term in sorted(terms)
                if (count := counts[term])
            )
            if score > 0:
                best[link] = max(best.get(link, 0.0), score)
        return sorted(best, key=lambda link: (-best[link], link))[:RANK_DEPTH]


def fuse(dense: list[str], lexical: list[str]) -> list[str]:
    """기사 단위 Reciprocal Rank Fusion."""
    scores = Counter()
    for ranking in (dense, lexical):
        for rank, link in enumerate(ranking, 1):
            scores[link] += 1 / (RRF_K + rank)
    return sorted(scores, key=lambda link: (-scores[link], link))[:RANK_DEPTH]


def run(items):
    rows = []
    topics = {}
    setup_seconds = {}
    for i, item in enumerate(items, 1):
        if not item.evidence:
            continue
        print(f"[{i}/{len(items)}] {item.id}", flush=True)
        key = (item.query, item.sense)
        if key not in topics:
            record = db.get_topic(*key)
            if record is None:
                topics[key] = None
            else:
                start = perf_counter()
                topics[key] = (record, BM25Index(record.id))
                setup_seconds[f"{item.query} → {item.sense}"] = round(perf_counter() - start, 3)
        if topics[key] is None:
            print(f"  주제가 없어 건너뜀: {item.query} → {item.sense}")
            continue
        record, index = topics[key]

        # 동일한 질문의 벡터·BM25 결과를 재사용한다. Hybrid 시간은 각 단계 시간의 합이다.
        dense_runs = {}
        for hyde in (False, True):
            start = perf_counter()
            ranking = article_ranking_db(record.id, item, record.selected, hyde, False)
            dense_runs[hyde] = (ranking, perf_counter() - start)
        start = perf_counter()
        lexical = index.rank(item.question)
        lexical_seconds = perf_counter() - start

        for name, (hyde, hybrid) in VARIANTS.items():
            ranking, elapsed = dense_runs[hyde]
            if hybrid:
                start = perf_counter()
                ranking = fuse(ranking, lexical)
                elapsed += lexical_seconds + perf_counter() - start
            hit5, rr, rank = score(ranking, item.evidence)
            rows.append({"id": item.id, "engine": "db", "variant": name, "hit@5": hit5,
                         "reciprocal_rank": rr, "rank": rank, "seconds": elapsed})

    summary = []
    for name in VARIANTS:
        matched = [row for row in rows if row["variant"] == name]
        count = len(matched)
        if count:
            summary.append({
                "engine": "db", "variant": name, "questions": count,
                "recall@5": round(sum(row["hit@5"] for row in matched) / count, 3),
                "mrr": round(sum(row["reciprocal_rank"] for row in matched) / count, 3),
                "avg_seconds": round(sum(row["seconds"] for row in matched) / count, 3),
            })
    return summary, rows, setup_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="검색어 하나만 비교")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--qa", type=Path, help="직접 지정한 골든셋 CSV")
    source.add_argument(
        "--dataset", choices=("original", "varied", "both"), default="both",
        help="original: 기존 질문, varied: 다양한 표현, both: 두 골든셋 비교",
    )
    parser.add_argument("--limit", type=int, help="골든셋마다 근거가 있는 질문 중 앞에서 N개만 실행")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit은 1 이상이어야 합니다")
    datasets = {
        "original": Path(__file__).with_name("qa_set_hyde_hybrid.csv"),
        "varied": Path(__file__).with_name("qa_set_hyde_hybrid_varied.csv"),
    }
    if args.qa is not None:
        datasets = {"custom": args.qa}
    elif args.dataset != "both":
        datasets = {args.dataset: datasets[args.dataset]}
    for path in datasets.values():
        if not path.is_file():
            parser.error(f"골든셋 파일이 없습니다: {path}")

    all_summary, all_rows = [], []
    setup_by_dataset = {}
    for name, path in datasets.items():
        print(f"\n골든셋 {name} — {path.name}", flush=True)
        items = [item for item in load_qa(path, query=args.target) if item.evidence]
        summary, rows, setup = run(items[:args.limit])
        for row in summary + rows:
            row["dataset"] = name
        all_summary.extend(summary)
        all_rows.extend(rows)
        setup_by_dataset[name] = setup
    if not all_rows:
        print("평가할 질문이 없습니다.")
        return
    print_table(all_summary, ["dataset", "engine", "variant", "questions", "recall@5", "mrr", "avg_seconds"])
    for row in all_rows:
        row["reciprocal_rank"] = round(row["reciprocal_rank"], 3)
        row["seconds"] = round(row["seconds"], 3)
    json_path, csv_path = save_results(
        f"exp_b_hyde_hybrid_{time_ns()}", all_summary + all_rows,
        {"rank_depth": RANK_DEPTH, "chunks_per_search": CHUNKS_PER_SEARCH,
         "rrf_k": RRF_K, "bm25_setup_seconds_by_dataset": setup_by_dataset,
         "target": args.target, "datasets": {name: str(path) for name, path in datasets.items()},
         "limit": args.limit},
        results_dir=RESULTS_DIR,
    )
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
