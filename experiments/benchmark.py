"""
속도 벤치마크 — 최적화 전·후를 "같은 입력"으로 비교한다 (DB에는 쓰지 않는다: 팀 공용 DB를 건드리지 않음)

    python experiments/benchmark.py --target 강남 --save-metas m.json --out before.json   처음: 검색 결과 저장
    python experiments/benchmark.py --target 강남 --metas m.json --out after.json         다음: 같은 기사 목록으로
    python experiments/benchmark.py --compare before.json after.json                      두 결과 비교 (시간 + 내용)

왜 run.py로 재지 않나: run.py는 한 번 수집한 기사를 DB에서 재사용하므로, 같은 검색어로 두 번째 돌리면
크롤링·판정을 건너뛰어 공정한 비교가 안 된다. 그래서 단계 함수를 DB 없이(메모리에서) 직접 부른다.

  1) 검색               search_news                       (네이버 API)
  2) 본문 크롤링         crawl_articles                    (--metas가 있으면 저장해 둔 기사 목록 그대로 → 전·후 같은 입력)
  3) 필터링             filter_articles                    (임베딩 → 중복 → 규칙 → LLM, 메모리 버전)
  4) 인덱싱             describe_images_many(최대 --images장) + build_article_chunks
  5) DB 왕복            연결 1번 비용 · topic 기사 불러오기   (읽기만)
  6) 질문 답변          search_topic(DB에 저장된 topic) + generate_answer_from_chunks  (qa_set.yaml 질문)

결과 JSON: 단계별 시간 + 내용 지문(크롤링 본문 해시, 판정, 본문 청크 텍스트, 검색된 청크, 답변)
→ --compare로 최적화 전·후에 "내용이 같은지"도 확인한다.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qupict.cli import setup_console  # noqa: E402

setup_console()

from qupict import db, timing  # noqa: E402
from qupict.collect import crawl_articles, search_news  # noqa: E402
from qupict.filtering import filter_articles  # noqa: E402
from qupict.indexing import MAX_IMAGES_PER_ARTICLE, image_chunk_text  # noqa: E402
from qupict.qa import generate_answer_from_chunks  # noqa: E402

try:
    from qupict.indexing import build_article_chunks, describe_images_many  # noqa: E402
except ImportError:
    # 최적화 전 코드(이 함수들이 생기기 전)에서도 돌 수 있게: 그때 index_topic 안에 있던 방식 그대로
    from qupict.indexing import IMAGE_CONCURRENCY, describe_image, embed_chunks, split_into_chunks  # noqa: E402

    async def build_article_chunks(articles, image_chunks):
        out = {}
        for article in articles:
            bodies = split_into_chunks(article.content)
            images = image_chunks.get(article.link, [])
            texts = bodies + [text for _, text in images]
            if not texts:
                continue
            vectors = await embed_chunks(article.title, texts)
            out[article.link] = ([db.Chunk(t, v) for t, v in zip(bodies, vectors)]
                                 + [db.Chunk(t, v, source="image", image_id=i)
                                    for (i, t), v in zip(images, vectors[len(bodies):])])
        return out

    async def describe_images_many(jobs):
        semaphore = asyncio.Semaphore(IMAGE_CONCURRENCY)

        async def one(image, title, link):
            async with semaphore:
                return await describe_image(image, title, link)
        return list(await asyncio.gather(*(one(*job) for job in jobs)))
from qupict.retrieval import search_topic  # noqa: E402
from qupict.schemas import NewsMeta  # noqa: E402

ANSWER_STATUSES = ("관련", "보류")


def _sha(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def _pick_topic(query: str, sense_name: str | None) -> db.TopicRecord:
    from qupict.pipeline import stored_topics
    records = [r for r in stored_topics(query) if sense_name is None or r.selected.sense.name == sense_name]
    if not records:
        raise SystemExit(f"DB에 '{query}' ({sense_name or '아무 의미'}) topic이 없습니다.")
    return records[0]


def _questions(query: str, sense_name: str, n: int) -> list[str]:
    from experiments.qa_set import load_qa
    return [item.question for item in load_qa(query=query) if item.sense == sense_name][:n]


async def run_benchmark(args) -> dict:
    record = _pick_topic(args.target, args.sense)
    selected = record.selected
    print(f"대상: {selected.query} → {selected.sense.name} (topic {record.id})")
    out: dict = {"target": f"{selected.query} → {selected.sense.name}", "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    timing.enable()
    timing.reset()

    # 1) 검색 — 항상 실제로 다시 한다 (시간 측정). 크롤링 입력은 --metas가 있으면 그것을 쓴다
    with timing.timed("1) 검색(네이버 API)") as t:
        searched = await search_news(selected, args.days, args.max)
        t.note = f"{len(searched)}건"
    out["searched_links"] = [m.link for m in searched]
    if args.metas:
        metas = [NewsMeta.model_validate(m) for m in json.loads(Path(args.metas).read_text(encoding="utf-8"))]
        print(f"크롤링 입력: {args.metas}의 기사 {len(metas)}건 (전·후 같은 입력)")
    else:
        metas = searched
    if args.save_metas:
        Path(args.save_metas).write_text(
            json.dumps([m.model_dump(mode="json") for m in metas], ensure_ascii=False, indent=1), encoding="utf-8")

    # 2) 크롤링
    with timing.timed("2) 본문 크롤링") as t:
        articles = await crawl_articles(metas)
        t.note = f"성공 {len(articles)} / 시도 {len(metas)}건"
    out["crawled"] = {a.link: {"content": _sha(a.content), "press": a.press, "images": len(a.images)}
                      for a in articles}

    # 3) 필터링 (메모리 버전)
    with timing.timed("3) 필터링"):
        outcome = await filter_articles(articles, selected)
    out["duplicates"] = outcome.duplicates
    out["verdicts"] = {v.link: v.status for v in outcome.verdicts}
    keep = outcome.links_with(*ANSWER_STATUSES)
    relevant = [a for a in articles if a.link in keep]

    # 4) 인덱싱: 이미지 설명(기사당 MAX_IMAGES_PER_ARTICLE장, 전체 --images장까지) + 청크 임베딩
    with timing.timed("4) 인덱싱"):
        jobs = [(img, a.title, a.link) for a in relevant for img in a.images[:MAX_IMAGES_PER_ARTICLE]][:args.images]
        with timing.timed("이미지 설명(VLM)") as t:
            descriptions = await describe_images_many(jobs)
            t.note = f"{sum(d is not None for d in descriptions)} / {len(jobs)}장"
        image_chunks: dict[str, list[tuple[int, str]]] = {}
        for i, ((img, _, link), desc) in enumerate(zip(jobs, descriptions)):
            if desc:
                image_chunks.setdefault(link, []).append((-(i + 1), image_chunk_text(img.caption, desc)))
        built = await build_article_chunks(relevant, image_chunks)
    out["body_chunks"] = {link: [_sha(c.content) for c in chunks if c.source == "body"]
                          for link, chunks in built.items()}
    # 임베딩이 같은지 보려고 기사마다 첫 청크 벡터 앞부분만 남긴다
    out["first_chunk_vec"] = {link: [round(x, 5) for x in chunks[0].embedding[:16]]
                              for link, chunks in built.items() if chunks}

    # 5) DB 왕복 (읽기만)
    with timing.timed("5) DB 왕복"):
        for i in range(3):
            with timing.timed(f"연결 + SELECT 1 (#{i + 1})"):
                with db.connect() as conn:
                    conn.execute("SELECT 1").fetchone()
        with timing.timed("topic 기사 불러오기") as t:
            usable = db.get_topic_articles(record.id, statuses=ANSWER_STATUSES)
            t.note = f"{len(usable)}건 (본문·임베딩 포함)"

    # 6) 질문 답변 (DB에 이미 인덱싱된 topic에서 검색)
    out["answers"] = []
    questions = _questions(selected.query, selected.sense.name, args.questions)
    with timing.timed("6) 질문 답변") as section:
        section.note = f"질문 {len(questions)}개"
        for i, question in enumerate(questions, 1):
            first: list[float] = []
            start = time.perf_counter()

            def on_token(_text: str) -> None:
                if not first:
                    first.append(time.perf_counter() - start)

            with timing.timed(f"질문 {i}") as qt:
                hits = search_topic(record.id, question, selected)
                try:
                    answer = generate_answer_from_chunks(question, hits, on_token=on_token)
                except TypeError:            # 최적화 전 코드에는 on_token(스트리밍)이 없다
                    answer = generate_answer_from_chunks(question, hits)
                qt.note = f"첫 글자 {first[0]:.2f}초" if first else "스트리밍 없음"
            out["answers"].append({"question": question, "hits": [f"{h.article.link}#{_sha(h.content)}" for h in hits],
                                   "answer": answer.answer, "first_token_seconds": first[0] if first else None})

    out["records"] = [{"label": r.label, "depth": r.depth, "seconds": round(r.seconds, 3), "note": r.note}
                      for r in timing.records()]
    timing.report(f"벤치마크 {out['target']}")
    return out


# ---------------------------------------------------------------------------
# 전·후 비교
# ---------------------------------------------------------------------------

def _top_records(data: dict) -> dict[str, float]:
    """단계 이름 → 시간. 이름이 겹치면(질문 1·2·3 안의 HyDE 등) 합친다."""
    totals: dict[str, float] = {}
    for r in data["records"]:
        key = ("  " * r["depth"]) + r["label"]
        totals[key] = totals.get(key, 0.0) + r["seconds"]
    return totals


def compare(before_path: str, after_path: str) -> None:
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))

    print(f"시간 비교 ({before['started']} → {after['started']})")
    b, a = _top_records(before), _top_records(after)
    for key in list(dict.fromkeys([*b, *a])):
        if key.strip().startswith("연결 + SELECT"):
            continue
        bs, as_ = b.get(key), a.get(key)
        diff = f"{(as_ - bs):+7.2f}초 ({(as_ / bs - 1) * 100:+4.0f}%)" if bs and as_ is not None and bs > 0 else ""
        print(f"  {key:<34} {bs if bs is not None else float('nan'):7.2f} → "
              f"{as_ if as_ is not None else float('nan'):7.2f}초  {diff}")

    print("\n내용 비교")
    common = before["crawled"].keys() & after["crawled"].keys()
    same_body = sum(before["crawled"][k]["content"] == after["crawled"][k]["content"] for k in common)
    print(f"  크롤링 성공: {len(before['crawled'])} → {len(after['crawled'])}건, 둘 다 성공 {len(common)}건 중 "
          f"본문 같음 {same_body}건")
    only_b = before["crawled"].keys() - after["crawled"].keys()
    only_a = after["crawled"].keys() - before["crawled"].keys()
    if only_b or only_a:
        print(f"    전에만 성공 {len(only_b)}건, 후에만 성공 {len(only_a)}건 (언론사 서버 응답에 따라 달라질 수 있음)")
    print(f"  중복 표시: {len(before['duplicates'])} → {len(after['duplicates'])}건, "
          f"같은 표시 {len(before['duplicates'].items() & after['duplicates'].items())}건")
    common_v = before["verdicts"].keys() & after["verdicts"].keys()
    same_v = sum(before["verdicts"][k] == after["verdicts"][k] for k in common_v)
    print(f"  판정: 둘 다 판정 {len(common_v)}건 중 같음 {same_v}건 "
          f"(LLM은 같은 입력에도 가끔 다르게 답한다 → 최적화와 무관한 흔들림 포함)")
    common_c = before["body_chunks"].keys() & after["body_chunks"].keys()
    same_c = sum(before["body_chunks"][k] == after["body_chunks"][k] for k in common_c)
    print(f"  본문 청크: 둘 다 있는 기사 {len(common_c)}건 중 청크 내용 같음 {same_c}건")
    diffs = [max(abs(x - y) for x, y in zip(before["first_chunk_vec"][k], after["first_chunk_vec"][k]))
             for k in before["first_chunk_vec"].keys() & after["first_chunk_vec"].keys()
             if before["body_chunks"].get(k) == after["body_chunks"].get(k)]
    if diffs:
        print(f"  임베딩: 같은 청크 {len(diffs)}건의 첫 벡터 앞 16칸 최대 차이 {max(diffs):.5f}")
    for qb, qa in zip(before["answers"], after["answers"]):
        same_hits = len(set(qb["hits"]) & set(qa["hits"]))
        print(f"  질문 '{qb['question'][:24]}': 검색 청크 {len(qb['hits'])} → {len(qa['hits'])}개 중 같은 청크 {same_hits}개"
              f", 첫 글자 {qa['first_token_seconds'] and round(qa['first_token_seconds'], 2)}초")


def main() -> None:
    p = argparse.ArgumentParser(prog="python experiments/benchmark.py")
    p.add_argument("--target", help="검색어 (DB에 저장된 topic)")
    p.add_argument("--sense", help="의미 이름 (생략하면 그 검색어의 첫 topic)")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--max", type=int, default=100, help="수집할 최대 기사 수 (run.py 기본과 같게 100)")
    p.add_argument("--metas", help="크롤링할 기사 목록 JSON (--save-metas로 저장한 것)")
    p.add_argument("--save-metas", help="이번 크롤링 입력(기사 목록)을 저장할 JSON")
    p.add_argument("--images", type=int, default=12, help="설명할 이미지 최대 수")
    p.add_argument("--questions", type=int, default=3)
    p.add_argument("--out", help="결과 JSON 경로")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if not args.target:
        p.error("--target이 필요합니다")
    result = asyncio.run(run_benchmark(args))
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
