"""
[4단계] 인덱싱 — 이미지 → 글 → 청크 → 임베딩 → article_chunks 저장

필터링에서 '관련'·'보류' 판정을 받은 기사만 질문 답변에 쓸 수 있게 준비한다.

  1) 이미지 → 글   둘 중 하나 (결과는 모두 article_images.llm_description에 저장)
                   - OCR(EasyOCR): 이미지 속 글자를 읽는다. LLM 비용 0. 서비스 기본 (qupict/ocr.py)
                   - VLM: LLM이 기사 이미지를 2~3문장으로 설명한다. 이미지 주소를 그대로 넘기고,
                     실패하면 파일을 내려받아 보낸다 (파일은 저장하지 않음)
  2) 청크 나누기   본문을 약 500자씩, 문장 단위로 자른다 (앞 청크의 마지막 문장을 겹쳐서 문맥 유지)
                   이미지 설명은 이미지 하나 = 청크 하나 (출처 유형 'image')
  3) 임베딩 저장   청크마다 임베딩을 만들어 article_chunks에 저장 → 5단계 pgvector 검색에 쓴다

  index_topic(topic_id)    DB 버전: 아직 청크가 없는 기사만 처리 (다시 실행해도 안전)
  split_into_chunks(text)  청크 나누기만 따로 (실험·테스트용)

비용: 이미지 설명은 이미지 1장에 LLM 호출 1번(의미 후보 만들기와 비슷한 비용)이라,
기사당 최대 MAX_IMAGES_PER_ARTICLE장만 설명한다. 청크 임베딩은 거의 무료다.
"""
from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from qupict import timing
from qupict.collect import download_images
from qupict.concurrency import run_parallel, timed_call
from qupict.config import settings
from qupict.llm import get_chat_model, get_embeddings
from qupict.schemas import Article, ArticleImage

INDEX_STATUSES = ("관련", "보류")   # 질문 답변에 쓰는 판정 (run.py의 ANSWER_STATUSES와 같게)
CHUNK_CHARS = 500                   # 청크 하나의 목표 길이 (한국어 기사 2~4문장)
MAX_IMAGES_PER_ARTICLE = 2          # 기사당 설명할 이미지 수 (LLM 비용 상한)
# 동시에 보내는 이미지 설명 요청 수. 4 → 8: 이미지 1장 설명에 수 초가 걸려서(측정: 8장 22초) 기다림이 대부분이다.
# 게이트웨이 한도(60초 120회) 안쪽. 필터링 LLM 판정과는 단계가 달라 동시에 겹치지 않는다.
IMAGE_CONCURRENCY = 8
EMBED_CONCURRENCY = 8               # 청크 임베딩 요청(기사마다 1번)을 동시에 보내는 수


# ---------------------------------------------------------------------------
# 청크 나누기
# ---------------------------------------------------------------------------

# 문장 끝: 마침표·물음표·느낌표 (뒤에 따옴표·괄호가 붙어도 됨) 다음의 공백
_SENTENCE_END = re.compile(r"(?<=[.!?。])[\"'”’)\]]*\s+")


def split_into_chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """본문을 문장 단위로 모아 약 size자 청크로 나눈다.

    - 문장 중간에서 자르지 않는다 (문장이 반으로 잘리면 검색·답변 품질이 떨어짐)
    - 다음 청크는 앞 청크의 마지막 문장으로 시작한다 (경계에 걸친 내용을 놓치지 않게)
    - size보다 긴 문장 하나는 그 문장만으로 청크가 된다
    """
    sentences = [s.strip() for s in _SENTENCE_END.split(text or "") if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > size:
            chunks.append(" ".join(current))
            current = [current[-1]] if len(current) > 1 else []   # 마지막 문장을 겹쳐서 시작
        current.append(sentence)
    if current and (not chunks or " ".join(current) not in chunks[-1]):
        chunks.append(" ".join(current))
    return chunks


def image_chunk_text(caption: str, description: str) -> str:
    """이미지 청크 내용. 답변에서 출처를 '기사 이미지 설명'으로 표시할 수 있게 표시를 붙인다."""
    parts = [f"사진 설명(기사): {caption}" if caption else "", f"사진 내용: {description}"]
    return "[기사 이미지] " + " / ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# 이미지 설명 (LLM)
# ---------------------------------------------------------------------------

_IMAGE_SYSTEM_PROMPT = """\
너는 뉴스 기사에 실린 사진을 글로 옮겨, 나중에 기사 내용을 검색할 때 쓸 수 있게 한다.

- 한국어 2~3문장으로, 사진에 보이는 것만 사실대로 적어라.
- 사진 속 글자·숫자·표·그래프가 있으면 그 내용을 꼭 적어라 (인포그래픽의 수치 등).
- 인물이 누구인지 얼굴로 추측하지 마라. 이름은 기사 제목이나 사진 설명에 나온 경우에만 써라.
- 사진에서 알 수 없는 내용(날짜, 장소 이름 등)을 지어내지 마라.
"""


async def _ask_image(image_url: str, title: str, caption: str) -> str:
    """LLM에 이미지 하나를 보여 주고 설명을 받는다. image_url은 웹 주소 또는 data: 주소. 실패하면 예외."""
    message = HumanMessage(content=[
        {"type": "text", "text": f"기사 제목: {title}\n사진 설명(기사): {caption or '(없음)'}"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ])
    result = await get_chat_model(temperature=0).ainvoke([SystemMessage(_IMAGE_SYSTEM_PROMPT), message])
    text = (result.content or "").strip()
    if not text:
        raise ValueError("빈 응답")
    return text


async def describe_image(image: ArticleImage, title: str, article_link: str = "") -> str | None:
    """이미지를 LLM으로 설명한다. 실패하면 None.

    1) 이미지 주소를 그대로 넘긴다. 내려받을 필요가 없어서 빠르다.
    2) 실패하면 파일을 내려받아 data: 주소(base64)로 보낸다. (외부 접근을 막는 사이트용)

    순서를 이렇게 정한 이유: 실제로 해 보니 게이트웨이가 일부 파일(WebP, 일부 JPEG)을 직접 올리면
    "Could not process image"로 거절했지만, 같은 이미지를 주소로 넘기면 설명이 됐다.
    """
    try:
        return await _ask_image(image.url, title, image.caption)
    except Exception:
        pass
    if image.data is None:
        await download_images([Article(title=title, link=article_link, images=[image])])
    if image.data:
        data_url = f"data:{image.content_type or 'image/jpeg'};base64,{base64.b64encode(image.data).decode()}"
        try:
            return await _ask_image(data_url, title, image.caption)
        except Exception as e:
            print(f"  [경고] 이미지 설명 실패 ({type(e).__name__}): {image.url[:60]}")
            return None
    print(f"  [경고] 이미지 설명 실패 (주소로도 안 되고 파일도 못 받음): {image.url[:60]}")
    return None


# ---------------------------------------------------------------------------
# 임베딩
# ---------------------------------------------------------------------------

async def embed_chunks(title: str, chunks: list[str]) -> list[list[float]]:
    """청크 임베딩. 기사 제목을 앞에 붙여서 임베딩한다 (짧은 청크에도 무슨 기사인지 정보가 들어가게).
    DB에는 청크 본문만 저장한다."""
    if not chunks:
        return []
    return await get_embeddings().aembed_documents([f"{title}\n{c}" for c in chunks])


async def build_article_chunks(articles: list[Article],
                               image_chunks: dict[str, list[tuple[int, str]]]) -> dict[str, list]:
    """기사들의 청크(본문 청크 + 이미지 청크)와 임베딩을 만든다. DB에 저장하지 않는다. {기사 link: [db.Chunk]}

    image_chunks: {기사 link: [(image_id, 이미지 청크 내용)]} (described_images 결과)
    청크가 하나도 없는 기사(본문·이미지 설명 모두 없음)는 결과에서 빠진다.

    기사마다 임베딩 요청 1번(예전과 같은 묶음)을 동시에 EMBED_CONCURRENCY개씩 보낸다.
    예전: 기사마다 요청 1번을 차례로 → 기사 수만큼 기다림.
    모든 청크를 한 요청에 몰아 보내면 더 빠르지만, 임베딩 API는 한 요청에 같이 들어간 글이 달라지면
    값이 아주 조금 달라진다 (측정: 최대 0.004). 결과를 예전과 똑같이 두려고 요청 묶음은 바꾸지 않았다.
    """
    from qupict import db

    plans: list[tuple[Article, list[str], list[tuple[int, str]]]] = []
    for article in articles:
        bodies = split_into_chunks(article.content)
        images = image_chunks.get(article.link, [])
        if bodies or images:
            plans.append((article, bodies, images))

    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)

    async def embed_one(article: Article, bodies: list[str], images: list[tuple[int, str]]) -> list[list[float]]:
        async with semaphore:
            return await embed_chunks(article.title, bodies + [text for _, text in images])

    all_vectors = await asyncio.gather(*(embed_one(*plan) for plan in plans))
    out: dict[str, list] = {}
    for (article, bodies, images), vecs in zip(plans, all_vectors):
        chunks = [db.Chunk(text, vec) for text, vec in zip(bodies, vecs)]
        chunks += [db.Chunk(text, vec, source="image", image_id=image_id)
                   for (image_id, text), vec in zip(images, vecs[len(bodies):])]
        out[article.link] = chunks
    return out


async def describe_images_many(jobs: list[tuple[ArticleImage, str, str]]) -> list[str | None]:
    """(이미지, 기사 제목, 기사 link) 목록을 동시에 IMAGE_CONCURRENCY개씩 설명한다. DB에 저장하지 않는다.
    돌려주는 값: jobs와 같은 순서의 설명 (실패하면 None)"""
    semaphore = asyncio.Semaphore(IMAGE_CONCURRENCY)

    async def one(image: ArticleImage, title: str, link: str) -> str | None:
        async with semaphore:
            return await describe_image(image, title, link)

    return list(await asyncio.gather(*(one(*job) for job in jobs)))


# ---------------------------------------------------------------------------
# DB 버전
# ---------------------------------------------------------------------------

@dataclass
class IndexResult:
    articles: int = 0            # 청크를 만든 기사 수
    body_chunks: int = 0
    image_chunks: int = 0
    images_described: int = 0
    images_failed: int = 0       # 내려받기·설명 실패 (VLM은 다음 실행 때 다시 시도, OCR은 다시 안 읽음)
    images_no_text: int = 0      # OCR로 읽었지만 글자가 없던 이미지 (인물·현장 사진 등)

    def summary(self) -> str:
        no_text = f", 글자 없음 {self.images_no_text}장" if self.images_no_text else ""
        return (f"기사 {self.articles}건 → 본문 청크 {self.body_chunks}개 + 이미지 청크 {self.image_chunks}개"
                f" (이미지 설명 {self.images_described}장{no_text}, 실패 {self.images_failed}장)")


def described_images(article_links: list[str]) -> dict[str, list[tuple[int, str]]]:
    """기사별로 이미 글(VLM 설명 또는 OCR 글자)이 있는 이미지. {기사 link: [(image_id, 이미지 청크 내용)]}
    (기사 안 순서). OCR로 읽었지만 글자가 없던 이미지("")는 청크를 만들지 않는다.

    db.py에 이 조회가 없어서 여기에 두었다. (DB 담당이 db.py로 옮겨도 된다 — 읽기 전용 SELECT 하나)
    """
    from qupict import db
    from qupict.ocr import OCR_MODEL_NAME, ocr_text_chunk

    if not article_links:
        return {}
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT a.link, i.id, i.caption, i.llm_description, i.described_model FROM article_images i
               JOIN articles a ON a.id = i.article_id
               WHERE a.link = ANY(%s) AND i.llm_description IS NOT NULL
               ORDER BY a.id, i.position""",
            (list(article_links),),
        ).fetchall()
    out: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        if not r["llm_description"]:
            continue
        if r.get("described_model") == OCR_MODEL_NAME:
            text = ocr_text_chunk(r["caption"], r["llm_description"])
        else:
            text = image_chunk_text(r["caption"], r["llm_description"])
        out.setdefault(r["link"], []).append((r["id"], text))
    return out


def _processed_image_counts(article_links: list[str]) -> dict[str, int]:
    """기사별로 이미 처리한(글이 비어 있어도 저장된) 이미지 수. OCR의 기사당 상한을 셀 때 쓴다."""
    from qupict import db

    if not article_links:
        return {}
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT a.link, count(*) AS n FROM article_images i
               JOIN articles a ON a.id = i.article_id
               WHERE a.link = ANY(%s) AND i.llm_description IS NOT NULL
               GROUP BY a.link""",
            (list(article_links),),
        ).fetchall()
    return {r["link"]: r["n"] for r in rows}


async def _describe_topic_images(topic_id: int, statuses: tuple[str, ...]) -> tuple[set[str], int, int]:
    """설명이 없는 이미지를 설명하고 저장한다. (설명이 새로 생긴 기사 link들, 설명한 수, 실패 수)

    기사당 MAX_IMAGES_PER_ARTICLE장까지만 설명한다. 이전 실행에서 이미 설명한 장수도 센다
    (안 그러면 실행할 때마다 같은 기사의 다음 이미지들을 계속 설명하게 된다).
    """
    from qupict import db

    todo = await asyncio.to_thread(db.get_images_to_describe, topic_id, statuses=statuses)
    if not todo:
        return set(), 0, 0
    # 이미 설명한 장수와 기사 제목은 서로 상관없는 DB 조회라 동시에 한다 (DB 연결 시간이 겹침)
    # 제목은 판정(statuses)을 통과한 기사에서만 찾는다: 설명할 이미지(todo)가 모두 그 기사들의 것이라
    # 결과는 topic 전체 기사를 불러오던 예전과 같고, 불러오는 양(본문·임베딩)만 준다.
    described, judged = await asyncio.to_thread(
        run_parallel,
        lambda: described_images(list({i.article_link for i in todo})),
        lambda: db.get_topic_articles(topic_id, statuses=statuses),
    )
    already = {link: len(v) for link, v in described.items()}
    per_article: dict[str, list] = {}
    for img in todo:                                    # get_images_to_describe는 기사 안 순서대로 준다
        picked = per_article.setdefault(img.article_link, [])
        if already.get(img.article_link, 0) + len(picked) < MAX_IMAGES_PER_ARTICLE:
            picked.append(img)
    picked = [img for imgs in per_article.values() for img in imgs]
    if not picked:
        return set(), 0, 0

    titles = {a.link: a.title for a in judged}
    descriptions = await describe_images_many(
        [(ArticleImage(url=img.url, caption=img.caption), titles.get(img.article_link, ""), img.article_link)
         for img in picked])

    ok = [(img, desc) for img, desc in zip(picked, descriptions) if desc is not None]
    # 설명 저장은 모아서 한 번에 (예전: 이미지마다 연결 1번)
    await asyncio.to_thread(
        db.save_image_descriptions, [(img.image_id, desc) for img, desc in ok], settings.chat_model)
    changed = {img.article_link for img, _ in ok}
    return changed, len(ok), len(picked) - len(ok)


async def _ocr_topic_images(topic_id: int, statuses: tuple[str, ...]) -> tuple[set[str], int, int, int]:
    """아직 글이 없는 이미지의 글자를 OCR로 읽고 저장한다.
    돌려주는 값: (글자가 새로 생긴 기사 link들, 글자 있음 수, 글자 없음 수, 내려받기 실패 수)

    - 기사당 MAX_IMAGES_PER_ARTICLE장까지 (이전 실행에서 처리한 장수도 센다)
    - 글자가 없거나 내려받지 못한 이미지는 ""로 저장한다 → 다음 실행 때 같은 이미지를 다시 읽지 않는다
    - 이미지 파일은 메모리에서만 쓰고 저장하지 않는다
    """
    from qupict import db, ocr

    todo = await asyncio.to_thread(db.get_images_to_describe, topic_id, statuses=statuses)
    if not todo:
        return set(), 0, 0, 0
    already = await asyncio.to_thread(_processed_image_counts, list({i.article_link for i in todo}))
    per_article: dict[str, list] = {}
    for img in todo:                                    # get_images_to_describe는 기사 안 순서대로 준다
        picked = per_article.setdefault(img.article_link, [])
        if already.get(img.article_link, 0) + len(picked) < MAX_IMAGES_PER_ARTICLE:
            picked.append(img)
    per_article = {link: imgs for link, imgs in per_article.items() if imgs}
    if not per_article:
        return set(), 0, 0, 0

    # 이미지 파일 내려받기 (기사 link를 Referer로 보내려고 기사별로 묶는다)
    files = {link: [ArticleImage(url=img.url, caption=img.caption) for img in imgs]
             for link, imgs in per_article.items()}
    await download_images([Article(title="", link=link, images=images) for link, images in files.items()])

    saved: list[tuple[int, str]] = []
    changed: set[str] = set()
    with_text = no_text = failed = 0
    for link, imgs in per_article.items():
        for img, file in zip(imgs, files[link]):
            if file.data is None:
                failed += 1
                saved.append((img.image_id, ""))
                continue
            text = await asyncio.to_thread(ocr.read_image_text, file.data)
            saved.append((img.image_id, text))
            if text:
                with_text += 1
                changed.add(link)
            else:
                no_text += 1
    await asyncio.to_thread(db.save_image_descriptions, saved, ocr.OCR_MODEL_NAME)
    return changed, with_text, no_text, failed


async def index_topic(
    topic_id: int,
    statuses: tuple[str, ...] = INDEX_STATUSES,
    describe_images: bool = True,
    rebuild: bool = False,
    ocr_images: bool = False,
) -> IndexResult:
    """[DB 버전] 판정을 통과한 기사의 청크·임베딩을 만들어 저장한다.

    처리하는 기사: ① 아직 청크가 없는 기사 ② 이번에 이미지 설명이 새로 생긴 기사 (이미지 청크를 넣으려고 다시 만든다)
    청크는 기사 단위로 통째로 교체 저장되므로 여러 번 실행해도 안전하다.
    describe_images=False면 이미지 설명을 새로 만들지 않는다 (LLM 비용 0, 이미 있는 설명은 청크에 넣음).
    ocr_images=True면 아직 글이 없는 이미지의 글자를 OCR로 읽는다 (describe_images와 함께 켜지 않는다).
    rebuild=True면 판정을 통과한 모든 기사의 청크를 다시 만든다 (청크 규칙을 바꿨을 때. 이미지 설명은 재사용).
    """
    from qupict import db

    result = IndexResult()
    changed: set[str] = set()

    def load_targets() -> dict[str, Article]:
        """청크를 만들 기본 대상. 이미지 설명과 상관없이 정해진다 (청크 유무·판정만 봄)."""
        if rebuild:
            return {a.link: a for a in db.get_topic_articles(topic_id, statuses=statuses)}
        return {a.link: a for a in db.get_articles_without_chunks(topic_id, statuses=statuses)}

    # 대상 기사 불러오기(DB)는 이미지 설명(LLM)을 기다리는 동안 동시에 해 둔다
    targets_task = asyncio.create_task(asyncio.to_thread(timed_call, load_targets))
    if describe_images:
        with timing.timed("이미지 설명(VLM)") as t:
            changed, result.images_described, result.images_failed = await _describe_topic_images(topic_id, statuses)
            t.note = f"설명 {result.images_described}장, 실패 {result.images_failed}장 (동시 {IMAGE_CONCURRENCY})"
    if ocr_images:
        with timing.timed("이미지 글자 읽기(OCR)") as t:
            changed_ocr, with_text, no_text, failed = await _ocr_topic_images(topic_id, statuses)
            changed |= changed_ocr
            result.images_described += with_text
            result.images_no_text += no_text
            result.images_failed += failed
            t.note = f"글자 있음 {with_text}장, 글자 없음 {no_text}장, 내려받기 실패 {failed}장"

    with timing.timed("청크 만들 기사 불러오기(DB)") as t:
        targets, load_sec = await targets_task
        if changed - targets.keys():
            for a in db.get_topic_articles(topic_id, statuses=statuses):
                if a.link in changed:
                    targets.setdefault(a.link, a)
        image_chunks = described_images(list(targets))
        t.note = f"기사 {len(targets)}건 (대상 조회 {load_sec:.2f}초는 이미지 처리와 동시에)"

    with timing.timed("청크 임베딩") as t:
        built = await build_article_chunks(list(targets.values()), image_chunks)
        t.note = f"기사 {len(built)}건, 청크 {sum(len(c) for c in built.values())}개 (기사별 요청을 동시에)"
    with timing.timed("청크 저장(DB)") as t:
        # 모든 기사의 청크를 모아서 한 번에 (예전: 기사마다 연결 1번 + 쿼리 3번)
        await asyncio.to_thread(db.save_chunks_many, list(built.items()))
        t.note = f"기사 {len(built)}건 (한 번에)"
    for chunks in built.values():
        result.articles += 1
        result.body_chunks += sum(c.source == "body" for c in chunks)
        result.image_chunks += sum(c.source == "image" for c in chunks)
    return result
