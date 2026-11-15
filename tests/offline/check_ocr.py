"""
이미지 속 글자 읽기(OCR) — 결과 정리, 인덱싱의 OCR 단계, 이미지 청크 글 (가짜 OCR·가짜 DB)

EasyOCR·DB·네트워크를 실제로 부르지 않는다. 실행: python tests/run_offline.py
"""
import asyncio
import sys
from contextlib import contextmanager

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from qupict import db, indexing as I, ocr  # noqa: E402
from qupict.schemas import Article  # noqa: E402


def box(x, y, w=40, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


# 1) OCR 결과 정리: 줄 순서(위 → 아래, 왼쪽 → 오른쪽), 자신 없는 조각 버림, 짧으면 "" (워터마크 등)
results = [(box(200, 10), "매출", 0.9), (box(10, 12), "카카오 3분기", 0.95), (box(10, 60), "2조 867억원", 0.88),
           (box(150, 60), "영업이익 2403억원", 0.8), (box(300, 300), "흐릿", 0.1)]
assert ocr.clean_ocr_result(results) == "카카오 3분기 매출 / 2조 867억원 영업이익 2403억원"
assert ocr.clean_ocr_result([(box(0, 0), "ⓒ뉴스1", 0.99)]) == ""           # 워터마크만 → 글자 없음
assert ocr.clean_ocr_result([]) == ""
long = [(box(0, i * 30), "가" * 50, 0.9) for i in range(30)]
assert len(ocr.clean_ocr_result(long)) == ocr.MAX_TEXT_CHARS
assert ocr.ocr_text_chunk("실적 그래프", "매출 2조") == "[기사 이미지] 사진 설명(기사): 실적 그래프 / 이미지 속 글자: 매출 2조"
assert ocr.ocr_text_chunk("", "매출 2조") == "[기사 이미지] 이미지 속 글자: 매출 2조"
print("1 ok: OCR 결과 정리 · 이미지 청크 글")

# 2) 인덱싱 OCR 단계: 기사당 상한, 글자 없음/내려받기 실패는 ""로 저장(다시 안 읽음), 글자 생긴 기사만 다시 청크
imgs = [db.ImageToDescribe(1, "A", "https://img/a1.jpg", "그래프"), db.ImageToDescribe(2, "A", "https://img/a2.jpg", ""),
        db.ImageToDescribe(3, "A", "https://img/a3.jpg", ""), db.ImageToDescribe(4, "B", "https://img/b1.jpg", ""),
        db.ImageToDescribe(5, "C", "https://img/c1.jpg", "")]
store: dict = {}                                     # 가짜 DB: image_id → (글, 모델)
db.get_images_to_describe = lambda tid, statuses=("관련",): [i for i in imgs if i.image_id not in store]
I._processed_image_counts = lambda links: {l: sum(1 for i in imgs if i.article_link == l and i.image_id in store)
                                           for l in links}
db.save_image_descriptions = lambda items, model=None: store.update({i: (t, model) for i, t in items})

async def fake_download(articles, concurrency=8):
    for a in articles:
        for img in a.images:
            if "c1" not in img.url:                  # C의 이미지는 내려받기 실패
                img.data = img.url.encode()
    return 0, 0
I.download_images = fake_download
ocr.read_image_text = lambda data: {"https://img/a1.jpg": "카카오 3분기 매출 2조 867억원"}.get(data.decode(), "")

changed, with_text, no_text, failed = asyncio.run(I._ocr_topic_images(9, ("관련", "보류")))
assert changed == {"A"} and (with_text, no_text, failed) == (1, 2, 1), (changed, with_text, no_text, failed)
assert store == {1: ("카카오 3분기 매출 2조 867억원", "easyocr"), 2: ("", "easyocr"),
                 4: ("", "easyocr"), 5: ("", "easyocr")}       # A는 2장까지(3번 안 읽음), C는 실패도 "" 저장
assert asyncio.run(I._ocr_topic_images(9, ("관련",))) == (set(), 0, 0, 0)   # 다시 돌려도 A의 3번은 상한이라 안 읽음
print("2 ok: OCR 단계 (기사당 상한 · 글자 없음 · 내려받기 실패 · 재실행)")

# 3) 이미지 청크 글: OCR로 읽은 건 "이미지 속 글자", VLM 설명은 "사진 내용", 글자 없음("")은 청크 없음
rows = [{"link": "A", "id": 1, "caption": "그래프", "llm_description": "매출 2조", "described_model": "easyocr"},
        {"link": "A", "id": 2, "caption": "", "llm_description": "", "described_model": "easyocr"},
        {"link": "B", "id": 4, "caption": "", "llm_description": "무대 위의 남성", "described_model": "gemini"}]

class Conn:
    def execute(self, sql, params=None): return self
    def fetchall(self): return rows

@contextmanager
def fake_connect():
    yield Conn()
real_connect, db.connect = db.connect, fake_connect
try:
    got = I.described_images(["A", "B"])
finally:
    db.connect = real_connect
assert got == {"A": [(1, "[기사 이미지] 사진 설명(기사): 그래프 / 이미지 속 글자: 매출 2조")],
               "B": [(4, "[기사 이미지] 사진 내용: 무대 위의 남성")]}, got
print("3 ok: 이미지 청크 글 (OCR / VLM / 글자 없음)")

# 4) index_topic(ocr_images=True): OCR로 글자가 생긴 기사(이미 청크 있음)도 다시 청크를 만든다
saved_chunks = {}
db.get_articles_without_chunks = lambda tid, statuses=("관련",): []          # 본문 청크는 이미 다 있음
db.get_topic_articles = lambda tid, **k: [Article(title="제목A", link="A", content="본문 하나. 본문 둘.")]
db.save_chunks_many = lambda items, model=None: saved_chunks.update({l: [(c.source, c.image_id) for c in cs]
                                                                     for l, cs in items})
store.clear()
I.described_images = lambda links: {"A": [(1, "[기사 이미지] 이미지 속 글자: 카카오 3분기 매출 2조 867억원")]} \
    if "A" in links and store.get(1, ("",))[0] else {}

class FakeEmb:
    async def aembed_documents(self, texts): return [[0.1] * 3 for _ in texts]
I.get_embeddings = lambda: FakeEmb()
r = asyncio.run(I.index_topic(9, describe_images=False, ocr_images=True))
assert saved_chunks == {"A": [("body", None), ("image", 1)]}, saved_chunks
assert (r.images_described, r.images_no_text, r.images_failed, r.image_chunks) == (1, 2, 1, 1), r
assert "글자 없음 2장" in r.summary()
print("4 ok: index_topic(ocr_images=True) → 이미지 청크 추가")
print("ALL PASS")
