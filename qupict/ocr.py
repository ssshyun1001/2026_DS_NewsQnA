"""
기사 이미지 속 글자 읽기 (OCR, EasyOCR) — VLM 대신 쓰는 이미지 → 글자 변환

  read_image_text(data)    이미지 파일(bytes) → 이미지 속 글자 (글자가 없거나 너무 짧으면 "")
  ocr_text_chunk(...)      이미지 청크에 넣을 글

VLM과 다른 점
  - 장면을 설명하지 않는다 (인물·현장 사진은 글자가 없어서 결과 없음).
  - 대신 인포그래픽·표·그래프·자막의 숫자와 문구를 그대로 읽는다. LLM 비용이 없다.

속도: GPU(CUDA)가 있으면 이미지 1장 0.2~0.5초, CPU면 1~3초. 그래서 서비스는 "준비 완료" 뒤
백그라운드에서 돌린다 (indexing.index_topic(ocr_images=True), run.add_image_text).

모델(한국어 + 영어, 약 100MB)은 처음 쓸 때 ~/.EasyOCR 폴더에 한 번 내려받는다.
"""
from __future__ import annotations

import re
import threading

OCR_LANGS = ["ko", "en"]
OCR_MODEL_NAME = "easyocr"          # DB(article_images.described_model)에 남기는 이름 — VLM 설명과 구분용
MIN_CONFIDENCE = 0.4                # 이 값보다 자신 없는 글자 조각은 버린다
MIN_TEXT_CHARS = 15                 # 글자(한글·영문·숫자)가 이보다 적으면 "글자 없는 이미지"로 본다
                                    #   (사진 구석의 언론사 워터마크 "ⓒ뉴스1" 같은 것만 있는 경우를 거르려고)
MAX_TEXT_CHARS = 600                # 청크 하나에 넣을 최대 길이 (본문 청크 500자와 비슷하게)

_reader = None
_reader_lock = threading.Lock()     # 모델 불러오기는 한 번만
_run_lock = threading.Lock()        # 한 번에 이미지 하나씩 (같은 모델을 여러 스레드가 동시에 쓰지 않게)


def get_reader():
    """EasyOCR 모델 (처음 한 번만 불러온다. GPU가 있으면 GPU로)."""
    global _reader
    with _reader_lock:
        if _reader is None:
            import easyocr
            import torch
            _reader = easyocr.Reader(OCR_LANGS, gpu=torch.cuda.is_available(), verbose=False)
        return _reader


def uses_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


_CHAR_RE = re.compile(r"[0-9A-Za-z가-힣]")


def clean_ocr_result(results: list) -> str:
    """EasyOCR 결과 [(좌표, 글자, 자신감)] → 위에서 아래, 왼쪽에서 오른쪽 순서로 이은 글.
    자신 없는 조각은 버리고, 글자가 MIN_TEXT_CHARS보다 적으면 ""."""
    pieces = []
    for box, text, conf in results:
        text = text.strip()
        if conf < MIN_CONFIDENCE or not text:
            continue
        top = min(p[1] for p in box)
        left = min(p[0] for p in box)
        pieces.append((top, left, text))
    if not pieces:
        return ""
    # 줄 단위로 묶기: 위쪽 좌표가 비슷하면(글자 높이의 절반 이내) 같은 줄로 보고 왼쪽부터
    pieces.sort()
    lines: list[list[tuple[float, float, str]]] = []
    for top, left, text in pieces:
        if lines and abs(lines[-1][0][0] - top) < 12:
            lines[-1].append((top, left, text))
        else:
            lines.append([(top, left, text)])
    joined = " / ".join(" ".join(t for _, _, t in sorted(line, key=lambda p: p[1])) for line in lines)
    if len(_CHAR_RE.findall(joined)) < MIN_TEXT_CHARS:
        return ""
    return joined[:MAX_TEXT_CHARS]


MAX_OCR_SIDE = 1600                 # 이보다 큰 이미지는 줄여서 읽는다 (속도·GPU 메모리. 기사 글자는 충분히 읽힘)


def _to_array(data: bytes):
    """이미지 파일 → RGB 배열. Pillow로 연다 (PNG 투명 배경 · WebP · GIF 첫 장 · CMYK JPEG도 처리).
    EasyOCR에 파일을 그대로 넘기면 OpenCV가 일부 형식(투명 PNG 등)을 못 다뤄 실패했다."""
    import io

    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im.seek(0)                              # 움직이는 GIF는 첫 장만
        if im.mode in ("RGBA", "LA", "P"):      # 투명 배경은 흰색으로
            im = im.convert("RGBA")
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im, mask=im.split()[-1])
            im = background
        else:
            im = im.convert("RGB")
        if max(im.size) > MAX_OCR_SIDE:
            im.thumbnail((MAX_OCR_SIDE, MAX_OCR_SIDE))
        return np.asarray(im)


def read_image_text(data: bytes) -> str:
    """이미지 파일 → 이미지 속 글자. 글자가 없거나(사진) 너무 짧으면 "". 읽기 실패도 ""."""
    reader = get_reader()
    try:
        array = _to_array(data)
        with _run_lock:
            results = reader.readtext(array)
    except Exception as e:                      # 깨진 이미지·지원하지 않는 형식
        print(f"  [경고] OCR 실패 ({type(e).__name__}: {str(e)[:80]})")
        return ""
    return clean_ocr_result(results)


def ocr_text_chunk(caption: str, text: str) -> str:
    """이미지 청크 내용 (OCR). VLM 설명 청크와 같은 표시([기사 이미지])로 시작해 답변 출처가 '이미지'로 보인다."""
    parts = [f"사진 설명(기사): {caption}" if caption else "", f"이미지 속 글자: {text}"]
    return "[기사 이미지] " + " / ".join(p for p in parts if p)
