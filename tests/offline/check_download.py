"""
1단계: 이미지 파일 내려받기 (로컬 테스트 서버)

LLM·네이버·DB를 실제로 부르지 않는다 (크레딧 0). 실행: python tests/run_offline.py
"""
import asyncio, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from aiohttp import web
from aiohttp.test_utils import TestServer
from qupict import collect
from qupict.schemas import Article, ArticleImage

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * (6 * 1024)
seen_referer = {}
async def ok(req):  seen_referer["v"] = req.headers.get("Referer"); return web.Response(body=PNG, content_type="image/png")
async def html(req): return web.Response(text="<html></html>", content_type="text/html")
async def missing(req): return web.Response(status=404)
async def tiny(req): return web.Response(body=b"x" * 500, content_type="image/jpeg")
async def huge(req): return web.Response(body=b"x" * (collect.MAX_IMAGE_BYTES + 10), content_type="image/jpeg")

async def main():
    app = web.Application()
    app.router.add_get("/ok.png", ok); app.router.add_get("/page", html)
    app.router.add_get("/missing.jpg", missing); app.router.add_get("/huge.jpg", huge); app.router.add_get("/tiny.jpg", tiny)
    async with TestServer(app) as server:
        u = lambda path: str(server.make_url(path))
        done = ArticleImage(url=u("/missing.jpg"), data=b"already")  # 이미 받은 건 건너뜀
        a = Article(title="t", link="https://press.com/article/1", images=[
            ArticleImage(url=u("/ok.png")), ArticleImage(url=u("/page")),
            ArticleImage(url=u("/missing.jpg")), ArticleImage(url=u("/huge.jpg")), done, ArticleImage(url=u("/tiny.jpg"))])
        got, total = await collect.download_images([a, Article(title="no images", link="x")])
        imgs = a.images
        assert (got, total) == (2, 6), (got, total)
        assert imgs[5].data is None   # 5KB 미만
        assert imgs[0].data == PNG and imgs[0].content_type == "image/png"
        assert imgs[1].data is None   # 이미지가 아닌 응답(HTML)
        assert imgs[2].data is None   # 404
        assert imgs[3].data is None   # 5MB 초과
        assert imgs[4].data == b"already"
        assert seen_referer["v"] == "https://press.com/article/1"
        assert "data=" not in repr(imgs[0])   # 출력할 때 바이트가 쏟아지지 않음
    print("ok: download_images (성공 / HTML / 404 / 5MB 초과 / 5KB 미만 / 이미 받은 것 / Referer / repr)")
asyncio.run(main())
