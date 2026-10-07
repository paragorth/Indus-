"""la82: load a JS page in Chromium, screenshot it and dump its visible text and links. usage: la82_shot.py URL OUTPREFIX [wait_ms]"""
import os, sys, asyncio, json
from playwright.async_api import async_playwright
url, out = sys.argv[1], sys.argv[2]; wait = int(sys.argv[3]) if len(sys.argv) > 3 else 9000
async def main():
    proxy = os.environ.get('HTTPS_PROXY')
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome', proxy={'server': proxy} if proxy else None, args=['--no-sandbox'], headless=True)
        pg = await (await b.new_context(viewport={'width': 1400, 'height': 2000})).new_page()
        try: await pg.goto(url, timeout=60000)
        except Exception as e: print('goto', str(e)[:100])
        await pg.wait_for_timeout(wait)
        await pg.screenshot(path=out + '.png', full_page=False)
        txt = await pg.evaluate('document.body ? document.body.innerText : ""')
        links = await pg.evaluate('Array.from(document.querySelectorAll("a")).map(a=>[a.innerText.trim().slice(0,60), a.href])')
        open(out + '.txt', 'w').write(txt); json.dump(links, open(out + '.links.json', 'w'))
        print('ok', len(txt), len(links)); await b.close()
asyncio.run(main())
