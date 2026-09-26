import os, sys, asyncio
from playwright.async_api import async_playwright
url=sys.argv[1]; n=int(sys.argv[2]); z=sys.argv[3]; out=sys.argv[4]
async def main():
    proxy=os.environ.get('HTTPS_PROXY')
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome', proxy={'server':proxy} if proxy else None, args=['--no-sandbox'],headless=False)
        ctx=await b.new_context(viewport={'width':1600,'height':2200})
        pg=await ctx.new_page()
        try: await pg.goto(f"{url}#page={n}&zoom={z}",timeout=60000)
        except Exception as e: print('goto',str(e)[:80])
        await pg.wait_for_timeout(9000)
        await pg.screenshot(path=out); print('ok')
        await b.close()
asyncio.run(main())
