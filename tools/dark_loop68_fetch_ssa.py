"""S-DARK-68: download the SSA 'names by state' zip (public domain, ssa.gov refuses non-browser clients) with Chromium.
Usage: xvfb-run -a python3 tools/dark_loop68_fetch_ssa.py [url] [out]"""
import asyncio, os, sys
from playwright.async_api import async_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else 'https://www.ssa.gov/oact/babynames/state/namesbystate.zip'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/home/user/Indus-/data/derived/dark/loop68_corpora/namesbystate.zip'

async def main():
    proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
                                    proxy={'server': proxy} if proxy else None, args=['--no-sandbox'], headless=False)
        ctx = await b.new_context(accept_downloads=True)
        page = await ctx.new_page()
        # visit the landing page first so the download request carries a referer and cookies
        await page.goto('https://www.ssa.gov/oact/babynames/limits.html', wait_until='domcontentloaded', timeout=90000)
        async with page.expect_download(timeout=300000) as dl:
            await page.evaluate(f"window.location.href = '{URL}'")
        d = await dl.value
        await d.save_as(OUT)
        print('saved', OUT, os.path.getsize(OUT))
        await b.close()

asyncio.run(main())
