"""Capture app screenshots for the deck/report (Playwright + pre-installed Chromium)."""
import asyncio, sys
from playwright.async_api import async_playwright
BASE = "http://127.0.0.1:8000"; OUT = "/home/claude/shots"
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = await b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        await pg.goto(BASE + "/"); await pg.wait_for_timeout(1500); await pg.screenshot(path=f"{OUT}/home.png")
        await pg.goto(BASE + "/capsule/"); await pg.wait_for_timeout(1200)
        btns = pg.locator("button[title*='SEE-AI']"); await btns.first.click()
        await pg.wait_for_selector("text=Most likely finding", timeout=60000); await pg.wait_for_timeout(1200)
        await pg.screenshot(path=f"{OUT}/capsule.png")
        await pg.goto(BASE + "/study/"); await pg.wait_for_timeout(800)
        await pg.get_by_text("Load demo study").click(); await pg.wait_for_selector("text=Review queue", timeout=180000); await pg.wait_for_timeout(1200)
        await pg.screenshot(path=f"{OUT}/study.png")
        await pg.goto(BASE + "/results/"); await pg.wait_for_timeout(2500); await pg.screenshot(path=f"{OUT}/results.png", full_page=True)
        await pg.goto(BASE + "/flare/"); await pg.wait_for_timeout(2500); await pg.screenshot(path=f"{OUT}/flare.png")
        await pg.goto(BASE + "/uc/"); await pg.wait_for_timeout(1000)
        await pg.locator("button[title*='HyperKvasir']").first.click(); await pg.wait_for_selector("text=Predicted grade", timeout=60000); await pg.wait_for_timeout(1000)
        await pg.screenshot(path=f"{OUT}/uc.png")
        await b.close()
asyncio.run(main())
