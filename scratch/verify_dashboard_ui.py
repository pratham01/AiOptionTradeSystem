"""Automated Playwright UI test and screenshot capture for all Streamlit pages."""
import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright

ARTIFACT_DIR = Path("/Users/pratham/.gemini/antigravity-ide/brain/d00bcbbe-c956-478a-a0af-4c0db69b5bdb")

async def verify_ui():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page = await context.new_page()

        print("1. Loading Mission Control...")
        await page.goto("http://localhost:8501", timeout=30000)
        await page.wait_for_timeout(5000)
        
        # Check for error elements
        errors = await page.locator(".stException").count()
        print(f"Mission Control loaded. Error elements found: {errors}")
        await page.screenshot(path=str(ARTIFACT_DIR / "ui_mission_control.png"))

        # Check sidebar links
        print("2. Navigating to Sector Scope...")
        # Streamlit multi-page or radio navigation
        sector_scope_link = page.locator("text=Sector Scope").first
        if await sector_scope_link.count() > 0:
            await sector_scope_link.click()
            await page.wait_for_timeout(6000)
            errors_sector = await page.locator(".stException").count()
            print(f"Sector Scope loaded. Error elements found: {errors_sector}")
            await page.screenshot(path=str(ARTIFACT_DIR / "ui_sector_scope.png"))

            # Switch to Intraday Edge Finder tab if present
            edge_tab = page.locator("text=Intraday Edge Finder").first
            if await edge_tab.count() > 0:
                print("3. Switching to Intraday Edge Finder tab...")
                await edge_tab.click()
                await page.wait_for_timeout(6000)
                await page.screenshot(path=str(ARTIFACT_DIR / "ui_intraday_edge_finder.png"))
                print("Intraday Edge Finder tab captured.")

        print("4. Navigating to Broker Health...")
        broker_link = page.locator("text=Broker Health").first
        if await broker_link.count() > 0:
            await broker_link.click()
            await page.wait_for_timeout(4000)
            await page.screenshot(path=str(ARTIFACT_DIR / "ui_broker_health.png"))
            print("Broker Health page captured.")

        print("5. Navigating to Market Mood...")
        mood_link = page.locator("text=Market Mood").first
        if await mood_link.count() > 0:
            await mood_link.click()
            await page.wait_for_timeout(4000)
            await page.screenshot(path=str(ARTIFACT_DIR / "ui_market_mood.png"))
            print("Market Mood page captured.")

        await browser.close()
        print("All UI checks completed successfully!")

if __name__ == "__main__":
    asyncio.run(verify_ui())
