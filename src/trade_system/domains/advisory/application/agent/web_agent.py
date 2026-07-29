import logging
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page, Browser, Playwright
import pandas as pd
from io import StringIO

LOGGER = logging.getLogger(__name__)

class WebScrapingAgent:
    """
    An agent capable of fetching and extracting data from webpages.
    It uses Playwright to handle Javascript-heavy pages dynamically, 
    and BeautifulSoup for robust HTML parsing.
    """
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None

    def start(self):
        """Start the Playwright browser session."""
        if self._playwright is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            LOGGER.info("WebScrapingAgent: Browser session started.")

    def stop(self):
        """Close the Playwright browser session."""
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        LOGGER.info("WebScrapingAgent: Browser session stopped.")

    def fetch_page_content(self, url: str, wait_for_selector: Optional[str] = None, timeout: int = 30000) -> str:
        """
        Navigates to the URL and returns the fully rendered HTML.
        Can optionally wait for a specific CSS selector to render.
        """
        if not self._browser:
            self.start()

        page = self._browser.new_page()
        try:
            LOGGER.info(f"Navigating to {url}")
            page.goto(url, timeout=timeout)
            
            if wait_for_selector:
                LOGGER.info(f"Waiting for selector: {wait_for_selector}")
                page.wait_for_selector(wait_for_selector, timeout=timeout)
                
            # Allow network to idle just in case
            page.wait_for_load_state("networkidle", timeout=timeout)
            
            html = page.content()
            return html
        except Exception as e:
            LOGGER.error(f"Failed to fetch {url}: {e}")
            raise
        finally:
            page.close()

    def extract_tables_to_pandas(self, html: str) -> List[pd.DataFrame]:
        """
        Extracts all HTML tables from the provided HTML and returns them as Pandas DataFrames.
        """
        try:
            dfs = pd.read_html(StringIO(html))
            LOGGER.info(f"Extracted {len(dfs)} tables from HTML.")
            return dfs
        except ValueError:
            # pd.read_html raises ValueError if no tables are found
            LOGGER.warning("No HTML tables found in the provided content.")
            return []

    def extract_text(self, html: str, selector: str) -> List[str]:
        """
        Extracts the text content of all elements matching the CSS selector.
        """
        soup = BeautifulSoup(html, 'html.parser')
        elements = soup.select(selector)
        return [el.get_text(strip=True) for el in elements]

    def interact_and_extract(self, url: str, interactions: List[dict]) -> str:
        """
        Performs a series of interactions (clicks, fills) and returns the resulting HTML.
        interactions format:
        [
            {"action": "fill", "selector": "#username", "value": "myuser"},
            {"action": "click", "selector": "#submit"},
            {"action": "wait", "selector": ".dashboard"}
        ]
        """
        if not self._browser:
            self.start()

        page = self._browser.new_page()
        try:
            page.goto(url)
            for step in interactions:
                action = step.get("action")
                selector = step.get("selector")
                
                if action == "fill":
                    page.fill(selector, step.get("value", ""))
                elif action == "click":
                    page.click(selector)
                elif action == "wait":
                    page.wait_for_selector(selector)
                elif action == "wait_for_timeout":
                    page.wait_for_timeout(step.get("value", 1000))
                    
            return page.content()
        finally:
            page.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
