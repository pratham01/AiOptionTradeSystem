from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WebsiteAgentConfig:
    url: str
    username: str
    password: str
    username_selector: str
    password_selector: str
    submit_selector: str
    wait_for_selector: str
    extract_selectors: list[str]
    headless: bool = True


class WebsiteDashboardAgent:
    """Generic website login + dashboard extractor using Playwright."""

    def __init__(self, config: WebsiteAgentConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Playwright is required. Install with: pip install playwright && playwright install chromium"
            ) from exc

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.config.headless)
            page = browser.new_page()
            page.goto(self.config.url, wait_until="domcontentloaded")

            page.fill(self.config.username_selector, self.config.username)
            page.fill(self.config.password_selector, self.config.password)
            page.click(self.config.submit_selector)
            page.wait_for_selector(self.config.wait_for_selector, timeout=30000)

            extracted: dict[str, str] = {}
            for selector in self.config.extract_selectors:
                try:
                    text = page.locator(selector).first.inner_text(timeout=5000)
                    extracted[selector] = text.strip()
                except Exception:
                    extracted[selector] = ""

            browser.close()

        return {
            "url": self.config.url,
            "status": "ok",
            "extracted": extracted,
        }


def load_config(path: str | Path) -> WebsiteAgentConfig:
    data = json.loads(Path(path).read_text())
    return WebsiteAgentConfig(
        url=data["url"],
        username=data["username"],
        password=data["password"],
        username_selector=data["username_selector"],
        password_selector=data["password_selector"],
        submit_selector=data["submit_selector"],
        wait_for_selector=data["wait_for_selector"],
        extract_selectors=data.get("extract_selectors", []),
        headless=bool(data.get("headless", True)),
    )
