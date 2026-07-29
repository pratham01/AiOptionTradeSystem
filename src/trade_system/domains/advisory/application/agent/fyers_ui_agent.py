import time
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import pyotp
from playwright.sync_api import sync_playwright

from trade_system.shared.config import Settings

LOGGER = logging.getLogger(__name__)

class FyersUiAgent:
    """
    Automated Agent to interact with Fyers Web UI.
    Can extract visual information or capture screenshots.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.load()
        # Direct login URL often avoids some splash screens
        self.url = "https://login.fyers.in/"
        self.user_id = self.settings.fyers.user_id
        self.password = self.settings.fyers.secret_key
        self.pin = self.settings.fyers.pin
        self.totp_secret = self.settings.fyers.totp_secret

    def login_and_capture(self, symbol: str = "NSE:NIFTY50-INDEX") -> str:
        """
        Logs into Fyers Web and captures a screenshot of the specified symbol.
        """
        LOGGER.info(f"FyersUiAgent: Attempting automated login for {self.user_id}...")
        
        with sync_playwright() as p:
            # 1. Launch Browser with a real-world User Agent
            browser = p.chromium.launch(headless=True)
            user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent=user_agent
            )
            page = context.new_page()
            
            try:
                # 2. Go to Login
                LOGGER.info(f"Navigating to {self.url}...")
                page.goto(self.url, wait_until="load", timeout=60000)
                
                # Diagnostic: Log title
                LOGGER.info(f"Page loaded. Title: {page.title()}")
                
                # Check for bot detection keywords
                body_text = page.inner_text("body")
                if "checking your browser" in body_text or "Access Denied" in body_text:
                    LOGGER.error("Fyers has blocked the automated browser (Cloudflare/Incapsula detection).")
                    raise RuntimeError("Bot detected by Fyers security.")

                # 3. Enter User ID
                try:
                    page.wait_for_selector("#login_id", timeout=5000)
                    page.fill("#login_id", self.user_id)
                except:
                    LOGGER.warning("Specific #login_id not found, inspecting page elements...")
                    # Log all inputs for debugging
                    inputs = page.query_selector_all("input")
                    for i, inp in enumerate(inputs):
                        LOGGER.info(f"Input {i}: id={inp.get_attribute('id')}, name={inp.get_attribute('name')}, placeholder={inp.get_attribute('placeholder')}")
                    
                    # Fallback: try common placeholders or first input
                    try:
                        page.get_by_placeholder("Enter Client ID").fill(self.user_id)
                    except:
                        try:
                            # Try to find any input that looks like a user id field
                            page.locator("input[type='text']").first.fill(self.user_id)
                        except:
                            raise RuntimeError("Could not find User ID input field.")
                
                # Find and click submit
                try:
                    page.click("#login_id_submit", timeout=5000)
                except:
                    # Try to find a button with 'Login' or 'Submit' or just the first button
                    page.locator("button:has-text('Login'), button:has-text('Submit'), button[type='submit']").first.click()
                
                # 4. Enter Password (if requested)
                try:
                    page.wait_for_selector("#password", timeout=5000)
                    page.fill("#password", self.password)
                    page.click("#password_submit")
                except:
                    LOGGER.info("Password field not requested (likely direct to TOTP).")

                # 5. Handle TOTP
                try:
                    # Look for TOTP input
                    page.wait_for_selector("input[name='otp1']", timeout=10000)
                    LOGGER.info("FyersUiAgent: Generating TOTP...")
                    totp = pyotp.TOTP(self.totp_secret).now()
                    for i, digit in enumerate(totp):
                        page.fill(f"input[name='otp{i+1}']", digit)
                    page.click("#totp_submit")
                except:
                    # Fallback for single input box
                    try:
                        page.wait_for_selector("#totp_input", timeout=5000)
                        totp = pyotp.TOTP(self.totp_secret).now()
                        page.fill("#totp_input", totp)
                        page.click("#totp_submit")
                    except:
                        LOGGER.warning("TOTP field not found or already logged in.")

                # 6. Enter PIN
                try:
                    page.wait_for_selector("input[name='pin1']", timeout=10000)
                    for i, digit in enumerate(self.pin):
                        page.fill(f"input[name='pin{i+1}']", digit)
                    page.click("#pin_submit")
                except:
                    LOGGER.warning("PIN field not found.")

                # 7. Navigate to Chart / Capture Data
                # Wait for the dashboard to load (look for standard trading layout elements)
                page.wait_for_selector(".main-content, .tradingview-chart, .dashboard-container", timeout=30000)
                LOGGER.info("FyersUiAgent: Login successful.")
                
                # Capture screenshot
                screenshot_path = Path("reports/fyers_ui_snapshot.png")
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot_path), full_page=True)
                
                return f"Captured Fyers UI state to {screenshot_path}"

            except Exception as e:
                # TAKE DIAGNOSTIC SCREENSHOT
                diag_path = Path("reports/diagnostic_fail.png")
                diag_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(diag_path))
                LOGGER.error(f"FyersUiAgent: Web automation failed. Diagnostic saved to {diag_path}. Error: {e}")
                return f"UI Extraction Error (See diagnostic_fail.png): {e}"
            finally:
                browser.close()

if __name__ == "__main__":
    agent = FyersUiAgent()
    print(agent.login_and_capture())
