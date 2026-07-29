"""
Direct scraper for top business news headlines from MoneyControl and Economic Times.
Used to provide high-signal macro context to the AI Trading Agents.
"""
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime

LOGGER = logging.getLogger(__name__)

class BusinessNewsScraper:
    """
    Scrapes high-impact financial news headlines from primary Indian sources.
    """
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def fetch_moneycontrol_top_stories(self) -> list[str]:
        """Scrape top stories from MoneyControl homepage."""
        url = "https://www.moneycontrol.com/news/business/"
        headlines = []
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # MoneyControl updated selectors (2024 structure)
            # 1. Main stories in the business section
            news_items = soup.select('.fleft h2 a, .news_listing li a')
            for item in news_items:
                title = item.get_text(strip=True)
                if len(title) > 20 and title not in headlines: # Basic filter for noise
                    headlines.append(title)
            
            # 2. Latest news items
            if not headlines:
                news_items = soup.find_all('li', class_='clearfix')
                for item in news_items:
                    link = item.find('a')
                    if link and link.get('title'):
                        headlines.append(link['title'])

        except Exception as e:
            LOGGER.error(f"Failed to scrape MoneyControl: {e}")
        
        return headlines[:15]

    def fetch_et_top_stories(self) -> list[str]:
        """Scrape top stories from Economic Times Business page."""
        url = "https://economictimes.indiatimes.com/news/economy"
        headlines = []
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # ET often uses specific tags for headlines in their news stream
            news_items = soup.select('.eachStory h3 a, .eachStory h2 a')
            for item in news_items[:8]:
                title = item.get_text(strip=True)
                if title:
                    headlines.append(title)

        except Exception as e:
            LOGGER.error(f"Failed to scrape Economic Times: {e}")
            
        return headlines

    def fetch_mint_top_stories(self) -> list[str]:
        """Scrape top business stories from LiveMint."""
        url = "https://www.livemint.com/market"
        headlines = []
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # LiveMint market section selectors
            news_items = soup.select('.headline a, .listview h2 a, .cardheading a')
            for item in news_items:
                title = item.get_text(strip=True)
                if len(title) > 25 and title not in headlines:
                    headlines.append(title)
        except Exception as e:
            LOGGER.error(f"Failed to scrape LiveMint: {e}")
        return headlines[:12]

    def fetch_ndtv_profit_top_stories(self) -> list[str]:
        """Scrape headlines from NDTV Profit."""
        url = "https://www.ndtvprofit.com/latest"
        headlines = []
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # NDTV Profit selectors
            news_items = soup.select('h2 a, .list-news-title a')
            for item in news_items:
                title = item.get_text(strip=True)
                if len(title) > 25 and title not in headlines:
                    headlines.append(title)
        except Exception as e:
            LOGGER.error(f"Failed to scrape NDTV Profit: {e}")
        return headlines[:10]

    def get_all_headlines(self) -> list[str]:
        """Consolidate headlines from multiple sources."""
        all_news = []
        all_news.extend(self.fetch_moneycontrol_top_stories())
        all_news.extend(self.fetch_et_top_stories())
        all_news.extend(self.fetch_mint_top_stories())
        all_news.extend(self.fetch_ndtv_profit_top_stories())
        return list(set(all_news)) # Deduplicate

if __name__ == "__main__":
    scraper = BusinessNewsScraper()
    print(f"--- MoneyControl Headlines ({datetime.now()}) ---")
    for h in scraper.fetch_moneycontrol_top_stories():
        print(f"• {h}")
