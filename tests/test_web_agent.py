import pytest
import pandas as pd
from trade_system.domains.advisory.application.agent.web_agent import WebScrapingAgent

def test_web_agent_fetches_content():
    # Simple test for a fast static page
    with WebScrapingAgent(headless=True) as agent:
        html = agent.fetch_page_content("https://example.com")
        assert "Example Domain" in html
        
        # Test text extraction
        texts = agent.extract_text(html, "h1")
        assert texts == ["Example Domain"]
        
def test_web_agent_tables():
    # Construct a simple HTML with a table
    html = """
    <html>
      <body>
        <table>
          <tr><th>Name</th><th>Price</th></tr>
          <tr><td>AAPL</td><td>150</td></tr>
          <tr><td>GOOG</td><td>2800</td></tr>
        </table>
      </body>
    </html>
    """
    
    with WebScrapingAgent(headless=True) as agent:
        dfs = agent.extract_tables_to_pandas(html)
        assert len(dfs) == 1
        df = dfs[0]
        assert "Name" in df.columns
        assert "Price" in df.columns
        assert df.iloc[0]["Name"] == "AAPL"
        assert df.iloc[0]["Price"] == 150
