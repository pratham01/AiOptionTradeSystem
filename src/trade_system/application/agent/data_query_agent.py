from __future__ import annotations

import pandas as pd
import json
import logging
from trade_system.application.advisory.llm import LlmAdvisorClient
from trade_system.infrastructure.database.connection import get_engine

LOGGER = logging.getLogger(__name__)

DB_SCHEMA_PROMPT = """
You are a data query agent for an algorithmic trading system.
Your job is to answer user questions about the market data or the system.
You have access to a SQLite database with the following key tables:

1. ohlcv_daily (symbol, timestamp, open, high, low, close, volume)
2. top_gainers_stocks (symbol, name, close, open, high, low, volume, change, change_pct, timestamp)
3. suggested_trades (id, date, symbol, direction, horizon, sector, entry_zone_low, entry_zone_high, target, stop_loss, confidence, outcome, actual_pnl_pct)
4. option_chain_data (timestamp, underlying_symbol, symbol, expiry, strike, option_type, ltp, oi, oi_change, volume, iv)

Note that symbols often have prefixes/suffixes like 'NSE:RELIANCE-EQ'. Use LIKE if unsure.

When the user asks a question, if it requires database data, output a valid SQL query starting with 'SQL:' on a new line. 
For example:
SQL: SELECT AVG(volume) FROM ohlcv_daily WHERE symbol LIKE '%RELIANCE%' AND timestamp >= date('now', '-1 month');

If it doesn't require database query, just answer the question directly. Do not output 'SQL:' if you are answering directly.
"""

class DataQueryAgent:
    def __init__(self, llm_client: LlmAdvisorClient | None = None) -> None:
        self.llm = llm_client or LlmAdvisorClient()
        self.engine = get_engine()
        
    async def process_query(self, user_prompt: str) -> str:
        prompt = f"{DB_SCHEMA_PROMPT}\n\nUser Question: {user_prompt}\n\nAnswer:"
        
        response = await self.llm.complete(prompt)
        
        if "SQL:" in response:
            try:
                # Extract the SQL query
                sql_lines = [line.strip() for line in response.splitlines() if line.strip().upper().startswith(("SELECT", "SQL:"))]
                sql_query = ""
                for line in response.splitlines():
                    if line.startswith("SQL:"):
                        sql_query = line.replace("SQL:", "").strip()
                        break
                
                if not sql_query:
                    # Fallback extraction
                    sql_query = response.split("SQL:")[1].strip().split(";")[0] + ";"
                
                # Clean up markdown if present
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

                with self.engine.connect() as conn:
                    df = pd.read_sql(sql_query, conn)
                
                # Ask LLM to format the response
                synthesis_prompt = f"User asked: {user_prompt}\nDatabase result:\n{df.to_string()}\n\nProvide a natural language summary of the result. If the result is empty, say so clearly."
                final_answer = await self.llm.complete(synthesis_prompt)
                return final_answer
            except Exception as e:
                LOGGER.error(f"Error executing SQL: {e}")
                return f"I encountered an error while querying the database: {e}\n\nAttempted query: {sql_query}"
                
        return response
