import sys
import os
import pandas as pd
from datetime import date
from sqlalchemy import text

sys.path.append(os.path.abspath('src'))
from trade_system.interfaces.dashboard.sector_scope_dashboard import _compute_entry_times
from trade_system.domains.market_data.infrastructure.database.connection import get_engine

# 1. Fetch live quotes (simulate merged_closes)
# Actually, let's just see if pivot_closes includes the live tick. No, it only uses tgt_df which is from `SELECT * FROM ohlcv_15m`.

