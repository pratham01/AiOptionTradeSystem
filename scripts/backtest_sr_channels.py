import sys
import logging
from datetime import datetime, date
from pathlib import Path
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.strategy.application.indicators.support_resistance_channels import SupportResistanceChannelDetector

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def run_backtest():
    engine = get_engine()
    
    query = text("""
        SELECT timestamp, open, high, low, close, volume 
        FROM ohlcv_15m 
        WHERE symbol = 'NSE:NIFTY50-INDEX' AND timestamp >= '2026-01-01'
        ORDER BY timestamp ASC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
    if df.empty:
        LOGGER.error("No data found for NIFTY50 since Jan 2026.")
        return
        
    LOGGER.info(f"Loaded {len(df)} 15m candles.")
    
    detector = SupportResistanceChannelDetector(loopback=290, pivot_period=10)
    snapshots = detector.calculate(df)
    
    stats = {
        "support_touches": 0,
        "support_respected": 0,
        "support_broken": 0,
        "resistance_touches": 0,
        "resistance_respected": 0,
        "resistance_broken": 0
    }
    
    active_touches = []
    detailed_events = []
    
    for i in range(len(df)):
        if i >= len(snapshots) or snapshots[i] is None:
            continue
            
        bar = df.iloc[i]
        snap = snapshots[i]
        high = float(bar['high'])
        low = float(bar['low'])
        close = float(bar['close'])
        timestamp = bar['timestamp']
        
        resolved_touches = []
        for t in active_touches:
            t['bars_elapsed'] += 1
            
            if t['type'] == 'SUPPORT':
                if close < t['bottom']:
                    stats["support_broken"] += 1
                    t['outcome'] = 'BROKEN'
                    t['resolution_time'] = timestamp
                    t['resolution_price'] = close
                    detailed_events.append(t)
                    resolved_touches.append(t)
                    continue
            else: # resistance
                if close > t['top']:
                    stats["resistance_broken"] += 1
                    t['outcome'] = 'BROKEN'
                    t['resolution_time'] = timestamp
                    t['resolution_price'] = close
                    detailed_events.append(t)
                    resolved_touches.append(t)
                    continue
                    
            if t['bars_elapsed'] >= 3:
                if t['type'] == 'SUPPORT':
                    stats["support_respected"] += 1
                else:
                    stats["resistance_respected"] += 1
                t['outcome'] = 'RESPECTED'
                t['resolution_time'] = timestamp
                t['resolution_price'] = close
                detailed_events.append(t)
                resolved_touches.append(t)
                
        for r in resolved_touches:
            active_touches.remove(r)
            
        for ch in snap.channels:
            buffer = close * 0.001
            ch_bottom = ch.low
            ch_top = ch.high
            
            if low <= ch_top + buffer and high >= ch_bottom - buffer:
                already_tracked = False
                for t in active_touches:
                    if abs(t['bottom'] - ch_bottom) < 10.0 and abs(t['top'] - ch_top) < 10.0:
                        already_tracked = True
                        break
                        
                if not already_tracked:
                    ch_type = ch.channel_type
                    if ch_type == "inside":
                        ch_type = "support" if close >= ch_top else "resistance"
                    
                    if ch_type == "support":
                        stats["support_touches"] += 1
                    elif ch_type == "resistance":
                        stats["resistance_touches"] += 1
                    else:
                        continue
                        
                    active_touches.append({
                        "touch_time": timestamp,
                        "type": ch_type.upper(),
                        "channel_bottom": round(ch_bottom, 2),
                        "channel_top": round(ch_top, 2),
                        "entry_price": round(close, 2),
                        "bars_elapsed": 0,
                        "bottom": ch_bottom,
                        "top": ch_top
                    })

    for t in active_touches:
        t['outcome'] = 'RESPECTED'
        t['resolution_time'] = df.iloc[-1]['timestamp']
        t['resolution_price'] = round(df.iloc[-1]['close'], 2)
        detailed_events.append(t)
        
        if t['type'] == 'SUPPORT':
            stats["support_respected"] += 1
        elif t['type'] == 'RESISTANCE':
            stats["resistance_respected"] += 1

    events_df = pd.DataFrame(detailed_events)
    cols = ['touch_time', 'type', 'channel_bottom', 'channel_top', 'entry_price', 'outcome', 'resolution_time', 'resolution_price']
    events_df = events_df[cols]
    
    artifact_dir = Path("/Users/pratham/.gemini/antigravity-ide/brain/7af0423f-5367-4a99-a800-f5bdfa28f4dd")
    events_df.to_csv(artifact_dir / "sr_detailed_report.csv", index=False)
    
    md_content = "# SR Detailed Backtest Report\n\n"
    md_content += "The full CSV report is saved as `sr_detailed_report.csv` in your artifacts folder.\n\n"
    md_content += "### Last 100 Touches\n\n"
    md_content += events_df.tail(100).to_markdown(index=False)
    
    with open(artifact_dir / "sr_detailed_report.md", "w") as f:
        f.write(md_content)
        
    print("Detailed report generated successfully.")
    
if __name__ == "__main__":
    run_backtest()
