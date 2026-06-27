from datetime import date
from trade_system.application.analysis.breakout_screener import BreakoutScreener

screener = BreakoutScreener()
alerts = screener.scan_for_breakouts(
    top_sectors_count=3,
    target_date=date(2026, 5, 27),
    use_vwap_filter=True,
    use_wick_filter=True,
    use_index_filter=True,
    use_sector_filter=False,  # Bypass sector filter to see all stock breakouts
    vol_surge_threshold=1.5
)

print(f"\nTotal alerts triggered: {len(alerts)}")
for a in alerts:
    print(f"[{a['trigger_time'].strftime('%H:%M')}] {a['symbol']} ({a['sector']}) - {a['direction']}: Close={a['close']:.2f}, Vol Ratio={a['volume']/a['vol_sma']:.2f}x, pChange={a['pchange']:.2f}%")
