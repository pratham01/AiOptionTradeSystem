import sys
from pathlib import Path

filepath = 'src/trade_system/interfaces/live/collector.py'
with open(filepath, 'r') as f:
    content = f.read()

# 1. Add Import
import_str = "from trade_system.domains.market_data.infrastructure.database.repository import save_market_data_batch, save_option_chain_batch"
new_import = import_str + "\nfrom trade_system.domains.analysis.application.analysis.breakout_screener import BreakoutScreener"
content = content.replace(import_str, new_import)

# 2. Add properties
init_str = "        self.ict_signal_events: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in symbols}"
new_init = init_str + """
        
        self.breakout_screener = BreakoutScreener()
        self.breakout_thread: threading.Thread | None = None
        self.last_breakout_alerts: dict[str, pd.Timestamp] = {}"""
content = content.replace(init_str, new_init)

# 3. Add to start()
start_str = "        self._start_option_chain_thread()"
new_start = start_str + "\n        self._start_breakout_thread()"
content = content.replace(start_str, new_start)

# 4. Add the thread logic
oc_thread_str = "    def _start_option_chain_thread(self) -> None:"
new_thread = """    def _start_breakout_thread(self) -> None:
        if self.breakout_thread and self.breakout_thread.is_alive():
            return
            
        def _breakout_loop():
            LOGGER.info("Starting live Intraday Sector Breakout Screener loop (5m interval).")
            while self.ws_running and not self.shutdown:
                try:
                    now = self._now_ist()
                    # Only run between 9:15 and 15:30
                    if self._parse_clock(self.settings.market_start) <= now.time() < self._parse_clock(self.settings.market_end):
                        breakouts = self.breakout_screener.scan_for_breakouts()
                        if breakouts:
                            self._send_breakout_alerts(breakouts)
                except Exception as e:
                    LOGGER.error(f"Error in breakout screener loop: {e}")
                
                time.sleep(300) # Wait 5 minutes before checking again
                
        self.breakout_thread = threading.Thread(target=_breakout_loop, daemon=True)
        self.breakout_thread.start()

    def _send_breakout_alerts(self, breakouts: list[dict]):
        now = self._now_ist()
        alerts_to_send = []
        for b in breakouts:
            sym = b['symbol']
            if sym in self.last_breakout_alerts:
                if (now - self.last_breakout_alerts[sym]).total_seconds() < 3600:
                    continue
            alerts_to_send.append(b)
            self.last_breakout_alerts[sym] = now
            
        if not alerts_to_send or not self.notifier:
            return
            
        lines = [
            f"🚀 <b>[LIVE SECTOR BREAKOUT]</b>",
            f"Time: {now.strftime('%H:%M')}",
            ""
        ]
        
        for b in alerts_to_send:
            clean_sym = b['symbol'].replace("NSE:", "").replace("-EQ", "")
            lines.append(f"📈 <b>{clean_sym}</b> ({b['sector']})")
            lines.append(f"Breakout Spot: ₹{b['close']:.2f} (Above ORB: ₹{b['orb_high']:.2f})")
            lines.append(f"Volume Surge: {b['volume']} (vs 20SMA {b['vol_sma']:.0f})")
            lines.append("")
            
        self.notifier.send("\\n".join(lines))
        LOGGER.info(f"Sent Live Breakout alerts for {len(alerts_to_send)} symbols.")

""" + oc_thread_str
content = content.replace(oc_thread_str, new_thread)

with open(filepath, 'w') as f:
    f.write(content)
print("Collector Patched Successfully!")
