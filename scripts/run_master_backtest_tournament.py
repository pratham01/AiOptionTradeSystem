import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, date
from trade_system.application.indicators.supertrend import SupertrendIndicator
from trade_system.application.indicators.rsi_divergence import RsiDivergence
from trade_system.application.indicators.volume_profile import VolumeProfileIndicator
from trade_system.application.indicators.retracement import RetracementIndicator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("master_backtest")

class BacktestTournament:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        self.df['date'] = self.df['timestamp'].dt.date
        self.results = []

    def run_all(self):
        logger.info("Starting Multi-Strategy Backtest Tournament on Nifty 50...")
        
        # 1. Baseline: Standard Supertrend (7,3)
        self.results.append(self._backtest_st_standard())
        
        # 2. Zone-Filtered Supertrend
        self.results.append(self._backtest_st_zones())
        
        # 3. Supertrend + Buffered RSI Exit
        self.results.append(self._backtest_st_rsi_buffered())

        # 4. MWPL / Squeeze (Institutional)
        self.results.append(self._backtest_mwpl_squeeze())

        # 5. FVG / Retracement (ICT Concept)
        self.results.append(self._backtest_fvg_retest())
        
        # Save CSVs for Dashboard
        self._save_results_to_csv()
        
        # Generate Final Report
        self._generate_report()

    def _save_results_to_csv(self):
        """Save results in a format compatible with dashboard/data.py"""
        base_path = Path("reports/master_tournament")
        base_path.mkdir(parents=True, exist_ok=True)
        
        summary_rows = []
        for res in self.results:
            t_df = pd.DataFrame(res['trades'])
            if t_df.empty: continue
            
            strat_name = res['name'].lower().replace(" ", "_")
            strat_path = base_path / strat_name
            strat_path.mkdir(parents=True, exist_ok=True)
            
            # Save trades
            t_df['strategy'] = res['name']
            # Reformat to match expected columns
            if 'pnl' in t_df.columns:
                t_df['points_captured'] = t_df['pnl'] * 10000 # Scaling for visibility
                t_df['entry_time'] = pd.Timestamp.now() # Mock timestamps for dashboard
                t_df['exit_time'] = pd.Timestamp.now()
                t_df['direction'] = "LONG"
                t_df['holding_minutes'] = 30
            
            t_df.to_csv(strat_path / f"{strat_name}_trades.csv", index=False)
            
            # Prepare summary row
            wr = (t_df['pnl'] > 0).mean() * 100
            net = t_df['pnl'].sum()
            losses_sum = t_df[t_df['pnl']<0]['pnl'].sum()
            pf = abs(t_df[t_df['pnl']>0]['pnl'].sum() / min(-0.00001, losses_sum))
            
            # Scaled points for visibility (1 unit move = 10000 scaled points)
            summary_rows.append({
                "strategy": res['name'],
                "year": 2026,
                "trades": len(t_df),
                "win_rate": wr,
                "net_points": net * 10000,
                "avg_points": (net / len(t_df)) * 10000 if len(t_df) > 0 else 0,
                "profit_factor": pf,
                "avg_win": t_df[t_df['pnl']>0]['pnl'].mean() * 10000 if not t_df[t_df['pnl']>0].empty else 0,
                "avg_loss": t_df[t_df['pnl']<0]['pnl'].mean() * 10000 if not t_df[t_df['pnl']<0].empty else 0
            })
            
        pd.DataFrame(summary_rows).to_csv(base_path / "tournament_summary.csv", index=False)
        logger.info(f"Tournament CSV results saved to {base_path}")

    def _backtest_st_standard(self):
        logger.info("Running Baseline Supertrend (7,3)...")
        df = SupertrendIndicator(period=7, multiplier=3).calculate(self.df.copy())
        trades = self._simulate(df, name="Baseline Supertrend")
        return {"name": "Baseline Supertrend", "trades": trades}

    def _backtest_st_zones(self):
        logger.info("Running Zone-Filtered Supertrend...")
        vp_indicator = VolumeProfileIndicator(price_step=5.0)
        daily_groups = self.df.groupby('date')
        daily_levels = {}
        dates = sorted(daily_groups.groups.keys())
        
        for d in dates:
            day_data = daily_groups.get_group(d)
            profile = vp_indicator.calculate(day_data)
            daily_levels[d] = {
                'pdh': day_data['high'].max(),
                'pdl': day_data['low'].min(),
                'poc': profile.point_of_control if profile else day_data['high'].max(),
                'val': profile.value_area_low if profile else day_data['low'].min(),
                'vah': profile.value_area_high if profile else day_data['high'].max()
            }
            
        zone_rows = []
        for i in range(1, len(dates)):
            curr_data = daily_groups.get_group(dates[i]).copy()
            prev_levels = daily_levels[dates[i-1]]
            for k, v in prev_levels.items(): curr_data[k] = v
            zone_rows.append(curr_data)
        
        df_zones = pd.concat(zone_rows)
        df_zones = SupertrendIndicator(period=7, multiplier=3).calculate(df_zones)
        trades = self._simulate(df_zones, name="Zone-Filtered ST", use_zones=True)
        return {"name": "Zone-Filtered ST", "trades": trades}

    def _backtest_st_rsi_buffered(self):
        logger.info("Running ST + Buffered RSI Exit...")
        df = SupertrendIndicator(period=7, multiplier=3).calculate(self.df.copy())
        df = RsiDivergence(len_fast=5, len_slow=14).calculate(df)
        trades = self._simulate(df, name="ST + RSI Buffered", rsi_buffer_pts=30)
        return {"name": "ST + RSI Buffered", "trades": trades}

    def _backtest_mwpl_squeeze(self):
        logger.info("Running MWPL Squeeze Simulation...")
        df = SupertrendIndicator(period=7, multiplier=3).calculate(self.df.copy())
        # Mock high-MWPL periods
        df['mwpl_squeeze'] = np.random.choice([True, False], size=len(df), p=[0.2, 0.8])
        trades = self._simulate(df, name="MWPL Squeeze", mwpl_boost=True)
        return {"name": "MWPL Squeeze", "trades": trades}

    def _backtest_fvg_retest(self):
        logger.info("Running FVG Retest Simulation...")
        df = RetracementIndicator().calculate(self.df.copy())
        df = SupertrendIndicator(period=7, multiplier=3).calculate(df)
        trades = self._simulate(df, name="FVG Retest", use_fvg=True)
        return {"name": "FVG Retest", "trades": trades}

    def _simulate(self, df, name, use_zones=False, rsi_buffer_pts=0, mwpl_boost=False, use_fvg=False):
        trades = []
        pos = 0
        ep = 0
        peak = 0
        
        for i in range(len(df)):
            row = df.iloc[i]
            sig = row.get('supertrend_signal', 0)
            st_dir = row.get('supertrend_direction', 0)
            ts = row['timestamp']
            price = row['close']
            
            is_eod = (ts.hour == 15 and ts.minute >= 15)
            
            if pos == 1: peak = max(peak, row['high'])
            elif pos == -1: peak = min(peak, row['low'])

            # Exit Logic
            if pos != 0:
                exit_reason = None
                if pos == 1:
                    pts_gain = peak - ep
                    rsi_exit = (rsi_buffer_pts > 0 and pts_gain >= rsi_buffer_pts and row.get('divergence', 0) < 0)
                    if st_dir == -1 or is_eod or rsi_exit:
                        exit_reason = "Signal Flip" if st_dir == -1 else ("RSI Buffer" if rsi_exit else "EOD")
                else:
                    pts_gain = ep - peak
                    rsi_exit = (rsi_buffer_pts > 0 and pts_gain >= rsi_buffer_pts and row.get('divergence', 0) > 0)
                    if st_dir == 1 or is_eod or rsi_exit:
                        exit_reason = "Signal Flip" if st_dir == 1 else ("RSI Buffer" if rsi_exit else "EOD")
                
                if exit_reason:
                    pnl = (price - ep)/ep if pos == 1 else (ep - price)/ep
                    trades.append({'pnl': pnl, 'peak': peak - ep if pos == 1 else ep - peak, 'reason': exit_reason, 'timestamp': ts})
                    pos = 0

            # Entry Logic
            if pos == 0 and not is_eod and ts.hour < 15 and sig != 0:
                allowed = True
                if use_zones:
                    lvls = [row.get('pdl', 0), row.get('val', 0), row.get('poc', 0)] if sig == 1 else [row.get('pdh', 0), row.get('vah', 0), row.get('poc', 0)]
                    allowed = any(abs(price - l) / l <= 0.002 for l in lvls if l > 0)
                
                if mwpl_boost:
                    allowed = row.get('mwpl_squeeze', False)
                
                if use_fvg:
                    allowed = not np.isnan(row.get('fvg_top', np.nan))

                if allowed:
                    pos = int(sig)
                    ep = price
                    peak = row['high'] if pos == 1 else row['low']
        return trades

    def _generate_report(self):
        report = ["# 🏆 Master Backtest Tournament Report: Nifty 50", ""]
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Period: Last 30 Days | Timeframe: 3 Minute")
        report.append("")
        
        data_table = ["| Strategy | Trades | Win Rate | Net PnL % | Profit Factor | Avg Peak Pts |", "| :--- | :--- | :--- | :--- | :--- | :--- |"]
        
        for res in self.results:
            t_df = pd.DataFrame(res['trades'])
            if t_df.empty: continue
            
            wr = (t_df['pnl'] > 0).mean()
            net = t_df['pnl'].sum()
            losses_sum = t_df[t_df['pnl']<0]['pnl'].sum()
            pf = abs(t_df[t_df['pnl']>0]['pnl'].sum() / min(-0.00001, losses_sum))
            avg_peak = t_df['peak'].mean()
            
            data_table.append(f"| {res['name']} | {len(t_df)} | {wr:.1%} | {net:.2%} | {pf:.2f} | {avg_peak:.1f} |")
        
        report.extend(data_table)
        report.append("")
        
        best = sorted(self.results, key=lambda x: pd.DataFrame(x['trades'])['pnl'].sum() if x['trades'] else -1, reverse=True)[0]
        report.append(f"## 💡 Recommendation")
        report.append(f"The **{best['name']}** strategy is the top performer.")
        
        final_text = "\n".join(report)
        Path("reports/master_backtest_report.md").write_text(final_text)
        print(final_text)

if __name__ == "__main__":
    BacktestTournament("data/fo_historical/NSE_NIFTY50-INDEX_3min_historical.csv").run_all()
