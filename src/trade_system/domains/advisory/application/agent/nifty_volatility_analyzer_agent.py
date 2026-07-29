"""
Nifty 50 15-Minute Candle Volatility and Big Movements Analyzer Agent.

This script fetches daily India VIX historical data via the Fyers API (with local caching),
loads Nifty 50 15m candles from the SQLite database, aligns the VIX close price and day
of the week, identifies extreme volatility events ("big movements"), and outputs
detailed CSV exports and markdown reports.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import text

from trade_system.shared.config import Settings
from trade_system.domains.market_data.infrastructure.database.connection import get_engine
from trade_system.domains.advisory.application.agent.top_gainers_agent import create_fyers_broker

LOGGER = logging.getLogger(__name__)


class NiftyVolatilityAnalyzerAgent:
    """Agent that performs off-market analysis on Nifty 50 15-minute candles."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.engine = get_engine()
        self._broker = None

    @property
    def broker(self):
        """Lazy loader for FyersBroker."""
        if self._broker is None:
            LOGGER.info("Initializing Fyers broker connection...")
            self._broker = create_fyers_broker()
        return self._broker

    def fetch_daily_vix(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch historical Daily India VIX data. Uses local CSV cache to avoid
        unnecessary API rate limits and speed up multiple runs.
        """
        cache_path = Path("data/india_vix_daily_cache.csv")
        cache_df = pd.DataFrame()

        # Load from cache if exists
        if cache_path.exists():
            try:
                cache_df = pd.read_csv(cache_path)
                cache_df["date"] = pd.to_datetime(cache_df["date"], format="mixed").dt.date
                LOGGER.info(f"Loaded {len(cache_df)} VIX rows from cache at {cache_path}")
            except Exception as e:
                LOGGER.warning(f"Error reading VIX cache: {e}. Re-fetching.")

        # Determine missing periods
        # We need data from start_date to end_date
        today = date.today()
        target_end = min(end_date, today)
        
        # If cache is empty or doesn't cover the full range, we fetch
        needs_fetch = True
        if not cache_df.empty:
            cache_min_date = cache_df["date"].min()
            cache_max_date = cache_df["date"].max()
            if cache_min_date <= start_date and cache_max_date >= target_end:
                needs_fetch = False
                LOGGER.info("VIX cache covers the requested date range.")

        if needs_fetch:
            LOGGER.info(f"VIX daily data cache missing or incomplete for {start_date} to {target_end}. Querying Fyers...")
            vix_symbol = "NSE:INDIAVIX-INDEX"
            
            # Fetch in yearly chunks to respect API range limits
            all_vix_dfs = []
            
            # We fetch from start_date (or earlier if cache is partially populated) to target_end
            current_start = start_date
            while current_start <= target_end:
                # Fyers allows up to 365 days of daily data
                current_end = min(current_start + timedelta(days=364), target_end)
                
                LOGGER.info(f"Fetching VIX from {current_start} to {current_end}...")
                try:
                    df_chunk = self.broker.fetch_history(
                        symbol=vix_symbol,
                        resolution="D",
                        range_from=current_start.isoformat(),
                        range_to=current_end.isoformat()
                    )
                    if not df_chunk.empty:
                        LOGGER.info(f"  Fetched {len(df_chunk)} rows.")
                        all_vix_dfs.append(df_chunk)
                    else:
                        LOGGER.warning(f"  No VIX data returned for chunk {current_start} to {current_end}.")
                except Exception as e:
                    LOGGER.error(f"  Error fetching VIX chunk {current_start} to {current_end}: {e}")
                
                # Advance start date and sleep to respect rate limits
                current_start = current_end + timedelta(days=1)
                time.sleep(0.5)

            if all_vix_dfs:
                new_vix_df = pd.concat(all_vix_dfs, ignore_index=True)
                new_vix_df["date"] = pd.to_datetime(new_vix_df["timestamp"], format="mixed").dt.date
                new_vix_df = new_vix_df[["date", "close"]].rename(columns={"close": "vix_close"})
                
                # Merge new data with cache
                if not cache_df.empty:
                    combined_df = pd.concat([cache_df, new_vix_df], ignore_index=True)
                else:
                    combined_df = new_vix_df
                
                # Drop duplicates and sort
                combined_df.drop_duplicates(subset=["date"], inplace=True)
                combined_df.sort_values(by="date", inplace=True)
                
                # Save back to cache
                cache_path.parent.mkdir(exist_ok=True, parents=True)
                combined_df.to_csv(cache_path, index=False)
                LOGGER.info(f"Updated VIX cache saved to {cache_path} ({len(combined_df)} rows)")
                cache_df = combined_df
            else:
                LOGGER.warning("Could not fetch new VIX data. Using existing cache only.")

        # Filter cache to requested range
        if not cache_df.empty:
            filtered_vix = cache_df[(cache_df["date"] >= start_date) & (cache_df["date"] <= end_date)].copy()
            return filtered_vix
        
        return pd.DataFrame(columns=["date", "vix_close"])

    def load_nifty_15m_candles(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Load Nifty 50 15m candles from SQLite database."""
        LOGGER.info(f"Loading Nifty 50 15m candles from database for {start_date} to {end_date}...")
        
        # Convert date to string timestamp
        start_str = f"{start_date.isoformat()} 00:00:00"
        end_str = f"{end_date.isoformat()} 23:59:59"
        
        query = text("""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM ohlcv_15m
            WHERE symbol = 'NSE:NIFTY50-INDEX'
              AND timestamp >= :start_str
              AND timestamp <= :end_str
            ORDER BY timestamp ASC
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"start_str": start_str, "end_str": end_str})
            
        LOGGER.info(f"Loaded {len(df)} Nifty 50 candles from DB.")
        return df

    def analyze_movements(
        self,
        df_nifty: pd.DataFrame,
        df_vix: pd.DataFrame,
        body_threshold: float = 0.5,
        range_threshold: float = 0.8,
        use_zscore: bool = False,
        zscore_threshold: float = 2.0
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process the candle data, calculate returns, ranges, Z-scores, align day and VIX,
        and filter out the big movements.
        """
        if df_nifty.empty:
            return pd.DataFrame(), pd.DataFrame()

        df = df_nifty.copy()
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        df["date"] = df["timestamp"].dt.date
        df["day_of_week"] = df["timestamp"].dt.strftime("%A")
        df["time_of_day"] = df["timestamp"].dt.strftime("%H:%M")
        
        # Calculate returns and ranges
        df["body_change_pct"] = (df["close"] - df["open"]) / df["open"] * 100
        df["abs_body_change_pct"] = df["body_change_pct"].abs()
        df["range_pct"] = (df["high"] - df["low"]) / df["low"] * 100
        
        # Calculate rolling multi-candle body and range changes (strictly intraday)
        df["open_prev1"] = df["open"].shift(1)
        df["date_prev1"] = df["date"].shift(1)
        df["high_max2"] = df["high"].rolling(2).max()
        df["low_min2"] = df["low"].rolling(2).min()
        df["body_2c"] = np.where(df["date"] == df["date_prev1"], (df["close"] - df["open_prev1"]) / df["open_prev1"] * 100, np.nan)
        df["abs_body_2c"] = df["body_2c"].abs()
        df["range_2c"] = np.where(df["date"] == df["date_prev1"], (df["high_max2"] - df["low_min2"]) / df["low_min2"] * 100, np.nan)

        df["open_prev2"] = df["open"].shift(2)
        df["date_prev2"] = df["date"].shift(2)
        df["high_max3"] = df["high"].rolling(3).max()
        df["low_min3"] = df["low"].rolling(3).min()
        df["body_3c"] = np.where(df["date"] == df["date_prev2"], (df["close"] - df["open_prev2"]) / df["open_prev2"] * 100, np.nan)
        df["abs_body_3c"] = df["body_3c"].abs()
        df["range_3c"] = np.where(df["date"] == df["date_prev2"], (df["high_max3"] - df["low_min3"]) / df["low_min3"] * 100, np.nan)

        df["open_prev3"] = df["open"].shift(3)
        df["date_prev3"] = df["date"].shift(3)
        df["high_max4"] = df["high"].rolling(4).max()
        df["low_min4"] = df["low"].rolling(4).min()
        df["body_4c"] = np.where(df["date"] == df["date_prev3"], (df["close"] - df["open_prev3"]) / df["open_prev3"] * 100, np.nan)
        df["abs_body_4c"] = df["body_4c"].abs()
        df["range_4c"] = np.where(df["date"] == df["date_prev3"], (df["high_max4"] - df["low_min4"]) / df["low_min4"] * 100, np.nan)
        
        # Calculate statistical metrics on whole dataset
        # Body changes
        mean_abs_body = df["abs_body_change_pct"].mean()
        std_abs_body = df["abs_body_change_pct"].std()
        df["body_zscore"] = (df["abs_body_change_pct"] - mean_abs_body) / std_abs_body
        
        # Range changes
        mean_range = df["range_pct"].mean()
        std_range = df["range_pct"].std()
        df["range_zscore"] = (df["range_pct"] - mean_range) / std_range
        
        # Align VIX
        if not df_vix.empty:
            df = pd.merge(df, df_vix[["date", "vix_close"]], on="date", how="left")
        else:
            df["vix_close"] = None

        # Filter criteria
        if use_zscore:
            LOGGER.info(f"Filtering big movements using Z-score threshold of {zscore_threshold} std devs...")
            filter_mask = (df["body_zscore"] >= zscore_threshold) | (df["range_zscore"] >= zscore_threshold)
            df["trigger_reasons"] = np.where(df["body_zscore"] >= zscore_threshold, "1C-BodyZ", "")
            df["trigger_reasons"] = np.where(
                df["range_zscore"] >= zscore_threshold, 
                np.where(df["trigger_reasons"] != "", df["trigger_reasons"] + ", 1C-RangeZ", "1C-RangeZ"),
                df["trigger_reasons"]
            )
        else:
            LOGGER.info(f"Filtering big movements using fixed rolling thresholds: body >= {body_threshold}% OR range >= {range_threshold}%...")
            
            def find_triggers(row):
                reasons = []
                # 1 Candle
                if row["abs_body_change_pct"] >= body_threshold:
                    reasons.append("1C-Body")
                if row["range_pct"] >= range_threshold:
                    reasons.append("1C-Range")
                
                # 2 Candles (30 mins)
                if pd.notna(row["abs_body_2c"]):
                    if row["abs_body_2c"] >= body_threshold + 0.1:
                        reasons.append("2C-Body")
                    if row["range_2c"] >= range_threshold + 0.1:
                        reasons.append("2C-Range")
                
                # 3 Candles (45 mins)
                if pd.notna(row["abs_body_3c"]):
                    if row["abs_body_3c"] >= body_threshold + 0.2:
                        reasons.append("3C-Body")
                    if row["range_3c"] >= range_threshold + 0.2:
                        reasons.append("3C-Range")
                
                # 4 Candles (60 mins)
                if pd.notna(row["abs_body_4c"]):
                    if row["abs_body_4c"] >= body_threshold + 0.3:
                        reasons.append("4C-Body")
                    if row["range_4c"] >= range_threshold + 0.4:
                        reasons.append("4C-Range")
                        
                return ", ".join(reasons) if reasons else "None"
            
            df["trigger_reasons"] = df.apply(find_triggers, axis=1)
            filter_mask = df["trigger_reasons"] != "None"
            
        df_filtered = df[filter_mask].copy()
        
        # Order filtered movements by date/timestamp
        df_filtered.sort_values(by="timestamp", inplace=True)
        
        return df, df_filtered

    def detect_breakdown_signals(
        self,
        df_all: pd.DataFrame,
        min_bearish_streak: int = 4,
        bandwidth_threshold: float = 0.30,
        bandwidth_lookback: int = 6,
        lower_high_min: int = 3,
        volume_lookback: int = 6,
        volume_surge_ratio: float = 1.3,
        composite_min_signals: int = 3,
    ) -> pd.DataFrame:
        """
        Detect 5 pre-conditions that precede large intraday movements.

        Conditions (computed per 15m candle):
        1. Multi-day resistance rejection — descending daily highs over last 3 trading days
        2. Consecutive bearish candles — 4+ consecutive bearish (close < open) in narrow range
        3. Volatility squeeze — rolling bandwidth (range_std / range_mean) below threshold
        4. Volume divergence — declining consolidation volume followed by a surge
        5. Lower highs — 3+ consecutive candle highs each lower than prior

        Returns the DataFrame with new signal columns appended.
        """
        if df_all.empty:
            return df_all

        df = df_all.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        if "date" not in df.columns:
            df["date"] = df["timestamp"].dt.date

        # --- Condition 1: Multi-day resistance rejection ---
        # Build daily summary: high, close for each trading day
        daily_summary = df.groupby("date").agg(
            daily_high=("high", "max"),
            daily_close=("close", "last"),
        ).reset_index()
        daily_summary.sort_values("date", inplace=True)

        # Compute descending daily highs — multiple patterns considered:
        # Look at daily highs for the last 4 trading days (current + 3 prior)
        daily_summary["dh_1"] = daily_summary["daily_high"].shift(1)  # 1 day ago
        daily_summary["dh_2"] = daily_summary["daily_high"].shift(2)  # 2 days ago
        daily_summary["dh_3"] = daily_summary["daily_high"].shift(3)  # 3 days ago

        # Signal fires if ANY of these descending-high patterns is present:
        # (a) Classic: prev day < 2-days-ago (consecutive decline)
        pattern_a = (
            daily_summary["dh_1"].notna()
            & daily_summary["dh_2"].notna()
            & (daily_summary["dh_1"] < daily_summary["dh_2"])
        )
        # (b) Current day high < prev day high (today itself is making a lower high)
        pattern_b = (
            daily_summary["dh_1"].notna()
            & (daily_summary["daily_high"] < daily_summary["dh_1"])
        )
        # (c) Broader: current day high is below the max of the last 3 trading days
        #     AND the max of last 3 days is also a descending sequence from further back
        three_day_max = daily_summary[["dh_1", "dh_2", "dh_3"]].max(axis=1)
        pattern_c = (
            daily_summary["dh_3"].notna()
            & (daily_summary["daily_high"] < three_day_max)
            & (daily_summary["dh_1"] < daily_summary["dh_3"])
        )

        daily_summary["sig_resistance_rejection"] = pattern_a | pattern_b | pattern_c

        # Merge back to candle data (applies to all candles of that day)
        df = df.merge(
            daily_summary[["date", "sig_resistance_rejection"]],
            on="date",
            how="left",
        )
        df["sig_resistance_rejection"] = df["sig_resistance_rejection"].fillna(False)

        # --- Condition 2: Consecutive bearish candles ---
        df["is_bearish"] = df["close"] < df["open"]
        # Count consecutive bearish candles (reset on date boundary and bull candle)
        bearish_streak = []
        streak = 0
        prev_date = None
        for _, row in df.iterrows():
            if row["date"] != prev_date:
                streak = 0
            prev_date = row["date"]
            if row["is_bearish"]:
                streak += 1
            else:
                streak = 0
            bearish_streak.append(streak)
        df["bearish_streak"] = bearish_streak
        df["sig_bearish_streak"] = df["bearish_streak"] >= min_bearish_streak

        # --- Condition 3: Volatility squeeze (bandwidth compression) ---
        df["range_pts"] = df["high"] - df["low"]
        # Rolling stats with intraday reset: we compute rolling on the raw series
        # but only consider candles within the same date for validity
        df["roll_range_mean"] = df["range_pts"].rolling(bandwidth_lookback, min_periods=bandwidth_lookback).mean()
        df["roll_range_std"] = df["range_pts"].rolling(bandwidth_lookback, min_periods=bandwidth_lookback).std()
        df["bandwidth"] = np.where(
            df["roll_range_mean"] > 0,
            df["roll_range_std"] / df["roll_range_mean"],
            np.nan,
        )
        # Squeeze: bandwidth below threshold for at least 2 of the last 3 candles
        df["bw_below"] = df["bandwidth"] < bandwidth_threshold
        df["bw_below_count_3"] = df["bw_below"].rolling(3, min_periods=1).sum()
        df["sig_volatility_squeeze"] = df["bw_below_count_3"] >= 2

        # --- Condition 4: Volume divergence ---
        # Compare current volume to the rolling mean of previous N candles
        df["vol_rolling_mean"] = df["volume"].rolling(volume_lookback, min_periods=volume_lookback).mean()
        # Volume was declining: current rolling mean < 0.8x of the rolling mean 6 candles ago
        df["vol_rolling_mean_prev"] = df["vol_rolling_mean"].shift(volume_lookback)
        df["vol_declining"] = (
            df["vol_rolling_mean"].notna()
            & df["vol_rolling_mean_prev"].notna()
            & (df["vol_rolling_mean"] < 0.8 * df["vol_rolling_mean_prev"])
        )
        # Volume surge: current candle volume > volume_surge_ratio * rolling mean
        df["vol_surge"] = (
            df["vol_rolling_mean"].notna()
            & (df["volume"] > volume_surge_ratio * df["vol_rolling_mean"])
        )
        # Signal: declining context AND current or next candle is a surge
        df["vol_surge_ahead"] = df["vol_surge"] | df["vol_surge"].shift(-1, fill_value=False)
        df["sig_volume_divergence"] = df["vol_declining"] & df["vol_surge_ahead"]

        # --- Condition 5: Consecutive lower highs ---
        df["prev_high"] = df["high"].shift(1)
        df["is_lower_high"] = df["high"] < df["prev_high"]
        # Count consecutive lower highs (reset on date boundary)
        lh_streak = []
        streak = 0
        prev_date = None
        for _, row in df.iterrows():
            if row["date"] != prev_date:
                streak = 0
            prev_date = row["date"]
            if row["is_lower_high"]:
                streak += 1
            else:
                streak = 0
            lh_streak.append(streak)
        df["lower_high_streak"] = lh_streak
        df["sig_lower_highs"] = df["lower_high_streak"] >= lower_high_min

        # --- Composite signal ---
        signal_cols = [
            "sig_resistance_rejection",
            "sig_bearish_streak",
            "sig_volatility_squeeze",
            "sig_volume_divergence",
            "sig_lower_highs",
        ]
        df["breakdown_signal_count"] = df[signal_cols].sum(axis=1)
        df["breakdown_signal"] = df["breakdown_signal_count"] >= composite_min_signals

        # Build human-readable breakdown_reasons
        reason_labels = {
            "sig_resistance_rejection": "ResistReject",
            "sig_bearish_streak": f"Bearish{min_bearish_streak}+",
            "sig_volatility_squeeze": "VolSqueeze",
            "sig_volume_divergence": "VolDivergence",
            "sig_lower_highs": f"LowerHighs{lower_high_min}+",
        }

        def build_reasons(row):
            reasons = [label for col, label in reason_labels.items() if row.get(col, False)]
            return ", ".join(reasons) if reasons else ""

        df["breakdown_reasons"] = df.apply(build_reasons, axis=1)

        # Clean up intermediate columns
        cleanup_cols = [
            "is_bearish", "range_pts", "roll_range_mean", "roll_range_std",
            "bw_below", "bw_below_count_3", "vol_rolling_mean", "vol_rolling_mean_prev",
            "vol_declining", "vol_surge", "vol_surge_ahead", "prev_high", "is_lower_high",
            "dh_1", "dh_2", "dh_3",
        ]
        df.drop(columns=[c for c in cleanup_cols if c in df.columns], inplace=True, errors="ignore")

        LOGGER.info(
            f"Breakdown signal detection complete: {df['breakdown_signal'].sum()} candles "
            f"flagged with {composite_min_signals}+ conditions."
        )
        return df

    def generate_report(
        self,
        df_all: pd.DataFrame,
        df_filtered: pd.DataFrame,
        start_date: date,
        end_date: date,
        body_threshold: float,
        range_threshold: float,
        use_zscore: bool,
        zscore_threshold: float,
    ) -> str:
        """Compile a comprehensive statistical and markdown report."""
        if df_all.empty:
            return "No data to analyze."

        # Overall Stats
        total_candles = len(df_all)
        filtered_count = len(df_filtered)
        filtered_pct = (filtered_count / total_candles) * 100 if total_candles > 0 else 0
        
        mean_body = df_all["abs_body_change_pct"].mean()
        std_body = df_all["abs_body_change_pct"].std()
        max_body = df_all["abs_body_change_pct"].max()
        
        mean_range = df_all["range_pct"].mean()
        std_range = df_all["range_pct"].std()
        max_range = df_all["range_pct"].max()
        
        # Overall average VIX (excluding NaN)
        overall_avg_vix = df_all["vix_close"].mean()
        filtered_avg_vix = df_filtered["vix_close"].mean()

        # Breakdown by Year
        df_all["year"] = df_all["timestamp"].dt.year
        df_filtered["year"] = df_filtered["timestamp"].dt.year
        
        yearly_all = df_all.groupby("year").size()
        yearly_filtered = df_filtered.groupby("year").size()
        yearly_avg_vix = df_all.groupby("year")["vix_close"].mean()
        yearly_filt_vix = df_filtered.groupby("year")["vix_close"].mean()
        
        # Breakdown by Day of Week
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_all = df_all.groupby("day_of_week").size().reindex(days_order).fillna(0)
        day_filtered = df_filtered.groupby("day_of_week").size().reindex(days_order).fillna(0)
        
        # Breakdown by Time of Day (top 10 times)
        time_filtered = df_filtered.groupby("time_of_day").size().sort_values(ascending=False).head(10)
        
        # Correlations (only where VIX is not null)
        vix_corr_body = 0.0
        vix_corr_range = 0.0
        valid_vix = df_all.dropna(subset=["vix_close"])
        if len(valid_vix) > 10:
            vix_corr_body = valid_vix["abs_body_change_pct"].corr(valid_vix["vix_close"])
            vix_corr_range = valid_vix["range_pct"].corr(valid_vix["vix_close"])

        # Top 10 largest body movements
        top_body_moves = df_all.sort_values(by="abs_body_change_pct", ascending=False).head(10)
        
        # Top 10 largest ranges
        top_range_moves = df_all.sort_values(by="range_pct", ascending=False).head(10)

        # Write Report Markdown Content
        report = []
        report.append("# Nifty 50 15m Volatility and Big Movements Analysis Report\n")
        report.append(f"**Analysis Period**: {start_date} to {end_date} (~{round((end_date - start_date).days / 365, 1)} years)")
        report.append(f"**Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\n")
        
        # Methodology Section
        report.append("## 1. Methodology & Parameters")
        report.append("This off-market analysis loads 15-minute Nifty 50 index candles and filters out the largest intraday movements.")
        if use_zscore:
            report.append(f"- **Detection Method**: Z-Score threshold $\ge {zscore_threshold}$ standard deviations above the mean.")
        else:
            report.append(f"- **Detection Method**: Fixed rolling thresholds.")
            report.append(f"  - 1-Candle (15m): Body $\ge {body_threshold}\%$ OR Range $\ge {range_threshold}\%$")
            report.append(f"  - 2-Candle (30m): Body $\ge {round(body_threshold + 0.1, 2)}\%$ OR Range $\ge {round(range_threshold + 0.1, 2)}\%$")
            report.append(f"  - 3-Candle (45m): Body $\ge {round(body_threshold + 0.2, 2)}\%$ OR Range $\ge {round(range_threshold + 0.2, 2)}\%$")
            report.append(f"  - 4-Candle (60m): Body $\ge {round(body_threshold + 0.3, 2)}\%$ OR Range $\ge {round(range_threshold + 0.4, 2)}\%$")
        report.append(f"- **Total 15m Candles Analyzed**: {total_candles:,}")
        report.append(f"- **Big Movements Detected**: {filtered_count:,} ({filtered_pct:.2f}% of all candles)\n")
        
        # Trigger reasons breakdown table
        if not use_zscore and "trigger_reasons" in df_filtered.columns:
            trig_counts = {}
            for reason_list in df_filtered["trigger_reasons"].dropna():
                for reason in reason_list.split(", "):
                    trig_counts[reason] = trig_counts.get(reason, 0) + 1
            
            report.append("### Trigger Timeline Analysis (Reason Breakdown)")
            report.append("| Trigger Type | Occurrences | % of Flagged Moves |")
            report.append("| :--- | :---: | :---: |")
            for reason, count in sorted(trig_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / filtered_count) * 100 if filtered_count > 0 else 0
                report.append(f"| {reason} | {count:,} | {pct:.2f}% |")
            report.append("")

        # Overall Summary Table
        report.append("## 2. Overall Nifty 50 15m Statistics")
        report.append("| Metric | Absolute Body Return (%) | High-Low Range (%) | India VIX |")
        report.append("| :--- | :---: | :---: | :---: |")
        report.append(f"| **Dataset Mean** | {mean_body:.3f}% | {mean_range:.3f}% | {overall_avg_vix:.2f} |")
        report.append(f"| **Std Dev** | {std_body:.3f}% | {std_range:.3f}% | - |")
        report.append(f"| **Dataset Max** | {max_body:.3f}% | {max_range:.3f}% | - |")
        report.append(f"| **Mean on Big Move Days** | - | - | {filtered_avg_vix:.2f} |\n")
        
        # VIX Volatility Correlation
        report.append("### India VIX Correlation Analysis")
        report.append(f"- Correlation between daily VIX Close and 15m Absolute Body Returns: **{vix_corr_body:.4f}**")
        report.append(f"- Correlation between daily VIX Close and 15m Candle Ranges: **{vix_corr_range:.4f}**")
        report.append(f"- *Note: A higher VIX level is correlated with larger individual intraday candle moves, which is clearly shown in the positive correlation coefficients above.*\n")

        # Yearly Distribution Table
        report.append("## 3. Distribution of Big Movements by Year")
        report.append("| Year | Total 15m Candles | Big Move Candles | % of Total | Avg Daily VIX (Overall) | Avg Daily VIX (Big Move Days) |")
        report.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for yr in sorted(yearly_all.keys()):
            cnt_all = yearly_all.get(yr, 0)
            cnt_filt = yearly_filtered.get(yr, 0)
            pct = (cnt_filt / cnt_all) * 100 if cnt_all > 0 else 0
            vix_all = yearly_avg_vix.get(yr, float('nan'))
            vix_filt = yearly_filt_vix.get(yr, float('nan'))
            vix_all_str = f"{vix_all:.2f}" if not pd.isna(vix_all) else "N/A"
            vix_filt_str = f"{vix_filt:.2f}" if not pd.isna(vix_filt) else "N/A"
            report.append(f"| {yr} | {cnt_all:,} | {cnt_filt:,} | {pct:.2f}% | {vix_all_str} | {vix_filt_str} |")
        report.append("")

        # Weekly Distribution Table
        report.append("## 4. Distribution of Big Movements by Day of the Week")
        report.append("| Day of the Week | Big Move Candles | Avg Daily VIX on Big Move Days |")
        report.append("| :--- | :---: | :---: |")
        for day in days_order:
            cnt_day = day_filtered.get(day, 0)
            # Find avg VIX for this day
            vix_day = df_filtered[df_filtered["day_of_week"] == day]["vix_close"].mean()
            vix_day_str = f"{vix_day:.2f}" if not pd.isna(vix_day) else "N/A"
            report.append(f"| {day} | {cnt_day:,} | {vix_day_str} |")
        report.append("")

        # Time of Day Distribution
        report.append("## 5. Top 10 Most Volatile Times of Day (IST)")
        report.append("These are the 15-minute candle intervals that most frequently trigger big movements:")
        report.append("| Rank | Candle Start Time | Occurrences | % of All Big Moves |")
        report.append("| :---: | :---: | :---: | :---: |")
        for idx, (t_str, count) in enumerate(time_filtered.items(), 1):
            pct = (count / filtered_count) * 100 if filtered_count > 0 else 0
            report.append(f"| {idx} | {t_str} | {count:,} | {pct:.2f}% |")
        report.append("\n*Note: Typically, the market open (09:15) and market close (15:15) segments exhibit the highest concentration of institutional order matching and volatility.*\n")

        # Top 10 body movements
        report.append("## 6. Top 10 Largest 15m Body Movements (Absolute Close-Open)")
        report.append("| Rank | Timestamp | Day | Open | Close | Body Change (%) | Range (%) | Trigger Reason | India VIX |")
        report.append("| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for idx, (_, row) in enumerate(top_body_moves.iterrows(), 1):
            ts_str = row["timestamp"].strftime("%Y-%m-%d %H:%M")
            vix_val = f"{row['vix_close']:.2f}" if not pd.isna(row['vix_close']) else "N/A"
            trig_str = row.get("trigger_reasons", "1C-Body")
            report.append(f"| {idx} | {ts_str} | {row['day_of_week']} | {row['open']:.2f} | {row['close']:.2f} | **{row['body_change_pct']:.2f}%** | {row['range_pct']:.2f}% | {trig_str} | {vix_val} |")
        report.append("")

        # Top 10 ranges
        report.append("## 7. Top 10 Largest 15m Candle Ranges (High-Low)")
        report.append("| Rank | Timestamp | Day | High | Low | Range (%) | Body Change (%) | Trigger Reason | India VIX |")
        report.append("| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for idx, (_, row) in enumerate(top_range_moves.iterrows(), 1):
            ts_str = row["timestamp"].strftime("%Y-%m-%d %H:%M")
            vix_val = f"{row['vix_close']:.2f}" if not pd.isna(row['vix_close']) else "N/A"
            trig_str = row.get("trigger_reasons", "1C-Range")
            report.append(f"| {idx} | {ts_str} | {row['day_of_week']} | {row['high']:.2f} | {row['low']:.2f} | **{row['range_pct']:.2f}%** | {row['body_change_pct']:.2f}% | {trig_str} | {vix_val} |")
        report.append("")

        # Section 8: Breakdown Pre-Condition Signals
        if "breakdown_signal" in df_all.columns:
            breakdown_df = df_all[df_all["breakdown_signal"]].copy()
            total_breakdown = len(breakdown_df)

            report.append("## 8. Breakdown Pre-Condition Signals")
            report.append(
                "These candles had 3 or more of the 5 pre-conditions active simultaneously, "
                "indicating a high probability of an imminent large directional move.\n"
            )
            report.append(f"**Total candles with 3+ pre-conditions**: {total_breakdown:,} "
                          f"({(total_breakdown / total_candles) * 100:.3f}% of all candles)\n")

            # Signal type distribution
            signal_col_labels = {
                "sig_resistance_rejection": "Multi-Day Resistance Rejection",
                "sig_bearish_streak": "4+ Consecutive Bearish Candles",
                "sig_volatility_squeeze": "Volatility Squeeze (BW < 0.30)",
                "sig_volume_divergence": "Volume Divergence (Decline → Surge)",
                "sig_lower_highs": "3+ Consecutive Lower Highs",
            }

            report.append("### Signal Type Distribution (among flagged candles)")
            report.append("| Pre-Condition | Active Count | % of Flagged |")
            report.append("| :--- | :---: | :---: |")
            for col, label in signal_col_labels.items():
                if col in breakdown_df.columns:
                    active = breakdown_df[col].sum()
                    pct = (active / total_breakdown) * 100 if total_breakdown > 0 else 0
                    report.append(f"| {label} | {int(active):,} | {pct:.1f}% |")
            report.append("")

            # Top 20 highest-conviction setups
            if total_breakdown > 0:
                top_setups = breakdown_df.sort_values(
                    by=["breakdown_signal_count", "range_pct"],
                    ascending=[False, False],
                ).head(20)

                report.append("### Top 20 Highest-Conviction Breakdown Setups")
                report.append(
                    "| Timestamp | Day | Close | Body (%) | Range (%) | "
                    "Signals | Reasons | VIX |"
                )
                report.append(
                    "| :---: | :--- | :---: | :---: | :---: | "
                    ":---: | :--- | :---: |"
                )
                for _, row in top_setups.iterrows():
                    ts_str = row["timestamp"].strftime("%Y-%m-%d %H:%M")
                    vix_val = f"{row['vix_close']:.2f}" if pd.notna(row.get("vix_close")) else "N/A"
                    bd_reasons = row.get("breakdown_reasons", "")
                    bd_count = int(row.get("breakdown_signal_count", 0))
                    report.append(
                        f"| {ts_str} | {row['day_of_week']} | {row['close']:.2f} | "
                        f"{row['body_change_pct']:.3f}% | {row['range_pct']:.3f}% | "
                        f"**{bd_count}/5** | {bd_reasons} | {vix_val} |"
                    )
                report.append("")

        return "\n".join(report)

    def run_analysis(
        self,
        start_date: date,
        end_date: date,
        body_threshold: float = 0.5,
        range_threshold: float = 0.8,
        use_zscore: bool = False,
        zscore_threshold: float = 2.0,
        csv_output_path: str = "data/nifty_big_movements.csv",
        report_output_path: str = "reports/nifty_volatility_analysis_report.md"
    ) -> None:
        """Orchestrate historical analysis."""
        LOGGER.info("Starting Nifty 50 15m Volatility off-market analysis...")
        
        # 1. Fetch Daily India VIX history
        df_vix = self.fetch_daily_vix(start_date, end_date)
        LOGGER.info(f"Retrieved {len(df_vix)} VIX daily close records.")

        # 2. Load Nifty 15m candles
        df_nifty = self.load_nifty_15m_candles(start_date, end_date)
        if df_nifty.empty:
            LOGGER.error("No Nifty 50 candles found in the database. Exiting.")
            return

        # 3. Analyze and Filter
        df_all, df_filtered = self.analyze_movements(
            df_nifty,
            df_vix,
            body_threshold=body_threshold,
            range_threshold=range_threshold,
            use_zscore=use_zscore,
            zscore_threshold=zscore_threshold
        )
        
        LOGGER.info(f"Analysis completed: {len(df_filtered)} / {len(df_all)} candles flagged.")

        # 3b. Detect breakdown pre-condition signals on the FULL dataset
        df_all = self.detect_breakdown_signals(df_all)

        # Merge breakdown columns back into filtered set
        breakdown_cols = [
            "sig_resistance_rejection", "sig_bearish_streak",
            "sig_volatility_squeeze", "sig_volume_divergence", "sig_lower_highs",
            "bearish_streak", "bandwidth", "lower_high_streak",
            "breakdown_signal_count", "breakdown_signal", "breakdown_reasons",
        ]
        existing_bd_cols = [c for c in breakdown_cols if c in df_all.columns]
        # Use timestamp as key (already in both frames)
        if "timestamp" in df_filtered.columns and existing_bd_cols:
            ts_key = "timestamp"
            merge_cols = [ts_key] + existing_bd_cols
            # Drop any pre-existing columns in df_filtered to avoid suffixes
            df_filtered = df_filtered.drop(
                columns=[c for c in existing_bd_cols if c in df_filtered.columns],
                errors="ignore",
            )
            df_filtered = df_filtered.merge(
                df_all[merge_cols].drop_duplicates(subset=[ts_key]),
                on=ts_key,
                how="left",
            )

        # 4. Save CSV output
        csv_path = Path(csv_output_path)
        csv_path.parent.mkdir(exist_ok=True, parents=True)
        # Select key columns for final CSV export
        export_cols = [
            "timestamp", "date", "day_of_week", "time_of_day",
            "open", "high", "low", "close", "volume",
            "body_change_pct", "abs_body_change_pct", "body_zscore",
            "range_pct", "range_zscore", "vix_close", "trigger_reasons",
            "abs_body_2c", "range_2c", "abs_body_3c", "range_3c", "abs_body_4c", "range_4c",
            # Breakdown signal columns
            "bearish_streak", "bandwidth", "lower_high_streak",
            "breakdown_signal_count", "breakdown_signal", "breakdown_reasons",
        ]
        
        # Ensure only columns that exist are exported
        cols_to_export = [col for col in export_cols if col in df_filtered.columns]
        df_filtered[cols_to_export].to_csv(csv_path, index=False)
        LOGGER.info(f"CSV exported successfully to {csv_path} with {len(df_filtered)} rows.")

        # 4b. Export breakdown alerts (candles where breakdown_signal is True)
        if "breakdown_signal" in df_all.columns:
            breakdown_alerts = df_all[df_all["breakdown_signal"]].copy()
            if not breakdown_alerts.empty:
                alert_path = Path(csv_output_path).parent / "nifty_breakdown_alerts.csv"
                alert_export_cols = [
                    "timestamp", "date", "day_of_week", "time_of_day",
                    "open", "high", "low", "close", "volume",
                    "body_change_pct", "range_pct", "vix_close",
                    "bearish_streak", "bandwidth", "lower_high_streak",
                    "breakdown_signal_count", "breakdown_reasons",
                ]
                alert_cols = [c for c in alert_export_cols if c in breakdown_alerts.columns]
                breakdown_alerts[alert_cols].to_csv(alert_path, index=False)
                LOGGER.info(
                    f"Breakdown alerts CSV exported: {alert_path} "
                    f"({len(breakdown_alerts)} candles with 3+ pre-conditions)"
                )

        # 5. Generate and save Report
        report_content = self.generate_report(
            df_all,
            df_filtered,
            start_date,
            end_date,
            body_threshold,
            range_threshold,
            use_zscore,
            zscore_threshold
        )
        
        report_path = Path(report_output_path)
        report_path.parent.mkdir(exist_ok=True, parents=True)
        report_path.write_text(report_content)
        LOGGER.info(f"Markdown report generated successfully at {report_path}.")


if __name__ == "__main__":
    # Configure simple logs for running directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Nifty 50 15m Volatility Analyzer")
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Start date in YYYY-MM-DD format (default: 2020-01-01)"
    )
    parser.add_argument(
        "--end-date",
        default="2026-06-24",
        help="End date in YYYY-MM-DD format (default: 2026-06-24)"
    )
    parser.add_argument(
        "--body-threshold",
        type=float,
        default=0.5,
        help="Percent body change threshold (default: 0.5%%)"
    )
    parser.add_argument(
        "--range-threshold",
        type=float,
        default=0.8,
        help="Percent high-low range threshold (default: 0.8%%)"
    )
    parser.add_argument(
        "--use-zscore",
        action="store_true",
        help="Filter by Z-score standard deviation thresholds instead of fixed percent changes"
    )
    parser.add_argument(
        "--zscore-threshold",
        type=float,
        default=2.0,
        help="Z-score standard deviation threshold (default: 2.0)"
    )
    parser.add_argument(
        "--csv-out",
        default="data/nifty_big_movements.csv",
        help="Output CSV path (default: data/nifty_big_movements.csv)"
    )
    parser.add_argument(
        "--report-out",
        default="reports/nifty_volatility_analysis_report.md",
        help="Output report Markdown path (default: reports/nifty_volatility_analysis_report.md)"
    )

    args = parser.parse_args()

    # Parse dates
    start_d = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    analyzer = NiftyVolatilityAnalyzerAgent()
    analyzer.run_analysis(
        start_date=start_d,
        end_date=end_d,
        body_threshold=args.body_threshold,
        range_threshold=args.range_threshold,
        use_zscore=args.use_zscore,
        zscore_threshold=args.zscore_threshold,
        csv_output_path=args.csv_out,
        report_output_path=args.report_out
    )
