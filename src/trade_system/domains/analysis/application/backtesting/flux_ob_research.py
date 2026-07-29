"""
Flux Charts Volumized Order Blocks Research Backtesting
=======================================================
Implements a historical price action backtest on top of the
translated FluxOrderBlockDetector.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from trade_system.domains.strategy.application.indicators.flux_order_blocks import FluxOrderBlockDetector

LOGGER = logging.getLogger(__name__)

ENTRY_CUTOFF_HOUR = 14
ENTRY_CUTOFF_MINUTE = 45
SQUARE_OFF_HOUR = 15
SQUARE_OFF_MINUTE = 15

@dataclass(slots=True)
class FluxObResearchArtifacts:
    summary: pd.DataFrame
    summary_path: Path
    report_path: Path

class FluxOrderBlockResearch:
    def __init__(
        self,
        swing_length: int = 10,
        max_atr_mult: float = 3.5,
        stop_buffer_points: float = 3.0,
        reward_risk: float = 2.0,
        min_reward_risk: float = 1.2,
        ob_end_method: str = "Wick",
    ) -> None:
        self.swing_length = swing_length
        self.max_atr_mult = max_atr_mult
        self.stop_buffer_points = stop_buffer_points
        self.reward_risk = reward_risk
        self.min_reward_risk = min_reward_risk
        self.ob_end_method = ob_end_method

    def run(self, data_dir: Path, output_dir: Path) -> FluxObResearchArtifacts:
        files = sorted(data_dir.glob("NSE_NIFTY50-INDEX_3min_*.csv"))
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        all_trades: list[pd.DataFrame] = []

        for file_path in files:
            last_part = file_path.stem.rsplit("_", 1)[-1]
            try:
                year = int(last_part)
            except ValueError:
                year = last_part.upper()
            trades = self.run_year(file_path)
            if not trades.empty:
                all_trades.append(trades.assign(year=year))
            trades.to_csv(output_dir / f"flux_ob_{year}_trades.csv", index=False)
            rows.append(self._summarize_year(year, trades))

        summary = pd.DataFrame(rows)
        if not summary.empty:
            summary["year_str"] = summary["year"].astype(str)
            summary = summary.sort_values("year_str").drop(columns=["year_str"]).reset_index(drop=True)
        if all_trades:
            combined = pd.concat(all_trades, ignore_index=True)
            combined.to_csv(output_dir / f"flux_ob_all_trades.csv", index=False)
            rows.append(self._summarize_year("ALL", combined))
            summary = pd.DataFrame(rows)
            
        summary_path = output_dir / "flux_ob_summary.csv"
        summary.to_csv(summary_path, index=False)
        report_path = output_dir / "flux_ob_report.md"
        report_path.write_text(self._build_report(summary))
        
        return FluxObResearchArtifacts(summary=summary, summary_path=summary_path, report_path=report_path)

    def run_year(self, file_path: Path) -> pd.DataFrame:
        frame = pd.read_csv(file_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            return pd.DataFrame()

        detector = FluxOrderBlockDetector(
            swing_length=self.swing_length,
            max_atr_mult=self.max_atr_mult,
            ob_end_method=self.ob_end_method,
            combine_obs=True
        )
        # Calculate order block history
        ob_history = detector.calculate(frame)

        trades: list[dict[str, object]] = []
        position: dict[str, object] | None = None

        for idx in range(len(frame)):
            row = frame.iloc[idx]
            ts = pd.Timestamp(row["timestamp"])
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])

            # Active OBs at this bar
            active_obs = ob_history[idx]
            
            # Find closest unmitigated bullish and bearish blocks
            active_bulls = [ob for ob in active_obs if ob.ob_type == "Bull" and not ob.breaker]
            active_bears = [ob for ob in active_obs if ob.ob_type == "Bear" and not ob.breaker]

            if position is not None:
                # Update excursions (MFE/MAE)
                self._update_excursions(position, row)
                closed = self._check_exit(position, row, ts)
                if closed is not None:
                    trades.append(closed)
                    position = None

            # Check if we can enter a new position
            if position is None:
                # Time filters
                if ts.time().hour > ENTRY_CUTOFF_HOUR or (
                    ts.time().hour == ENTRY_CUTOFF_HOUR and ts.time().minute > ENTRY_CUTOFF_MINUTE
                ):
                    continue
                if ts.time().hour < 9 or (ts.time().hour == 9 and ts.time().minute < 30):
                    continue

                # Check long entry: Price tests a bullish order block zone (dips into or touches top of zone)
                # target is the nearest bearish order block
                bull_zone = None
                for ob in active_bulls:
                    if low <= ob.top and high >= ob.bottom:
                        bull_zone = ob
                        break
                
                bear_zone = None
                for ob in active_bears:
                    if high >= ob.bottom and low <= ob.top:
                        bear_zone = ob
                        break

                if bull_zone is not None:
                    # Target is nearest bear zone top
                    target = min([ob.bottom for ob in active_bears]) if active_bears else None
                    position = self._build_position(
                        direction=1,
                        row=row,
                        ts=ts,
                        ob_zone=bull_zone,
                        target_level=target
                    )
                elif bear_zone is not None:
                    # Target is nearest bull zone bottom
                    target = max([ob.top for ob in active_bulls]) if active_bulls else None
                    position = self._build_position(
                        direction=-1,
                        row=row,
                        ts=ts,
                        ob_zone=bear_zone,
                        target_level=target
                    )

        if position is not None:
            last = frame.iloc[-1]
            trades.append(self._close_position(position, pd.Timestamp(last["timestamp"]), float(last["close"]), "FORCED_CLOSE"))

        return pd.DataFrame(trades)

    def _build_position(
        self,
        direction: int,
        row: pd.Series,
        ts: pd.Timestamp,
        ob_zone: Any,
        target_level: float | None,
    ) -> dict[str, object] | None:
        entry_price = float(row["close"])
        if direction == 1:
            stop_loss = ob_zone.bottom - self.stop_buffer_points
            risk = entry_price - stop_loss
        else:
            stop_loss = ob_zone.top + self.stop_buffer_points
            risk = stop_loss - entry_price

        if risk <= 0:
            return None

        # Take Profit Calculation
        take_profit = entry_price + direction * risk * self.reward_risk
        if target_level is not None:
            target_reward = (target_level - entry_price) * direction
            if target_reward / risk >= self.min_reward_risk:
                take_profit = target_level

        return {
            "direction": direction,
            "entry_time": ts,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "initial_risk": risk,
            "ob_top": ob_zone.top,
            "ob_bottom": ob_zone.bottom,
            "ob_volume": ob_zone.ob_volume,
            "mfe_points": 0.0,
            "mae_points": 0.0,
        }

    @staticmethod
    def _update_excursions(position: dict[str, object], row: pd.Series) -> None:
        entry = float(position["entry_price"])
        direction = int(position["direction"])
        if direction == 1:
            mfe = float(row["high"]) - entry
            mae = entry - float(row["low"])
        else:
            mfe = entry - float(row["low"])
            mae = float(row["high"]) - entry
        position["mfe_points"] = max(float(position["mfe_points"]), mfe)
        position["mae_points"] = max(float(position["mae_points"]), mae)

    def _check_exit(
        self,
        position: dict[str, object],
        row: pd.Series,
        ts: pd.Timestamp,
    ) -> dict[str, object] | None:
        direction = int(position["direction"])
        stop_loss = float(position["stop_loss"])
        take_profit = float(position["take_profit"])

        high = float(row["high"])
        low = float(row["low"])

        if direction == 1:
            if low <= stop_loss:
                return self._close_position(position, ts, stop_loss, "SL_HIT")
            if high >= take_profit:
                return self._close_position(position, ts, take_profit, "TARGET_HIT")
        else:
            if high >= stop_loss:
                return self._close_position(position, ts, stop_loss, "SL_HIT")
            if low <= take_profit:
                return self._close_position(position, ts, take_profit, "TARGET_HIT")

        if ts.time().hour > SQUARE_OFF_HOUR or (ts.time().hour == SQUARE_OFF_HOUR and ts.time().minute >= SQUARE_OFF_MINUTE):
            return self._close_position(position, ts, float(row["close"]), "EOD_SQUARE_OFF")

        return None

    def _close_position(
        self,
        position: dict[str, object],
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
    ) -> dict[str, object]:
        direction = int(position["direction"])
        entry_price = float(position["entry_price"])
        risk = float(position["initial_risk"])
        pnl = (exit_price - entry_price) * direction
        return {
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_time": position["entry_time"],
            "entry_price": round(entry_price, 2),
            "stop_loss": round(float(position["stop_loss"]), 2),
            "take_profit": round(float(position["take_profit"]), 2),
            "exit_time": exit_time,
            "exit_price": round(exit_price, 2),
            "exit_reason": exit_reason,
            "ob_top": round(float(position["ob_top"]), 2),
            "ob_bottom": round(float(position["ob_bottom"]), 2),
            "ob_volume": round(float(position["ob_volume"]), 2),
            "points_captured": round(pnl, 2),
            "risk_points": round(risk, 2),
            "r_multiple": round(pnl / risk, 2) if risk else 0.0,
            "mfe_points": round(float(position["mfe_points"]), 2),
            "mae_points": round(float(position["mae_points"]), 2),
        }

    @staticmethod
    def _equity_drawdown(trades: pd.DataFrame) -> float:
        if trades.empty:
            return 0.0
        equity = trades["points_captured"].cumsum()
        running_peak = equity.cummax()
        drawdown = equity - running_peak
        return round(float(drawdown.min()), 2)

    def _summarize_year(self, year: int | str, trades: pd.DataFrame) -> dict[str, object]:
        if trades.empty:
            return {
                "year": year,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_points": 0.0,
                "avg_points": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "total_r": 0.0,
                "avg_r": 0.0,
                "target_hits": 0,
                "sl_hits": 0,
                "eod_exits": 0,
                "max_drawdown_points": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
            }

        wins = trades[trades["points_captured"] > 0]
        losses = trades[trades["points_captured"] < 0]
        gross_profit = float(wins["points_captured"].sum()) if not wins.empty else 0.0
        gross_loss = abs(float(losses["points_captured"].sum())) if not losses.empty else 0.0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss else round(gross_profit, 2)

        return {
            "year": year,
            "trades": int(len(trades)),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "win_rate": round(float((trades["points_captured"] > 0).mean() * 100), 2),
            "net_points": round(float(trades["points_captured"].sum()), 2),
            "avg_points": round(float(trades["points_captured"].mean()), 2),
            "avg_win": round(float(wins["points_captured"].mean()), 2) if not wins.empty else 0.0,
            "avg_loss": round(float(losses["points_captured"].mean()), 2) if not losses.empty else 0.0,
            "profit_factor": profit_factor,
            "expectancy": round(float(trades["points_captured"].mean()), 2),
            "total_r": round(float(trades["r_multiple"].sum()), 2),
            "avg_r": round(float(trades["r_multiple"].mean()), 2),
            "target_hits": int((trades["exit_reason"] == "TARGET_HIT").sum()),
            "sl_hits": int((trades["exit_reason"] == "SL_HIT").sum()),
            "eod_exits": int((trades["exit_reason"] == "EOD_SQUARE_OFF").sum()),
            "max_drawdown_points": self._equity_drawdown(trades),
            "max_win": round(float(trades["points_captured"].max()), 2),
            "max_loss": round(float(trades["points_captured"].min()), 2),
        }

    @staticmethod
    def _build_report(summary: pd.DataFrame) -> str:
        try:
            table = summary.to_markdown(index=False)
        except ImportError:
            table = summary.to_string(index=False)
        return "\n".join(
            [
                "# Flux Charts Volumized Order Blocks Backtest Report",
                "",
                "This report summarizes the performance of the Python-translated Flux Volumized Order Blocks Strategy.",
                "",
                "## Strategy Parameters",
                "- Swing Length: 10",
                "- Max ATR Multiplier: 3.5",
                "- Stop Buffer: 3.0 points",
                "- Target R-to-R: 2.0 (fallback)",
                "- Square-off at: 15:15 IST",
                "",
                "## Yearly Results Summary",
                "",
                table,
                "",
                "## Trading Methodology",
                "1. **Long entry:** Price touches an unmitigated Bullish Order Block.",
                "2. **Short entry:** Price touches an unmitigated Bearish Order Block.",
                "3. **Long stop-loss:** Placed below the bottom of the active order block.",
                "4. **Short stop-loss:** Placed above the top of the active order block.",
                "5. **Exits:** Target opposing order block or 2.0R multiplier, or intraday square-off at 15:15.",
            ]
        )
