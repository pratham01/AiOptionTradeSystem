from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path

import pandas as pd

from trade_system.application.analysis import prepare_price_action_concepts
from trade_system.application.backtesting.ict_fvg_liquidity_research import FVGDetector, OrderBlockDetector
from trade_system.application.indicators import calculate_supertrend


ENTRY_CUTOFF = time(14, 45)
SQUARE_OFF_BAR = time(15, 15)


@dataclass(slots=True)
class ZoneEngineArtifacts:
    summary: pd.DataFrame
    summary_path: Path
    trades_path: Path
    report_path: Path


class ZoneEngineResearch:
    def __init__(
        self,
        *,
        fvg_min_size: float = 8.0,
        ob_impulse_thresh: float = 12.0,
        zone_buffer_points: float = 8.0,
        stop_buffer_points: float = 5.0,
        min_volume_ratio: float = 1.15,
        min_body_ratio: float = 0.4,
        reward_risk: float = 2.0,
        allowed_zone_types: set[str] | None = None,
    ) -> None:
        self.fvg_min_size = fvg_min_size
        self.ob_impulse_thresh = ob_impulse_thresh
        self.zone_buffer_points = zone_buffer_points
        self.stop_buffer_points = stop_buffer_points
        self.min_volume_ratio = min_volume_ratio
        self.min_body_ratio = min_body_ratio
        self.reward_risk = reward_risk
        self.allowed_zone_types = allowed_zone_types

    def run_year(self, file_path: Path, output_dir: Path) -> ZoneEngineArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        trades = self._backtest(file_path)
        summary = pd.DataFrame([self._summarize(trades)])
        year = int(file_path.stem.rsplit("_", 1)[-1])
        trades_path = output_dir / f"zone_engine_{year}_trades.csv"
        summary_path = output_dir / f"zone_engine_{year}_summary.csv"
        report_path = output_dir / f"zone_engine_{year}_report.md"
        trades.to_csv(trades_path, index=False)
        summary.to_csv(summary_path, index=False)
        report_path.write_text(self._build_report(year, summary, trades))
        return ZoneEngineArtifacts(summary=summary, summary_path=summary_path, trades_path=trades_path, report_path=report_path)

    def _backtest(self, file_path: Path) -> pd.DataFrame:
        source = pd.read_csv(file_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        source["trade_date"] = source["timestamp"].dt.date
        source["session_vwap"] = (
            (source["close"] * source["volume"]).groupby(source["trade_date"]).cumsum()
            / source["volume"].groupby(source["trade_date"]).cumsum().replace(0, pd.NA)
        )
        source["volume_sma20"] = source["volume"].rolling(20, min_periods=5).mean()
        source["volume_ratio"] = source["volume"] / source["volume_sma20"].replace(0, pd.NA)
        concepts = prepare_price_action_concepts(source.set_index("timestamp")[["open", "high", "low", "close", "volume"]])
        concepts = concepts.reset_index()
        st = calculate_supertrend(source[["timestamp", "open", "high", "low", "close", "volume"]], period=7, multiplier=3)
        data = source.merge(concepts.drop(columns=["open", "high", "low", "close", "volume"], errors="ignore"), on="timestamp", how="left")
        data = data.merge(st[["timestamp", "supertrend", "supertrend_direction"]], on="timestamp", how="left")

        day_zone_map = self._build_previous_day_zones(source)
        fvg = FVGDetector(min_size=self.fvg_min_size)
        ob = OrderBlockDetector(impulse_thresh=self.ob_impulse_thresh)

        trades: list[dict[str, object]] = []
        position: dict[str, object] | None = None

        for idx in range(len(data)):
            window = data.iloc[: idx + 1].copy()
            row = data.iloc[idx]
            ts = pd.Timestamp(row["timestamp"])
            current_date = ts.date()

            active_fvgs = fvg.update(window[["timestamp", "open", "high", "low", "close", "volume"]])
            active_obs = ob.update(window[["timestamp", "open", "high", "low", "close", "volume"]])
            _ = active_obs
            zones = self._candidate_zones(
                row=row,
                current_date=current_date,
                previous_day_zones=day_zone_map.get(current_date, []),
                active_fvgs=active_fvgs,
                ob_detector=ob,
            )

            if position is not None:
                closed = self._maybe_close(position, row, ts)
                if closed is not None:
                    trades.append(closed)
                    position = None

            if position is not None:
                continue
            if ts.time() > ENTRY_CUTOFF:
                continue

            long_zone = self._best_zone_match(row, zones, direction=1)
            short_zone = self._best_zone_match(row, zones, direction=-1)

            if long_zone is not None and self._long_confirmation(row):
                position = self._open_position(row, ts, long_zone, direction=1)
            elif short_zone is not None and self._short_confirmation(row):
                position = self._open_position(row, ts, short_zone, direction=-1)

        if position is not None:
            last = data.iloc[-1]
            trades.append(self._close(position, pd.Timestamp(last["timestamp"]), float(last["close"]), "FORCED_CLOSE"))

        frame = pd.DataFrame(trades)
        if not frame.empty:
            frame["holding_minutes"] = (
                pd.to_datetime(frame["exit_time"]) - pd.to_datetime(frame["entry_time"])
            ).dt.total_seconds() / 60.0
        return frame

    def _build_previous_day_zones(self, source: pd.DataFrame) -> dict[object, list[dict[str, object]]]:
        by_date = {trade_date: frame.copy() for trade_date, frame in source.groupby("trade_date")}
        ordered_dates = sorted(by_date)
        zone_map: dict[object, list[dict[str, object]]] = {}
        for idx in range(1, len(ordered_dates)):
            prev_date = ordered_dates[idx - 1]
            curr_date = ordered_dates[idx]
            prev_frame = by_date[prev_date]
            profile = self._volume_profile(prev_frame)
            prev_open = float(prev_frame["open"].iloc[0])
            prev_high = float(prev_frame["high"].max())
            prev_low = float(prev_frame["low"].min())
            prev_close = float(prev_frame["close"].iloc[-1])
            zone_map[curr_date] = [
                self._make_zone("prev_day_open", prev_open, "support"),
                self._make_zone("prev_day_close", prev_close, "support"),
                self._make_zone("prev_day_high", prev_high, "resistance"),
                self._make_zone("prev_day_low", prev_low, "support"),
                self._make_zone("prev_day_poc", profile["poc"], "support"),
                self._make_zone("prev_day_val", profile["val"], "support"),
                self._make_zone("prev_day_vah", profile["vah"], "resistance"),
            ]
        return zone_map

    def _volume_profile(self, frame: pd.DataFrame) -> dict[str, float]:
        bins = (frame["close"].astype(float) / 10.0).round() * 10.0
        profile = frame.assign(bucket=bins).groupby("bucket")["volume"].sum().sort_index()
        total_volume = float(profile.sum())
        prices = list(profile.index.astype(float))
        volumes = [float(profile.loc[price]) for price in prices]
        poc_index = max(range(len(prices)), key=lambda i: volumes[i])
        selected = {poc_index}
        running = volumes[poc_index]
        low = poc_index - 1
        high = poc_index + 1
        while running < total_volume * 0.70 and (low >= 0 or high < len(prices)):
            low_vol = volumes[low] if low >= 0 else -1.0
            high_vol = volumes[high] if high < len(prices) else -1.0
            if high_vol >= low_vol:
                selected.add(high)
                running += volumes[high]
                high += 1
            else:
                selected.add(low)
                running += volumes[low]
                low -= 1
        chosen = [prices[idx] for idx in sorted(selected)]
        return {"poc": prices[poc_index], "val": min(chosen), "vah": max(chosen)}

    def _make_zone(self, zone_type: str, level: float, side: str) -> dict[str, object]:
        return {
            "zone_type": zone_type,
            "side": side,
            "center": level,
            "lower": level - self.zone_buffer_points,
            "upper": level + self.zone_buffer_points,
        }

    def _candidate_zones(
        self,
        *,
        row: pd.Series,
        current_date: object,
        previous_day_zones: list[dict[str, object]],
        active_fvgs: list[dict[str, object]],
        ob_detector: OrderBlockDetector,
    ) -> list[dict[str, object]]:
        zones = list(previous_day_zones)
        if pd.notna(row.get("range_low_20")):
            zones.append(self._make_zone("range_support_20", float(row["range_low_20"]), "support"))
        if pd.notna(row.get("range_high_20")):
            zones.append(self._make_zone("range_resistance_20", float(row["range_high_20"]), "resistance"))

        price = float(row["close"])
        bull_ob = ob_detector.in_zone(price, 1)
        bear_ob = ob_detector.in_zone(price, -1)
        if bull_ob is not None:
            zones.append(
                {
                    "zone_type": "bull_order_block",
                    "side": "support",
                    "center": (float(bull_ob["top"]) + float(bull_ob["bottom"])) / 2.0,
                    "lower": float(bull_ob["bottom"]),
                    "upper": float(bull_ob["top"]),
                }
            )
        if bear_ob is not None:
            zones.append(
                {
                    "zone_type": "bear_order_block",
                    "side": "resistance",
                    "center": (float(bear_ob["top"]) + float(bear_ob["bottom"])) / 2.0,
                    "lower": float(bear_ob["bottom"]),
                    "upper": float(bear_ob["top"]),
                }
            )
        for fvg in active_fvgs[-4:]:
            zones.append(
                {
                    "zone_type": "bull_fvg" if fvg["type"] == "BFVG" else "bear_fvg",
                    "side": "support" if fvg["type"] == "BFVG" else "resistance",
                    "center": float(fvg["mid"]),
                    "lower": float(fvg["bottom"]),
                    "upper": float(fvg["top"]),
                }
            )
        if self.allowed_zone_types is None:
            return zones
        return [zone for zone in zones if str(zone["zone_type"]) in self.allowed_zone_types]

    def _best_zone_match(self, row: pd.Series, zones: list[dict[str, object]], direction: int) -> dict[str, object] | None:
        side = "support" if direction == 1 else "resistance"
        candidates = [zone for zone in zones if zone["side"] == side]
        if not candidates:
            return None
        if direction == 1:
            touched = [
                zone
                for zone in candidates
                if float(row["low"]) <= float(zone["upper"]) and float(row["close"]) >= float(zone["lower"])
            ]
        else:
            touched = [
                zone
                for zone in candidates
                if float(row["high"]) >= float(zone["lower"]) and float(row["close"]) <= float(zone["upper"])
            ]
        if not touched:
            return None
        price = float(row["close"])
        return min(touched, key=lambda zone: abs(price - float(zone["center"])))

    @staticmethod
    def _scalar(row: pd.Series, key: str, default: float = 0.0) -> float:
        value = row.get(key, default)
        if pd.isna(value):
            return default
        return float(value)

    def _long_confirmation(self, row: pd.Series) -> bool:
        volume_ratio = self._scalar(row, "volume_ratio", 0.0)
        return (
            int(row.get("supertrend_direction", 0) or 0) == 1
            and self._scalar(row, "close") > self._scalar(row, "open")
            and self._scalar(row, "body_ratio") >= self.min_body_ratio
            and volume_ratio >= self.min_volume_ratio
            and self._scalar(row, "close") >= self._scalar(row, "ema20", self._scalar(row, "close"))
            and (
                bool(row.get("bull_liquidity_sweep"))
                or int(row.get("internal_choch", 0) or 0) == 1
                or int(row.get("swing_choch", 0) or 0) == 1
                or bool(row.get("bull_ob_retest"))
                or bool(row.get("bull_fvg_recent"))
            )
        )

    def _short_confirmation(self, row: pd.Series) -> bool:
        volume_ratio = self._scalar(row, "volume_ratio", 0.0)
        return (
            int(row.get("supertrend_direction", 0) or 0) == -1
            and self._scalar(row, "close") < self._scalar(row, "open")
            and self._scalar(row, "body_ratio") >= self.min_body_ratio
            and volume_ratio >= self.min_volume_ratio
            and self._scalar(row, "close") <= self._scalar(row, "ema20", self._scalar(row, "close"))
            and (
                bool(row.get("bear_liquidity_sweep"))
                or int(row.get("internal_choch", 0) or 0) == -1
                or int(row.get("swing_choch", 0) or 0) == -1
                or bool(row.get("bear_ob_retest"))
                or bool(row.get("bear_fvg_recent"))
            )
        )

    def _open_position(self, row: pd.Series, ts: pd.Timestamp, zone: dict[str, object], direction: int) -> dict[str, object]:
        entry = float(row["close"])
        if direction == 1:
            stop = min(float(row["low"]), float(zone["lower"])) - self.stop_buffer_points
            risk = entry - stop
            target = entry + risk * self.reward_risk
        else:
            stop = max(float(row["high"]), float(zone["upper"])) + self.stop_buffer_points
            risk = stop - entry
            target = entry - risk * self.reward_risk
        return {
            "side": direction,
            "entry_time": ts.to_pydatetime(),
            "entry_price": entry,
            "stop_loss": stop,
            "take_profit": target,
            "zone_type": zone["zone_type"],
            "zone_center": float(zone["center"]),
            "entry_reason": f"{'LONG' if direction == 1 else 'SHORT'}_{zone['zone_type']}",
            "risk_points": risk,
        }

    def _maybe_close(self, position: dict[str, object], row: pd.Series, ts: pd.Timestamp) -> dict[str, object] | None:
        side = int(position["side"])
        if side == 1:
            if float(row["low"]) <= float(position["stop_loss"]):
                return self._close(position, ts, float(position["stop_loss"]), "SL_HIT")
            if float(row["high"]) >= float(position["take_profit"]):
                return self._close(position, ts, float(position["take_profit"]), "TARGET_HIT")
            if int(row.get("supertrend_direction", 0) or 0) == -1:
                return self._close(position, ts, float(row["close"]), "TREND_CHANGE")
        else:
            if float(row["high"]) >= float(position["stop_loss"]):
                return self._close(position, ts, float(position["stop_loss"]), "SL_HIT")
            if float(row["low"]) <= float(position["take_profit"]):
                return self._close(position, ts, float(position["take_profit"]), "TARGET_HIT")
            if int(row.get("supertrend_direction", 0) or 0) == 1:
                return self._close(position, ts, float(row["close"]), "TREND_CHANGE")
        if ts.time() >= SQUARE_OFF_BAR:
            return self._close(position, ts, float(row["close"]), "EOD_SQUARE_OFF")
        return None

    def _close(self, position: dict[str, object], exit_time: pd.Timestamp, exit_price: float, exit_reason: str) -> dict[str, object]:
        side = int(position["side"])
        pnl = (exit_price - float(position["entry_price"])) * side
        risk = float(position["risk_points"])
        return {
            "entry_time": position["entry_time"],
            "entry_price": round(float(position["entry_price"]), 2),
            "stop_loss": round(float(position["stop_loss"]), 2),
            "take_profit": round(float(position["take_profit"]), 2),
            "exit_time": exit_time,
            "exit_price": round(float(exit_price), 2),
            "exit_reason": exit_reason,
            "side": "LONG" if side == 1 else "SHORT",
            "zone_type": position["zone_type"],
            "zone_center": round(float(position["zone_center"]), 2),
            "entry_reason": position["entry_reason"],
            "points_captured": round(pnl, 2),
            "risk_points": round(risk, 2),
            "r_multiple": round(pnl / risk, 2) if risk else 0.0,
            "outcome": "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT",
        }

    @staticmethod
    def _summarize(trades: pd.DataFrame) -> dict[str, object]:
        if trades.empty:
            return {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_points": 0.0,
                "avg_points": 0.0,
                "avg_r": 0.0,
                "target_hits": 0,
                "sl_hits": 0,
                "trend_exits": 0,
                "eod_exits": 0,
                "max_win": 0.0,
                "max_loss": 0.0,
            }
        return {
            "trades": int(len(trades)),
            "wins": int((trades["points_captured"] > 0).sum()),
            "losses": int((trades["points_captured"] < 0).sum()),
            "win_rate": round(float((trades["points_captured"] > 0).mean() * 100), 2),
            "net_points": round(float(trades["points_captured"].sum()), 2),
            "avg_points": round(float(trades["points_captured"].mean()), 2),
            "avg_r": round(float(trades["r_multiple"].mean()), 2),
            "target_hits": int((trades["exit_reason"] == "TARGET_HIT").sum()),
            "sl_hits": int((trades["exit_reason"] == "SL_HIT").sum()),
            "trend_exits": int((trades["exit_reason"] == "TREND_CHANGE").sum()),
            "eod_exits": int((trades["exit_reason"] == "EOD_SQUARE_OFF").sum()),
            "max_win": round(float(trades["points_captured"].max()), 2),
            "max_loss": round(float(trades["points_captured"].min()), 2),
        }

    @staticmethod
    def _to_markdown(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except ImportError:
            return frame.to_string(index=False)

    def _build_report(self, year: int, summary: pd.DataFrame, trades: pd.DataFrame) -> str:
        zone_breakdown = pd.DataFrame()
        monthly = pd.DataFrame()
        if not trades.empty:
            zone_breakdown = (
                trades.groupby("zone_type", as_index=False)
                .agg(
                    trades=("zone_type", "count"),
                    net_points=("points_captured", "sum"),
                    avg_points=("points_captured", "mean"),
                    win_rate=("points_captured", lambda values: (values > 0).mean() * 100),
                    avg_r=("r_multiple", "mean"),
                )
                .sort_values("net_points", ascending=False)
            )
            monthly_frame = trades.copy()
            monthly_frame["month"] = pd.to_datetime(monthly_frame["entry_time"]).dt.to_period("M").astype(str)
            monthly = (
                monthly_frame.groupby("month", as_index=False)
                .agg(
                    trades=("month", "count"),
                    net_points=("points_captured", "sum"),
                    avg_points=("points_captured", "mean"),
                    win_rate=("points_captured", lambda values: (values > 0).mean() * 100),
                )
                .sort_values("month")
            )
        return "\n".join(
            [
                f"# Zone Engine Report {year}",
                "",
                "Method:",
                "- previous-day levels: open, close, high, low",
                "- previous-day volume profile: POC, VAL, VAH",
                "- rolling range extremes as dynamic support/resistance",
                "- active bull/bear order blocks and FVG zones",
                "- entries only when price rejects a zone with volume, body strength, and Supertrend alignment",
                "",
                "## Summary",
                "",
                self._to_markdown(summary),
                "",
                "## Zone Breakdown",
                "",
                self._to_markdown(zone_breakdown) if not zone_breakdown.empty else "No trades",
                "",
                "## Monthly Breakdown",
                "",
                self._to_markdown(monthly) if not monthly.empty else "No trades",
            ]
        )
