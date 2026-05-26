from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_system.application.indicators import calculate_supertrend


ENTRY_CUTOFF_HOUR = 14
ENTRY_CUTOFF_MINUTE = 45
SQUARE_OFF_HOUR = 15
SQUARE_OFF_MINUTE = 15


@dataclass(slots=True)
class VolumeProfileSummary:
    point_of_control: float
    value_area_low: float
    value_area_high: float


@dataclass(slots=True)
class IctFvgLiquidityArtifacts:
    summary: pd.DataFrame
    summary_path: Path
    report_path: Path


class FVGDetector:
    def __init__(self, min_size: float = 8.0) -> None:
        self.min_size = min_size
        self._fvgs: list[dict[str, object]] = []

    def update(self, window: pd.DataFrame) -> list[dict[str, object]]:
        if len(window) < 3:
            return []
        c1 = window.iloc[-3]
        c2 = window.iloc[-2]
        c3 = window.iloc[-1]

        for fvg in self._fvgs:
            if bool(fvg["filled"]):
                continue
            if fvg["type"] == "BFVG" and float(c3["low"]) <= float(fvg["bottom"]) + self.min_size * 0.5:
                fvg["filled"] = True
            if fvg["type"] == "BKFVG" and float(c3["high"]) >= float(fvg["top"]) - self.min_size * 0.5:
                fvg["filled"] = True

        new_fvg: dict[str, object] | None = None
        if float(c3["low"]) > float(c1["high"]) and float(c3["low"]) - float(c1["high"]) >= self.min_size:
            new_fvg = {
                "type": "BFVG",
                "timestamp": pd.Timestamp(c2["timestamp"]),
                "top": float(c3["low"]),
                "bottom": float(c1["high"]),
                "mid": (float(c3["low"]) + float(c1["high"])) / 2.0,
                "size": float(c3["low"]) - float(c1["high"]),
                "filled": False,
            }
        elif float(c3["high"]) < float(c1["low"]) and float(c1["low"]) - float(c3["high"]) >= self.min_size:
            new_fvg = {
                "type": "BKFVG",
                "timestamp": pd.Timestamp(c2["timestamp"]),
                "top": float(c1["low"]),
                "bottom": float(c3["high"]),
                "mid": (float(c1["low"]) + float(c3["high"])) / 2.0,
                "size": float(c1["low"]) - float(c3["high"]),
                "filled": False,
            }
        if new_fvg is not None and not any(pd.Timestamp(f["timestamp"]) == pd.Timestamp(new_fvg["timestamp"]) for f in self._fvgs):
            self._fvgs.append(new_fvg)

        self._fvgs = self._fvgs[-20:]
        return [f for f in self._fvgs if not bool(f["filled"])]

    def nearest_unfilled(self, price: float, fvg_type: str | None = None) -> dict[str, object] | None:
        candidates = [f for f in self._fvgs if not bool(f["filled"]) and (fvg_type is None or f["type"] == fvg_type)]
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(float(item["mid"]) - price))

    def in_zone(self, price: float, fvg_type: str | None = None) -> dict[str, object] | None:
        for fvg in reversed(self._fvgs):
            if bool(fvg["filled"]):
                continue
            if fvg_type and fvg["type"] != fvg_type:
                continue
            if float(fvg["bottom"]) <= price <= float(fvg["top"]):
                return fvg
        return None


class OrderBlockDetector:
    def __init__(self, impulse_thresh: float = 12.0, lookforward: int = 3) -> None:
        self.impulse_thresh = impulse_thresh
        self.lookforward = lookforward
        self._obs: list[dict[str, object]] = []

    def update(self, window: pd.DataFrame) -> list[dict[str, object]]:
        if len(window) < self.lookforward + 2:
            return self._obs
        c = window.iloc[-self.lookforward - 1]
        future = window.iloc[-self.lookforward :]
        latest = window.iloc[-1]

        if float(c["close"]) > float(c["open"]):
            down_move = float(c["close"]) - float(future["low"].min())
            if down_move >= self.impulse_thresh:
                ob = {
                    "type": "BearOB",
                    "timestamp": pd.Timestamp(c["timestamp"]),
                    "top": float(c["high"]),
                    "bottom": float(c["open"]),
                    "strength": down_move,
                    "tested": False,
                    "valid": True,
                }
                if not any(abs(float(item["top"]) - ob["top"]) < 5 for item in self._obs if item["type"] == "BearOB"):
                    self._obs.append(ob)

        if float(c["close"]) < float(c["open"]):
            up_move = float(future["high"].max()) - float(c["close"])
            if up_move >= self.impulse_thresh:
                ob = {
                    "type": "BullOB",
                    "timestamp": pd.Timestamp(c["timestamp"]),
                    "top": float(c["open"]),
                    "bottom": float(c["close"]),
                    "strength": up_move,
                    "tested": False,
                    "valid": True,
                }
                if not any(abs(float(item["bottom"]) - ob["bottom"]) < 5 for item in self._obs if item["type"] == "BullOB"):
                    self._obs.append(ob)

        for ob in self._obs:
            if not bool(ob["tested"]):
                if ob["type"] == "BullOB" and float(ob["bottom"]) <= float(latest["low"]) <= float(ob["top"]):
                    ob["tested"] = True
                elif ob["type"] == "BearOB" and float(ob["bottom"]) <= float(latest["high"]) <= float(ob["top"]):
                    ob["tested"] = True

            if ob["type"] == "BullOB" and float(latest["close"]) < float(ob["bottom"]):
                ob["valid"] = False
            elif ob["type"] == "BearOB" and float(latest["close"]) > float(ob["top"]):
                ob["valid"] = False

        self._obs = [item for item in self._obs[-30:] if bool(item["valid"])]
        return self._obs

    def in_zone(self, price: float, direction: int) -> dict[str, object] | None:
        wanted = "BullOB" if direction == 1 else "BearOB"
        candidates = [item for item in self._obs if item["type"] == wanted and bool(item["valid"])]
        for ob in sorted(candidates, key=lambda item: float(item["strength"]), reverse=True):
            if float(ob["bottom"]) <= price <= float(ob["top"]):
                return ob
        return None


class LiquidityTracker:
    def __init__(self, tolerance: float = 5.0, min_touches: int = 2, lookback: int = 20) -> None:
        self.tolerance = tolerance
        self.min_touches = min_touches
        self.lookback = lookback
        self._levels: list[dict[str, object]] = []

    def update(self, window: pd.DataFrame) -> dict[str, object]:
        if len(window) < self.lookback + 1:
            return {"levels": [], "swept": None}

        recent = window.iloc[-self.lookback - 1 : -1]
        latest = window.iloc[-1]
        highs = [float(value) for value in recent["high"]]
        lows = [float(value) for value in recent["low"]]
        levels: list[dict[str, object]] = []

        for high in highs:
            touches = sum(1 for candidate in highs if abs(candidate - high) < self.tolerance)
            if touches >= self.min_touches and not any(
                abs(float(item["level"]) - high) < self.tolerance for item in levels if item["type"] == "BSL"
            ):
                levels.append({"type": "BSL", "level": high, "touches": touches})

        for low in lows:
            touches = sum(1 for candidate in lows if abs(candidate - low) < self.tolerance)
            if touches >= self.min_touches and not any(
                abs(float(item["level"]) - low) < self.tolerance for item in levels if item["type"] == "SSL"
            ):
                levels.append({"type": "SSL", "level": low, "touches": touches})

        self._levels = levels
        swept = None
        for level in levels:
            if level["type"] == "BSL" and float(latest["high"]) > float(level["level"]) and float(latest["close"]) < float(level["level"]):
                swept = {
                    "type": "BSL_SWEEP",
                    "level": float(level["level"]),
                    "direction": -1,
                    "strength": int(level["touches"]),
                }
            if level["type"] == "SSL" and float(latest["low"]) < float(level["level"]) and float(latest["close"]) > float(level["level"]):
                swept = {
                    "type": "SSL_SWEEP",
                    "level": float(level["level"]),
                    "direction": 1,
                    "strength": int(level["touches"]),
                }
        return {"levels": levels, "swept": swept}

    def nearest_ssl(self, price: float) -> float | None:
        levels = [float(item["level"]) for item in self._levels if item["type"] == "SSL" and float(item["level"]) < price]
        return max(levels) if levels else None

    def nearest_bsl(self, price: float) -> float | None:
        levels = [float(item["level"]) for item in self._levels if item["type"] == "BSL" and float(item["level"]) > price]
        return min(levels) if levels else None


class MarketStructureTracker:
    def __init__(self, lookback: int = 5) -> None:
        self.lookback = lookback
        self._swing_highs: list[float] = []
        self._swing_lows: list[float] = []
        self._trend: int = 0
        self._last_pivot_time: pd.Timestamp | None = None
        self._last_event_time: pd.Timestamp | None = None

    def update(self, window: pd.DataFrame) -> dict[str, object]:
        if len(window) < self.lookback * 2 + 1:
            return {"trend": self._trend, "bos": None, "choch": None}

        look_window = window.iloc[-(self.lookback * 2 + 1) :]
        pivot = look_window.iloc[self.lookback]
        pivot_time = pd.Timestamp(pivot["timestamp"])
        if self._last_pivot_time != pivot_time:
            left = look_window.iloc[: self.lookback]
            right = look_window.iloc[self.lookback + 1 :]
            is_high = all(float(pivot["high"]) >= float(value) for value in left["high"]) and all(
                float(pivot["high"]) >= float(value) for value in right["high"]
            )
            is_low = all(float(pivot["low"]) <= float(value) for value in left["low"]) and all(
                float(pivot["low"]) <= float(value) for value in right["low"]
            )
            if is_high:
                self._swing_highs.append(float(pivot["high"]))
                self._swing_highs = self._swing_highs[-10:]
            if is_low:
                self._swing_lows.append(float(pivot["low"]))
                self._swing_lows = self._swing_lows[-10:]
            self._last_pivot_time = pivot_time

        latest = window.iloc[-1]
        previous = window.iloc[-2]
        latest_time = pd.Timestamp(latest["timestamp"])
        bos = None
        choch = None

        if self._swing_highs and float(previous["close"]) <= self._swing_highs[-1] < float(latest["close"]):
            if self._last_event_time != latest_time:
                if self._trend != 1:
                    choch = {"type": "CHOCH_UP", "level": self._swing_highs[-1], "price": float(latest["close"])}
                    self._trend = 1
                else:
                    bos = {"type": "BOS_UP", "level": self._swing_highs[-1], "price": float(latest["close"])}
                self._last_event_time = latest_time
        elif self._swing_lows and float(previous["close"]) >= self._swing_lows[-1] > float(latest["close"]):
            if self._last_event_time != latest_time:
                if self._trend != -1:
                    choch = {"type": "CHOCH_DOWN", "level": self._swing_lows[-1], "price": float(latest["close"])}
                    self._trend = -1
                else:
                    bos = {"type": "BOS_DOWN", "level": self._swing_lows[-1], "price": float(latest["close"])}
                self._last_event_time = latest_time

        return {"trend": self._trend, "bos": bos, "choch": choch}


class IctFvgLiquidityResearch:
    def __init__(
        self,
        *,
        fvg_min_size: float = 8.0,
        ob_impulse_thresh: float = 12.0,
        liquidity_tolerance: float = 5.0,
        bos_lookback: int = 5,
        stop_buffer_points: float = 5.0,
        reward_risk: float = 2.0,
        min_reward_risk: float = 1.2,
    ) -> None:
        self.fvg_min_size = fvg_min_size
        self.ob_impulse_thresh = ob_impulse_thresh
        self.liquidity_tolerance = liquidity_tolerance
        self.bos_lookback = bos_lookback
        self.stop_buffer_points = stop_buffer_points
        self.reward_risk = reward_risk
        self.min_reward_risk = min_reward_risk

    def run(self, data_dir: Path, output_dir: Path) -> IctFvgLiquidityArtifacts:
        files = sorted(data_dir.glob("NSE_NIFTY50-INDEX_3min_*.csv"))
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        all_trades: list[pd.DataFrame] = []

        for file_path in files:
            year = int(file_path.stem.rsplit("_", 1)[-1])
            trades = self.run_year(file_path)
            if not trades.empty:
                all_trades.append(trades.assign(year=year))
            trades.to_csv(output_dir / f"ict_fvg_liquidity_{year}_trades.csv", index=False)
            rows.append(self._summarize_year(year, trades))

        summary = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
        if all_trades:
            combined = pd.concat(all_trades, ignore_index=True)
            combined.to_csv(output_dir / "ict_fvg_liquidity_all_trades.csv", index=False)
            rows.append(self._summarize_year("ALL", combined))
            summary = pd.DataFrame(rows)
        summary_path = output_dir / "ict_fvg_liquidity_summary.csv"
        summary.to_csv(summary_path, index=False)
        report_path = output_dir / "ict_fvg_liquidity_report.md"
        report_path.write_text(self._build_report(summary))
        return IctFvgLiquidityArtifacts(summary=summary, summary_path=summary_path, report_path=report_path)

    def run_year(self, file_path: Path) -> pd.DataFrame:
        frame = pd.read_csv(file_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if frame.empty:
            return pd.DataFrame()
        st = calculate_supertrend(frame[["timestamp", "open", "high", "low", "close", "volume"]], period=7, multiplier=3)
        frame = frame.merge(st[["timestamp", "supertrend", "supertrend_direction"]], on="timestamp", how="left")
        frame["trade_date"] = frame["timestamp"].dt.date

        fvg = FVGDetector(min_size=self.fvg_min_size)
        ob = OrderBlockDetector(impulse_thresh=self.ob_impulse_thresh)
        liq = LiquidityTracker(tolerance=self.liquidity_tolerance)
        ms = MarketStructureTracker(lookback=self.bos_lookback)

        trades: list[dict[str, object]] = []
        position: dict[str, object] | None = None

        for idx in range(len(frame)):
            window = frame.iloc[: idx + 1]
            row = frame.iloc[idx]
            ts = pd.Timestamp(row["timestamp"])

            active_fvgs = fvg.update(window)
            active_obs = ob.update(window)
            _ = active_obs
            liq_snapshot = liq.update(window)
            ms_snapshot = ms.update(window)

            price = float(row["close"])
            st_dir = int(row["supertrend_direction"]) if pd.notna(row["supertrend_direction"]) else 0
            bull_fvg = fvg.in_zone(price, "BFVG")
            bear_fvg = fvg.in_zone(price, "BKFVG")
            bull_ob = ob.in_zone(price, 1)
            bear_ob = ob.in_zone(price, -1)
            sweep = liq_snapshot["swept"]
            bullish_structure = bool(ms_snapshot["bos"] and ms_snapshot["bos"]["type"] == "BOS_UP") or bool(
                ms_snapshot["choch"] and ms_snapshot["choch"]["type"] == "CHOCH_UP"
            ) or int(ms_snapshot["trend"]) == 1
            bearish_structure = bool(ms_snapshot["bos"] and ms_snapshot["bos"]["type"] == "BOS_DOWN") or bool(
                ms_snapshot["choch"] and ms_snapshot["choch"]["type"] == "CHOCH_DOWN"
            ) or int(ms_snapshot["trend"]) == -1

            if position is not None:
                self._update_excursions(position, row)
                closed = self._check_exit(position, row, ts, sweep, bullish_structure, bearish_structure)
                if closed is not None:
                    trades.append(closed)
                    position = None

            if position is not None:
                continue
            if ts.time().hour > ENTRY_CUTOFF_HOUR or (
                ts.time().hour == ENTRY_CUTOFF_HOUR and ts.time().minute > ENTRY_CUTOFF_MINUTE
            ):
                continue
            if ts.time().hour < 9 or (ts.time().hour == 9 and ts.time().minute < 30):
                continue

            if sweep and int(sweep["direction"]) == 1:
                score = int(bullish_structure) + int(bull_fvg is not None) + int(bull_ob is not None) + int(st_dir == 1)
                if score >= 2:
                    position = self._build_position(
                        direction=1,
                        row=row,
                        ts=ts,
                        confluence_score=score,
                        trigger="SSL_SWEEP",
                        sweep=sweep,
                        fvg_zone=bull_fvg,
                        ob_zone=bull_ob,
                        target_level=liq.nearest_bsl(price),
                    )
            elif sweep and int(sweep["direction"]) == -1:
                score = int(bearish_structure) + int(bear_fvg is not None) + int(bear_ob is not None) + int(st_dir == -1)
                if score >= 2:
                    position = self._build_position(
                        direction=-1,
                        row=row,
                        ts=ts,
                        confluence_score=score,
                        trigger="BSL_SWEEP",
                        sweep=sweep,
                        fvg_zone=bear_fvg,
                        ob_zone=bear_ob,
                        target_level=liq.nearest_ssl(price),
                    )

        if position is not None:
            last = frame.iloc[-1]
            trades.append(self._close_position(position, pd.Timestamp(last["timestamp"]), float(last["close"]), "FORCED_CLOSE"))

        return pd.DataFrame(trades)

    def _build_position(
        self,
        *,
        direction: int,
        row: pd.Series,
        ts: pd.Timestamp,
        confluence_score: int,
        trigger: str,
        sweep: dict[str, object],
        fvg_zone: dict[str, object] | None,
        ob_zone: dict[str, object] | None,
        target_level: float | None,
    ) -> dict[str, object] | None:
        entry_price = float(row["close"])
        if direction == 1:
            stop_candidates = [float(row["low"]), float(sweep["level"])]
            if fvg_zone is not None:
                stop_candidates.append(float(fvg_zone["bottom"]))
            if ob_zone is not None:
                stop_candidates.append(float(ob_zone["bottom"]))
            stop_loss = min(stop_candidates) - self.stop_buffer_points
            risk = entry_price - stop_loss
        else:
            stop_candidates = [float(row["high"]), float(sweep["level"])]
            if fvg_zone is not None:
                stop_candidates.append(float(fvg_zone["top"]))
            if ob_zone is not None:
                stop_candidates.append(float(ob_zone["top"]))
            stop_loss = max(stop_candidates) + self.stop_buffer_points
            risk = stop_loss - entry_price

        if risk <= 0:
            return None

        rr_target = entry_price + direction * risk * self.reward_risk
        take_profit = rr_target
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
            "confluence_score": confluence_score,
            "trigger": trigger,
            "fvg_type": fvg_zone["type"] if fvg_zone is not None else "",
            "ob_type": ob_zone["type"] if ob_zone is not None else "",
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
        sweep: dict[str, object] | None,
        bullish_structure: bool,
        bearish_structure: bool,
    ) -> dict[str, object] | None:
        direction = int(position["direction"])
        stop_loss = float(position["stop_loss"])
        take_profit = float(position["take_profit"])

        if direction == 1:
            if float(row["low"]) <= stop_loss:
                return self._close_position(position, ts, stop_loss, "SL_HIT")
            if float(row["high"]) >= take_profit:
                return self._close_position(position, ts, take_profit, "TARGET_HIT")
            if sweep and int(sweep["direction"]) == -1 and bearish_structure:
                return self._close_position(position, ts, float(row["close"]), "OPPOSITE_SIGNAL")
        else:
            if float(row["high"]) >= stop_loss:
                return self._close_position(position, ts, stop_loss, "SL_HIT")
            if float(row["low"]) <= take_profit:
                return self._close_position(position, ts, take_profit, "TARGET_HIT")
            if sweep and int(sweep["direction"]) == 1 and bullish_structure:
                return self._close_position(position, ts, float(row["close"]), "OPPOSITE_SIGNAL")

        if ts.time().hour > SQUARE_OFF_HOUR or (ts.time().hour == SQUARE_OFF_HOUR and ts.time().minute >= SQUARE_OFF_MINUTE):
            return self._close_position(position, ts, float(row["close"]), "EOD_SQUARE_OFF")
        return None

    @staticmethod
    def _compute_profile(day_frame: pd.DataFrame) -> VolumeProfileSummary | None:
        if day_frame.empty:
            return None
        buckets = (day_frame["close"].astype(float) / 10.0).round() * 10.0
        profile = day_frame.assign(bucket=buckets).groupby("bucket")["volume"].sum().sort_index()
        if profile.empty:
            return None
        total = float(profile.sum())
        prices = list(profile.index.astype(float))
        volumes = [float(profile.loc[price]) for price in prices]
        poc_idx = max(range(len(prices)), key=lambda idx: volumes[idx])
        selected = {poc_idx}
        cumulative = volumes[poc_idx]
        low_idx = poc_idx - 1
        high_idx = poc_idx + 1
        while cumulative < total * 0.70 and (low_idx >= 0 or high_idx < len(prices)):
            low_vol = volumes[low_idx] if low_idx >= 0 else -1.0
            high_vol = volumes[high_idx] if high_idx < len(prices) else -1.0
            if high_vol >= low_vol:
                selected.add(high_idx)
                cumulative += volumes[high_idx]
                high_idx += 1
            else:
                selected.add(low_idx)
                cumulative += volumes[low_idx]
                low_idx -= 1
        chosen = [prices[idx] for idx in sorted(selected)]
        return VolumeProfileSummary(point_of_control=prices[poc_idx], value_area_low=min(chosen), value_area_high=max(chosen))

    def _close_position(self, position: dict[str, object], exit_time: pd.Timestamp, exit_price: float, exit_reason: str) -> dict[str, object]:
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
            "exit_price": round(float(exit_price), 2),
            "exit_reason": exit_reason,
            "trigger": position["trigger"],
            "fvg_type": position["fvg_type"],
            "ob_type": position["ob_type"],
            "confluence_score": int(position["confluence_score"]),
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
                "opposite_exits": 0,
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
            "opposite_exits": int((trades["exit_reason"] == "OPPOSITE_SIGNAL").sum()),
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
                "# ICT / FVG / Liquidity Backtest",
                "",
                "Rules used in this repo-native backtest:",
                "- 3-minute NIFTY candles",
                "- entry only after a liquidity sweep",
                "- bullish trades require bullish structure plus at least one of FVG / OB / Supertrend alignment",
                "- bearish trades require bearish structure plus at least one of FVG / OB / Supertrend alignment",
                "- stop-loss uses sweep level / bar extreme / FVG or OB invalidation with a small buffer",
                "- target uses nearest opposing liquidity if it offers at least the minimum R multiple, otherwise fixed 2R",
                "- square-off at 15:15 IST",
                "",
                "This is a historical price-action approximation of the donor ICT/SMC logic. It does not include options-chain or external premarket flow filters.",
                "",
                "## Summary",
                "",
                table,
            ]
        )
