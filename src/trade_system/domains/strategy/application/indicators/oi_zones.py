"""
OI-Based Support & Resistance Zone Detector
============================================
Derives high-probability intraday S/R zones from the previous day's option
chain Open Interest and Volume data.

Theory
------
Options market-makers must delta-hedge at strikes with heavy OI, creating
price stickiness around those levels. This module identifies:

1. Max CE OI Strike       → Resistance (call writers defend)
2. Max PE OI Strike       → Support (put writers defend)
3. Max Pain Strike        → Expiry magnet (price drifts here)
4. Volume Cluster Strikes → Active zones (aggressive intraday flow)
5. GEX Gamma Wall         → Strong mean-reversion S/R
6. Previous Day H/L/Close → Classical S/R with OI confirmation

Usage
-----
    from trade_system.domains.strategy.application.indicators.oi_zones import OIZoneDetector

    detector = OIZoneDetector()
    zones = detector.compute_zones(
        oc_df=prev_day_oc_df,         # option chain snapshot DataFrame
        prev_day_high=24550.0,
        prev_day_low=24280.0,
        prev_day_close=24510.0,
        spot_price=24500.0,           # today's open / current spot
        lot_size=50,                   # Nifty=50, BankNifty=15
        strike_step=50,               # Nifty=50, BankNifty=100
    )
    for z in zones:
        print(z)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OIZone:
    """A single Option-Interest derived S/R zone."""

    strike: float
    """The price level / strike of the zone."""

    zone_type: str
    """
    One of:
      RESISTANCE      — Max CE OI or previous day high
      SUPPORT         — Max PE OI or previous day low
      MAX_PAIN        — Expiry magnet strike
      ACTIVE_CLUSTER  — High combined volume strike
      GAMMA_WALL      — Highest absolute GEX strike
      PREV_CLOSE      — Previous day close (pivot)
    """

    source: str
    """Detailed source label, e.g. 'CE_OI_MAX', 'PREV_HIGH'."""

    oi_value: float = 0.0
    """Raw OI (contracts) at this strike."""

    volume_value: float = 0.0
    """Raw volume at this strike."""

    strength: float = 0.0
    """
    Normalized strength 0.0–1.0.
    Higher = more likely to act as S/R.
    """

    is_resistance: bool = True
    """True = price resistance, False = price support."""

    band_top: float = field(init=False)
    """Top of the zone band (strike + tolerance)."""

    band_bottom: float = field(init=False)
    """Bottom of the zone band (strike - tolerance)."""

    # Tolerance is set by the detector based on ATR
    _tolerance: float = field(default=25.0, repr=False)

    def __post_init__(self) -> None:
        self.band_top = self.strike + self._tolerance
        self.band_bottom = self.strike - self._tolerance

    def set_tolerance(self, tolerance: float) -> None:
        """Adjust the band around the strike (call after init)."""
        self._tolerance = tolerance
        self.band_top = self.strike + tolerance
        self.band_bottom = self.strike - tolerance

    def contains(self, price: float) -> bool:
        """Return True if price is within this zone's band."""
        return self.band_bottom <= price <= self.band_top

    def __str__(self) -> str:
        direction = "R" if self.is_resistance else "S"
        return (
            f"[{direction}] {self.zone_type:<16} @ {self.strike:>8.1f} "
            f"| strength={self.strength:.2f} | OI={self.oi_value/1e5:.1f}L "
            f"| source={self.source}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Detector
# ═══════════════════════════════════════════════════════════════════════════════

class OIZoneDetector:
    """
    Computes Support / Resistance zones from a previous-day option chain snapshot.

    Parameters
    ----------
    num_top_strikes : int
        Number of top OI strikes to consider for clustering (default 5).
    num_volume_clusters : int
        Number of high-volume strikes to promote as ACTIVE_CLUSTER zones.
    tolerance_pct : float
        Zone band half-width as percentage of spot price (default 0.15%).
    min_oi_threshold : float
        Minimum OI (in contracts) to consider a strike significant.
    """

    def __init__(
        self,
        num_top_strikes: int = 5,
        num_volume_clusters: int = 3,
        tolerance_pct: float = 0.0015,
        min_oi_threshold: float = 500_000,
    ) -> None:
        self.num_top_strikes = num_top_strikes
        self.num_volume_clusters = num_volume_clusters
        self.tolerance_pct = tolerance_pct
        self.min_oi_threshold = min_oi_threshold

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def compute_zones(
        self,
        oc_df: pd.DataFrame,
        prev_day_high: Optional[float] = None,
        prev_day_low: Optional[float] = None,
        prev_day_close: Optional[float] = None,
        spot_price: Optional[float] = None,
        lot_size: int = 50,
        strike_step: int = 50,
    ) -> list[OIZone]:
        """
        Compute all S/R zones from the option chain snapshot.

        Parameters
        ----------
        oc_df : DataFrame
            Parsed option chain with columns: strike, option_type, oi (or open_interest),
            volume, ltp, gamma.  Uses the *last* timestamp snapshot if multiple
            timestamps are present.
        prev_day_high / low / close : float, optional
            Previous day OHLCV levels for classical S/R overlay.
        spot_price : float, optional
            Current spot price (for normalization and ATM calculation).
        lot_size : int
            Contract lot size for GEX calculation.
        strike_step : int
            Strike interval (50 for Nifty, 100 for BankNifty).

        Returns
        -------
        List of OIZone sorted by strength descending.
        """
        if oc_df is None or oc_df.empty:
            LOGGER.warning("OIZoneDetector: empty option chain dataframe received.")
            return []

        # ── Normalise column names ───────────────────────────────────────────
        df = oc_df.copy()
        df.columns = [c.lower() for c in df.columns]
        if "open_interest" in df.columns and "oi" not in df.columns:
            df.rename(columns={"open_interest": "oi"}, inplace=True)

        # ── Use the latest snapshot if multiple timestamps exist ─────────────
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            latest_ts = df["timestamp"].max()
            df = df[df["timestamp"] == latest_ts].copy()
            LOGGER.info("Using OC snapshot at %s (%d rows)", latest_ts, len(df))

        # ── Require minimum columns ──────────────────────────────────────────
        required = ["strike", "option_type", "oi", "volume"]
        for col in required:
            if col not in df.columns:
                LOGGER.error("OIZoneDetector: missing column '%s'. Aborting.", col)
                return []

        df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df = df.dropna(subset=["strike"])

        calls = df[df["option_type"].str.upper() == "CE"]
        puts = df[df["option_type"].str.upper() == "PE"]

        # Derive spot from data if not provided
        if spot_price is None or spot_price <= 0:
            if "spot_price" in df.columns:
                spot_price = float(df["spot_price"].replace(0, np.nan).dropna().iloc[-1]) if not df["spot_price"].replace(0, np.nan).dropna().empty else None
            if spot_price is None:
                spot_price = float(df["strike"].median())
        
        LOGGER.info("OIZoneDetector: spot_price=%.1f", spot_price)
        tolerance = spot_price * self.tolerance_pct

        zones: list[OIZone] = []

        # ── 1. Max CE OI → Resistance ────────────────────────────────────────
        zones += self._top_oi_strikes(
            calls, "CE", spot_price, tolerance, self.num_top_strikes, is_resistance=True
        )

        # ── 2. Max PE OI → Support ───────────────────────────────────────────
        zones += self._top_oi_strikes(
            puts, "PE", spot_price, tolerance, self.num_top_strikes, is_resistance=False
        )

        # ── 3. Max Pain Strike ────────────────────────────────────────────────
        mp = self._calculate_max_pain(df, spot_price)
        if mp is not None:
            z = OIZone(
                strike=float(mp),
                zone_type="MAX_PAIN",
                source="MAX_PAIN",
                oi_value=0.0,
                strength=0.6,
                is_resistance=spot_price > mp,
                _tolerance=tolerance,
            )
            z.__post_init__()
            zones.append(z)

        # ── 4. Volume Cluster Zones ──────────────────────────────────────────
        zones += self._volume_cluster_zones(df, spot_price, tolerance)

        # ── 5. GEX Gamma Wall ─────────────────────────────────────────────────
        gex_wall = self._calculate_gex_wall(df, spot_price, lot_size)
        if gex_wall is not None:
            is_res = gex_wall >= spot_price
            z = OIZone(
                strike=float(gex_wall),
                zone_type="GAMMA_WALL",
                source="GEX_WALL",
                oi_value=0.0,
                strength=0.85,
                is_resistance=is_res,
                _tolerance=tolerance,
            )
            z.__post_init__()
            zones.append(z)

        # ── 6. Previous Day Levels ────────────────────────────────────────────
        if prev_day_high:
            zones.append(self._prev_level_zone("RESISTANCE", "PREV_HIGH", prev_day_high, spot_price, tolerance, 0.70))
        if prev_day_low:
            zones.append(self._prev_level_zone("SUPPORT", "PREV_LOW", prev_day_low, spot_price, tolerance, 0.70))
        if prev_day_close:
            zones.append(self._prev_level_zone("PREV_CLOSE", "PREV_CLOSE", prev_day_close, spot_price, tolerance, 0.55))

        # ── Deduplicate overlapping zones ─────────────────────────────────────
        zones = self._deduplicate(zones, tolerance)

        # ── Sort by strength descending ───────────────────────────────────────
        zones.sort(key=lambda z: z.strength, reverse=True)
        LOGGER.info("OIZoneDetector: %d zones computed.", len(zones))
        return zones

    def get_nearest_zone(
        self, zones: list[OIZone], price: float
    ) -> Optional[OIZone]:
        """Return the closest zone to the given price."""
        if not zones:
            return None
        return min(zones, key=lambda z: abs(z.strike - price))

    def get_active_zones(
        self, zones: list[OIZone], price: float, window_pct: float = 0.01
    ) -> list[OIZone]:
        """Return zones within window_pct of the current price."""
        half = price * window_pct
        return [z for z in zones if abs(z.strike - price) <= half]

    def classify_price(
        self, zones: list[OIZone], price: float
    ) -> str:
        """
        Returns a bias label based on the strongest nearby zones:
        'AT_SUPPORT' | 'AT_RESISTANCE' | 'BETWEEN_ZONES' | 'NO_DATA'
        """
        if not zones:
            return "NO_DATA"
        active = self.get_active_zones(zones, price, window_pct=0.002)
        if not active:
            return "BETWEEN_ZONES"
        # Check if majority of nearby zones are support or resistance
        n_res = sum(1 for z in active if z.is_resistance)
        n_sup = sum(1 for z in active if not z.is_resistance)
        if n_res > n_sup:
            return "AT_RESISTANCE"
        elif n_sup > n_res:
            return "AT_SUPPORT"
        return "BETWEEN_ZONES"

    # ──────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _top_oi_strikes(
        self,
        df: pd.DataFrame,
        opt_type: str,
        spot: float,
        tolerance: float,
        n: int,
        is_resistance: bool,
    ) -> list[OIZone]:
        """Return top-N strikes by OI for a given option type."""
        if df.empty:
            return []
        oi_by_strike = df.groupby("strike")["oi"].sum().sort_values(ascending=False)
        total_oi = oi_by_strike.sum()
        zones = []
        for rank, (strike, oi_val) in enumerate(oi_by_strike.head(n).items()):
            if oi_val < self.min_oi_threshold:
                continue
            # Strength: normalized OI fraction + rank decay
            strength = min(1.0, (oi_val / max(total_oi, 1)) * n * 2) * (1.0 - rank * 0.1)
            z = OIZone(
                strike=float(strike),
                zone_type="RESISTANCE" if is_resistance else "SUPPORT",
                source=f"{opt_type}_OI_MAX" if rank == 0 else f"{opt_type}_OI_TOP{rank+1}",
                oi_value=float(oi_val),
                volume_value=float(df[df["strike"] == strike]["volume"].sum()),
                strength=round(strength, 3),
                is_resistance=is_resistance,
                _tolerance=tolerance,
            )
            z.__post_init__()
            zones.append(z)
        return zones

    @staticmethod
    def _calculate_max_pain(df: pd.DataFrame, spot: float) -> Optional[float]:
        """
        Max Pain = strike where total options payoff is minimized.
        Iterates all strikes and sums [CE: max(spot-K,0)*OI + PE: max(K-spot,0)*OI].
        """
        try:
            strikes = sorted(df["strike"].unique())
            total_pain = {}
            for k in strikes:
                ce_row = df[(df["strike"] == k) & (df["option_type"].str.upper() == "CE")]
                pe_row = df[(df["strike"] == k) & (df["option_type"].str.upper() == "PE")]
                ce_oi = float(ce_row["oi"].sum()) if not ce_row.empty else 0
                pe_oi = float(pe_row["oi"].sum()) if not pe_row.empty else 0

                pain = 0.0
                for s in strikes:
                    ce_s = df[(df["strike"] == s) & (df["option_type"].str.upper() == "CE")]
                    pe_s = df[(df["strike"] == s) & (df["option_type"].str.upper() == "PE")]
                    pain += max(k - s, 0) * float(ce_s["oi"].sum() if not ce_s.empty else 0)
                    pain += max(s - k, 0) * float(pe_s["oi"].sum() if not pe_s.empty else 0)
                total_pain[k] = pain

            if not total_pain:
                return None
            return min(total_pain, key=total_pain.get)
        except Exception as e:
            LOGGER.warning("Max pain calculation failed: %s", e)
            return None

    def _volume_cluster_zones(
        self, df: pd.DataFrame, spot: float, tolerance: float
    ) -> list[OIZone]:
        """Find strikes with highest combined CE+PE volume."""
        try:
            vol_by_strike = df.groupby("strike")["volume"].sum().sort_values(ascending=False)
            total_vol = vol_by_strike.sum()
            zones = []
            for rank, (strike, vol) in enumerate(vol_by_strike.head(self.num_volume_clusters).items()):
                if vol <= 0:
                    continue
                strength = min(0.75, (vol / max(total_vol, 1)) * self.num_volume_clusters * 2)
                is_res = float(strike) >= spot
                z = OIZone(
                    strike=float(strike),
                    zone_type="ACTIVE_CLUSTER",
                    source=f"VOL_CLUSTER_{rank+1}",
                    oi_value=float(df[df["strike"] == strike]["oi"].sum()),
                    volume_value=float(vol),
                    strength=round(strength, 3),
                    is_resistance=is_res,
                    _tolerance=tolerance,
                )
                z.__post_init__()
                zones.append(z)
            return zones
        except Exception as e:
            LOGGER.warning("Volume cluster calculation failed: %s", e)
            return []

    @staticmethod
    def _calculate_gex_wall(
        df: pd.DataFrame, spot: float, lot_size: int
    ) -> Optional[float]:
        """
        GEX by strike = spot * gamma * OI * lot_size
        GEX is positive for calls, negative for puts.
        Returns the strike with highest |GEX| (the gamma wall).
        """
        try:
            if "gamma" not in df.columns:
                return None
            df = df.copy()
            df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce").fillna(0)
            df["gex"] = spot * df["gamma"] * df["oi"] * lot_size
            df.loc[df["option_type"].str.upper() == "PE", "gex"] *= -1
            gex_by_strike = df.groupby("strike")["gex"].sum()
            if gex_by_strike.empty:
                return None
            return float(gex_by_strike.abs().idxmax())
        except Exception as e:
            LOGGER.warning("GEX wall calculation failed: %s", e)
            return None

    @staticmethod
    def _prev_level_zone(
        zone_type: str,
        source: str,
        price: float,
        spot: float,
        tolerance: float,
        strength: float,
    ) -> OIZone:
        is_res = price >= spot
        z = OIZone(
            strike=float(price),
            zone_type=zone_type,
            source=source,
            strength=strength,
            is_resistance=is_res,
            _tolerance=tolerance,
        )
        z.__post_init__()
        return z

    @staticmethod
    def _deduplicate(zones: list[OIZone], tolerance: float) -> list[OIZone]:
        """
        Merge zones within 2x tolerance of each other, keeping the strongest.
        """
        if not zones:
            return []
        # Sort by strike
        zones = sorted(zones, key=lambda z: z.strike)
        merged: list[OIZone] = []
        skip = set()
        for i, z1 in enumerate(zones):
            if i in skip:
                continue
            group = [z1]
            for j, z2 in enumerate(zones[i + 1:], start=i + 1):
                if j in skip:
                    continue
                if abs(z2.strike - z1.strike) <= tolerance * 2:
                    group.append(z2)
                    skip.add(j)
            # Keep the strongest in the group
            best = max(group, key=lambda z: z.strength)
            merged.append(best)
        return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience function
# ═══════════════════════════════════════════════════════════════════════════════

def compute_oi_zones(
    oc_df: pd.DataFrame,
    prev_day_high: Optional[float] = None,
    prev_day_low: Optional[float] = None,
    prev_day_close: Optional[float] = None,
    spot_price: Optional[float] = None,
    lot_size: int = 50,
    strike_step: int = 50,
) -> list[OIZone]:
    """Convenience wrapper around OIZoneDetector.compute_zones()."""
    return OIZoneDetector().compute_zones(
        oc_df=oc_df,
        prev_day_high=prev_day_high,
        prev_day_low=prev_day_low,
        prev_day_close=prev_day_close,
        spot_price=spot_price,
        lot_size=lot_size,
        strike_step=strike_step,
    )
