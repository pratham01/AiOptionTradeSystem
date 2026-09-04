"""
OptionCausalGraph — In-Memory Computational Directed Graph (DAG) for Strike Price Dynamics.

Models the multi-dimensional causal web governing option strike prices, dealer gamma hedging,
and institutional wall barriers using NetworkX and interactive Plotly visualization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

LOGGER = logging.getLogger(__name__)


# Standard Nifty 50 and Bank Nifty Top Heavyweight Weights
NIFTY_WEIGHTS = {
    "HDFCBANK": 0.145,
    "RELIANCE": 0.098,
    "ICICIBANK": 0.075,
    "INFY": 0.058,
    "ITC": 0.042,
    "TCS": 0.038,
}

BANKNIFTY_WEIGHTS = {
    "HDFCBANK": 0.290,
    "ICICIBANK": 0.240,
    "SBIN": 0.115,
    "AXISBANK": 0.102,
    "KOTAKBANK": 0.091,
}


@dataclass
class StrikeShockResult:
    """Projected metrics for a single strike under a simulated shock."""
    node_id: str
    strike: float
    option_type: str
    current_ltp: float
    projected_ltp: float
    ltp_change: float
    ltp_change_pct: float
    delta: float
    gamma: float
    vega: float
    theta: float
    oi: int
    net_gex: float


@dataclass
class ShockwaveSimulationResult:
    """Consolidated outcome of multi-hop causal graph shockwave propagation."""
    spot_shock_pct: float
    vix_shock_pct: float
    initial_spot: float
    projected_spot: float
    spot_delta: float
    stock_contributions: Dict[str, float]
    strike_projections: List[StrikeShockResult]
    dealer_hedge_flow_cr: float       # In Crores ₹ (Futures buying/selling required)
    dealer_hedge_direction: str       # "DEALERS BUYING FUTURES", "DEALERS SELLING FUTURES", "NEUTRAL"
    call_wall_status: str             # "DEFENDED", "TESTED", "BREACHED (GAMMA BLAST RISK!)"
    put_wall_status: str              # "DEFENDED", "TESTED", "BREACHED (LIQUIDATION PANIC RISK!)"
    gravitational_pull_max_pain: float # Distance to Max Pain


class OptionCausalGraph:
    """
    In-Memory Computational Graph representing the causal interconnected web
    between Underlying Spot, India VIX, Heavyweight Equities, Walls, and Option Strikes.
    """

    def __init__(self, symbol: str = "NSE:NIFTY50-INDEX") -> None:
        self.symbol = symbol
        self.graph = nx.DiGraph()
        self.node_positions: Dict[str, Tuple[float, float]] = {}

    def build_graph(
        self,
        spot_price: float,
        oc_df: pd.DataFrame,
        vix_level: float = 11.5,
        days_to_expiry: int = 5,
        strike_step: float = 50.0,
        n_strikes: int = 4,
        call_wall: Optional[float] = None,
        put_wall: Optional[float] = None,
        max_pain: Optional[float] = None,
    ) -> nx.DiGraph:
        """
        Construct the directed causal graph across all layers.
        """
        self.graph.clear()
        self.node_positions.clear()

        is_banknifty = "BANK" in self.symbol
        weights = BANKNIFTY_WEIGHTS if is_banknifty else NIFTY_WEIGHTS
        lot_size = 30 if is_banknifty else 75

        # Normalize Option Chain
        df = oc_df.copy() if oc_df is not None and not oc_df.empty else pd.DataFrame()
        if not df.empty:
            if "strike" not in df.columns and "strike_price" in df.columns:
                df["strike"] = pd.to_numeric(df["strike_price"], errors="coerce")
            if "delta" not in df.columns:
                df["delta"] = df["option_type"].apply(lambda t: 0.5 if t == "CE" else -0.5)
            if "gamma" not in df.columns:
                df["gamma"] = 0.0005
            if "vega" not in df.columns:
                df["vega"] = 2.0
            if "theta" not in df.columns:
                df["theta"] = -5.0

        atm_strike = round(spot_price / strike_step) * strike_step
        if call_wall is None:
            call_wall = atm_strike + (2 * strike_step)
        if put_wall is None:
            put_wall = atm_strike - (2 * strike_step)
        if max_pain is None:
            max_pain = atm_strike

        # ── Layer 1: Anchor Drivers ──────────────────────────────────────────
        # Spot Node (Center anchor)
        self.graph.add_node(
            "SPOT",
            label=f"{self.symbol.split(':')[-1]}\n₹{spot_price:,.1f}",
            category="SPOT",
            price=spot_price,
            size=28,
            color="#3a86ff",
            desc="Underlying Index Spot Anchor",
        )
        self.node_positions["SPOT"] = (0.0, 0.0)

        # India VIX Node
        self.graph.add_node(
            "VIX",
            label=f"India VIX\n{vix_level:.2f}",
            category="VOLATILITY",
            value=vix_level,
            size=22,
            color="#9d4edd",
            desc="Implied Volatility Regime",
        )
        self.node_positions["VIX"] = (0.0, 2.2)

        # Time / Theta Decay Node
        self.graph.add_node(
            "TIME",
            label=f"Theta / DTE\n{days_to_expiry} Days",
            category="TIME",
            value=days_to_expiry,
            size=18,
            color="#ffbe0b",
            desc="Exogenous Theta Decay Clock",
        )
        self.node_positions["TIME"] = (2.2, 2.2)

        # Heavyweight Constituent Nodes
        stock_x = np.linspace(-2.2, -0.6, len(weights))
        for idx, (stk, w) in enumerate(weights.items()):
            node_id = f"STK_{stk}"
            self.graph.add_node(
                node_id,
                label=f"{stk}\n({w*100:.1f}%)",
                category="CONSTITUENT",
                weight=w,
                size=16,
                color="#00b4d8",
                desc=f"Constituent Equity (Weight: {w*100:.1f}%)",
            )
            self.node_positions[node_id] = (stock_x[idx], 2.2)
            # Edge: Constituent -> Spot
            self.graph.add_edge(node_id, "SPOT", weight=w, label=f"{w*100:.1f}% wt", edge_type="WEIGHT")

        # ── Layer 2: Institutional Structures ────────────────────────────────
        # Call Wall
        self.graph.add_node(
            "CALL_WALL",
            label=f"Call Wall\n₹{call_wall:,.0f}",
            category="STRUCTURE",
            level=call_wall,
            size=22,
            color="#ff006e",
            desc="Primary Overhead Resistance & Dealer Short Gamma Wall",
        )
        self.node_positions["CALL_WALL"] = (-1.8, 0.9)
        self.graph.add_edge("CALL_WALL", "SPOT", weight=-0.8, label="Resistance Pin", edge_type="FRICTION")

        # Put Wall
        self.graph.add_node(
            "PUT_WALL",
            label=f"Put Wall\n₹{put_wall:,.0f}",
            category="STRUCTURE",
            level=put_wall,
            size=22,
            color="#06d6a0",
            desc="Primary Structural Support Floor & Put Writer Defense",
        )
        self.node_positions["PUT_WALL"] = (1.8, 0.9)
        self.graph.add_edge("PUT_WALL", "SPOT", weight=0.8, label="Support Floor", edge_type="FRICTION")

        # Max Pain
        self.graph.add_node(
            "MAX_PAIN",
            label=f"Max Pain\n₹{max_pain:,.0f}",
            category="STRUCTURE",
            level=max_pain,
            size=20,
            color="#ffaa00",
            desc="Option Writer Gravitational Settle Zone",
        )
        self.node_positions["MAX_PAIN"] = (0.0, 0.9)
        self.graph.add_edge("MAX_PAIN", "SPOT", weight=0.5, label="Gravitational Pull", edge_type="GRAVITY")

        # ── Layer 3: Strike Nodes (ATM ± n_strikes) ──────────────────────────
        strikes_to_add = [atm_strike + (i * strike_step) for i in range(-n_strikes, n_strikes + 1)]
        strikes_to_add = sorted(strikes_to_add)

        ce_y = np.linspace(-1.0, -2.6, len(strikes_to_add))
        pe_y = np.linspace(-1.0, -2.6, len(strikes_to_add))

        for idx, k in enumerate(strikes_to_add):
            # Extract CE row
            ce_id = f"{int(k)}_CE"
            ce_row = df[(df["strike"] == k) & (df["option_type"] == "CE")] if not df.empty else pd.DataFrame()
            ce_ltp = float(ce_row["ltp"].iloc[0]) if not ce_row.empty and "ltp" in ce_row.columns and pd.notna(ce_row["ltp"].iloc[0]) else max(10.0, (spot_price - k) if spot_price > k else 45.0)
            ce_oi = int(ce_row["oi"].iloc[0]) if not ce_row.empty and "oi" in ce_row.columns and pd.notna(ce_row["oi"].iloc[0]) else 500000
            ce_delta = float(ce_row["delta"].iloc[0]) if not ce_row.empty and "delta" in ce_row.columns and pd.notna(ce_row["delta"].iloc[0]) else 0.50
            ce_gamma = float(ce_row["gamma"].iloc[0]) if not ce_row.empty and "gamma" in ce_row.columns and pd.notna(ce_row["gamma"].iloc[0]) else 0.0005
            ce_vega = float(ce_row["vega"].iloc[0]) if not ce_row.empty and "vega" in ce_row.columns and pd.notna(ce_row["vega"].iloc[0]) else 2.5
            ce_theta = float(ce_row["theta"].iloc[0]) if not ce_row.empty and "theta" in ce_row.columns and pd.notna(ce_row["theta"].iloc[0]) else -5.0

            # CE Node
            self.graph.add_node(
                ce_id,
                label=f"{int(k)} CE\n₹{ce_ltp:.1f}",
                category="STRIKE_CE",
                strike=k,
                option_type="CE",
                ltp=ce_ltp,
                oi=ce_oi,
                delta=ce_delta,
                gamma=ce_gamma,
                vega=ce_vega,
                theta=ce_theta,
                net_gex=ce_gamma * ce_oi * lot_size * (spot_price ** 2) * 0.01,
                size=14 + min(12, int(ce_oi / 1000000)),
                color="#e63946",
                desc=f"Call Option {int(k)} CE (Δ={ce_delta:.2f}, OI={ce_oi:,})",
            )
            self.node_positions[ce_id] = (-1.6, ce_y[idx])

            # Edge: SPOT -> CE (via Delta)
            self.graph.add_edge("SPOT", ce_id, weight=ce_delta, label=f"Δ {ce_delta:.2f}", edge_type="DELTA")
            # Edge: VIX -> CE (via Vega)
            self.graph.add_edge("VIX", ce_id, weight=ce_vega, label=f"ν {ce_vega:.1f}", edge_type="VEGA")
            # Edge: TIME -> CE (via Theta)
            self.graph.add_edge("TIME", ce_id, weight=ce_theta, label=f"θ {ce_theta:.1f}", edge_type="THETA")

            # Extract PE row
            pe_id = f"{int(k)}_PE"
            pe_row = df[(df["strike"] == k) & (df["option_type"] == "PE")] if not df.empty else pd.DataFrame()
            pe_ltp = float(pe_row["ltp"].iloc[0]) if not pe_row.empty and "ltp" in pe_row.columns and pd.notna(pe_row["ltp"].iloc[0]) else max(10.0, (k - spot_price) if k > spot_price else 45.0)
            pe_oi = int(pe_row["oi"].iloc[0]) if not pe_row.empty and "oi" in pe_row.columns and pd.notna(pe_row["oi"].iloc[0]) else 500000
            pe_delta = float(pe_row["delta"].iloc[0]) if not pe_row.empty and "delta" in pe_row.columns and pd.notna(pe_row["delta"].iloc[0]) else -0.50
            pe_gamma = float(pe_row["gamma"].iloc[0]) if not pe_row.empty and "gamma" in pe_row.columns and pd.notna(pe_row["gamma"].iloc[0]) else 0.0005
            pe_vega = float(pe_row["vega"].iloc[0]) if not pe_row.empty and "vega" in pe_row.columns and pd.notna(pe_row["vega"].iloc[0]) else 2.5
            pe_theta = float(pe_row["theta"].iloc[0]) if not pe_row.empty and "theta" in pe_row.columns and pd.notna(pe_row["theta"].iloc[0]) else -5.0

            # PE Node
            self.graph.add_node(
                pe_id,
                label=f"{int(k)} PE\n₹{pe_ltp:.1f}",
                category="STRIKE_PE",
                strike=k,
                option_type="PE",
                ltp=pe_ltp,
                oi=pe_oi,
                delta=pe_delta,
                gamma=pe_gamma,
                vega=pe_vega,
                theta=pe_theta,
                net_gex=-pe_gamma * pe_oi * lot_size * (spot_price ** 2) * 0.01,
                size=14 + min(12, int(pe_oi / 1000000)),
                color="#2a9d8f",
                desc=f"Put Option {int(k)} PE (Δ={pe_delta:.2f}, OI={pe_oi:,})",
            )
            self.node_positions[pe_id] = (1.6, pe_y[idx])

            # Edge: SPOT -> PE (via Delta)
            self.graph.add_edge("SPOT", pe_id, weight=pe_delta, label=f"Δ {pe_delta:.2f}", edge_type="DELTA")
            # Edge: VIX -> PE (via Vega)
            self.graph.add_edge("VIX", pe_id, weight=pe_vega, label=f"ν {pe_vega:.1f}", edge_type="VEGA")
            # Edge: TIME -> PE (via Theta)
            self.graph.add_edge("TIME", pe_id, weight=pe_theta, label=f"θ {pe_theta:.1f}", edge_type="THETA")

        # ── Layer 4: Dealer Hedging Node ─────────────────────────────────────
        self.graph.add_node(
            "DEALER_HEDGE",
            label="Dealer Delta\nHedge Pool",
            category="HEDGE",
            size=24,
            color="#fb5607",
            desc="Dynamic Delta Rehedging Flow forced on Market Makers",
        )
        self.node_positions["DEALER_HEDGE"] = (0.0, -3.2)

        # Edges from strikes to Dealer Hedge
        for k in strikes_to_add:
            ce_id = f"{int(k)}_CE"
            pe_id = f"{int(k)}_PE"
            if ce_id in self.graph:
                self.graph.add_edge(ce_id, "DEALER_HEDGE", weight=self.graph.nodes[ce_id]["gamma"], edge_type="GAMMA_FLOW")
            if pe_id in self.graph:
                self.graph.add_edge(pe_id, "DEALER_HEDGE", weight=self.graph.nodes[pe_id]["gamma"], edge_type="GAMMA_FLOW")

        # Feedback edge from Dealer Hedge back to SPOT
        self.graph.add_edge("DEALER_HEDGE", "SPOT", weight=1.0, label="Hedging Feedback", edge_type="FEEDBACK")

        return self.graph

    def simulate_shockwave(
        self,
        spot_shock_pct: float = 0.0,
        vix_shock_pct: float = 0.0,
        constituent_shocks: Optional[Dict[str, float]] = None,
    ) -> ShockwaveSimulationResult:
        """
        Execute multi-hop shockwave propagation across all graph edges.
        """
        if "SPOT" not in self.graph.nodes:
            raise ValueError("Causal graph must be built before running simulation.")

        initial_spot = float(self.graph.nodes["SPOT"]["price"])
        vix_node = self.graph.nodes.get("VIX", {})
        initial_vix = float(vix_node.get("value", 11.5))
        vix_delta = initial_vix * (vix_shock_pct / 100.0)

        # 1. Propagate Constituent Shocks to Spot
        constituent_shocks = constituent_shocks or {}
        stock_contributions = {}
        stock_spot_shift = 0.0

        for node_id in self.graph.nodes:
            if node_id.startswith("STK_"):
                stk_symbol = node_id.replace("STK_", "")
                w = float(self.graph.nodes[node_id]["weight"])
                shk = float(constituent_shocks.get(stk_symbol, 0.0))
                contrib = initial_spot * w * (shk / 100.0)
                stock_contributions[stk_symbol] = contrib
                stock_spot_shift += contrib

        # Net Spot Change
        direct_spot_shift = initial_spot * (spot_shock_pct / 100.0)
        total_spot_delta = direct_spot_shift + stock_spot_shift
        projected_spot = initial_spot + total_spot_delta

        # 2. Propagate Spot & VIX Shifts to Option Strikes
        strike_results: List[StrikeShockResult] = []
        is_banknifty = "BANK" in self.symbol
        lot_size = 30 if is_banknifty else 75
        total_futures_shares_rehedge = 0.0

        for node_id, data in self.graph.nodes(data=True):
            if data.get("category") in ("STRIKE_CE", "STRIKE_PE"):
                k = float(data["strike"])
                opt_type = data["option_type"]
                ltp = float(data["ltp"])
                d = float(data["delta"])
                g = float(data["gamma"])
                v = float(data["vega"])
                th = float(data["theta"])
                oi = int(data["oi"])
                net_gex = float(data["net_gex"])

                # Second-order Taylor expansion for option price change
                # dP = Delta * dS + 0.5 * Gamma * (dS)^2 + Vega * dVol
                delta_p = (d * total_spot_delta) + (0.5 * g * (total_spot_delta ** 2)) + (v * (vix_delta / 10.0))
                proj_ltp = max(0.05, round(ltp + delta_p, 2))
                ltp_ch = round(proj_ltp - ltp, 2)
                ltp_ch_pct = round((ltp_ch / max(ltp, 0.05)) * 100.0, 1)

                # Dealer Delta Hedging Impact
                # Delta shift = Gamma * dS -> Rehedging contracts needed = Gamma * dS * OI * LotSize
                rehedge_shares = g * total_spot_delta * oi * lot_size
                if opt_type == "CE":
                    total_futures_shares_rehedge += rehedge_shares
                else:
                    total_futures_shares_rehedge -= rehedge_shares

                strike_results.append(
                    StrikeShockResult(
                        node_id=node_id,
                        strike=k,
                        option_type=opt_type,
                        current_ltp=ltp,
                        projected_ltp=proj_ltp,
                        ltp_change=ltp_ch,
                        ltp_change_pct=ltp_ch_pct,
                        delta=d,
                        gamma=g,
                        vega=v,
                        theta=th,
                        oi=oi,
                        net_gex=net_gex,
                    )
                )

        # Dealer Rehedging Flow in Crores
        dealer_flow_cr = round((total_futures_shares_rehedge * projected_spot) / 10000000.0, 2)
        if dealer_flow_cr > 25.0:
            dealer_dir = f"DEALERS FORCED TO BUY (+₹{dealer_flow_cr:.1f} Cr) — SHORT SQUEEZE FUEL 🚀"
        elif dealer_flow_cr < -25.0:
            dealer_dir = f"DEALERS FORCED TO SELL (-₹{abs(dealer_flow_cr):.1f} Cr) — CASCADE LIQUIDATION 🔻"
        else:
            dealer_dir = f"BALANCED / RESILIENT (₹{dealer_flow_cr:+.1f} Cr)"

        # 3. Assess Wall Breaches
        call_wall = float(self.graph.nodes.get("CALL_WALL", {}).get("level", projected_spot + 200))
        put_wall = float(self.graph.nodes.get("PUT_WALL", {}).get("level", projected_spot - 200))
        max_pain = float(self.graph.nodes.get("MAX_PAIN", {}).get("level", projected_spot))

        if projected_spot >= call_wall + 25:
            cw_status = "⚠️ BREACHED! Call Writers Trapped in Gamma Blast"
        elif projected_spot >= call_wall - 30:
            cw_status = "⚡ TESTING Call Wall Resistance"
        else:
            cw_status = "🛡️ DEFENDED"

        if projected_spot <= put_wall - 25:
            pw_status = "⚠️ BREACHED! Put Writers Trapped in Panic Unwinding"
        elif projected_spot <= put_wall + 30:
            pw_status = "⚡ TESTING Put Wall Support"
        else:
            pw_status = "🛡️ DEFENDED"

        return ShockwaveSimulationResult(
            spot_shock_pct=spot_shock_pct,
            vix_shock_pct=vix_shock_pct,
            initial_spot=initial_spot,
            projected_spot=round(projected_spot, 2),
            spot_delta=round(total_spot_delta, 2),
            stock_contributions=stock_contributions,
            strike_projections=sorted(strike_results, key=lambda x: (x.strike, x.option_type)),
            dealer_hedge_flow_cr=dealer_flow_cr,
            dealer_hedge_direction=dealer_dir,
            call_wall_status=cw_status,
            put_wall_status=pw_status,
            gravitational_pull_max_pain=round(projected_spot - max_pain, 2),
        )

    def build_plotly_figure(
        self,
        simulation_result: Optional[ShockwaveSimulationResult] = None,
    ) -> go.Figure:
        """
        Render the entire graph network into an interactive dark-themed Plotly figure.
        """
        edge_x = []
        edge_y = []
        edge_colors = []

        # Prepare node projection lookup
        proj_map = {}
        if simulation_result:
            for s in simulation_result.strike_projections:
                proj_map[s.node_id] = s

        # Draw Edges
        for u, v, data in self.graph.edges(data=True):
            if u in self.node_positions and v in self.node_positions:
                x0, y0 = self.node_positions[u]
                x1, y1 = self.node_positions[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.2, color="#30363d"),
            hoverinfo="none",
            mode="lines",
        )

        # Draw Nodes
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_texts = []
        node_hovers = []

        for node_id, data in self.graph.nodes(data=True):
            x, y = self.node_positions[node_id]
            node_x.append(x)
            node_y.append(y)
            node_colors.append(data.get("color", "#ffffff"))
            node_sizes.append(data.get("size", 16))
            node_texts.append(data.get("label", node_id).replace("\n", "<br>"))

            # Hover Tooltip
            cat = data.get("category", "")
            if cat in ("STRIKE_CE", "STRIKE_PE"):
                p_info = proj_map.get(node_id)
                proj_str = f"<br>⚡ Projected LTP: ₹{p_info.projected_ltp} ({p_info.ltp_change_pct:+.1f}%)" if p_info else ""
                hover = (
                    f"<b>{data.get('desc')}</b><br>"
                    f"Current LTP: ₹{data.get('ltp', 0):.2f}{proj_str}<br>"
                    f"Open Interest: {data.get('oi', 0):,} contracts<br>"
                    f"Delta (Δ): {data.get('delta', 0):.2f}<br>"
                    f"Gamma (Γ): {data.get('gamma', 0):.4f}<br>"
                    f"Vega (ν): {data.get('vega', 0):.2f}<br>"
                    f"Theta (θ): {data.get('theta', 0):.1f}"
                )
            elif cat == "SPOT":
                proj_spot_str = f"<br>⚡ Projected Spot: ₹{simulation_result.projected_spot} ({simulation_result.spot_delta:+.1f} pts)" if simulation_result else ""
                hover = f"<b>{data.get('desc')}</b><br>Current Spot: ₹{data.get('price', 0):,.2f}{proj_spot_str}"
            else:
                hover = f"<b>{data.get('desc')}</b>"

            node_hovers.append(hover)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_texts,
            textposition="top center",
            textfont=dict(size=9, color="#e6edf3", family="Arial"),
            hoverinfo="text",
            hovertext=node_hovers,
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color="#0d1117"),
                opacity=0.95,
            ),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=30),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                height=650,
            ),
        )

        return fig
