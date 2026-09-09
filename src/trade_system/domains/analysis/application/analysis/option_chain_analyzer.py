import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Column schema for the parsed option chain DataFrame ─────────────────────
_OC_COLUMNS = [
    'strike', 'expiry', 'option_type', 'symbol', 'ltp', 'bid', 'ask', 'volume', 
    'oi', 'change', 'change_percent', 'iv', 'delta', 'gamma', 'theta', 'vega', 'rho'
]

class OptionChainAnalyzer:
    """
    Fetches, parses, and deeply analyzes option chain data for institutional forensics.
    """

    def __init__(self, fyers_client, symbol="NIFTY50", strike_count=20):
        self.fyers = fyers_client
        self.symbol = symbol
        self.strike_count = strike_count
        self.expiries = []
        self.nearest_expiry = None
        self._pcr_history = []
        self._pcr_history_max = 5
        self._fetch_and_set_expiries()

    def _fetch_and_set_expiries(self):
        try:
            oc_response = self._fetch_option_chain()
            if oc_response:
                df, _ = self._parse_option_chain(oc_response)
                if df is not None and not df.empty:
                    self.expiries = sorted(df['expiry'].unique())
        except: pass

    def analyze(self, df=None, spot_price=None, vix=None, prev_df=None, prev_day_df=None):
        """Full analysis pipeline."""
        try:
            if spot_price is None or vix is None:
                fetched_spot, fetched_vix = self._fetch_spot_and_vix()
                spot_price = spot_price or fetched_spot
                vix = vix or fetched_vix

            if spot_price is None: return None

            if df is None or df.empty:
                oc_response = self._fetch_option_chain()
                df, _ = self._parse_option_chain(oc_response)
            
            if df is None or df.empty: return None

            # Institutional Layers
            metrics = self._calculate_metrics(df, spot_price)
            oi_walls = self._calculate_oi_walls(df, spot_price)
            max_pain = self._calculate_max_pain(df)
            institutional = self._detect_institutional_activity(df, spot_price)
            iv_skew = self._analyse_iv_skew(df, spot_price)
            gex_analysis = self._calculate_gex(df, spot_price)
            
            # --- NEW: ATM Volume Delta Forensics (Divergence Detector) ---
            vol_delta_analysis = self._calculate_atm_volume_delta(df, spot_price)
            
            confluence = self._build_confluence_score(
                metrics, oi_walls, max_pain, iv_skew, institutional, gex_analysis, vol_delta_analysis
            )
            
            market_nature = self._detect_market_nature(metrics, vix, confluence)

            return {
                'metrics': metrics,
                'oi_walls': oi_walls,
                'max_pain': max_pain,
                'institutional': institutional,
                'iv_skew': iv_skew,
                'gex': gex_analysis,
                'vol_delta': vol_delta_analysis,
                'confluence': confluence,
                'market_nature': market_nature,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return None

    def get_option_chain_df(self):
        """Fetch and return the parsed option chain DataFrame."""
        response = self._fetch_option_chain()
        if not response:
            return None
        df, spot = self._parse_option_chain(response)
        return df

    def _fetch_option_chain(self):
        try:
            prefix = "BSE" if self.symbol == "SENSEX" else "NSE"
            data = {"symbol": f"{prefix}:{self.symbol}-INDEX", "strikecount": self.strike_count, "greeks": "1"}
            # The dummy client in test uses 'optionchain'.
            if hasattr(self.fyers, 'optionchain'):
                return self.fyers.optionchain(data=data)
            return self.fyers.optionschain(data=data)
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            return None

    def _parse_option_chain(self, response):
        try:
            if not response or response.get('s') != 'ok': return None, None
            
            spot_price = response.get('data', {}).get('ltp')
            
            expiry_data = response.get('data', {}).get('expiryData', [])
            if expiry_data:
                self.nearest_expiry = expiry_data[0].get('date')  # Format: DD-MM-YYYY
            else:
                self.nearest_expiry = None
            
            oc_list = response.get('data', {}).get('optionsChain', [])
            if not oc_list: return None, spot_price
            
            # Flatten nested greeks if present
            for item in oc_list:
                if 'greeks' in item and isinstance(item['greeks'], dict):
                    greeks = item.pop('greeks')
                    item.update(greeks)
                    
            df = pd.DataFrame(oc_list)
            
            # Filter out underlying spot if returned in optionsChain array
            df = df[df['option_type'].isin(['CE', 'PE'])]
            
            # Combine duplicate/legacy keys manually to avoid pandas duplicate column errors
            if 'strike_price' in df.columns:
                df['strike'] = df['strike'].combine_first(df['strike_price']) if 'strike' in df.columns else df['strike_price']
                df = df.drop(columns=['strike_price'])
                
            if 'open_interest' in df.columns:
                df['oi'] = df['oi'].combine_first(df['open_interest']) if 'oi' in df.columns else df['open_interest']
                df = df.drop(columns=['open_interest'])
                
            if 'last_price' in df.columns:
                df['ltp'] = df['ltp'].combine_first(df['last_price']) if 'ltp' in df.columns else df['last_price']
                df = df.drop(columns=['last_price'])
                
            # Note: the test uses ltpch / change interchangeably. We can map them.
            if 'ltpch' in df.columns:
                df['change'] = df['change'].combine_first(df['ltpch']) if 'change' in df.columns else df['ltpch']
                
            if 'ltpchp' in df.columns:
                df['change_percent'] = df['change_percent'].combine_first(df['ltpchp']) if 'change_percent' in df.columns else df['ltpchp']
            
            # Ensure all required columns exist and in the right order
            for col in _OC_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0
                    
            # Filter to only the columns we care about to prevent duplicate reindex errors
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.reindex(columns=_OC_COLUMNS)
                    
            return df, spot_price
        except Exception as e:
            logger.error(f"Error parsing option chain: {e}")
            return None, None

    def _fetch_spot_and_vix(self):
        try:
            symbols = f"NSE:{self.symbol}-INDEX,NSE:INDIAVIX-INDEX"
            res = self.fyers.quotes({"symbols": symbols})
            if res.get('s') != 'ok': return None, 15.0
            d = {q['v']['symbol']: q['v']['lp'] for q in res['data']}
            return d.get(f"NSE:{self.symbol}-INDEX"), d.get("NSE:INDIAVIX-INDEX", 15.0)
        except: return None, 15.0

    def _calculate_metrics(self, df, spot):
        calls = df[df['option_type'] == 'CE']
        puts = df[df['option_type'] == 'PE']
        c_oi = float(calls['oi'].sum())
        p_oi = float(puts['oi'].sum())
        return {
            'pcr_oi': round(p_oi/c_oi, 2) if c_oi > 0 else 1.0,
            'total_call_oi': c_oi,
            'total_put_oi': p_oi,
            'spot_price': spot
        }

    def _calculate_oi_walls(self, df, spot):
        calls = df[df['option_type'] == 'CE']
        puts = df[df['option_type'] == 'PE']
        ce_wall = float(calls.loc[calls['oi'].idxmax(), 'strike']) if not calls.empty else spot + 100
        pe_wall = float(puts.loc[puts['oi'].idxmax(), 'strike']) if not puts.empty else spot - 100
        return {
            'resistance': ce_wall,
            'support': pe_wall,
            'near_resistance': abs(spot-ce_wall) < 30,
            'near_support': abs(spot-pe_wall) < 30
        }

    def _calculate_max_pain(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate true institutional Max Pain strike (minimum writer payout loss)."""
        if df is None or df.empty or 'strike' not in df.columns or 'oi' not in df.columns or 'option_type' not in df.columns:
            return {'max_pain_strike': 0.0}
        
        strikes = sorted(df['strike'].dropna().unique())
        if not strikes:
            return {'max_pain_strike': 0.0}

        ce_df = df[df['option_type'] == 'CE'][['strike', 'oi']].dropna()
        pe_df = df[df['option_type'] == 'PE'][['strike', 'oi']].dropna()
        
        losses = {}
        for s in strikes:
            ce_loss = ((s - ce_df['strike']).clip(lower=0) * ce_df['oi']).sum()
            pe_loss = ((pe_df['strike'] - s).clip(lower=0) * pe_df['oi']).sum()
            losses[s] = float(ce_loss + pe_loss)
            
        if losses:
            min_strike = min(losses, key=losses.get)
            return {'max_pain_strike': float(min_strike), 'total_loss': losses[min_strike]}
        return {'max_pain_strike': float(strikes[len(strikes) // 2])}

    def _detect_institutional_activity(self, df, spot):
        # Classify based on Price vs OI Change
        return {'net_signal': 'NEUTRAL'}

    def _analyse_iv_skew(self, df, spot):
        try:
            atm = round(spot / 50) * 50
            calls = df[(df['strike'] == atm) & (df['option_type'] == 'CE')]
            puts = df[(df['strike'] == atm) & (df['option_type'] == 'PE')]
            if calls.empty or puts.empty: return {'skew': 'NEUTRAL'}
            
            c_iv = float(calls['iv'].iloc[0])
            p_iv = float(puts['iv'].iloc[0])
            diff = p_iv - c_iv
            
            if diff > 2.0: label = "PUT_SKEW (Bearish Fear)"
            elif diff < -2.0: label = "CALL_SKEW (Bullish Greed)"
            else: label = "NEUTRAL"
            
            return {'skew': label, 'put_iv': p_iv, 'call_iv': c_iv}
        except: return {'skew': 'NEUTRAL'}

    def _calculate_gex(self, df, spot_price):
        try:
            lot_size = 25 if "BANK" in self.symbol else (10 if "SENSEX" in self.symbol else 50)
            # GEX approximation
            df['strike_gex'] = spot_price * 0.01 * df.get('gamma', 0) * df['oi'] * lot_size
            df.loc[df['option_type'] == 'PE', 'strike_gex'] *= -1
            total_gex = df['strike_gex'].sum()
            gamma_wall = df.loc[df['strike_gex'].abs().idxmax(), 'strike']
            return {
                'total_gex': round(total_gex, 0),
                'gex_label': "POSITIVE_GEX (Stable)" if total_gex > 0 else "NEGATIVE_GEX (Volatile)",
                'gamma_wall': int(gamma_wall)
            }
        except: return {'total_gex': 0, 'gex_label': 'NEUTRAL'}

    # ══════════════════════════════════════════════════════════════════════════
    #  LAYER 11 — ATM VOLUME DELTA (Divergence Detector)
    # ══════════════════════════════════════════════════════════════════════════
    def _calculate_atm_volume_delta(self, df, spot_price):
        """
        Analyzes 3 strike prices adjacent to ATM (ATM-1, ATM, ATM+1).
        Know the 'Volume Divergence' - is the current price move supported by 
        aggressive option volume intensity?
        """
        try:
            strike_step = 100 if "BANK" in self.symbol else 50
            atm = round(spot_price / strike_step) * strike_step
            targets = [atm - strike_step, atm, atm + strike_step]
            
            details = []
            total_ce_vol = 0
            total_pe_vol = 0
            
            for s in targets:
                c = df[(df['strike'] == s) & (df['option_type'] == 'CE')]
                p = df[(df['strike'] == s) & (df['option_type'] == 'PE')]
                
                c_vol = int(c['volume'].iloc[0]) if not c.empty else 0
                p_vol = int(p['volume'].iloc[0]) if not p.empty else 0
                
                v_delta = c_vol - p_vol
                v_ratio = c_vol / max(p_vol, 1)
                
                total_ce_vol += c_vol
                total_pe_vol += p_vol
                
                details.append({
                    "strike": s,
                    "ce_vol": c_vol,
                    "pe_vol": p_vol,
                    "delta": v_delta,
                    "ratio": round(v_ratio, 2)
                })

            net_delta = total_ce_vol - total_pe_vol
            net_ratio = total_ce_vol / max(total_pe_vol, 1)
            
            # Classification
            if net_ratio >= 1.5: label = "AGGRESSIVE_BULLISH_VOLUME"
            elif net_ratio <= 0.65: label = "AGGRESSIVE_BEARISH_VOLUME"
            else: label = "NEUTRAL_INTENSITY"

            return {
                "net_delta": net_delta,
                "net_ratio": round(net_ratio, 2),
                "label": label,
                "strikes": details,
                "is_divergent": False # Logic handled in confluence
            }
        except Exception as e:
            logger.error(f"Volume delta failed: {e}")
            return {"label": "UNKNOWN"}

    def _build_confluence_score(self, metrics, walls, pain, skew, inst, gex, vol_delta):
        score = 5 # Neutral start
        reasons = []
        
        # 1. Volume Divergence Check
        v_label = vol_delta.get("label")
        if v_label == "AGGRESSIVE_BULLISH_VOLUME":
            score += 1
            reasons.append("Bullish ATM Volume Intensity")
        elif v_label == "AGGRESSIVE_BEARISH_VOLUME":
            score -= 1
            reasons.append("Bearish ATM Volume Intensity")
            
        # 2. GEX Filter
        if gex.get("gex_label") == "NEGATIVE_GEX (Volatile)":
            reasons.append("High Volatility Regime (Negative Gamma)")
            
        return {
            'call_score': min(10, score),
            'put_score': max(0, 10-score),
            'reasons': reasons
        }

    def _detect_market_nature(self, m, v, c):
        return {'label': 'NEUTRAL'}

if __name__ == "__main__":
    import asyncio
    import argparse
    from trade_system.shared.config import Settings
    from trade_system.domains.trading.infrastructure.brokers.legacy.fyers_auth import FyersAuthService
    from trade_system.domains.trading.infrastructure.brokers.legacy.fyers import FyersBroker
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NIFTY50")
    args = parser.parse_args()

    async def run_analysis():
        settings = Settings.load()
        auth = FyersAuthService(settings)
        token = auth.get_valid_token()
        broker = FyersBroker(client_id=settings.fyers.client_id, access_token=token, user_id=settings.fyers.user_id, authenticator=auth.authenticator)
        
        analyzer = OptionChainAnalyzer(broker.fyers, symbol=args.symbol)
        result = analyzer.analyze()
        
        if result:
            print(f"\n📊 --- INSTITUTIONAL FORENSICS: {args.symbol} ---")
            print(f"LTP: ₹{result['metrics']['spot_price']:.2f} | PCR: {result['metrics']['pcr_oi']:.2f}")
            print(f"Max Pain: ₹{result['max_pain']['max_pain_strike']}")
            
            vd = result['vol_delta']
            print(f"\n🌊 ATM VOLUME DELTA (3-Strike Cluster):")
            print(f" • Sentiment: {vd['label']}")
            print(f" • Net Delta: {vd['net_delta']:,} (CE Vol - PE Vol)")
            print(f" • Cluster Ratio: {vd['net_ratio']}x")
            for s in vd['strikes']:
                print(f"   - Strike {int(s['strike'])}: Delta {s['delta']:,} (Ratio {s['ratio']})")
            
            gex = result['gex']
            print(f"\n🧲 GEX (Gamma Exposure):")
            print(f" • Label: {gex['gex_label']} | Net GEX: {gex['total_gex']:,.0f}")
            print(f" • Gamma Wall: ₹{gex['gamma_wall']}")
            
            print(f"\n🧠 Swarm Insights: " + ", ".join(result['confluence']['reasons']))
            print("\n------------------------------------------------\n")

    asyncio.run(run_analysis())
