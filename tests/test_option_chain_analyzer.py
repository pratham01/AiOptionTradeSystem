import pandas as pd
import logging

from trade_system.application.analysis.option_chain_analyzer import OptionChainAnalyzer


class _DummyFyersClient:
    def __init__(self):
        self.last_optionchain_data = None

    def optionchain(self, data):
        self.last_optionchain_data = data
        return {"s": "ok", "data": {"optionsChain": []}}


def test_parse_option_chain_keeps_greeks_from_top_level_and_nested_payload(monkeypatch):
    monkeypatch.setattr(OptionChainAnalyzer, "_fetch_and_set_expiries", lambda self: None)
    analyzer = OptionChainAnalyzer(fyers_client=_DummyFyersClient(), symbol="NIFTY50", strike_count=10)

    response = {
        "s": "ok",
        "data": {
            "ltp": 22950.0,
            "optionsChain": [
                {
                    "symbol": "NSE:NIFTY50-INDEX",
                    "option_type": "",
                    "strike_price": -1,
                },
                {
                    "symbol": "NSE:NIFTY26APR23000CE",
                    "option_type": "CE",
                    "strike_price": 23000,
                    "ltp": 120.5,
                    "bid": 120.4,
                    "ask": 120.6,
                    "volume": 100000,
                    "oi": 150000,
                    "ltpch": 10.0,
                    "ltpchp": 9.0,
                    "iv": 12.5,
                    "delta": 0.52,
                    "gamma": 0.0012,
                    "theta": -8.4,
                    "vega": 11.3,
                    "rho": 0.7,
                },
                {
                    "symbol": "NSE:NIFTY26APR23000PE",
                    "option_type": "PE",
                    "strike": 23000,
                    "ltp": 98.5,
                    "bid": 98.4,
                    "ask": 98.6,
                    "volume": 90000,
                    "open_interest": 175000,
                    "change": 7.0,
                    "change_percent": 8.0,
                    "greeks": {
                        "iv": 13.1,
                        "delta": -0.48,
                        "gamma": 0.0011,
                        "theta": -7.9,
                        "vega": 10.9,
                        "rho": -0.6,
                    },
                },
            ],
        },
    }

    df, spot_price = analyzer._parse_option_chain(response)

    assert spot_price == 22950.0
    assert list(df.columns) == [
        "strike",
        "expiry",
        "option_type",
        "symbol",
        "ltp",
        "bid",
        "ask",
        "volume",
        "oi",
        "change",
        "change_percent",
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
    ]
    assert len(df) == 2

    ce_row = df[df["option_type"] == "CE"].iloc[0]
    pe_row = df[df["option_type"] == "PE"].iloc[0]

    assert ce_row["change"] == 10.0
    assert ce_row["change_percent"] == 9.0
    assert ce_row["delta"] == 0.52
    assert ce_row["gamma"] == 0.0012
    assert ce_row["theta"] == -8.4
    assert ce_row["vega"] == 11.3
    assert ce_row["rho"] == 0.7

    assert pe_row["oi"] == 175000
    assert pe_row["iv"] == 13.1
    assert pe_row["delta"] == -0.48
    assert pe_row["gamma"] == 0.0011
    assert pe_row["theta"] == -7.9
    assert pe_row["vega"] == 10.9
    assert pe_row["rho"] == -0.6

    assert isinstance(df, pd.DataFrame)


def test_fetch_option_chain_requests_greeks(monkeypatch):
    monkeypatch.setattr(OptionChainAnalyzer, "_fetch_and_set_expiries", lambda self: None)
    client = _DummyFyersClient()
    analyzer = OptionChainAnalyzer(fyers_client=client, symbol="NIFTY50", strike_count=12)

    analyzer._fetch_option_chain()

    assert client.last_optionchain_data == {
        "symbol": "NSE:NIFTY50-INDEX",
        "strikecount": 12,
        "greeks": "1",
    }





def test_analyze_builds_buyer_signal_with_previous_day_reference(monkeypatch):
    monkeypatch.setattr(OptionChainAnalyzer, "_fetch_and_set_expiries", lambda self: None)
    analyzer = OptionChainAnalyzer(fyers_client=_DummyFyersClient(), symbol="NIFTY50", strike_count=10)

    current_df = pd.DataFrame(
        [
            {"strike": 22950, "expiry": "", "option_type": "CE", "symbol": "CE1", "ltp": 140, "bid": 139, "ask": 141, "volume": 120000, "oi": 100000, "change": 10, "change_percent": 8, "iv": 12, "delta": 0.6, "gamma": 0.001, "theta": -7, "vega": 10, "rho": 0.5},
            {"strike": 23000, "expiry": "", "option_type": "CE", "symbol": "CE2", "ltp": 110, "bid": 109, "ask": 111, "volume": 90000, "oi": 180000, "change": 7, "change_percent": 5, "iv": 12, "delta": 0.5, "gamma": 0.001, "theta": -7, "vega": 10, "rho": 0.5},
            {"strike": 23150, "expiry": "", "option_type": "CE", "symbol": "CE3", "ltp": 80, "bid": 79, "ask": 81, "volume": 75000, "oi": 300000, "change": 5, "change_percent": 4, "iv": 11, "delta": 0.4, "gamma": 0.001, "theta": -6, "vega": 9, "rho": 0.4},
            {"strike": 22850, "expiry": "", "option_type": "PE", "symbol": "PE1", "ltp": 95, "bid": 94, "ask": 96, "volume": 110000, "oi": 150000, "change": 8, "change_percent": 6, "iv": 13, "delta": -0.45, "gamma": 0.001, "theta": -8, "vega": 11, "rho": -0.5},
            {"strike": 22900, "expiry": "", "option_type": "PE", "symbol": "PE2", "ltp": 120, "bid": 119, "ask": 121, "volume": 140000, "oi": 220000, "change": 10, "change_percent": 8, "iv": 13, "delta": -0.5, "gamma": 0.001, "theta": -8, "vega": 11, "rho": -0.5},
            {"strike": 22950, "expiry": "", "option_type": "PE", "symbol": "PE3", "ltp": 145, "bid": 144, "ask": 146, "volume": 130000, "oi": 320000, "change": 11, "change_percent": 9, "iv": 14, "delta": -0.6, "gamma": 0.001, "theta": -9, "vega": 12, "rho": -0.6},
        ]
    )
    prev_df = current_df.copy()
    prev_df.loc[(prev_df["strike"] == 22950) & (prev_df["option_type"] == "PE"), "oi"] = 250000
    prev_day_df = current_df.copy()
    prev_day_df.loc[(prev_day_df["strike"] == 22900) & (prev_day_df["option_type"] == "PE"), "oi"] = 340000
    prev_day_df.loc[(prev_day_df["strike"] == 22950) & (prev_day_df["option_type"] == "PE"), "oi"] = 180000
    prev_day_df.loc[(prev_day_df["strike"] == 23000) & (prev_day_df["option_type"] == "CE"), "oi"] = 330000
    prev_day_df.loc[(prev_day_df["strike"] == 23150) & (prev_day_df["option_type"] == "CE"), "oi"] = 200000

    analysis = analyzer.analyze(
        df=current_df,
        spot_price=23040.0,
        vix=14.0,
        prev_df=prev_df,
        prev_day_df=prev_day_df,
    )

    assert analysis is not None
    assert "confluence" in analysis
    assert "vol_delta" in analysis
    assert analysis["metrics"]["spot_price"] == 23040.0
    assert analysis["gex"]["total_gex"] is not None
    assert analysis["vol_delta"]["label"] in ["AGGRESSIVE_BULLISH_VOLUME", "AGGRESSIVE_BEARISH_VOLUME", "NEUTRAL_INTENSITY", "UNKNOWN"]
