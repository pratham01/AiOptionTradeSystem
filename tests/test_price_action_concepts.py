from pathlib import Path

import pandas as pd

from trade_system.domains.analysis.application.analysis import prepare_price_action_concepts
from trade_system.domains.analysis.application.backtesting.price_action_concepts_research import PriceActionConceptsResearch


def test_prepare_price_action_concepts_builds_expected_columns():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:15:00", periods=25, freq="15min"),
            "open": [100, 101, 102, 101, 100, 99, 101, 103, 104, 105, 104, 106, 108, 107, 109, 111, 112, 111, 113, 114, 113, 115, 116, 117, 118],
            "high": [101, 102, 103, 102, 101, 100, 103, 104, 105, 106, 106, 108, 109, 109, 111, 112, 113, 113, 114, 115, 115, 116, 117, 118, 119],
            "low": [99, 100, 101, 99, 98, 97, 100, 102, 103, 103, 103, 105, 107, 106, 108, 110, 111, 110, 112, 113, 112, 114, 115, 116, 117],
            "close": [100.5, 101.5, 101.2, 100.1, 98.5, 99.8, 102.5, 103.8, 104.6, 105.7, 105.8, 107.2, 108.5, 108.1, 110.6, 111.8, 112.2, 112.6, 113.7, 114.8, 114.2, 115.6, 116.5, 117.4, 118.3],
            "volume": [100] * 25,
        }
    ).set_index("timestamp")

    result = prepare_price_action_concepts(frame)

    for column in [
        "internal_bos",
        "internal_choch",
        "swing_bos",
        "swing_choch",
        "bull_choch_plus",
        "bear_choch_plus",
        "bull_liquidity_sweep",
        "bull_fvg",
        "bull_ob_created",
        "bull_ob_retest",
        "premium_pct",
        "in_discount",
        "in_premium",
    ]:
        assert column in result.columns


def test_price_action_concepts_research_runs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    timestamps = pd.date_range("2026-01-01 09:15:00", periods=240, freq="3min")
    closes = []
    price = 100.0
    for idx in range(len(timestamps)):
        if idx < 80:
            price += 0.25
        elif idx < 120:
            price -= 0.45
        elif idx < 180:
            price += 0.55
        else:
            price -= 0.15
        closes.append(round(price, 2))
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": pd.Series(closes).shift(1).fillna(closes[0]).round(2),
            "high": [value + 0.35 for value in closes],
            "low": [value - 0.35 for value in closes],
            "close": closes,
            "volume": [100 + (i % 10) * 5 for i in range(len(closes))],
        }
    )
    file_path = data_dir / "NSE_NIFTY50-INDEX_3min_2026.csv"
    frame.to_csv(file_path, index=False)

    artifacts = PriceActionConceptsResearch().run(data_dir, tmp_path / "reports")

    assert artifacts.summary_path.exists()
    assert artifacts.report_path.exists()
    assert set(artifacts.summary["strategy"]) == {
        "pac_choch_reversal",
        "pac_choch_plus_ob_reversal",
        "pac_bos_continuation",
        "pac_ob_fvg_continuation",
        "pac_sweep_fvg_reversal",
    }
