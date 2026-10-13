import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
from trade_system.interfaces.dashboard.sector_scope_dashboard import IntradayFlowReversalEngine

# Simulate merged_closes rows
data = [
    {'symbol': 'NSE:RELIANCE-EQ', 'sector': 'ENERGY', 'close_last': 2500, 'pChange': 0.0, 'low_today': 2450, 'high_today': 2550}, # 2.04% from low, 1.96% from high
    {'symbol': 'NSE:TCS-EQ', 'sector': 'IT', 'close_last': 4000, 'pChange': 0.5, 'low_today': 3995, 'high_today': 4010}, # 0.12% from low, 0.24% from high
]

df = pd.DataFrame(data)
for idx, row in df.iterrows():
    move_from_low = ((row['close_last'] - row['low_today']) / row['low_today'] * 100) if row.get('low_today', 0) > 0 else 0.0
    drop_from_high = ((row['high_today'] - row['close_last']) / row['high_today'] * 100) if row.get('high_today', 0) > 0 else 0.0
    
    flow_quad, _ = IntradayFlowReversalEngine.classify_derivative_quadrant(
        price_change=row.get('pChange', 0.0),
        move_from_low=move_from_low,
        drop_from_high=drop_from_high
    )
    print(f"{row['symbol']}: MoveFromLow={move_from_low:.2f}%, DropFromHigh={drop_from_high:.2f}% -> Flow={flow_quad}")
