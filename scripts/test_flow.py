import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
from trade_system.interfaces.dashboard.sector_scope_dashboard import IntradayFlowReversalEngine

print("Testing classify_derivative_quadrant...")
print(IntradayFlowReversalEngine.classify_derivative_quadrant(price_change=1.5, move_from_low=2.0, drop_from_high=0.1))
print(IntradayFlowReversalEngine.classify_derivative_quadrant(price_change=1.5, move_from_low=0.02, drop_from_high=0.001))
