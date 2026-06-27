import sys
from pathlib import Path
from datetime import datetime

# Add src to python path
sys.path.append(str(Path("src").resolve()))

from trade_system.application.agent.conviction_fuser_agent import ConvictionFuserAgent
from trade_system.core import SessionPlan, TradeSuggestion, MarketContext, TradeDirection, TradeHorizon
from trade_system.core.models.domain import MarketRegime, OptionParams, SetupFeatures, TradeOutcome

def run_test():
    fuser = ConvictionFuserAgent(max_trades_per_day=3)
    
    plan = SessionPlan(date="2026-05-25")
    plan.market_context = MarketContext(
        timestamp=datetime.now(),
        regime=MarketRegime.TRENDING_BULL,
        bias="BULLISH",
        score=85.0
    )
    
    # Create some dummy suggestions
    opt = OptionParams(direction=TradeDirection.CALL)
    feat = SetupFeatures()
    
    s1 = TradeSuggestion(
        id="1", symbol="NSE:NIFTY50-INDEX", timestamp=datetime.now(), direction=TradeDirection.CALL, 
        horizon=TradeHorizon.INTRADAY, entry_zone_low=0, entry_zone_high=0, target=0, stop_loss=0, 
        option_params=opt, confidence=0.70, narrative="Nifty breakout", setup_features=feat, is_nifty=True
    )
    
    s2 = TradeSuggestion(
        id="2", symbol="NSE:RELIANCE-EQ", timestamp=datetime.now(), direction=TradeDirection.CALL, 
        horizon=TradeHorizon.INTRADAY, entry_zone_low=0, entry_zone_high=0, target=0, stop_loss=0, 
        option_params=opt, confidence=0.60, narrative="Rel breakout", setup_features=feat, is_nifty=False
    )
    
    s3 = TradeSuggestion(
        id="3", symbol="NSE:RELIANCE-EQ", timestamp=datetime.now(), direction=TradeDirection.CALL, 
        horizon=TradeHorizon.INTRADAY, entry_zone_low=0, entry_zone_high=0, target=0, stop_loss=0, 
        option_params=opt, confidence=0.85, narrative="Rel huge breakout duplicate", setup_features=feat, is_nifty=False
    )
    
    opt_bear = OptionParams(direction=TradeDirection.PUT)
    s4 = TradeSuggestion(
        id="4", symbol="NSE:HDFCBANK-EQ", timestamp=datetime.now(), direction=TradeDirection.PUT, 
        horizon=TradeHorizon.INTRADAY, entry_zone_low=0, entry_zone_high=0, target=0, stop_loss=0, 
        option_params=opt_bear, confidence=0.80, narrative="HDFC bear", setup_features=feat, is_nifty=False
    )
    
    s5 = TradeSuggestion(
        id="5", symbol="NSE:TCS-EQ", timestamp=datetime.now(), direction=TradeDirection.CALL, 
        horizon=TradeHorizon.INTRADAY, entry_zone_low=0, entry_zone_high=0, target=0, stop_loss=0, 
        option_params=opt, confidence=0.55, narrative="TCS bull", setup_features=feat, is_nifty=False
    )
    
    plan.nifty_suggestions = [s1]
    plan.fo_suggestions = [s2, s3, s4, s5]
    
    print(f"Before Fusion: {len(plan.all_suggestions())} suggestions")
    
    fuser.fuse_and_filter(plan)
    
    print(f"After Fusion: {len(plan.all_suggestions())} suggestions (Max is 3)")
    
    for s in plan.all_suggestions():
        print(f" - {s.symbol} | Dir: {s.direction.value} | Conf: {s.confidence:.2f} | Tags: {s.tags}")

if __name__ == "__main__":
    run_test()
