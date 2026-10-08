import pandas as pd
from typing import List, Dict

def identify_order_blocks(df: pd.DataFrame, atr_multiplier: float = 1.5) -> Dict[str, List[Dict]]:
    """
    Identifies unmitigated Bullish and Bearish Order Blocks in a given OHLCV DataFrame.
    
    A Bullish OB is defined as the last bearish candle before a strong bullish impulse.
    A Bearish OB is defined as the last bullish candle before a strong bearish impulse.
    
    Returns a dictionary:
    {
        'bullish': [{'start_date': timestamp, 'top': float, 'bottom': float}],
        'bearish': [{'start_date': timestamp, 'top': float, 'bottom': float}]
    }
    """
    if len(df) < 15:
        return {'bullish': [], 'bearish': []}
        
    df = df.copy()
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(14).mean()
    
    bullish_obs = []
    bearish_obs = []
    
    for i in range(14, len(df)-1):
        c0 = df.iloc[i]     # Potential OB candle
        c1 = df.iloc[i+1]   # Impulse candle
        avg_body = df['avg_body'].iloc[i]
        
        # Check for Bullish OB: c0 is bearish, c1 is bullish and massive
        if c0['close'] < c0['open'] and c1['close'] > c1['open'] and c1['body'] > atr_multiplier * avg_body:
            ob = {
                'start_date': c0['timestamp'],
                'start_idx': i,
                'top': max(c0['open'], c0['high']), # Some prefer High, some Open. High is safer.
                'bottom': c0['low'],
                'mitigated': False,
                'broken': False
            }
            ob['top'] = c0['high'] # standard ICT uses High to Low
            bullish_obs.append(ob)
            
        # Check for Bearish OB: c0 is bullish, c1 is bearish and massive
        elif c0['close'] > c0['open'] and c1['close'] < c1['open'] and c1['body'] > atr_multiplier * avg_body:
            ob = {
                'start_date': c0['timestamp'],
                'start_idx': i,
                'top': c0['high'],
                'bottom': c0['low'],
                'mitigated': False,
                'broken': False
            }
            bearish_obs.append(ob)
            
    # Check Mitigation
    for ob in bullish_obs:
        # Check all candles AFTER the impulse candle (i+2 onwards)
        for j in range(ob['start_idx'] + 2, len(df)):
            curr_low = df.iloc[j]['low']
            curr_close = df.iloc[j]['close']
            if curr_close < ob['bottom']:
                ob['broken'] = True
                break
            if curr_low <= ob['top']:
                ob['mitigated'] = True
                
    for ob in bearish_obs:
        for j in range(ob['start_idx'] + 2, len(df)):
            curr_high = df.iloc[j]['high']
            curr_close = df.iloc[j]['close']
            if curr_close > ob['top']:
                ob['broken'] = True
                break
            if curr_high >= ob['bottom']:
                ob['mitigated'] = True
                
    # Filter to only Unmitigated and Unbroken OBs
    valid_bullish = [ob for ob in bullish_obs if not ob['broken'] and not ob['mitigated']]
    valid_bearish = [ob for ob in bearish_obs if not ob['broken'] and not ob['mitigated']]
    
    return {
        'bullish': valid_bullish,
        'bearish': valid_bearish
    }
