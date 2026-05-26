import sys

filepath = 'src/trade_system/application/agent/fo_option_buyer_agent.py'
with open(filepath, 'r') as f:
    content = f.read()

old_reason = """    rationale = (
        f"momo={momentum:.2f}%, brk={intraday_breakout:.2f}%, "
        f"rng={range_expansion:.2f}%, vol={quote.volume}, vix={vix}, pcr={pcr}, sec={sector_perf:.2f}%"
        if sector_perf is not None else
        f"momo={momentum:.2f}%, brk={intraday_breakout:.2f}%, "
        f"rng={range_expansion:.2f}%, vol={quote.volume}, vix={vix}, pcr={pcr}"
    )"""

new_reason = """    rationale = (
        f"Strong momentum ({momentum:.2f}%) with intraday breakout ({intraday_breakout:.2f}%). "
        f"Range expanded by {range_expansion:.2f}% with volume of {quote.volume}. "
        f"Sector Performance: {sector_perf:.2f}%."
        if sector_perf is not None else
        f"Strong momentum ({momentum:.2f}%) with intraday breakout ({intraday_breakout:.2f}%). "
        f"Range expanded by {range_expansion:.2f}% with volume of {quote.volume}."
    )"""

if old_reason in content:
    content = content.replace(old_reason, new_reason)
    with open(filepath, 'w') as f:
        f.write(content)
    print("Successfully patched reason!")
else:
    print("Old reason not found!")
