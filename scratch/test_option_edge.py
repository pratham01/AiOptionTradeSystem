"""Full end-to-end pipeline test with adjusted thresholds."""
from trade_system.domains.analysis.application.analysis.intraday_option_edge import IntradayOptionEdgePipeline

pipeline = IntradayOptionEdgePipeline(max_alerts_per_cycle=5, min_risk_reward=1.0)

print("=" * 80)
print("🎯 INTRADAY OPTION EDGE PIPELINE — FULL TEST (adjusted thresholds)")
print("=" * 80)

alerts = pipeline.scan()

if not alerts:
    print("\n⚠️ Still no alerts. Checking with R:R = 0 to see if entry triggers are the issue...")
    pipeline2 = IntradayOptionEdgePipeline(max_alerts_per_cycle=10, min_risk_reward=0.0)
    alerts2 = pipeline2.scan()
    if alerts2:
        print(f"\n✅ {len(alerts2)} alerts found with R:R=0 (entry triggers are working, R:R filter was too strict)")
        for a in alerts2[:3]:
            clean = a.symbol.replace("NSE:", "").replace("-EQ", "")
            print(f"  {clean}: Score={a.edge_score:.0f} Dir={a.direction} Entry={a.entry_type} R:R={a.risk_reward_stock:.1f}")
    else:
        print("  No alerts even with R:R=0 — SmartEntryTrigger is filtering everything out")
else:
    print(f"\n✅ Generated {len(alerts)} alerts!\n")
    for i, alert in enumerate(alerts, 1):
        print(f"{'─' * 70}")
        print(f"Alert #{i}")
        print(alert.format_telegram().replace("<b>", "**").replace("</b>", "**"))
        print()
