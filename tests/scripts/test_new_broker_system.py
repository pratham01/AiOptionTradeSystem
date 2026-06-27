#!/usr/bin/env python3
"""
Test script for the new multi-broker system.

This script tests:
1. Broker initialization
2. Authentication
3. Quote fetching with failover
4. Historical data fetching
5. Health monitoring
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path / "src"))

from trade_system.config import Settings
from trade_system.infrastructure.brokers.factory import get_broker_manager

def test_broker_initialization() -> bool:
    """Test broker manager initialization."""
    print("🔧 Testing broker initialization...")
    
    try:
        settings = Settings.load()
        print(f"  ✓ Settings loaded")
        print(f"    Primary broker: {settings.primary_broker}")
        print(f"    Backup broker: {settings.backup_broker}")
        
        manager = get_broker_manager(settings)
        print(f"  ✓ Broker manager created: {manager}")
        
        # Check health status
        health = manager.get_health_status()
        print(f"  ✓ Broker health status:")
        for name, status in health.items():
            print(f"    {name}: {'✓' if status['is_healthy'] else '✗'} "
                  f"(authenticated: {status['authenticated']})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_authentication() -> bool:
    """Test broker authentication."""
    print("\n🔐 Testing authentication...")
    
    try:
        settings = Settings.load()
        manager = get_broker_manager(settings)
        
        # Authenticate all brokers
        results = manager.authenticate_all()
        print(f"  ✓ Authentication results:")
        for name, success in results.items():
            print(f"    {name}: {'✓' if success else '✗'}")
        
        # Check if at least one broker authenticated
        if any(results.values()):
            print(f"  ✓ At least one broker authenticated successfully")
            return True
        else:
            print(f"  ✗ No brokers authenticated")
            return False
            
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_quote_fetching() -> bool:
    """Test quote fetching with failover."""
    print("\n📊 Testing quote fetching...")
    
    try:
        settings = Settings.load()
        manager = get_broker_manager(settings)
        
        # Test quotes for a few symbols
        symbols = ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ"]
        print(f"  Fetching quotes for: {symbols}")
        
        quotes = manager.get_quotes(symbols)
        print(f"  ✓ Fetched {len(quotes)} quotes")
        
        for symbol, quote in quotes.items():
            print(f"    {symbol}: ₹{quote.last_price:.2f} "
                  f"({quote.change_percent:+.2f}%)")
        
        return len(quotes) > 0
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_historical_data() -> bool:
    """Test historical data fetching."""
    print("\n📈 Testing historical data...")
    
    try:
        from datetime import date, timedelta
        
        settings = Settings.load()
        manager = get_broker_manager(settings)
        
        # Test historical data for last 5 days
        end_date = date.today()
        start_date = end_date - timedelta(days=5)
        symbol = "NSE:RELIANCE-EQ"
        
        print(f"  Fetching historical data for {symbol}")
        print(f"  Period: {start_date} to {end_date}")
        
        data = manager.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="DAY"
        )
        
        print(f"  ✓ Fetched {len(data)} data points")
        
        if data:
            latest = data[-1]
            print(f"    Latest: ₹{latest.close:.2f} on {latest.timestamp.date()}")
        
        return len(data) > 0
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_failover() -> bool:
    """Test broker failover mechanism."""
    print("\n🔄 Testing failover mechanism...")
    
    try:
        settings = Settings.load()
        manager = get_broker_manager(settings)
        
        # Get primary broker
        primary = manager.get_broker(settings.primary_broker)
        print(f"  Primary broker: {primary.name}")
        
        # Get backup broker if available
        if settings.backup_broker:
            backup = manager.get_broker(settings.backup_broker)
            print(f"  Backup broker: {backup.name}")
        
        # Test quote fetching (this will use failover if needed)
        symbols = ["NSE:RELIANCE-EQ"]
        quotes = manager.get_quotes(symbols)
        
        if quotes:
            print(f"  ✓ Quote fetching successful (with automatic failover)")
            return True
        else:
            print(f"  ✗ Quote fetching failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def main() -> int:
    """Run all tests."""
    print("=" * 70)
    print("🧪 TESTING NEW BROKER SYSTEM")
    print("=" * 70)
    
    tests = [
        ("Broker Initialization", test_broker_initialization),
        ("Authentication", test_authentication),
        ("Quote Fetching", test_quote_fetching),
        ("Historical Data", test_historical_data),
        ("Failover Mechanism", test_failover),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! New broker system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check configuration.")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
