#!/usr/bin/env python3
"""
Comprehensive test script for live trading bot
Tests all components before allowing live trading
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

def test_alpaca_connection():
    """Test connection to Alpaca"""
    print("\n" + "="*60)
    print("TEST 1: Alpaca API Connection")
    print("="*60)

    try:
        from alpaca.trading.client import TradingClient

        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')

        if not api_key or not api_secret:
            print("❌ FAILED: API credentials not found in .env")
            return False

        # Test paper trading client
        client = TradingClient(api_key, api_secret, paper=True)
        account = client.get_account()

        print(f"✅ PASSED: Connected to Alpaca Paper Trading")
        print(f"   Account ID: {account.id}")
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_market_data():
    """Test market data retrieval"""
    print("\n" + "="*60)
    print("TEST 2: Market Data Retrieval")
    print("="*60)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta

        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')

        client = StockHistoricalDataClient(api_key, api_secret)

        # Request recent data for SPY
        end = datetime.now()
        start = end - timedelta(days=5)

        request = StockBarsRequest(
            symbol_or_symbols='SPY',
            timeframe=TimeFrame.Minute,
            start=start,
            end=end
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        print(f"✅ PASSED: Retrieved {len(df)} bars for SPY")
        print(f"   Latest close: ${df['close'].iloc[-1]:.2f}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_paper_order():
    """Test placing a paper trading order"""
    print("\n" + "="*60)
    print("TEST 3: Paper Trading Order")
    print("="*60)

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')

        client = TradingClient(api_key, api_secret, paper=True)

        # Place a small test order
        order_data = MarketOrderRequest(
            symbol='SPY',
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        print("   Submitting test buy order for 1 share of SPY...")
        order = client.submit_order(order_data)

        print(f"✅ PASSED: Order submitted successfully")
        print(f"   Order ID: {order.id}")
        print(f"   Status: {order.status}")
        print(f"   Symbol: {order.symbol}")
        print(f"   Qty: {order.qty}")

        # Cancel the order
        print("   Cancelling test order...")
        client.cancel_order_by_id(order.id)
        print("   ✅ Order cancelled")

        return True

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_position_check():
    """Test position checking"""
    print("\n" + "="*60)
    print("TEST 4: Position Management")
    print("="*60)

    try:
        from alpaca.trading.client import TradingClient

        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')

        client = TradingClient(api_key, api_secret, paper=True)

        positions = client.get_all_positions()

        print(f"✅ PASSED: Retrieved positions")
        print(f"   Current positions: {len(positions)}")

        for pos in positions:
            print(f"   - {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_clock():
    """Test market clock"""
    print("\n" + "="*60)
    print("TEST 5: Market Clock")
    print("="*60)

    try:
        from alpaca.trading.client import TradingClient

        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')

        client = TradingClient(api_key, api_secret, paper=True)

        clock = client.get_clock()

        print(f"✅ PASSED: Market clock retrieved")
        print(f"   Market is: {'OPEN' if clock.is_open else 'CLOSED'}")
        print(f"   Timestamp: {clock.timestamp}")
        print(f"   Next open: {clock.next_open}")
        print(f"   Next close: {clock.next_close}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_bot_initialization():
    """Test bot initialization"""
    print("\n" + "="*60)
    print("TEST 6: Bot Initialization")
    print("="*60)

    try:
        # Import the bot
        from run_live_bot import LiveTradingBot

        bot = LiveTradingBot(
            ticker='SPY',
            strategy='smc',
            mode='paper',
            config_path='configs/config_spy_optimized.yml'
        )

        print(f"✅ PASSED: Bot initialized successfully")
        print(f"   Ticker: {bot.ticker}")
        print(f"   Strategy: {bot.strategy}")
        print(f"   Mode: {bot.mode}")
        print(f"   Max daily trades: {bot.max_daily_trades}")
        print(f"   Max daily loss: ${bot.max_daily_loss:,.2f}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("LIVE TRADING BOT - COMPREHENSIVE TEST SUITE")
    print("="*80)

    tests = [
        ("Alpaca Connection", test_alpaca_connection),
        ("Market Data", test_market_data),
        ("Paper Order", test_paper_order),
        ("Position Management", test_position_check),
        ("Market Clock", test_clock),
        ("Bot Initialization", test_bot_initialization),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {str(e)}")
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\nResults: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED - Bot is ready for paper trading")
        print("⚠️  To enable live trading:")
        print("   1. Ensure you have sufficient funds in your Alpaca account")
        print("   2. Test thoroughly with paper trading first")
        print("   3. Start with small position sizes")
        print("   4. Monitor closely during the first trading session")
    else:
        print("\n❌ SOME TESTS FAILED - Do not use for live trading yet")
        print("   Fix the failing tests before proceeding")

    return passed_count == total_count


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
