import os
import sys
from datetime import datetime

try:
    import inquirer
except ImportError:  # pragma: no cover
    inquirer = None

from .utils import load_config
from .cli import main as cli_main


class TradingBotMenu:
    def __init__(self, config_path="config.yml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.default_symbol = self.config['data'].get('symbol', 'XAUUSD')
    
    def run(self):
        if inquirer is None:
            print("Interactive menu requires 'inquirer'. Install with: pip install inquirer")
            return

        print("\n" + "="*60)
        print("  Trading Bot - Strategy Validator & Live Trading")
        print("="*60 + "\n")

        while True:
            # Ask for mode: WFA or Live Trading
            mode_answer = inquirer.prompt([
                inquirer.List(
                    'mode',
                    message="Select mode",
                    choices=[
                        ('Walk-Forward Analysis (WFA)', 'wfa'),
                        ('Live Trading Bot', 'live'),
                        ('Exit', 'EXIT')
                    ]
                )
            ])

            if not mode_answer or mode_answer['mode'] == 'EXIT':
                print("\nGood luck and manage risk! 👋")
                break

            if mode_answer['mode'] == 'wfa':
                self._run_wfa_mode()
            elif mode_answer['mode'] == 'live':
                self._run_live_mode()

    def _run_wfa_mode(self):
        """Run Walk-Forward Analysis mode"""
        while True:
            # Ask for ticker
            ticker_answer = inquirer.prompt([
                inquirer.List(
                    'ticker',
                    message="Select ticker symbol",
                    choices=[
                        ('SPY - S&P 500 ETF', 'SPY'),
                        ('QQQ - Nasdaq 100 ETF', 'QQQ'),
                        ('IWM - Russell 2000 ETF', 'IWM'),
                        ('DIA - Dow Jones ETF', 'DIA'),
                        ('Custom ticker...', 'CUSTOM'),
                        ('Back to main menu', 'BACK')
                    ]
                )
            ])

            if not ticker_answer or ticker_answer['ticker'] == 'BACK':
                return

            ticker = ticker_answer['ticker']
            if ticker == 'CUSTOM':
                custom_answer = inquirer.prompt([
                    inquirer.Text('custom_ticker', message="Enter ticker symbol (e.g., AAPL, TSLA)")
                ])
                if not custom_answer or not custom_answer.get('custom_ticker'):
                    continue
                ticker = custom_answer['custom_ticker'].upper()

            # Ask for strategy
            strategy_answer = inquirer.prompt([
                inquirer.List(
                    'strategy',
                    message=f"Select strategy for {ticker}",
                    choices=[
                        ('Smart Money Concepts (SMC)', 'smc'),
                        ('Back to ticker selection', 'BACK')
                    ]
                )
            ])

            if not strategy_answer or strategy_answer['strategy'] == 'BACK':
                continue

            strategy = strategy_answer['strategy']

            # Ask for years to test
            years_answer = inquirer.prompt([
                inquirer.List(
                    'years_option',
                    message=f"Select time period for {ticker} WFA",
                    choices=[
                        ('Single Year', 'single'),
                        ('Multiple Years', 'multiple'),
                        ('All Years (2020-2025)', 'all'),
                        ('Back to strategy selection', 'BACK')
                    ]
                )
            ])

            if not years_answer or years_answer['years_option'] == 'BACK':
                continue

            option = years_answer['years_option']
            years_str = None

            if option == 'single':
                year_answer = inquirer.prompt([
                    inquirer.Text('year', message="Enter year (e.g., 2024)", default=str(datetime.now().year))
                ])
                if year_answer and year_answer.get('year'):
                    years_str = year_answer['year']
            elif option == 'multiple':
                years_input = inquirer.prompt([
                    inquirer.Text('years', message="Enter years comma-separated (e.g., 2023,2024,2025)",
                                 default="2023,2024,2025")
                ])
                if years_input and years_input.get('years'):
                    years_str = years_input['years']
            else:  # all
                years_str = '2020,2021,2022,2023,2024,2025'

            if not years_str:
                continue

            # Run WFA
            self._run_wfa(ticker, strategy, years_str)
    
    def _ask_symbol(self):
        answer = inquirer.prompt([
            inquirer.Text('symbol', message="Symbol", default=self.default_symbol)
        ])
        return answer['symbol'] if answer and answer.get('symbol') else self.default_symbol
    
    def _ask_dates(self):
        answer = inquirer.prompt([
            inquirer.Text('start', message="From date (YYYY-MM-DD, blank=all)", default=""),
            inquirer.Text('end', message="To date (YYYY-MM-DD, blank=all)", default="")
        ])
        if not answer:
            return None, None
        start = answer.get('start') or None
        end = answer.get('end') or None
        for label, value in [('start', start), ('end', end)]:
            if value:
                try:
                    datetime.strptime(value, "%Y-%m-%d")
                except ValueError:
                    print(f"Invalid {label} date '{value}'. Ignoring.")
                    if label == 'start':
                        start = None
                    else:
                        end = None
        return start, end
    
    def _invoke_cli(self, arg_list):
        sys.argv = [sys.argv[0]] + arg_list
        cli_main()
    
    def _reset_model(self):
        model_path = self.config['ml']['model_path']
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(project_root, model_path)
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"✅ Removed ML model at {model_path}")
        else:
            print(f"ℹ️ No model found at {model_path}")

    def _train_latest_window(self):
        symbol = self._ask_symbol()
        args = [
            '--symbol', symbol,
            '--config', self.config_path,
            '--action', 'train_latest',
            '--data-source', 'alpaca',
        ]
        print(f"\n>>> Running: {' '.join(args)}\n")
        self._invoke_cli(args)

    def _start_alpaca_live(self):
        symbol = self._ask_symbol()
        dry_run_answer = inquirer.prompt([
            inquirer.Confirm('dry', message="Run in dry-run mode?", default=True)
        ])
        dry_flag = dry_run_answer and dry_run_answer.get('dry', True)
        args = [
            '--symbol', symbol,
            '--config', self.config_path,
            '--action', 'alpaca_live',
        ]
        if dry_flag:
            args.append('--dry-run')
        print(f"\n>>> Running: {' '.join(args)}\n")
        self._invoke_cli(args)

    def _run_wfa(self, ticker, strategy, years_str):
        """Run Walk-Forward Analysis with comprehensive reporting"""
        import subprocess

        # Determine config file based on ticker and strategy
        config_map = {
            ('SPY', 'smc'): 'configs/config_spy_optimized.yml',
            ('QQQ', 'smc'): 'configs/config_spy_optimized.yml',  # Can use same SMC strategy
            ('IWM', 'smc'): 'configs/config_spy_optimized.yml',
            ('DIA', 'smc'): 'configs/config_spy_optimized.yml',
        }

        config_file = config_map.get((ticker, strategy), 'configs/config_spy_optimized.yml')

        cmd = [
            'python3', 'scripts/run_wfa_with_reports.py',
            '--ticker', ticker,
            '--strategy', strategy,
            '--years', years_str,
            '--config', config_file
        ]

        print(f"\n{'='*60}")
        print(f"  Running WFA: {ticker} | Strategy: {strategy.upper()}")
        print(f"  Years: {years_str}")
        print(f"{'='*60}\n")

        subprocess.run(cmd)

    def _run_live_mode(self):
        """Run Live Trading mode"""
        import subprocess

        while True:
            # Ask paper or live
            mode_answer = inquirer.prompt([
                inquirer.List(
                    'trading_mode',
                    message="Select trading mode",
                    choices=[
                        ('Paper Trading (Simulated)', 'paper'),
                        ('Live Trading (Real Money)', 'live'),
                        ('Back to main menu', 'BACK')
                    ]
                )
            ])

            if not mode_answer or mode_answer['trading_mode'] == 'BACK':
                return

            trading_mode = mode_answer['trading_mode']

            # Ask for ticker
            ticker_answer = inquirer.prompt([
                inquirer.List(
                    'ticker',
                    message="Select ticker symbol",
                    choices=[
                        ('SPY - S&P 500 ETF', 'SPY'),
                        ('QQQ - Nasdaq 100 ETF', 'QQQ'),
                        ('IWM - Russell 2000 ETF', 'IWM'),
                        ('DIA - Dow Jones ETF', 'DIA'),
                        ('Custom ticker...', 'CUSTOM'),
                        ('Back', 'BACK')
                    ]
                )
            ])

            if not ticker_answer or ticker_answer['ticker'] == 'BACK':
                continue

            ticker = ticker_answer['ticker']
            if ticker == 'CUSTOM':
                custom_answer = inquirer.prompt([
                    inquirer.Text('custom_ticker', message="Enter ticker symbol (e.g., AAPL, TSLA)")
                ])
                if not custom_answer or not custom_answer.get('custom_ticker'):
                    continue
                ticker = custom_answer['custom_ticker'].upper()

            # Ask for strategy
            strategy_answer = inquirer.prompt([
                inquirer.List(
                    'strategy',
                    message=f"Select strategy for {ticker}",
                    choices=[
                        ('Smart Money Concepts (SMC)', 'smc'),
                        ('Back', 'BACK')
                    ]
                )
            ])

            if not strategy_answer or strategy_answer['strategy'] == 'BACK':
                continue

            strategy = strategy_answer['strategy']

            # Confirm live trading
            if trading_mode == 'live':
                confirm = inquirer.prompt([
                    inquirer.Confirm(
                        'confirm_live',
                        message="⚠️  WARNING: This will trade with REAL MONEY. Are you sure?",
                        default=False
                    )
                ])
                if not confirm or not confirm.get('confirm_live'):
                    print("\n❌ Live trading cancelled. Use Paper Trading to test first.\n")
                    continue

            # Determine config file
            config_map = {
                ('SPY', 'smc'): 'configs/config_spy_optimized.yml',
                ('QQQ', 'smc'): 'configs/config_spy_optimized.yml',
                ('IWM', 'smc'): 'configs/config_spy_optimized.yml',
                ('DIA', 'smc'): 'configs/config_spy_optimized.yml',
            }
            config_file = config_map.get((ticker, strategy), 'configs/config_spy_optimized.yml')

            # Run live trading bot
            cmd = [
                'python3', 'scripts/run_live_bot.py',
                '--ticker', ticker,
                '--strategy', strategy,
                '--mode', trading_mode,
                '--config', config_file
            ]

            print(f"\n{'='*60}")
            print(f"  Starting {'PAPER' if trading_mode == 'paper' else 'LIVE'} Trading Bot")
            print(f"  Ticker: {ticker} | Strategy: {strategy.upper()}")
            print(f"{'='*60}\n")

            subprocess.run(cmd)
            return  # Exit after bot stops


def main():
    menu = TradingBotMenu()
    menu.run()


