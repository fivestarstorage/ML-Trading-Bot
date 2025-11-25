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
        
        while True:
            action_answer = inquirer.prompt([
                inquirer.List(
                    'action',
                    message="Select an action",
                    choices=[
                        ('Backtest (Normal)', 'backtest_normal'),
                        ('Backtest (Prop Firm)', 'backtest_prop'),
                        ('Train / Update ML Model', 'train'),
                        ('Train latest 5y (Alpaca)', 'train_latest'),
                        ('Walk-Forward Analysis', 'wfa'),
                        ('Optimize Variables (grid)', 'optimize'),
                        ('Start Alpaca Live Bot', 'alpaca_live'),
                        ('Reset ML Model', 'reset_model'),
                        ('Exit', 'exit')
                    ]
                )
            ])
            
            if not action_answer or action_answer['action'] == 'exit':
                print("Good luck and manage risk! 👋")
                break
            
            action = action_answer['action']
            
            if action == 'reset_model':
                self._reset_model()
                continue
            if action == 'train_latest':
                self._train_latest_window()
                continue
            if action == 'alpaca_live':
                self._start_alpaca_live()
                continue
            
            symbol = self._ask_symbol()
            start_date, end_date = self._ask_dates()
            
            args = ['--symbol', symbol, '--config', self.config_path]
            if start_date:
                args.extend(['--from', start_date])
            if end_date:
                args.extend(['--to', end_date])
            
            if action == 'backtest_normal':
                args.extend(['--mode', 'normal', '--action', 'backtest'])
            elif action == 'backtest_prop':
                args.extend(['--mode', 'propfirm', '--action', 'backtest'])
            elif action == 'train':
                args.extend(['--action', 'train_model'])
            elif action == 'wfa':
                if not start_date or not end_date:
                    print("Walk-forward requires both start and end dates.")
                    continue
                args.append('--wfa')
            elif action == 'optimize':
                if not start_date or not end_date:
                    print("Optimization requires both start and end dates. Please try again.")
                    continue
                args.append('--find-optimised-variables')
                args.extend(['--mode', self.config['backtest'].get('mode', 'normal')])
            
            print(f"\n>>> Running: {' '.join(args)}\n")
            self._invoke_cli(args)
    
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


def main():
    menu = TradingBotMenu()
    menu.run()


