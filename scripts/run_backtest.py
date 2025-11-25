#!/usr/bin/env python
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.cli import main as run_cli
from src.menu import TradingBotMenu


def main():
    if len(sys.argv) > 1:
        run_cli()
    else:
        menu = TradingBotMenu()
        menu.run()


if __name__ == "__main__":
    main()


