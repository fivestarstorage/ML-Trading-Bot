import argparse
import pandas as pd
import os
from .utils import load_config, setup_logging
from .data_adapter import DataAdapter
from .structure import StructureAnalyzer
from .features import FeatureEngineer
from .entries import EntryGenerator
from .backtester import Backtester
from .ml_model import MLModel
from .wfa import WalkForwardAnalyzer
from .orb_strategy import ORBEntryGenerator
from .orb_backtester import ORBBacktester
from .optimizer import VariableOptimizer
from .adaptive import AdaptiveManager
from .adaptive_filters import FilterProfileManager
from .ig_client import IGClient
from .ig_streaming import IGStreamingClient
from .live_trader import IGLiveTrader
from .live_runner import LiveStreamingRunner
from .alpaca_client import AlpacaClient
from .alpaca_live_trader import AlpacaLiveTrader
from .alpaca_live_runner import AlpacaPollingRunner
from .model_metadata import (
    save_training_metadata,
    load_metadata,
    training_covers_recent,
)

def main():
    parser = argparse.ArgumentParser(description="ML Trading Bot CLI")
    parser.add_argument("--symbol", help="Trading symbol", default=None)
    parser.add_argument("--from", dest="start_date", help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="end_date", help="End date YYYY-MM-DD")
    parser.add_argument("--mode", choices=['normal', 'propfirm'], help="Backtest mode")
    parser.add_argument("--wfa", action="store_true", help="Run Walk-Forward Analysis")
    parser.add_argument(
        "--action",
        choices=['backtest', 'train_model', 'train_latest', 'reset_model', 'live_demo', 'stream_live', 'alpaca_live'],
        default='backtest',
        help="Action to perform"
    )
    parser.add_argument("--find-optimised-variables", action="store_true", help="Find optimal variable combinations")
    parser.add_argument("--max-drawdown", type=float, default=None, help="Max drawdown percentage (e.g., 10.0 for 10%%)")
    parser.add_argument("--initial-capital", type=float, default=None, help="Initial trading capital (e.g., 1000)")
    parser.add_argument("--config", default="config.yml", help="Path to config file")
    parser.add_argument("--data-source", choices=['csv', 'ig', 'alpaca'], help="Override data source")
    parser.add_argument("--dry-run", action="store_true", help="Skip sending live orders (log only)")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    adaptive_manager = AdaptiveManager(config)
    adaptive_manager.apply_to_config(config)
    logger = setup_logging()
    
    # Overrides
    if args.symbol: config['data']['symbol'] = args.symbol
    if args.mode: config['backtest']['mode'] = args.mode
    if args.max_drawdown is not None: 
        config['backtest']['propfirm']['max_drawdown_pct'] = args.max_drawdown / 100.0
    if args.initial_capital is not None:
        config['backtest']['propfirm']['initial_capital'] = args.initial_capital
    if args.data_source:
        config['data']['source'] = args.data_source.lower()

    if args.action == 'train_latest':
        today = pd.Timestamp.utcnow().normalize()
        args.end_date = today.strftime('%Y-%m-%d')
        args.start_date = (today - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
        logger.info("Auto training window selected: %s -> %s", args.start_date, args.end_date)

    if args.action in ('train_model', 'train_latest'):
        if not args.start_date or not args.end_date:
            logger.error("Training requires both --from and --to dates.")
            return
        run_training_job(config, logger, args.start_date, args.end_date)
        return

    filter_manager = FilterProfileManager(config)
    
    # 0. Live streaming shortcut (needs IG data, no heavy CSV preprocessing)
    if args.action == 'stream_live':
        run_streaming_mode(config, logger, args.start_date, args.end_date)
        return
    if args.action == 'alpaca_live':
        run_alpaca_live(config, logger, args.start_date, args.end_date, args.dry_run)
        return

    # 1. Data Loading
    adapter = DataAdapter(config, start_date=args.start_date, end_date=args.end_date)
    # Load the base timeframe data as specified in config (now 5m)
    # Pass the timeframe suffix so it looks for the correct file
    base_tf_str = config['data']['timeframe_base'] # "5m"
    df_base = adapter.load_data(timeframe_suffix=base_tf_str)
    
    # Filter by date if provided
    def _normalize_ts(value):
        if not value:
            return None
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    start_ts = _normalize_ts(args.start_date)
    end_ts = _normalize_ts(args.end_date)

    if start_ts is not None:
        df_base = df_base[df_base.index >= start_ts]
    if end_ts is not None:
        df_base = df_base[df_base.index <= end_ts]
    
    if df_base.empty:
        logger.warning("No data found for the specified date range.")
        return
    
    logger.info(f"Data loaded: {len(df_base)} rows (Base Timeframe {base_tf_str})")
    
    # 2. Load H4 and Daily data directly (instead of resampling)
    df_h4 = adapter.load_h4_data(
        start_date=args.start_date if args.start_date else None,
        end_date=args.end_date if args.end_date else None
    )
    df_d1 = adapter.load_daily_data(
        start_date=args.start_date if args.start_date else None,
        end_date=args.end_date if args.end_date else None
    )
    
    # Resample H1 if needed (for features)
    h1_min = config['timeframes']['h1']
    df_h1 = adapter.resample_data(df_base, h1_min)
    
    logger.info(f"H4={len(df_h4)}, H1={len(df_h1)}")
    
    # 3. Feature Engineering & Candidate Generation
    strategy_type = config['strategy'].get('strategy_type', 'smc')
    
    features = FeatureEngineer(config)
    df_base = features.calculate_technical_features(df_base)
    
    if strategy_type == 'orb':
        # ORB Strategy (similar to New_XAU_Bot.mq5)
        logger.info("Using ORB (Opening Range Breakout) strategy")
        entry_gen = ORBEntryGenerator(config)
        candidates = entry_gen.generate_candidates(df_base)
    else:
        # SMC Strategy (original)
        logger.info("Using SMC (Smart Money Concepts) strategy")
        structure = StructureAnalyzer(df_h4, config, daily_df=df_d1)
        fvgs = features.detect_fvgs(df_base)
        obs = features.detect_obs(df_base)
        logger.info(f"Detected {len(fvgs)} FVGs and {len(obs)} OBs on Base Timeframe")
        entry_gen = EntryGenerator(config, structure, features)
        candidates = entry_gen.generate_candidates(df_base, df_h4, fvgs, obs)
    
    logger.info(f"Generated {len(candidates)} candidates")
    if strategy_type != 'orb':
        active_profile = filter_manager.active_profile_from_config(config)
        if active_profile:
            candidates, meta = filter_manager.apply(active_profile, candidates)
            logger.info(f"Applied adaptive filter profile '{active_profile}': {meta['filtered']} / {meta['initial']} candidates retained.")
            if candidates.empty:
                logger.warning("No candidates remain after applying the active filter profile. Exiting.")
                return
    
    if candidates.empty:
        logger.warning("No candidates generated. Exiting.")
        return

    # 5. Action
    if args.find_optimised_variables:
        # Variable optimization
        if not args.start_date or not args.end_date:
            logger.error("--find-optimised-variables requires --from and --to dates")
            return
        
        max_dd_limit = args.max_drawdown / 100.0 if args.max_drawdown is not None else config['backtest']['propfirm']['max_drawdown_pct']
        optimizer = VariableOptimizer(config, max_drawdown_limit=max_dd_limit)
        top_5 = optimizer.optimize(args.start_date, args.end_date)
        
        # Results are already saved by the optimizer (including equity graphs)
        if top_5 is not None:
            logger.info(f"\nOptimization complete. Check reports/optimization_* for results.")
        
    elif args.action == 'reset_model':
        model_path = config['ml']['model_path']
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(project_root, model_path)
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info("ML model '%s' deleted.", model_path)
        else:
            logger.info("No ML model found at '%s'.", model_path)
        return

    elif args.action == 'live_demo':
        live_cfg = config.get('live_trading', {})
        if not live_cfg.get('enabled', False):
            logger.error("Enable live_trading.enabled in config.yml before running live trades.")
            return

        model = MLModel(config)
        try:
            model.load(config['ml']['model_path'])
        except FileNotFoundError:
            logger.error("No trained model found at %s. Train first before live trading.", config['ml']['model_path'])
            return

        ig_client = IGClient(
            api_key=live_cfg.get('api_key'),
            username=live_cfg.get('username'),
            password=live_cfg.get('password'),
            account_id=live_cfg.get('account_id'),
            use_demo=live_cfg.get('use_demo', True),
        )

        probs = model.predict_proba(candidates)
        live_trader = IGLiveTrader(config, live_cfg, ig_client=ig_client)
        payload = live_trader.run(candidates, probs)
        if payload is None:
            logger.info("Live demo check complete. No trade sent.")
        else:
            logger.info("IG response: %s", payload)
        return

    elif args.action == 'stream_live':
        stream_cfg = config['data'].get('ig_stream')
        live_cfg = config.get('live_trading', {})
        if not stream_cfg:
            logger.error("Configure data.ig_stream in config.yml to enable streaming.")
            return
        if not live_cfg.get('epic') and not stream_cfg.get('epic'):
            logger.error("Set either live_trading.epic or data.ig_stream.epic.")
            return

        ig_client = IGClient(
            api_key=live_cfg.get('api_key'),
            username=live_cfg.get('username'),
            password=live_cfg.get('password'),
            account_id=live_cfg.get('account_id'),
            use_demo=live_cfg.get('use_demo', True),
        )

        epic = stream_cfg.get('epic') or live_cfg.get('epic')
        streamer = IGStreamingClient(
            ig_client=ig_client,
            epic=epic,
            resolution=stream_cfg.get('resolution', '5MINUTE'),
            stream_url=stream_cfg.get('stream_url', "https://demo-apd.marketdatasystems.com"),
            adapter_set=stream_cfg.get('adapter_set', 'DEFAULT'),
        )

        feature_engineer = FeatureEngineer(config)
        structure = StructureAnalyzer(df_h4, config, daily_df=df_d1)
        entry_gen = EntryGenerator(config, structure, feature_engineer)
        model = MLModel(config)
        model.load(config['ml']['model_path'])
        live_trader = IGLiveTrader(config, live_cfg, ig_client=ig_client)

        runner = LiveStreamingRunner(
            config=config,
            base_df=df_base,
            df_h4=df_h4,
            df_d1=df_d1,
            feature_engineer=feature_engineer,
            entry_generator=entry_gen,
            model=model,
            live_trader=live_trader,
            streamer=streamer,
        )
        runner.run()
        return

    elif args.wfa:
        if not args.start_date or not args.end_date:
            logger.error("Walk-forward mode requires --from and --to dates.")
            return
        wfa = WalkForwardAnalyzer(config)
        scored_candidates, fold_summaries = wfa.sequential_walk(candidates, args.start_date, args.end_date)
        if scored_candidates.empty:
            logger.warning("Walk-forward produced no trades. Exiting.")
            return
        backtester = Backtester(config)
        history, trades = backtester.run(scored_candidates, scored_candidates['wfa_prob'])
        save_reports(history, trades, config, prefix="wfa_")
        adaptive_manager.update_from_folds(fold_summaries)
        
    else:
        # Normal Backtest
        model = MLModel(config)
        try:
            model.load(config['ml']['model_path'])
            probs = model.predict_proba(candidates)
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Training on first 50% of data as fallback.")
            split = int(len(candidates) * 0.5)
            train_df = candidates.iloc[:split]
            model.train(train_df)
            probs = model.predict_proba(candidates)
        
        # Use ORB backtester if ORB strategy
        if strategy_type == 'orb':
            backtester = ORBBacktester(config)
            history, trades = backtester.run(candidates, probs, df_base)
        else:
            backtester = Backtester(config)
            history, trades = backtester.run(candidates, probs)
        
        save_reports(history, trades, config, prefix="backtest_")

def save_reports(history, trades, config, prefix=""):
    """Save plots and CSVs."""
    import matplotlib.pyplot as plt
    import datetime
    
    # Ensure logger is available
    logger = setup_logging() 
    from .utils import get_logger
    logger = get_logger()
    
    # Create unique timestamped folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_base_dir = "reports"
    report_dir = os.path.join(report_base_dir, timestamp)
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    # Save CSVs
    trades.to_csv(os.path.join(report_dir, f"{prefix}trades.csv"))
    history.to_csv(os.path.join(report_dir, f"{prefix}history.csv"))
    
    # Plot Equity
    if not history.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(history['time'], history['equity'], label='Equity')
        
        # Drawdown line if propfirm
        if config['backtest']['mode'] == 'propfirm':
            initial = config['backtest']['propfirm']['initial_capital']
            dd_limit = initial * (1 - config['backtest']['propfirm']['max_drawdown_pct'])
            plt.axhline(y=dd_limit, color='r', linestyle='--', label='Hard Breach Level')
            
        plt.title("Equity Curve")
        plt.legend()
        plt.savefig(os.path.join(report_dir, f"{prefix}equity.png"))
        plt.close() # Close the plot to free memory
        
        logger.info(f"Reports saved to {report_dir}")
        
    # Summary Metrics
    if not trades.empty:
        wins = trades[trades['net_pnl'] > 0]
        losses = trades[trades['net_pnl'] <= 0]
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
        total_profit = trades['net_pnl'].sum()
        profit_factor = abs(wins['net_pnl'].sum() / losses['net_pnl'].sum()) if losses['net_pnl'].sum() != 0 else float('inf')
        
        summary = {
            'total_trades': len(trades),
            'net_profit': total_profit,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
        pd.Series(summary).to_csv(os.path.join(report_dir, f"{prefix}summary.csv"))


def run_streaming_mode(config, logger, start_date=None, end_date=None):
    stream_cfg = config['data'].get('ig_stream', {})
    live_cfg = config.get('live_trading', {})

    if not stream_cfg:
        logger.error("Configure data.ig_stream before running stream_live.")
        return

    if not live_cfg:
        logger.error("Configure live_trading settings before running stream_live.")
        return

    if not live_cfg.get('enabled', False):
        logger.warning("live_trading.enabled is False. No live orders will be sent.")

    config['data']['source'] = 'ig'
    ig_client = IGClient(
        api_key=live_cfg.get('api_key'),
        username=live_cfg.get('username'),
        password=live_cfg.get('password'),
        account_id=live_cfg.get('account_id'),
        use_demo=live_cfg.get('use_demo', True),
    )
    adapter = DataAdapter(
        config,
        ig_client=ig_client,
        start_date=start_date,
        end_date=end_date,
    )

    base_tf = config['data']['timeframe_base']
    df_base = adapter.load_data(timeframe_suffix=base_tf)
    if df_base.empty:
        logger.error("IG REST returned no data for %s. Cannot start stream.", stream_cfg.get('epic'))
        return

    df_h4 = adapter.load_h4_data(start_date=start_date, end_date=end_date)
    df_d1 = adapter.load_daily_data(start_date=start_date, end_date=end_date)

    feature_engineer = FeatureEngineer(config)
    structure = StructureAnalyzer(df_h4, config, daily_df=df_d1)
    entry_gen = EntryGenerator(config, structure, feature_engineer)

    model = MLModel(config)
    try:
        model.load(config['ml']['model_path'])
    except FileNotFoundError:
        logger.error("ML model not found at %s. Train the model before streaming.", config['ml']['model_path'])
        return

    epic = stream_cfg.get('epic') or live_cfg.get('epic')
    if not epic:
        logger.error("Set live_trading.epic or data.ig_stream.epic before streaming.")
        return

    streamer = IGStreamingClient(
        ig_client=ig_client,
        epic=epic,
        resolution=stream_cfg.get('resolution', '5MINUTE'),
        stream_url=stream_cfg.get('stream_url', "https://demo-apd.marketdatasystems.com"),
        adapter_set=stream_cfg.get('adapter_set', 'DEFAULT'),
    )

    live_trader = IGLiveTrader(config, live_cfg, ig_client=ig_client)

    runner = LiveStreamingRunner(
        config=config,
        base_df=df_base,
        df_h4=df_h4,
        df_d1=df_d1,
        feature_engineer=feature_engineer,
        entry_generator=entry_gen,
        model=model,
        live_trader=live_trader,
        streamer=streamer,
    )
    runner.run()

def run_alpaca_live(config, logger, start_date=None, end_date=None, dry_run=False):
    live_cfg = config.get('live_trading', {})
    if live_cfg.get('broker', 'alpaca').lower() != 'alpaca':
        logger.error("Set live_trading.broker to 'alpaca' for Alpaca live trading.")
        return

    metadata = load_metadata(config)
    if not training_covers_recent(metadata, years=5, tolerance_days=2):
        logger.error("Model is not trained on the last 5 years. Run 'train_latest' or the menu action before starting live trading.")
        return

    history_days = live_cfg.get('history_days', 120)
    if start_date:
        history_start = start_date
    else:
        history_start = (pd.Timestamp.utcnow() - pd.Timedelta(days=history_days)).strftime('%Y-%m-%d')

    config['data']['source'] = 'alpaca'
    adapter = DataAdapter(config, start_date=history_start, end_date=end_date)
    base_tf = config['data']['timeframe_base']
    df_base = adapter.load_data(timeframe_suffix=base_tf)
    if df_base.empty:
        logger.error("Alpaca REST returned no data for %s. Cannot start live mode.", config['data']['symbol'])
        return

    feature_engineer = FeatureEngineer(config)
    model = MLModel(config)
    try:
        model.load(config['ml']['model_path'])
    except FileNotFoundError:
        logger.error("Train the SPY model first (%s) before running live trading.", config['ml']['model_path'])
        return

    alpaca_client = AlpacaClient(
        api_key=live_cfg.get('api_key'),
        api_secret=live_cfg.get('api_secret'),
        trading_url=live_cfg.get('trading_url'),
        data_url=live_cfg.get('data_url'),
    )
    live_trader = AlpacaLiveTrader(config, live_cfg, alpaca_client=alpaca_client, dry_run=dry_run)
    runner = AlpacaPollingRunner(
        config=config,
        adapter=adapter,
        base_df=df_base,
        feature_engineer=feature_engineer,
        model=model,
        live_trader=live_trader,
        alpaca_client=alpaca_client,
        dry_run=dry_run,
    )
    runner.run()


def run_training_job(config, logger, start_date, end_date):
    logger.info("Training window: %s -> %s", start_date, end_date)
    adapter = DataAdapter(config, start_date=start_date, end_date=end_date)
    base_tf_str = config['data']['timeframe_base']
    df_base = adapter.load_data(timeframe_suffix=base_tf_str)
    if df_base.empty:
        logger.error("No data found for training window.")
        return

    df_h4 = adapter.load_h4_data(start_date=start_date, end_date=end_date)
    df_d1 = adapter.load_daily_data(start_date=start_date, end_date=end_date)
    features = FeatureEngineer(config)
    df_base = features.calculate_technical_features(df_base)

    strategy_type = config['strategy'].get('strategy_type', 'smc')
    if strategy_type == 'orb':
        logger.info("Using ORB strategy for training.")
        entry_gen = ORBEntryGenerator(config)
        candidates = entry_gen.generate_candidates(df_base)
    else:
        structure = StructureAnalyzer(df_h4, config, daily_df=df_d1)
        fvgs = features.detect_fvgs(df_base)
        obs = features.detect_obs(df_base)
        entry_gen = EntryGenerator(config, structure, features)
        candidates = entry_gen.generate_candidates(df_base, df_h4, fvgs, obs)

    filter_manager = FilterProfileManager(config)
    if strategy_type != 'orb':
        active_profile = filter_manager.active_profile_from_config(config)
        if active_profile:
            candidates, meta = filter_manager.apply(active_profile, candidates)
            logger.info(
                "Applied filter profile '%s' during training: %s -> %s candidates",
                active_profile,
                meta['initial'],
                meta['filtered'],
            )

    if candidates.empty:
        logger.error("Training aborted: no candidates generated for the specified window.")
        return

    model = MLModel(config)
    model.train(candidates)
    model.save(config['ml']['model_path'])
    logger.info("Model saved to %s", config['ml']['model_path'])

    data_start = df_base.index.min()
    data_end = df_base.index.max()
    candidate_start = pd.to_datetime(candidates['entry_time']).min()
    candidate_end = pd.to_datetime(candidates['entry_time']).max()
    meta_path = save_training_metadata(
        config,
        data_start,
        data_end,
        candidate_start,
        candidate_end,
        len(candidates),
    )
    logger.info("Training metadata updated at %s", meta_path)

if __name__ == "__main__":
    main()




