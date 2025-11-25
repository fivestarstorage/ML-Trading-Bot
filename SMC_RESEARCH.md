# SMC + ML Research Notes

These notes capture the institutional concepts reviewed before rebuilding the strategy. They form the basis for the new candidate-generation logic in `src/entries.py` and the surrounding infrastructure.

## What top-performing SMC desks focus on

1. **Multi-timeframe Context**
   - Institutional order flow is framed using a slow HTF bias (Daily 50/200 EMA crossover with price above/below EMA200 to avoid taking trades against a macro trend).
   - Execution happens on intraday flows (H4 structure for BOS / CHoCH, 5m–15m execution) so trades are only taken when HTF + execution timeframe align.

2. **Liquidity Engineering**
   - Liquidity pools (equal highs/lows, Asian range extremes) are swept intentionally before displacement. Reliable setups wait for a sweep + rejection and only engage inside discount (for longs) or premium (for shorts) zones of the current range.
   - Inducement + mitigation: the displacement candle must leave a Fair Value Gap or refined Order Block that price can revisit for a low-risk entry.

3. **Session / Killzone timing**
   - Most institutional flows occur during London open (07:00–11:00 UTC) and New York open (13:00–17:00 UTC). Outside of these windows, SMC setups have materially lower follow-through.

4. **Displacement quality**
   - Smart money waits for an impulsive candle whose body/ATR ratio exceeds 1.0–1.5 before trusting a newly created FVG/OB. Weak displacement is ignored.

5. **Adaptive risk and volatility regimes**
   - Position sizing is reduced when ATR compresses below its rolling mean or when volume z-score is negative; complacent sessions tend to mean-revert and invalidate momentum entries.

6. **Data-driven validation**
   - SMC rules on their own are discretionary. Pairing them with feature-rich ML (LightGBM) to rank signals by posterior probability provides a consistent gatekeeper layer.

## Implementation highlights

- Daily trend detection and EMA context prepared inside `StructureAnalyzer`, exposing `daily_trend` to candidate generation and allowing the `use_daily_trend_filter` toggle in `config.yml`.
- Liquidity sweeps & displacement filters now gate every signal (`EntryGenerator._detect_liquidity_sweep`, `body_factor` threshold).
- Killzone/session tagging happens in `FeatureEngineer`, enabling strict session filters (`require_killzone_filter` + `session_filter`).
- Adaptive risk ties ATR regimes to the prop-firm risk framework in `Backtester`.
- The menu-driven CLI (`trading-bot`) wraps every workflow so research / training / optimisation can be triggered consistently.

These principles were distilled from prop-firm / institutional playbooks and matched against the historic diagnostics already present in the repository. The code now mirrors the playbook instead of applying a loosely defined “SMC” label.


