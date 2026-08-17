"""
PAIRS TRADING MONITOR — SCALED (100-stock universe)
=====================================================
Tracks a universe of ~100 stocks organized into economically-linked
sector groups, tests every within-sector pair for real (cointegrated)
relationships, and alerts on statistically unusual divergence.

Why sector-restricted, not "all possible pairs":
    With ~100 stocks, ALL possible pairs = ~4,950 combinations. Testing
    that many pairs at a 5% significance threshold means ~247 of them
    would show "significant" cointegration by RANDOM CHANCE ALONE, even
    if nothing were actually related (the "multiple testing problem").
    Restricting to within-sector pairs cuts this to a few hundred
    economically-motivated tests, which is both more meaningful and
    safer statistically. A stricter Bonferroni-corrected threshold is
    also reported alongside the standard one — see OUTPUT below.

Run modes:
    python pairs_monitor.py --scan             Full universe scan (sector-restricted pairs)
    python pairs_monitor.py --watch             Loop and re-check every N minutes
    python pairs_monitor.py --backtest AAA BBB   Deep-dive on one specific pair
    python pairs_monitor.py --scan --verbose     Print full detail for every pair, not just hits

Requires: pip install yfinance statsmodels pandas numpy scipy
"""

import time
import sys
import argparse
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    "formation_lookback_days": 252,
    "zscore_window": 30,
    "entry_zscore": 2.0,
    "exit_zscore": 0.5,
    "stop_zscore": 3.5,
    "max_coint_pvalue": 0.05,          # standard threshold, per-pair
    "min_correlation": 0.6,
    "min_avg_dollar_volume": 5_000_000,
    "earnings_blackout_days": 3,
    "recalibration_days": 63,
    "watch_interval_minutes": 60,
    "log_file": "pairs_alerts_log.csv",

    # --- Walk-forward backtest settings ---
    "wf_total_years": 5,          # total history to pull for the backtest
    "wf_formation_days": 252,     # ~1 trading year used to fit hedge ratio & test cointegration
    "wf_trading_days": 63,        # ~1 quarter traded forward on those frozen parameters
    "wf_transaction_cost_bps": 5, # one-way cost per leg, in basis points (5bps = 0.05%)

    # --- Co-movement / relative-strength scanner (for trending, non-cointegrated pairs) ---
    "comovement_window_days": 60,  # rolling window for correlation/beta, in trading days
}

# ============================================================================
# UNIVERSE — ~100 stocks grouped by real economic linkage.
# Pairs are only ever generated WITHIN a group, never across groups —
# that's the whole point: a semiconductor stock and a beverage stock
# might randomly correlate for a month, but there's no economic reason
# they'd stay cointegrated, so testing that pair is a waste and a
# statistical liability.
# ============================================================================

UNIVERSE = {
    "Memory/Storage Semis":     ["MU", "WDC", "STX"],
    "Broad Semiconductors":     ["AMD", "INTC", "TXN", "QCOM", "AVGO", "NXPI", "ON", "MRVL", "MCHP", "ADI"],
    "Copper/Base Metals Mining":["FCX", "SCCO", "TECK", "VALE", "RIO", "BHP"],
    "Integrated Oil Majors":    ["XOM", "CVX", "COP", "OXY", "PSX", "MPC", "VLO"],
    "Data Center/Power Infra":  ["VRT", "ETN", "MOD", "PWR", "AME", "HUBB"],
    "Beverages":                ["KO", "PEP", "KDP", "MNST", "STZ"],
    "Big Banks":                ["JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "COF"],
    "Home Improvement/Big Box": ["HD", "LOW", "TGT", "WMT", "COST"],
    "Payment Processors":       ["V", "MA", "AXP", "PYPL", "FIS"],
    "Airlines":                 ["DAL", "UAL", "AAL", "LUV", "ALK"],
    "Pharma Giants":            ["PFE", "MRK", "JNJ", "ABT", "BMY", "LLY", "ABBV"],
    "Insurance":                ["TRV", "ALL", "PGR", "CB", "AIG"],
    "Telecom":                  ["T", "VZ", "TMUS"],
    "Utilities":                ["DUK", "SO", "NEE", "AEP", "D"],
    "REITs":                    ["PLD", "AMT", "EQIX", "DLR", "O"],
    "Restaurants":               ["MCD", "YUM", "CMG", "SBUX", "DPZ"],
    "Enterprise Software/Cloud":["MSFT", "ORCL", "CRM", "ADBE", "NOW"],
    "Industrials/Machinery":    ["CAT", "DE", "HON", "MMM", "GE"],
    "Aerospace/Defense":        ["LMT", "RTX", "NOC", "GD"],
}


def build_candidate_pairs(universe):
    """Generate all within-group combinations. Cross-group pairs are
    never generated — that's the sector-restriction guardrail."""
    pairs = []
    for sector, tickers in universe.items():
        for a, b in itertools.combinations(tickers, 2):
            pairs.append((a, b, sector))
    return pairs


def all_tickers(universe):
    seen = []
    for tickers in universe.values():
        for t in tickers:
            if t not in seen:
                seen.append(t)
    return seen


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PairAnalysis:
    ticker_a: str
    ticker_b: str
    sector: str = ""
    correlation: float = None
    coint_pvalue: float = None
    hedge_ratio: float = None
    half_life_days: float = None
    current_zscore: float = None
    is_statistically_valid: bool = False
    bonferroni_significant: bool = False
    alert_level: str = "none"
    earnings_conflict: bool = False
    notes: list = field(default_factory=list)


# ============================================================================
# BATCH DATA FETCHING — download every ticker ONCE, reuse across all pairs
# ============================================================================

def fetch_universe_data(tickers, lookback_days):
    """Single batch download for the whole universe. Returns:
       prices: {ticker: pd.Series of adjusted close}
       volumes: {ticker: pd.Series of volume}
    Handles both the multi-index and flat column formats different
    yfinance versions return."""
    end = datetime.today()
    start = end - timedelta(days=int(lookback_days * 1.6))
    print(f"Downloading data for {len(tickers)} tickers in one batch (this is the slow part, ~30-90 sec)...")
    raw = yf.download(tickers, start=start, end=end, progress=False,
                       auto_adjust=True, group_by="ticker", threads=True)

    prices, volumes = {}, {}
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t in raw.columns.get_level_values(0):
                    sub = raw[t]
                elif t in raw.columns.get_level_values(1):
                    sub = raw.xs(t, axis=1, level=1)
                else:
                    continue
                close = sub["Close"].dropna()
                vol = sub["Volume"].dropna()
            else:
                # only happens if a single ticker was requested
                close = raw["Close"].dropna()
                vol = raw["Volume"].dropna()
            if len(close) > 20:
                prices[t] = close
                volumes[t] = vol
        except Exception as e:
            print(f"  (skipping {t}: {e})")
            continue

    missing = set(tickers) - set(prices.keys())
    if missing:
        print(f"  No usable data for: {', '.join(sorted(missing))}")
    print(f"Got data for {len(prices)}/{len(tickers)} tickers.\n")
    return prices, volumes


def fetch_earnings_calendar(tickers):
    """One call per ticker (unavoidable — yfinance has no batch endpoint
    for this), but only ONCE per ticker total, not once per pair."""
    days_to_earnings = {}
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar
            next_date = None
            if isinstance(cal, dict) and "Earnings Date" in cal:
                dates = cal["Earnings Date"]
                next_date = dates[0] if isinstance(dates, list) else dates
            elif hasattr(cal, "loc") and "Earnings Date" in getattr(cal, "index", []):
                next_date = cal.loc["Earnings Date"][0]
            if isinstance(next_date, (datetime, pd.Timestamp)):
                days_to_earnings[t] = (next_date.date() - datetime.today().date()).days
        except Exception:
            pass
    return days_to_earnings


# ============================================================================
# STATISTICAL CORE (same tests as before, just fed pre-fetched data now)
# ============================================================================

def compute_hedge_ratio(price_a, price_b):
    log_a, log_b = np.log(price_a), np.log(price_b)
    X = sm.add_constant(log_b)
    model = sm.OLS(log_a, X).fit()
    beta = model.params.iloc[1] if hasattr(model.params, "iloc") else model.params[1]
    spread = log_a - beta * log_b
    return float(beta), spread


def engle_granger_test(price_a, price_b):
    _, pvalue, _ = coint(np.log(price_a), np.log(price_b))
    return float(pvalue)


def compute_half_life(spread):
    spread_lag = spread.shift(1).dropna()
    spread_ret = (spread - spread.shift(1)).dropna()
    spread_lag = spread_lag.loc[spread_ret.index]
    X = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, X).fit()
    theta = model.params.iloc[1] if hasattr(model.params, "iloc") else model.params[1]
    if theta >= 0:
        return None
    return float(-np.log(2) / theta)


def rolling_zscore(spread, window):
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


# ============================================================================
# PAIR ANALYSIS (now takes pre-fetched series instead of hitting the network)
# ============================================================================

def analyze_pair(ticker_a, ticker_b, sector, prices, volumes, earnings, cfg, n_tests):
    result = PairAnalysis(ticker_a=ticker_a, ticker_b=ticker_b, sector=sector)

    if ticker_a not in prices or ticker_b not in prices:
        result.notes.append("Missing price data for one or both legs.")
        return result

    df = pd.concat([prices[ticker_a], prices[ticker_b]], axis=1, join="inner")
    df.columns = ["A", "B"]
    if len(df) < cfg["formation_lookback_days"] * 0.7:
        result.notes.append("Insufficient overlapping history.")
        return result

    result.correlation = float(df["A"].corr(df["B"]))
    if result.correlation < cfg["min_correlation"]:
        result.notes.append(f"Correlation {result.correlation:.2f} below floor.")
        return result

    try:
        result.coint_pvalue = engle_granger_test(df["A"], df["B"])
        beta, spread = compute_hedge_ratio(df["A"], df["B"])
        result.hedge_ratio = beta
        result.half_life_days = compute_half_life(spread)
    except Exception as e:
        result.notes.append(f"Statistical test failed (bad/missing data): {e}")
        return result

    vol_a = volumes.get(ticker_a)
    vol_b = volumes.get(ticker_b)
    dollar_vol_a = float((prices[ticker_a] * vol_a).tail(30).mean()) if vol_a is not None else 0
    dollar_vol_b = float((prices[ticker_b] * vol_b).tail(30).mean()) if vol_b is not None else 0
    illiquid = dollar_vol_a < cfg["min_avg_dollar_volume"] or dollar_vol_b < cfg["min_avg_dollar_volume"]
    if illiquid:
        result.notes.append("Below liquidity floor.")

    de_a = earnings.get(ticker_a)
    de_b = earnings.get(ticker_b)
    if (de_a is not None and 0 <= de_a <= cfg["earnings_blackout_days"]) or \
       (de_b is not None and 0 <= de_b <= cfg["earnings_blackout_days"]):
        result.earnings_conflict = True
        result.notes.append("Earnings imminent for one leg.")

    result.is_statistically_valid = (
        result.coint_pvalue < cfg["max_coint_pvalue"]
        and result.half_life_days is not None
        and 1 < result.half_life_days < cfg["formation_lookback_days"]
        and not illiquid
    )

    # Bonferroni-corrected threshold across ALL tests run this scan —
    # a much stricter bar, shown for context on how robust a "pass" really is.
    bonferroni_threshold = cfg["max_coint_pvalue"] / max(n_tests, 1)
    result.bonferroni_significant = result.coint_pvalue < bonferroni_threshold

    z_series = rolling_zscore(spread, cfg["zscore_window"])
    current_z = z_series.dropna().iloc[-1] if not z_series.dropna().empty else None
    result.current_zscore = float(current_z) if current_z is not None else None

    if result.current_zscore is not None and result.is_statistically_valid:
        az = abs(result.current_zscore)
        if az >= cfg["stop_zscore"]:
            result.alert_level = "extreme"
        elif az >= cfg["entry_zscore"]:
            result.alert_level = "divergence"
        elif az >= cfg["entry_zscore"] * 0.6:
            result.alert_level = "watch"

    return result


# ============================================================================
# REPORTING
# ============================================================================

def format_report(result: PairAnalysis):
    lines = [f"\n{'='*60}", f"{result.ticker_a} / {result.ticker_b}  [{result.sector}]"]
    if result.correlation is not None:
        lines.append(f"  Correlation:        {result.correlation:.3f}")
    if result.coint_pvalue is not None:
        tag = "PASS" if result.coint_pvalue < CONFIG["max_coint_pvalue"] else "FAIL"
        bonf = " (Bonferroni-significant too)" if result.bonferroni_significant else ""
        lines.append(f"  Cointegration p:    {result.coint_pvalue:.4f}  [{tag}]{bonf}")
    if result.hedge_ratio is not None:
        lines.append(f"  Hedge ratio (beta): {result.hedge_ratio:.3f}")
    if result.half_life_days is not None:
        lines.append(f"  Half-life:          {result.half_life_days:.1f} days")
    if result.current_zscore is not None:
        lines.append(f"  Current z-score:    {result.current_zscore:+.2f}")
    lines.append(f"  Statistically valid pair: {result.is_statistically_valid}")
    lines.append(f"  Alert level: {result.alert_level.upper()}")
    for n in result.notes:
        lines.append(f"  note: {n}")
    if result.alert_level in ("divergence", "extreme") and result.is_statistically_valid:
        z = result.current_zscore
        if z > 0:
            lines.append(f"  >>> {result.ticker_a} rich vs {result.ticker_b}: "
                         f"SHORT {result.ticker_a}, LONG {result.ticker_b}")
        else:
            lines.append(f"  >>> {result.ticker_b} rich vs {result.ticker_a}: "
                         f"SHORT {result.ticker_b}, LONG {result.ticker_a}")
    return "\n".join(lines)


def write_github_outputs(results, cfg):
    """When running inside GitHub Actions, write a nice run summary and a
    JSON file of active alerts so the workflow can decide whether to open
    an issue. No-ops harmlessly when run locally."""
    import os, json

    alerts = [r for r in results if r.alert_level in ("divergence", "extreme")]
    valid = [r for r in results if r.is_statistically_valid]

    # JSON file the workflow step reads to decide on issue creation
    alert_payload = [
        {
            "pair": f"{r.ticker_a}/{r.ticker_b}",
            "sector": r.sector,
            "zscore": round(r.current_zscore, 2) if r.current_zscore is not None else None,
            "coint_pvalue": round(r.coint_pvalue, 4) if r.coint_pvalue is not None else None,
            "half_life_days": round(r.half_life_days, 1) if r.half_life_days is not None else None,
            "alert_level": r.alert_level,
            "direction": (
                f"SHORT {r.ticker_a}, LONG {r.ticker_b}" if r.current_zscore and r.current_zscore > 0
                else f"SHORT {r.ticker_b}, LONG {r.ticker_a}"
            ) if r.current_zscore is not None else None,
        }
        for r in alerts
    ]
    with open("latest_alerts.json", "w") as f:
        json.dump(alert_payload, f, indent=2)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with open(summary_path, "a") as f:
        f.write(f"## Pairs scan — {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(f"- Pairs tested: {len(results)}\n")
        f.write(f"- Statistically valid pairs: {len(valid)}\n")
        f.write(f"- Active alerts: {len(alerts)}\n\n")
        if alerts:
            f.write("| Pair | Sector | z-score | Direction |\n|---|---|---|---|\n")
            for r in alerts:
                f.write(f"| {r.ticker_a}/{r.ticker_b} | {r.sector} | {r.current_zscore:+.2f} | "
                        f"{'SHORT ' + r.ticker_a + ', LONG ' + r.ticker_b if r.current_zscore > 0 else 'SHORT ' + r.ticker_b + ', LONG ' + r.ticker_a} |\n")
        if valid:
            f.write("\n### All statistically valid pairs this run\n\n")
            f.write("| Pair | Sector | Correlation | Coint. p | Half-life | z-score |\n|---|---|---|---|---|---|\n")
            for r in valid:
                f.write(f"| {r.ticker_a}/{r.ticker_b} | {r.sector} | {r.correlation:.2f} | "
                        f"{r.coint_pvalue:.4f} | {r.half_life_days:.1f}d | {r.current_zscore:+.2f} |\n")



def log_alert(result: PairAnalysis, cfg):
    if result.alert_level == "none":
        return
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pair": f"{result.ticker_a}/{result.ticker_b}",
        "sector": result.sector,
        "zscore": result.current_zscore,
        "coint_pvalue": result.coint_pvalue,
        "half_life_days": result.half_life_days,
        "alert_level": result.alert_level,
    }
    import os
    df_row = pd.DataFrame([row])
    df_row.to_csv(cfg["log_file"], mode="a", header=not os.path.exists(cfg["log_file"]), index=False)


# ============================================================================
# MAIN RUN MODES
# ============================================================================

def run_scan(cfg, verbose=False):
    candidate_pairs = build_candidate_pairs(UNIVERSE)
    tickers = all_tickers(UNIVERSE)
    print(f"Universe: {len(tickers)} tickers across {len(UNIVERSE)} sectors")
    print(f"Candidate pairs to test (within-sector only): {len(candidate_pairs)}")
    print(f"(For comparison, ALL possible pairs among {len(tickers)} stocks would be "
          f"{len(tickers)*(len(tickers)-1)//2} — sector-restriction avoids testing "
          f"the ~{int(0.05*(len(tickers)*(len(tickers)-1)//2))} false positives "
          f"that pure chance would produce at that scale.)\n")

    prices, volumes = fetch_universe_data(tickers, cfg["formation_lookback_days"])
    print("Checking earnings calendars (one call per ticker)...")
    earnings = fetch_earnings_calendar(list(prices.keys()))

    print(f"\nRunning {len(candidate_pairs)} pairwise tests...\n")
    results = []
    for a, b, sector in candidate_pairs:
        try:
            r = analyze_pair(a, b, sector, prices, volumes, earnings, cfg, n_tests=len(candidate_pairs))
        except Exception as e:
            r = PairAnalysis(ticker_a=a, ticker_b=b, sector=sector, notes=[f"Unexpected error: {e}"])
        results.append(r)
        log_alert(r, cfg)

    # Always show anything statistically valid or alerting
    highlights = [r for r in results if r.is_statistically_valid or r.alert_level != "none"]
    for r in highlights:
        print(format_report(r))

    if verbose:
        others = [r for r in results if r not in highlights]
        print(f"\n\n--- Remaining {len(others)} tested pairs (failed validity gate) ---")
        for r in others:
            print(format_report(r))

    print(f"\n\n{'='*60}\nSUMMARY")
    print(f"  Pairs tested:              {len(results)}")
    print(f"  Statistically valid pairs: {len(highlights)}")
    alerts = [r for r in results if r.alert_level in ("divergence", "extreme")]
    if alerts:
        print(f"  ACTIVE ALERTS: {len(alerts)}")
        for r in alerts:
            print(f"    {r.ticker_a}/{r.ticker_b}  z={r.current_zscore:+.2f}  [{r.alert_level.upper()}]")
    else:
        print("  No divergence alerts this scan.")

    write_github_outputs(results, cfg)


def run_watch(cfg):
    print(f"Starting watch loop, checking every {cfg['watch_interval_minutes']} min. Ctrl+C to stop.")
    while True:
        run_scan(cfg)
        time.sleep(cfg["watch_interval_minutes"] * 60)


def run_backtest(ticker_a, ticker_b, cfg):
    sector = "manual"
    for sec, tickers in UNIVERSE.items():
        if ticker_a in tickers and ticker_b in tickers:
            sector = sec
            break
    prices, volumes = fetch_universe_data([ticker_a, ticker_b], cfg["formation_lookback_days"])
    earnings = fetch_earnings_calendar([ticker_a, ticker_b])
    r = analyze_pair(ticker_a, ticker_b, sector, prices, volumes, earnings, cfg, n_tests=1)
    print(format_report(r))


# ============================================================================
# WALK-FORWARD BACKTEST
# ============================================================================
#
# Why walk-forward instead of testing on the same window used to find the
# pair: fitting the hedge ratio AND measuring performance on the same data
# is a classic way backtests lie to you — the strategy "knows" the answer
# in advance. Walk-forward fixes this by only ever using PAST data (a
# formation window) to decide how to trade, then testing that frozen
# decision on data the strategy hasn't seen yet (the trading window), then
# rolling forward and repeating. This is closer to how the pair would
# actually have been traded in real time.

@dataclass
class WalkForwardTrade:
    entry_date: object
    exit_date: object
    direction: str          # "short_a_long_b" or "long_a_short_b"
    entry_zscore: float
    exit_zscore: float
    trade_return: float     # net of transaction costs, in decimal (0.01 = 1%)
    exit_reason: str        # "target" | "stop" | "block_end"


def _simulate_trading_window(spread_trading, formation_mean, formation_std, cfg):
    """Given a frozen mean/std from the formation window, walk day-by-day
    through the trading window and simulate entries/exits. Returns a list
    of completed trades and a daily P&L series (0 when flat) for equity
    curve / Sharpe / drawdown purposes."""
    z = (spread_trading - formation_mean) / formation_std
    trades = []
    daily_pnl = pd.Series(0.0, index=spread_trading.index)

    position = None  # None, or dict with entry info
    cost = cfg["wf_transaction_cost_bps"] / 10000.0

    for i in range(1, len(spread_trading)):
        date = spread_trading.index[i]
        prev_date = spread_trading.index[i - 1]
        current_z = z.iloc[i]

        if position is None:
            if current_z >= cfg["entry_zscore"]:
                position = {"direction": "short_a_long_b", "entry_date": date,
                            "entry_spread": spread_trading.iloc[i], "entry_z": current_z}
                daily_pnl.loc[date] -= 2 * cost  # entry cost (2 legs)
            elif current_z <= -cfg["entry_zscore"]:
                position = {"direction": "long_a_short_b", "entry_date": date,
                            "entry_spread": spread_trading.iloc[i], "entry_z": current_z}
                daily_pnl.loc[date] -= 2 * cost
        else:
            # daily mark-to-market P&L: change in spread, signed by direction.
            # This treats the dollar-neutral log-spread position's daily
            # change as the position's daily return — a standard simplifying
            # approximation in academic pairs-trading backtests (it ignores
            # financing/borrow costs and assumes frictionless rebalancing
            # of the hedge ratio; real-world results would be somewhat
            # lower after those frictions).
            spread_change = spread_trading.iloc[i] - spread_trading.loc[prev_date]
            sign = -1 if position["direction"] == "short_a_long_b" else 1
            daily_pnl.loc[date] += sign * spread_change

            exit_reason = None
            if abs(current_z) <= cfg["exit_zscore"]:
                exit_reason = "target"
            elif abs(current_z) >= cfg["stop_zscore"]:
                exit_reason = "stop"
            elif i == len(spread_trading) - 1:
                exit_reason = "block_end"

            if exit_reason:
                daily_pnl.loc[date] -= 2 * cost  # exit cost
                total_spread_move = sign * (spread_trading.iloc[i] - position["entry_spread"])
                trade_return = total_spread_move - 4 * cost  # round-trip costs, both legs, both sides
                trades.append(WalkForwardTrade(
                    entry_date=position["entry_date"], exit_date=date,
                    direction=position["direction"], entry_zscore=position["entry_z"],
                    exit_zscore=current_z, trade_return=trade_return, exit_reason=exit_reason,
                ))
                position = None

    return trades, daily_pnl


@dataclass
class WalkForwardResult:
    ticker_a: str
    ticker_b: str
    sector: str = ""
    period_start: str = ""
    period_end: str = ""
    observations: int = 0
    n_blocks: int = 0
    n_eligible: int = 0
    completed_trades: int = 0
    total_return: float = None
    ann_vol: float = None
    sharpe: float = None
    max_drawdown: float = None
    win_rate: float = None
    trades: list = field(default_factory=list)
    error: str = None


def compute_walkforward(log_a, log_b, cfg):
    """Pure computation over already-fetched log-price series. Returns a
    WalkForwardResult (ticker/sector left blank — caller fills those in)."""
    formation_n = cfg["wf_formation_days"]
    trading_n = cfg["wf_trading_days"]

    all_trades = []
    all_daily_pnl = []
    n_blocks = 0
    n_eligible = 0

    start = 0
    while start + formation_n + trading_n <= len(log_a):
        n_blocks += 1
        form_a, form_b = log_a.iloc[start:start + formation_n], log_b.iloc[start:start + formation_n]
        trade_a, trade_b = (log_a.iloc[start + formation_n:start + formation_n + trading_n],
                             log_b.iloc[start + formation_n:start + formation_n + trading_n])
        try:
            _, pvalue, _ = coint(form_a, form_b)
        except Exception:
            start += trading_n
            continue

        if pvalue < cfg["max_coint_pvalue"]:
            n_eligible += 1
            X = sm.add_constant(form_b)
            model = sm.OLS(form_a, X).fit()
            beta = model.params.iloc[1] if hasattr(model.params, "iloc") else model.params[1]
            formation_spread = form_a - beta * form_b
            formation_mean, formation_std = formation_spread.mean(), formation_spread.std()
            trading_spread = trade_a - beta * trade_b
            trades, daily_pnl = _simulate_trading_window(trading_spread, formation_mean, formation_std, cfg)
            all_trades.extend(trades)
            all_daily_pnl.append(daily_pnl)
        else:
            all_daily_pnl.append(pd.Series(0.0, index=trade_a.index))

        start += trading_n

    result = WalkForwardResult(
        ticker_a="", ticker_b="", observations=len(log_a),
        n_blocks=n_blocks, n_eligible=n_eligible, completed_trades=len(all_trades),
        trades=all_trades,
    )

    if not all_daily_pnl:
        result.error = "insufficient_history"
        return result

    equity_returns = pd.concat(all_daily_pnl).sort_index()
    cumulative = (1 + equity_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    result.total_return = float(cumulative.iloc[-1] - 1)
    result.ann_vol = float(equity_returns.std() * np.sqrt(252))
    result.sharpe = float(equity_returns.mean() / equity_returns.std() * np.sqrt(252)) if equity_returns.std() > 0 else None
    result.max_drawdown = float(drawdown.min())
    wins = sum(1 for t in all_trades if t.trade_return > 0)
    result.win_rate = (wins / len(all_trades)) if all_trades else None
    return result


def run_walkforward_backtest(ticker_a, ticker_b, cfg):
    total_days = int(cfg["wf_total_years"] * 365)
    prices, _ = fetch_universe_data([ticker_a, ticker_b], total_days)

    if ticker_a not in prices or ticker_b not in prices:
        print(f"Could not fetch data for {ticker_a} and/or {ticker_b}.")
        return

    df = pd.concat([prices[ticker_a], prices[ticker_b]], axis=1, join="inner")
    df.columns = ["A", "B"]
    log_a, log_b = np.log(df["A"]), np.log(df["B"])

    r = compute_walkforward(log_a, log_b, cfg)
    r.ticker_a, r.ticker_b = ticker_a, ticker_b
    r.period_start, r.period_end = str(df.index[0].date()), str(df.index[-1].date())

    if r.error == "insufficient_history":
        print("Not enough history for even one formation+trading block. Try a longer wf_total_years.")
        return

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD BACKTEST: {ticker_a}/{ticker_b}")
    print(f"{'='*60}")
    print(f"  Formation window:    {cfg['wf_formation_days']} trading days")
    print(f"  Trading window:      {cfg['wf_trading_days']} trading days")
    print(f"  Transaction cost:    {cfg['wf_transaction_cost_bps']} bps per leg, one-way")
    print(f"  Observations:        {r.observations} daily price points ({r.period_start} to {r.period_end})")
    print(f"  Eligible blocks:     {r.n_eligible} / {r.n_blocks} "
          f"(passed cointegration test on that block's formation window)")
    print(f"  Completed trades:    {r.completed_trades}")
    print(f"  Total return:        {r.total_return:+.2%}")
    print(f"  Annualized volatility: {r.ann_vol:.2%}")
    print(f"  Sharpe ratio:        {r.sharpe:.2f}" if r.sharpe is not None else "  Sharpe ratio:        n/a (no trades taken)")
    print(f"  Max drawdown:        {r.max_drawdown:.2%}")
    print(f"  Win rate:            {r.win_rate:.1%}" if r.win_rate is not None else "  Win rate:            n/a (no completed trades)")

    if r.trades:
        print(f"\n  Trade log:")
        for t in r.trades:
            print(f"    {t.entry_date.date()} -> {t.exit_date.date()}  "
                  f"{t.direction:16s}  return={t.trade_return:+.2%}  exit={t.exit_reason}")

    print(f"\n  Notes:")
    print(f"   - This treats the dollar-neutral spread's daily change as the position's")
    print(f"     daily return — a standard simplifying approximation. Real-world results")
    print(f"     would likely be somewhat lower after financing/borrow costs and hedge")
    print(f"     ratio rebalancing frictions this doesn't model.")
    print(f"   - 'Eligible blocks' below ~50% means this pair's relationship is unstable")
    print(f"     over time — treat any live signals from it with extra caution.")
    print(f"   - Past performance in a backtest, even a careful walk-forward one, is not")
    print(f"     a guarantee of future results.")


def run_walkforward_all(cfg):
    """Runs the walk-forward backtest across every within-sector candidate
    pair in the universe and prints a single comparison table."""
    candidate_pairs = build_candidate_pairs(UNIVERSE)
    tickers = all_tickers(UNIVERSE)
    total_days = int(cfg["wf_total_years"] * 365)

    print(f"Fetching {cfg['wf_total_years']} years of data for {len(tickers)} tickers "
          f"(one batch download)...")
    prices, _ = fetch_universe_data(tickers, total_days)

    print(f"Running walk-forward backtests on {len(candidate_pairs)} candidate pairs "
          f"(this is CPU-bound, not network-bound, so it's faster than the data download)...\n")

    results = []
    for a, b, sector in candidate_pairs:
        if a not in prices or b not in prices:
            continue
        df = pd.concat([prices[a], prices[b]], axis=1, join="inner")
        if len(df) < cfg["wf_formation_days"] + cfg["wf_trading_days"]:
            continue
        df.columns = ["A", "B"]
        log_a, log_b = np.log(df["A"]), np.log(df["B"])
        try:
            r = compute_walkforward(log_a, log_b, cfg)
        except Exception as e:
            continue
        r.ticker_a, r.ticker_b, r.sector = a, b, sector
        r.period_start, r.period_end = str(df.index[0].date()), str(df.index[-1].date())
        if r.error != "insufficient_history":
            results.append(r)

    # Only worth showing pairs that actually got at least one eligible block —
    # otherwise "0 trades, 0% everything" rows just add noise.
    active = [r for r in results if r.n_eligible > 0]
    active.sort(key=lambda r: (r.sharpe if r.sharpe is not None else -999), reverse=True)

    period = f"{active[0].period_start} to {active[0].period_end}" if active else "n/a"
    header = f"{'Pair':<12}{'Period':<24}{'Obs':>6}{'Elig.':>8}{'Trades':>8}{'TotalRet':>10}{'AnnVol':>9}{'Sharpe':>8}{'MaxDD':>9}{'WinRate':>9}"
    print(header)
    print("-" * len(header))
    for r in active:
        sharpe_str = f"{r.sharpe:.2f}" if r.sharpe is not None else "n/a"
        win_str = f"{r.win_rate:.0%}" if r.win_rate is not None else "n/a"
        print(f"{r.ticker_a + '/' + r.ticker_b:<12}{r.period_start + ' to ' + r.period_end:<24}"
              f"{r.observations:>6}{r.n_eligible:>5}/{r.n_blocks:<3}{r.completed_trades:>8}"
              f"{r.total_return:>+10.2%}{r.ann_vol:>9.2%}{sharpe_str:>8}{r.max_drawdown:>9.2%}{win_str:>9}")

    skipped = len(candidate_pairs) - len(results)
    print(f"\n{len(active)} pairs had at least one eligible (cointegrated) block out of "
          f"{len(results)} pairs with sufficient data ({skipped} skipped for insufficient history).")
    print("Sorted by Sharpe ratio, descending. A pair with 0 eligible blocks never showed up here at all —")
    print("that's a pair that was never tradeable by this strategy across its whole history.")


# ============================================================================
# CO-MOVEMENT / RELATIVE-STRENGTH SCANNER
# ============================================================================
#
# This is a DIFFERENT strategy from everything above, for pairs that move
# together because of a SHARED TREND (same sector supercycle, same macro
# driver) rather than a stable long-run equilibrium. Classic pairs trading
# (everything above this section) requires cointegration — a stable spread
# that reverts to a fixed mean. That fails, correctly, for a pair like a
# recently-spun-off company riding the same secular trend as its former
# parent's biggest peer: there's no long, stable history to test, and the
# relationship isn't "reverting to normal" — both are just trending together.
#
# What this measures instead: on a SHORT rolling window (days to weeks, not
# years), how tightly do these two stocks' DAILY RETURNS move together
# (correlation + beta), and on any given day, did one of them move without
# the other confirming yet? That's a real, different, tradeable pattern —
# but it is NOT market-neutral like classic pairs trading. Both stocks can
# fall together if the shared narrative (e.g. an AI/memory demand cycle)
# reverses. Treat this as a momentum/co-movement signal, not an arbitrage.

@dataclass
class CoMovementSnapshot:
    ticker_a: str
    ticker_b: str
    window_days: int
    rolling_correlation: float = None
    rolling_beta: float = None       # B's typical move per 1% move in A, over the window
    today_return_a: float = None
    today_return_b: float = None
    expected_return_b: float = None  # what B "should" have done today, given A's move and beta
    residual_today: float = None     # actual B return minus expected B return
    residual_zscore: float = None    # how unusual today's residual is vs the window's own residual history
    note: str = None


def compute_comovement(ticker_a, ticker_b, cfg, window_days=None):
    window_days = window_days or cfg.get("comovement_window_days", 60)
    lookback_days = max(window_days * 3, 180)  # extra history so the rolling window itself has room to roll
    prices, _ = fetch_universe_data([ticker_a, ticker_b], lookback_days)

    if ticker_a not in prices or ticker_b not in prices:
        return CoMovementSnapshot(ticker_a, ticker_b, window_days, note="Could not fetch data for one or both tickers.")

    df = pd.concat([prices[ticker_a], prices[ticker_b]], axis=1, join="inner")
    df.columns = ["A", "B"]
    if len(df) < window_days + 5:
        return CoMovementSnapshot(
            ticker_a, ticker_b, window_days,
            note=f"Only {len(df)} overlapping trading days available — too little history for a "
                 f"{window_days}-day rolling window. This is common right after a spin-off/IPO; "
                 f"try a shorter --comovement-window."
        )

    ret_a = df["A"].pct_change().dropna()
    ret_b = df["B"].pct_change().dropna()
    ret_a, ret_b = ret_a.align(ret_b, join="inner")

    roll_corr = ret_a.rolling(window_days).corr(ret_b)
    roll_cov = ret_a.rolling(window_days).cov(ret_b)
    roll_var = ret_a.rolling(window_days).var()
    roll_beta = roll_cov / roll_var

    predicted_b = roll_beta.shift(1) * ret_a  # beta known as of yesterday, applied to today's A move — no lookahead
    residual = ret_b - predicted_b
    resid_mean = residual.rolling(window_days).mean()
    resid_std = residual.rolling(window_days).std()
    resid_z = (residual - resid_mean) / resid_std

    snap = CoMovementSnapshot(
        ticker_a=ticker_a, ticker_b=ticker_b, window_days=window_days,
        rolling_correlation=float(roll_corr.iloc[-1]) if not pd.isna(roll_corr.iloc[-1]) else None,
        rolling_beta=float(roll_beta.iloc[-1]) if not pd.isna(roll_beta.iloc[-1]) else None,
        today_return_a=float(ret_a.iloc[-1]),
        today_return_b=float(ret_b.iloc[-1]),
        expected_return_b=float(predicted_b.iloc[-1]) if not pd.isna(predicted_b.iloc[-1]) else None,
        residual_today=float(residual.iloc[-1]) if not pd.isna(residual.iloc[-1]) else None,
        residual_zscore=float(resid_z.iloc[-1]) if not pd.isna(resid_z.iloc[-1]) else None,
    )
    return snap


def run_comovement_scan(ticker_a, ticker_b, cfg, window_days=None):
    snap = compute_comovement(ticker_a, ticker_b, cfg, window_days)

    print(f"\n{'='*60}")
    print(f"CO-MOVEMENT SCAN: {ticker_a}/{ticker_b}")
    print(f"{'='*60}")
    print("(This is a momentum/relative-strength signal, NOT market-neutral")
    print(" pairs trading — see the notes at the bottom before using it.)\n")

    if snap.note and snap.rolling_correlation is None:
        print(f"  {snap.note}")
        return

    print(f"  Rolling window:      {snap.window_days} trading days")
    print(f"  Rolling correlation: {snap.rolling_correlation:.3f}")
    print(f"  Rolling beta ({ticker_b} per 1% {ticker_a} move): {snap.rolling_beta:.2f}")
    print(f"  Today's {ticker_a} return:  {snap.today_return_a:+.2%}")
    print(f"  Today's {ticker_b} return:  {snap.today_return_b:+.2%}")
    if snap.expected_return_b is not None:
        print(f"  Expected {ticker_b} return (given {ticker_a}'s move + recent beta): {snap.expected_return_b:+.2%}")
    if snap.residual_today is not None:
        print(f"  Residual (actual - expected): {snap.residual_today:+.2%}")
    if snap.residual_zscore is not None:
        print(f"  Residual z-score:    {snap.residual_zscore:+.2f}")
        az = abs(snap.residual_zscore)
        if az >= 2.0:
            lagger = ticker_b if snap.residual_zscore < 0 else ticker_a
            leader = ticker_a if snap.residual_zscore < 0 else ticker_b
            print(f"\n  >>> {lagger} has notably lagged {leader}'s recent move (|z|={az:.1f}).")
            print(f"      If the co-movement pattern holds, {lagger} may be due to catch up —")
            print(f"      this is a momentum/catch-up read, not a guarantee.")
        else:
            print(f"\n  No notable lag right now — the two are moving roughly as expected together.")

    print(f"\n  Notes:")
    print(f"   - This is fundamentally different from the cointegration-based pairs trading")
    print(f"     above. It assumes these two will KEEP moving together (shared trend/narrative),")
    print(f"     not that they'll converge to some stable long-run spread.")
    print(f"   - Because of that, it is NOT market-neutral or hedged the way a true pairs trade")
    print(f"     is. If the shared narrative reverses (e.g. an AI/memory demand pullback), both")
    print(f"     legs can fall together — this strategy does not protect you from that.")
    print(f"   - Rolling correlation/beta can shift quickly, especially for a recently-listed")
    print(f"     stock (thin history) or a stock in an extreme run (like a post-spinoff repricing).")
    print(f"     Re-run this regularly rather than trusting one snapshot.")
    print(f"   - This is not financial advice.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--backtest", nargs=2, metavar=("TICKER_A", "TICKER_B"))
    parser.add_argument("--walkforward", nargs=2, metavar=("TICKER_A", "TICKER_B"),
                         help="Run a walk-forward backtest on one pair (out-of-sample, no lookahead).")
    parser.add_argument("--walkforward-all", action="store_true",
                         help="Run walk-forward backtests across every candidate pair, print a comparison table.")
    parser.add_argument("--comovement", nargs=2, metavar=("TICKER_A", "TICKER_B"),
                         help="Momentum/relative-strength scan for trending (non-cointegrated) pairs.")
    parser.add_argument("--comovement-window", type=int, default=None,
                         help="Rolling window in trading days for --comovement (default 60).")
    args = parser.parse_args()

    if args.walkforward_all:
        run_walkforward_all(CONFIG)
    elif args.comovement:
        run_comovement_scan(args.comovement[0], args.comovement[1], CONFIG, args.comovement_window)
    elif args.walkforward:
        run_walkforward_backtest(args.walkforward[0], args.walkforward[1], CONFIG)
    elif args.backtest:
        run_backtest(args.backtest[0], args.backtest[1], CONFIG)
    elif args.watch:
        run_watch(CONFIG)
    else:
        run_scan(CONFIG, verbose=args.verbose)
