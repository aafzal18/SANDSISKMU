# Pairs Trading Monitor

Scans a ~100-stock universe (grouped into ~19 economically-linked sectors),
tests every within-sector pair for genuine cointegration (not just
correlation), and alerts when a statistically real pair diverges further
than its own history says is normal.

This version runs automatically on **GitHub Actions** — no need to keep
your own laptop running or open a terminal every day. GitHub's servers
run the scan on a schedule, and it opens a GitHub Issue in your repo
whenever something alerts.

## One-time setup

### 1. Create the repo
- Go to https://github.com/new
- Name it something like `pairs-trading-monitor`
- Choose **Public** (public repos get unlimited free GitHub Actions
  minutes; private repos get 2,000 free minutes/month, which is still
  plenty for this, but public is simplest if you don't mind the code
  being visible — none of this code touches your money or accounts).
- Don't initialize with a README (we're bringing our own files).
- Click **Create repository**.

### 2. Upload the files
On the new repo's page, click **"uploading an existing file"** (or
**Add file → Upload files**). Drag in:
- `pairs_monitor.py`
- `requirements.txt`
- `.github/workflows/scan.yml` — **important:** GitHub's upload UI
  usually preserves folder structure if you drag the whole `.github`
  folder in at once. If it doesn't, see "If the folder doesn't upload
  right" below.

Commit directly to the `main` branch.

### 3. If the folder doesn't upload right
GitHub's drag-and-drop can be finicky with the hidden `.github` folder.
The reliable alternative is via `git` on your own machine:

```bash
cd path/to/where/you/downloaded/these/files
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pairs-trading-monitor.git
git push -u origin main
```

(Replace `YOUR_USERNAME` with your actual GitHub username — you'll find
the exact URL to use on your new repo's page, under the green "Code"
button.)

### 4. Confirm Actions is enabled
Go to the **Actions** tab on your repo. GitHub Actions is on by default
for new repos, but if you see a prompt to enable workflows, click to
enable it.

### 5. Do a manual test run
Still in the **Actions** tab, click **"Pairs Trading Scan"** on the left,
then **"Run workflow"** (dropdown button, top right) → **Run workflow**.
This triggers it immediately instead of waiting for the schedule — good
for confirming everything works before waiting for the automated
schedule to kick in.

It'll take a few minutes (downloading data for 100+ tickers). Click into
the running job to watch its progress live. When it finishes, check the
**Summary** page (top of that same run) for a clean markdown report —
this comes from the `GITHUB_STEP_SUMMARY` output I built into the script
specifically for this.

## How the automatic schedule works

`.github/workflows/scan.yml` is currently set to run **every hour,
Monday-Friday, 9am-9pm UTC**. That's the line:

```yaml
- cron: "0 9-21 * * 1-5"
```

To change the frequency or hours, edit that cron line. A few examples:
- Once a day at 3pm UTC: `"0 15 * * 1-5"`
- Every 30 minutes during market hours: `"*/30 13-21 * * 1-5"`
  (13-21 UTC ≈ 9am-5pm US Eastern, adjust for daylight saving)
- [crontab.guru](https://crontab.guru) is a good tool for building/checking
  cron expressions if you want something custom.

**Note:** GitHub Actions scheduled runs can be delayed by a few minutes
during high load — this is a known platform limitation, not a bug in
your setup. Fine for this use case, not fine if you needed
millisecond-precision triggers (you don't, for daily-timeframe pairs
trading).

## How alerts reach you

Every run that finds an active divergence automatically **opens a GitHub
Issue** in your repo with the pair, z-score, suggested direction, and
stats. To get notified the moment that happens:
- Go to your repo → click **Watch** (top right) → choose **Custom** →
  check **Issues**. GitHub will email you whenever a new issue opens.
- Or install the GitHub mobile app and enable notifications — you'll get
  a push notification for new issues in repos you watch.

## Checking results without waiting for an alert

- **Actions tab → any past run → Summary**: full markdown report of that
  run's statistically valid pairs, even if nothing alerted.
- **`pairs_alerts_log.csv`** in the repo root: every alert ever fired,
  automatically committed back to the repo after each run — a running
  history you can open in Excel/Sheets anytime.

## Editing your stock universe

Open `pairs_monitor.py` in the GitHub web editor (click the file, then
the pencil icon) or locally, find the `UNIVERSE` dictionary near the
top, and add/remove tickers or whole sector groups. Commit the change —
the next scheduled run (or a manual "Run workflow") will pick it up
automatically, no redeployment needed.

## Cost

This is free for public repos (GitHub Actions gives unlimited minutes to
public repos) and well within the free tier even for private repos —
each run takes a few minutes, roughly ~200 runs/month at the default
hourly-during-market-hours schedule, way under the 2,000 free
minutes/month private-repo allowance.

## Reminder

Nothing here is financial advice. This tool flags statistically unusual
divergence in historically-linked pairs — it doesn't know *why* a
divergence happened, and can't tell you whether it's a temporary
mispricing or a permanent repricing. That judgment is still yours.

## Walk-forward backtest — does this pair actually make money historically?

Everything above tells you a pair *looks* statistically real right now.
It doesn't tell you whether trading on that signal would have actually
made money in the past. That's what this does — and it does it the
honest way: **walk-forward**, meaning it only ever uses past data to
decide how to trade, then tests that decision on data it hasn't seen
yet, then rolls forward and repeats. This avoids the classic backtesting
trap of fitting a strategy to the exact data you're "testing" it on,
which always looks great and usually doesn't hold up in real trading.

```
python3 pairs_monitor.py --walkforward JPM BAC
```

**What it does:** splits your pair's price history into rolling blocks —
a ~1 year "formation" window to fit the hedge ratio and test
cointegration, followed by a ~1 quarter "trading" window where it
actually simulates entries/exits using only what was known at the start
of that quarter. Rolls forward through your whole history this way.

**What the output means:**
- **Observations** — total daily price points used
- **Eligible blocks** — how many of the rolling formation windows passed
  the cointegration test (this is itself useful info: if it's low, the
  relationship isn't stable over time, and any live signal from this
  pair deserves extra skepticism)
- **Completed trades** — how many full entry-to-exit trades happened
  across all eligible windows
- **Total return** — compounded return of the strategy over the whole
  backtest period, net of estimated transaction costs
- **Annualized volatility** — how bumpy the ride was
- **Sharpe ratio** — return per unit of risk taken (roughly: above 1 is
  good, above 2 is very good, negative means it lost money relative to
  its volatility)
- **Max drawdown** — the worst peak-to-trough decline the strategy would
  have experienced — useful for gut-checking whether you could actually
  stomach holding through the worst stretch
- **Win rate** — percentage of completed trades that were profitable

**Important honesty note built into the tool itself:** this treats the
day-to-day change in the dollar-neutral spread as the position's return,
which is the standard simplification used in most academic pairs-trading
backtests — it does not model financing/borrow costs or the friction of
periodically rebalancing the hedge ratio. Real-world results would
likely run somewhat below what the backtest shows. Treat this as a
"is this idea worth investigating further," not "here's my exact
expected return."

## Comparing many pairs at once: --walkforward-all

Instead of backtesting one pair at a time, this runs the walk-forward
backtest across every within-sector candidate pair in your universe and
prints a single sortable comparison table:

```
python3 pairs_monitor.py --walkforward-all
```

```
Pair        Period                     Obs   Elig.  Trades  TotalRet   AnnVol  Sharpe    MaxDD  WinRate
-----------------------------------------------------------------------------------------------------
JPM/BAC     2020-08-17 to 2025-08-14  1258   12/16       31    +18.4%    4.2%    1.31    -5.1%      68%
...
```

Sorted by Sharpe ratio, highest first. Only pairs with at least one
eligible (cointegrated) block appear — a pair that was never tradeable
across its whole history simply doesn't show up, which is itself useful
information (it means don't bother watching that pair for live alerts
either).

**Heads up on runtime:** this downloads ~5 years of data for all ~104
tickers once, then runs the walk-forward simulation across all ~257
candidate pairs. The download is the slow part (a couple minutes); the
actual backtesting math is fast. Budget several minutes total.

## Co-movement scanner — for pairs that trend together, not mean-revert (e.g. MU/SNDK)

Some pairs genuinely move together but will never pass a cointegration
test — usually because one is a recently spun-off/IPO'd company with too
little history, or because both are riding the same secular trend (an
AI/memory supercycle, a sector-wide re-rating) rather than sitting at a
stable equilibrium. Forcing the cointegration framework onto a pair like
that doesn't work, and shouldn't — but there's a real, different
strategy for this case: momentum/relative-strength co-movement.

```
python3 pairs_monitor.py --comovement MU SNDK
python3 pairs_monitor.py --comovement MU SNDK --comovement-window 30
```

**What it measures:** on a short rolling window (60 trading days by
default — much shorter than the multi-year window cointegration wants),
how tightly do the two stocks' *daily returns* move together
(correlation + beta)? Then: given today's move in stock A and that
recent beta, did stock B move as much as expected, or has it lagged?
A large lag (flagged via a z-score on the "residual") is the signal —
the idea being if the co-movement pattern holds, the laggard may catch
up.

**Read this before using it — this is NOT the same risk profile as
everything else in this tool:**
- Classic pairs trading (cointegration-based, everything else in this
  repo) is market-neutral — long one leg, short the other, so a broad
  market or sector move doesn't hurt you either way.
- This co-movement scanner is **not** market-neutral. It's a bet that
  the shared trend continues. If the underlying narrative reverses (an
  AI/memory demand pullback, for example), both legs can fall together
  and this strategy does nothing to protect you from that.
- Rolling correlation/beta over a short window can shift fast,
  especially right after a spin-off or IPO when there's limited history
  to smooth things out. Re-run this regularly — a snapshot from a week
  ago may not reflect today's relationship.

If a pair errors out with "too little history" (common for a stock
that's only been trading independently for a year or so), try a shorter
window: `--comovement-window 20` or `30`.
