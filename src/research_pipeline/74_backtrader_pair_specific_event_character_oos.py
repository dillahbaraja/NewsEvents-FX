"""Sequential Backtrader-style validation for pair-specific event-character signals."""
from pathlib import Path
import math
import os

import pandas as pd

from common import ARTIFACTS, DATA, pair_file, pip_size, progress_done, progress_start, progress_step, read_prices

SCRIPT = Path(__file__).name
DATATEST = DATA / "datatest"
OUT = ARTIFACTS / "event_character_pair_specific" / "engine_validation"
SIGNALS = Path(os.environ.get("PAIR_ENGINE_SIGNALS_FILE", str(OUT / "pair_specific_event_character_signals_oos.csv")))
RUN_NAME = os.environ.get("PAIR_ENGINE_RUN_NAME", "raw")


def safe_pf(values):
    s = pd.Series(values).dropna()
    gp = float(s[s > 0].sum())
    gl = float(-s[s < 0].sum())
    return math.inf if gl == 0 and gp > 0 else (gp / gl if gl > 0 else 0.0)


def max_dd(values):
    eq = pd.Series(values, dtype=float).cumsum()
    if eq.empty:
        return 0.0
    return float((eq.cummax() - eq).max())


def run_pair(pair, signals):
    price = read_prices(DATATEST / pair_file(pair)).set_index("time").sort_index()
    signals = signals.sort_values("entry_time")
    active_until = None
    orders = []
    skipped = 0
    for _, row in signals.iterrows():
        entry_pos = price.index.searchsorted(pd.Timestamp(row["entry_time"]), side="left")
        if entry_pos >= len(price):
            continue
        entry_time = price.index[entry_pos]
        if active_until is not None and entry_time < active_until:
            skipped += 1
            continue
        exit_pos = entry_pos + int(row["holding_bars"])
        if exit_pos >= len(price):
            continue
        exit_time = price.index[exit_pos]
        entry_price = float(price["Close"].iloc[entry_pos])
        exit_price = float(price["Close"].iloc[exit_pos])
        direction = 1 if row["side"] == "BUY" else -1
        pnl = ((exit_price - entry_price) / pip_size(pair)) * direction
        active_until = exit_time
        orders.append({
            "pair": pair,
            "model": row["model"],
            "event_time": row["event_time"],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": row["side"],
            "event_name": row.get("Name", ""),
            "country": row.get("CountryCode", ""),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pips": pnl,
            "holding_bars": int(row["holding_bars"]),
        })
    orders_df = pd.DataFrame(orders)
    pair_dir = OUT / f"backtrader_{RUN_NAME}" / pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    orders_df.to_csv(pair_dir / "backtrader_orders.csv", index=False)
    pips = orders_df["pnl_pips"] if not orders_df.empty else pd.Series(dtype=float)
    return {
        "pair": pair,
        "model": ",".join(sorted(signals["model"].dropna().unique())),
        "signals": int(len(signals)),
        "trades": int(len(orders_df)),
        "skipped_overlap": int(skipped),
        "net_pips": float(pips.sum()) if len(pips) else 0.0,
        "profit_factor": safe_pf(pips),
        "winrate": float((pips > 0).mean()) if len(pips) else 0.0,
        "max_drawdown_pips": max_dd(pips),
    }


def main():
    started = progress_start(SCRIPT)
    signals = pd.read_csv(SIGNALS, parse_dates=["entry_time", "event_time"], low_memory=False)
    progress_step("BACKTRADER_CONFIG", signals_file=str(SIGNALS), run_name=RUN_NAME, rows=len(signals))
    rows = []
    pairs = sorted(signals["pair"].dropna().unique())
    for i, pair in enumerate(pairs, 1):
        progress_step("BACKTRADER_PAIR_START", i, len(pairs), pair=pair)
        row = run_pair(pair, signals[signals["pair"].eq(pair)].copy())
        rows.append(row)
        progress_step("BACKTRADER_PAIR_DONE", i, len(pairs), pair=pair, trades=row["trades"], net_pips=f"{row['net_pips']:.1f}", pf=f"{row['profit_factor']:.4f}")
    pd.DataFrame(rows).sort_values("net_pips", ascending=False).to_csv(OUT / f"backtrader_pair_specific_summary_{RUN_NAME}.csv", index=False)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
