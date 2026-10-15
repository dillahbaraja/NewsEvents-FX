"""VectorBT portfolio validation for pair-specific event-character signals."""
from pathlib import Path
import math
import os

import numpy as np
import pandas as pd

from common import ARTIFACTS, DATA, pair_file, pip_size, progress_done, progress_start, progress_step, read_prices

try:
    import vectorbt as vbt
except Exception as exc:
    raise SystemExit(f"vectorbt import failed: {exc}")

SCRIPT = Path(__file__).name
DATATEST = DATA / "datatest"
OUT = ARTIFACTS / "event_character_pair_specific" / "engine_validation"
SIGNALS = Path(os.environ.get("PAIR_ENGINE_SIGNALS_FILE", str(OUT / "pair_specific_event_character_signals_oos.csv")))
RUN_NAME = os.environ.get("PAIR_ENGINE_RUN_NAME", "raw")
INITIAL_CASH = 10000.0


def safe_pf(values):
    s = pd.Series(values).dropna()
    gp = float(s[s > 0].sum())
    gl = float(-s[s < 0].sum())
    return math.inf if gl == 0 and gp > 0 else (gp / gl if gl > 0 else 0.0)


def read_price(pair):
    d = read_prices(DATATEST / pair_file(pair))
    return d.set_index("time").sort_index()


def idx_pos(index, t):
    pos = index.searchsorted(pd.Timestamp(t), side="left")
    return None if pos >= len(index) else int(pos)


def run_pair(pair, signals):
    price = read_price(pair)
    idx = price.index
    long_entries = pd.Series(False, index=idx)
    long_exits = pd.Series(False, index=idx)
    short_entries = pd.Series(False, index=idx)
    short_exits = pd.Series(False, index=idx)
    audit = []
    for _, row in signals.sort_values("entry_time").iterrows():
        ep = idx_pos(idx, row["entry_time"])
        if ep is None:
            continue
        xp = ep + int(row["holding_bars"])
        if xp >= len(idx):
            continue
        entry_price = float(price["Close"].iloc[ep])
        exit_price = float(price["Close"].iloc[xp])
        side = row["side"]
        direction = 1 if side == "BUY" else -1
        pnl_pips = ((exit_price - entry_price) / pip_size(pair)) * direction
        if side == "BUY":
            long_entries.iloc[ep] = True
            long_exits.iloc[xp] = True
        else:
            short_entries.iloc[ep] = True
            short_exits.iloc[xp] = True
        audit.append({
            "pair": pair,
            "model": row["model"],
            "event_time": row["event_time"],
            "entry_time": idx[ep],
            "exit_time": idx[xp],
            "side": side,
            "event_name": row.get("Name", ""),
            "country": row.get("CountryCode", ""),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pips_event_audit": pnl_pips,
            "holding_bars": int(row["holding_bars"]),
        })

    pf = vbt.Portfolio.from_signals(
        close=price["Close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=INITIAL_CASH,
        fees=0.0,
        slippage=0.0,
        freq="1h",
    )
    pair_dir = OUT / f"vectorbt_{RUN_NAME}" / pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(pair_dir / "vectorbt_event_audit.csv", index=False)
    pf.value().rename("equity").to_csv(pair_dir / "vectorbt_equity_curve.csv")
    try:
        pf.trades.records_readable.to_csv(pair_dir / "vectorbt_trades.csv", index=False)
    except Exception:
        pd.DataFrame().to_csv(pair_dir / "vectorbt_trades.csv", index=False)
    stats = pf.stats()
    pips = audit_df["pnl_pips_event_audit"] if not audit_df.empty else pd.Series(dtype=float)
    return {
        "pair": pair,
        "model": ",".join(sorted(signals["model"].dropna().unique())),
        "signals": int(len(signals)),
        "event_audit_trades": int(len(audit_df)),
        "event_audit_net_pips": float(pips.sum()) if len(pips) else 0.0,
        "event_audit_profit_factor": safe_pf(pips),
        "event_audit_winrate": float((pips > 0).mean()) if len(pips) else 0.0,
        "vectorbt_total_return_pct": stats.get("Total Return [%]", np.nan),
        "vectorbt_max_drawdown_pct": stats.get("Max Drawdown [%]", np.nan),
        "vectorbt_total_trades": stats.get("Total Trades", np.nan),
        "vectorbt_win_rate_pct": stats.get("Win Rate [%]", np.nan),
        "vectorbt_profit_factor": stats.get("Profit Factor", np.nan),
    }


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    signals = pd.read_csv(SIGNALS, parse_dates=["entry_time", "event_time"], low_memory=False)
    progress_step("VECTORBT_CONFIG", signals_file=str(SIGNALS), run_name=RUN_NAME, rows=len(signals))
    rows = []
    pairs = sorted(signals["pair"].dropna().unique())
    for i, pair in enumerate(pairs, 1):
        progress_step("VECTORBT_PAIR_START", i, len(pairs), pair=pair)
        row = run_pair(pair, signals[signals["pair"].eq(pair)].copy())
        rows.append(row)
        progress_step("VECTORBT_PAIR_DONE", i, len(pairs), pair=pair, net_pips=f"{row['event_audit_net_pips']:.1f}", pf=f"{row['event_audit_profit_factor']:.4f}")
    summary = pd.DataFrame(rows).sort_values("event_audit_net_pips", ascending=False)
    summary.to_csv(OUT / f"vectorbt_pair_specific_summary_{RUN_NAME}.csv", index=False)
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
