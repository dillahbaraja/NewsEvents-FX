from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd

from common import DATA, ARTIFACTS, pip_size, pair_file

try:
    import vectorbt as vbt
except Exception as exc:
    raise SystemExit(
        "vectorbt is not installed or failed to import. Run: pip install vectorbt\n"
        f"Import error: {exc}"
    )


DATATEST = DATA / "datatest"
SIGNAL_TABLE = ARTIFACTS / "event_study_v3_signal_table.csv"
OUT_DIR = ARTIFACTS / "event_study_v3" / "vectorbt"
INITIAL_CASH = 10000.0
HOLDING_HOURS = 2
FREQ = "1h"


def read_price(pair):
    path = DATATEST / pair_file(pair)
    if not path.exists():
        raise FileNotFoundError(f"Missing test OHLCV file: {path}")
    d = pd.read_csv(path)
    d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    return d.dropna(subset=["Time", "Close"]).sort_values("Time").drop_duplicates("Time").set_index("Time")


def signal_to_price_index(index, event_time):
    pos = index.searchsorted(pd.Timestamp(event_time), side="left")
    if pos >= len(index):
        return None
    return pos


def build_signals(pair, price, signals):
    idx = price.index
    long_entries = pd.Series(False, index=idx)
    long_exits = pd.Series(False, index=idx)
    short_entries = pd.Series(False, index=idx)
    short_exits = pd.Series(False, index=idx)
    audit_rows = []

    subset = signals[signals["SYMBOL"].astype(str).str.upper().eq(pair)].copy()
    subset = subset.sort_values("EVENT_TIME")
    for _, row in subset.iterrows():
        entry_pos = signal_to_price_index(idx, row["EVENT_TIME"])
        if entry_pos is None:
            continue
        exit_pos = entry_pos + HOLDING_HOURS
        if exit_pos >= len(idx):
            continue

        signal = int(row["SIGNAL"])
        entry_time = idx[entry_pos]
        exit_time = idx[exit_pos]
        entry_price = float(price["Close"].iloc[entry_pos])
        exit_price = float(price["Close"].iloc[exit_pos])
        direction = 1 if signal > 0 else -1
        pnl_pips = ((exit_price - entry_price) / pip_size(pair)) * direction

        if signal > 0:
            long_entries.iloc[entry_pos] = True
            long_exits.iloc[exit_pos] = True
            side = "BUY"
        elif signal < 0:
            short_entries.iloc[entry_pos] = True
            short_exits.iloc[exit_pos] = True
            side = "SELL"
        else:
            continue

        audit_rows.append({
            "pair": pair,
            "event_time": row["EVENT_TIME"],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "event_country": row.get("COUNTRY", ""),
            "event_name": row.get("EVENT_NAME", ""),
            "source_model": row.get("SOURCE_MODEL", ""),
            "probability": row.get("PROBABILITY", np.nan),
            "signal": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pips_event_study": pnl_pips,
        })

    return long_entries, long_exits, short_entries, short_exits, pd.DataFrame(audit_rows)


def scalar_stat(stats, key, default=np.nan):
    try:
        value = stats.loc[key]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return value
    except Exception:
        return default


def run_pair(pair, signals):
    price = read_price(pair)
    le, lx, se, sx, audit = build_signals(pair, price, signals)

    pf = vbt.Portfolio.from_signals(
        close=price["Close"],
        entries=le,
        exits=lx,
        short_entries=se,
        short_exits=sx,
        init_cash=INITIAL_CASH,
        fees=0.0,
        slippage=0.0,
        freq=FREQ,
    )

    pair_dir = OUT_DIR / pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(pair_dir / "vectorbt_event_audit.csv", index=False)
    pf.value().rename("equity").to_csv(pair_dir / "vectorbt_equity_curve.csv")

    try:
        trades = pf.trades.records_readable
        trades.to_csv(pair_dir / "vectorbt_trades.csv", index=False)
    except Exception:
        trades = pd.DataFrame()
        pd.DataFrame().to_csv(pair_dir / "vectorbt_trades.csv", index=False)

    try:
        fig = pf.plot()
        fig.write_html(str(pair_dir / "vectorbt_equity_curve.html"))
    except Exception as exc:
        (pair_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")

    stats = pf.stats()
    event_pips = float(audit["pnl_pips_event_study"].sum()) if not audit.empty else 0.0
    event_trades = int(len(audit))
    event_win_rate = float((audit["pnl_pips_event_study"] > 0).mean()) if event_trades else 0.0
    gross_profit = float(audit.loc[audit["pnl_pips_event_study"] > 0, "pnl_pips_event_study"].sum()) if event_trades else 0.0
    gross_loss = float(-audit.loc[audit["pnl_pips_event_study"] < 0, "pnl_pips_event_study"].sum()) if event_trades else 0.0
    profit_factor_pips = math.inf if gross_loss == 0 and gross_profit > 0 else (gross_profit / gross_loss if gross_loss > 0 else 0.0)

    return {
        "pair": pair,
        "signals": int(len(signals[signals["SYMBOL"].astype(str).str.upper().eq(pair)])),
        "event_audit_trades": event_trades,
        "event_audit_win_rate": event_win_rate,
        "event_audit_net_pips": event_pips,
        "event_audit_profit_factor": profit_factor_pips,
        "vectorbt_total_return_pct": scalar_stat(stats, "Total Return [%]"),
        "vectorbt_benchmark_return_pct": scalar_stat(stats, "Benchmark Return [%]"),
        "vectorbt_max_drawdown_pct": scalar_stat(stats, "Max Drawdown [%]"),
        "vectorbt_total_trades": scalar_stat(stats, "Total Trades"),
        "vectorbt_win_rate_pct": scalar_stat(stats, "Win Rate [%]"),
        "vectorbt_profit_factor": scalar_stat(stats, "Profit Factor"),
        "vectorbt_expectancy": scalar_stat(stats, "Expectancy"),
        "status": "ok",
    }


def write_report(summary):
    cols = [
        "pair",
        "signals",
        "event_audit_trades",
        "event_audit_win_rate",
        "event_audit_net_pips",
        "event_audit_profit_factor",
        "vectorbt_total_return_pct",
        "vectorbt_max_drawdown_pct",
        "vectorbt_total_trades",
        "vectorbt_win_rate_pct",
        "vectorbt_profit_factor",
    ]
    lines = [
        "# VectorBT Event Study V3 Backtest",
        "",
        "Backtest ini memakai `vectorbt` untuk menguji signal table Event Study V3 pada data OHLCV H1 test.",
        "",
        f"Signal table: `{SIGNAL_TABLE}`",
        f"Holding period: `{HOLDING_HOURS}H`",
        "",
    ]
    view = summary[cols].copy()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in view.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if np.isinf(v):
                    vals.append("inf")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines += [
        "",
        f"Total event-audit net pips: `{summary['event_audit_net_pips'].sum():.1f}`",
        f"Pairs tested: `{len(summary)}`",
        "",
        "Catatan: event-audit pips dihitung langsung dari entry/exit event. Metric vectorbt adalah portfolio simulation metric dan bisa berbeda karena aturan posisi/overlap di engine vectorbt.",
    ]
    (OUT_DIR / "LAPORAN_VECTORBT_EVENT_STUDY_V3.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("[PROGRESS] START 33_vectorbt_event_study_v3_backtest.py", flush=True)
    if not SIGNAL_TABLE.exists():
        raise FileNotFoundError(f"Missing signal table: {SIGNAL_TABLE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    signals = pd.read_csv(SIGNAL_TABLE)
    signals["EVENT_TIME"] = pd.to_datetime(signals["EVENT_TIME"], errors="coerce")
    signals = signals.dropna(subset=["EVENT_TIME"]).copy()
    signals["SYMBOL"] = signals["SYMBOL"].astype(str).str.upper()
    pairs = sorted(signals["SYMBOL"].dropna().unique().tolist())
    print(f"[DATA] pairs={pairs}", flush=True)
    print(f"[DATA] signals={len(signals)}", flush=True)

    rows = []
    for i, pair in enumerate(pairs, start=1):
        print("=" * 72, flush=True)
        print(f"[VECTORBT] START pair={i}/{len(pairs)} {pair}", flush=True)
        try:
            row = run_pair(pair, signals)
        except Exception as exc:
            row = {"pair": pair, "status": "failed", "error": str(exc)}
            print(f"[VECTORBT] FAILED pair={pair} error={exc}", flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_DIR / "vectorbt_event_study_v3_summary.csv", index=False)
        print(f"[VECTORBT] DONE pair={pair} status={row.get('status')} trades={row.get('event_audit_trades', 0)}", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "vectorbt_event_study_v3_summary.csv", index=False)
    write_report(summary)
    metadata = {
        "signal_table": str(SIGNAL_TABLE),
        "out_dir": str(OUT_DIR),
        "holding_hours": HOLDING_HOURS,
        "initial_cash": INITIAL_CASH,
        "pairs": pairs,
    }
    (OUT_DIR / "vectorbt_event_study_v3_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'vectorbt_event_study_v3_summary.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_VECTORBT_EVENT_STUDY_V3.md'}", flush=True)


if __name__ == "__main__":
    main()
