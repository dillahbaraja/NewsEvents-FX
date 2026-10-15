from common import *

import math

import backtrader as bt

try:
    import vectorbt as vbt
except Exception as exc:
    raise SystemExit(f"vectorbt is required. Import error: {exc}")


DATATEST = DATA / "datatest"
SIGNAL_TABLE = ARTIFACTS / "catboost_block_inverse_mt5_signal_table.csv"
OUT_DIR = ARTIFACTS / "catboost_block_inverse" / "backtrader_vectorbt_test"

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-08-31 23:59:59")
INITIAL_CASH = 10000.0
HOLDING_BARS = 1
FREQ = "1h"


class PandasH1(bt.feeds.PandasData):
    lines = ("volume",)
    params = (
        ("datetime", None),
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
        ("openinterest", None),
    )


class SequentialSignalStrategy(bt.Strategy):
    params = (("signals", None), ("pair", ""))

    def __init__(self):
        self.signal_map = self.p.signals if self.p.signals is not None else {}
        self.active_trade = None
        self.orders_out = []

    def _dt(self):
        return pd.Timestamp(bt.num2date(self.datas[0].datetime[0])).tz_localize(None)

    def next(self):
        now = self._dt()
        close_price = float(self.datas[0].close[0])
        if self.active_trade is not None and now >= self.active_trade["exit_time"]:
            trade = dict(self.active_trade)
            trade["exit_time_actual"] = now
            trade["exit_price"] = close_price
            direction = 1.0 if trade["signal"] == "BUY" else -1.0
            trade["pnl_pips"] = ((close_price - trade["entry_price"]) / pip_size(self.p.pair)) * direction
            self.orders_out.append(trade)
            self.active_trade = None

        if self.active_trade is not None:
            return

        rows = self.signal_map.get(now)
        if not rows:
            return
        row = rows[0]
        signal = "BUY" if int(row["SIGNAL"]) > 0 else "SELL"
        self.active_trade = {
            "model": row.get("MODEL", "catboost_block_inverse"),
            "pair": self.p.pair,
            "runtime": "backtrader_sequential_close_1h",
            "event_time": row.get("EVENT_TIME"),
            "entry_time": now,
            "exit_time": now + pd.Timedelta(hours=HOLDING_BARS),
            "country": row.get("COUNTRY", ""),
            "event_name": row.get("EVENT_NAME", ""),
            "signal": signal,
            "entry_price": close_price,
            "probability": row.get("PROBABILITY", math.nan),
            "skipped_same_bar_signals": max(len(rows) - 1, 0),
        }

    def stop(self):
        if self.active_trade is not None:
            now = self._dt()
            close_price = float(self.datas[0].close[0])
            trade = dict(self.active_trade)
            trade["exit_time_actual"] = now
            trade["exit_price"] = close_price
            direction = 1.0 if trade["signal"] == "BUY" else -1.0
            trade["pnl_pips"] = ((close_price - trade["entry_price"]) / pip_size(self.p.pair)) * direction
            trade["forced_close"] = True
            self.orders_out.append(trade)


def read_price(pair):
    path = DATATEST / pair_file(pair)
    if not path.exists():
        return None
    d = read_prices(path)
    d = d[(d["time"] >= START) & (d["time"] <= END)].copy()
    return d.set_index("time") if not d.empty else None


def load_signals():
    if not SIGNAL_TABLE.exists():
        raise FileNotFoundError(f"Missing signal table: {SIGNAL_TABLE}")
    d = pd.read_csv(SIGNAL_TABLE)
    d["EVENT_TIME"] = pd.to_datetime(d["EVENT_TIME"], errors="coerce")
    d = d.dropna(subset=["EVENT_TIME"]).copy()
    d["SYMBOL"] = d["SYMBOL"].astype(str).str.upper()
    d = d[(d["EVENT_TIME"] >= START) & (d["EVENT_TIME"] <= END)].copy()
    d["SIGNAL"] = pd.to_numeric(d["SIGNAL"], errors="coerce").fillna(0).astype(int)
    d = d[d["SIGNAL"].isin([1, -1])].copy()
    return d


def max_drawdown(values):
    if len(values) == 0:
        return 0.0
    curve = pd.Series(values, dtype=float).cumsum()
    return float((curve.cummax() - curve).max())


def summarize_pnl(engine, pair, pnl, trades, buy_count, sell_count, status="ok", skipped=0):
    pnl = pd.Series(pnl, dtype=float).dropna()
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    net = float(pnl.sum())
    n = int(trades)
    return {
        "model": "catboost_block_inverse",
        "engine": engine,
        "pair": pair,
        "status": status,
        "trades": n,
        "buy_count": int(buy_count),
        "sell_count": int(sell_count),
        "net_pips": net,
        "avg_pips": net / n if n else 0.0,
        "winrate": float((pnl > 0).mean()) if n else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else (math.inf if gross_profit > 0 else 0.0),
        "max_drawdown_pips": max_drawdown(pnl),
        "skipped_same_bar_signals": int(skipped),
    }


def signal_map(signals):
    out = {}
    for _, row in signals.sort_values("EVENT_TIME").iterrows():
        out.setdefault(pd.Timestamp(row["EVENT_TIME"]).floor("h"), []).append(row.to_dict())
    return out


def run_backtrader_pair(pair, signals):
    price = read_price(pair)
    if price is None:
        return pd.DataFrame(), summarize_pnl("backtrader", pair, [], 0, 0, 0, "missing_price")
    subset = signals[signals["SYMBOL"].eq(pair)].copy()
    if subset.empty:
        return pd.DataFrame(), summarize_pnl("backtrader", pair, [], 0, 0, 0, "no_signals")
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.adddata(PandasH1(dataname=price))
    cerebro.addstrategy(SequentialSignalStrategy, signals=signal_map(subset), pair=pair)
    result = cerebro.run()
    orders = pd.DataFrame(result[0].orders_out)
    if orders.empty:
        return orders, summarize_pnl("backtrader", pair, [], 0, 0, 0, "no_trades")
    return orders, summarize_pnl(
        "backtrader",
        pair,
        orders["pnl_pips"],
        len(orders),
        int(orders["signal"].eq("BUY").sum()),
        int(orders["signal"].eq("SELL").sum()),
        "ok",
        int(pd.to_numeric(orders.get("skipped_same_bar_signals", 0), errors="coerce").fillna(0).sum()),
    )


def run_vectorbt_pair(pair, signals):
    price = read_price(pair)
    if price is None:
        return pd.DataFrame(), summarize_pnl("vectorbt_event_audit", pair, [], 0, 0, 0, "missing_price"), {}
    subset = signals[signals["SYMBOL"].eq(pair)].copy().sort_values("EVENT_TIME")
    if subset.empty:
        return pd.DataFrame(), summarize_pnl("vectorbt_event_audit", pair, [], 0, 0, 0, "no_signals"), {}

    idx = price.index
    long_entries = pd.Series(False, index=idx)
    long_exits = pd.Series(False, index=idx)
    short_entries = pd.Series(False, index=idx)
    short_exits = pd.Series(False, index=idx)
    audit = []
    for _, row in subset.iterrows():
        entry_pos = idx.searchsorted(pd.Timestamp(row["EVENT_TIME"]), side="left")
        if entry_pos >= len(idx):
            continue
        exit_pos = entry_pos + HOLDING_BARS
        if exit_pos >= len(idx):
            continue
        entry_time = idx[entry_pos]
        exit_time = idx[exit_pos]
        entry = float(price["Close"].iloc[entry_pos])
        exitp = float(price["Close"].iloc[exit_pos])
        direction = 1 if int(row["SIGNAL"]) > 0 else -1
        pnl = ((exitp - entry) / pip_size(pair)) * direction
        if direction > 0:
            long_entries.iloc[entry_pos] = True
            long_exits.iloc[exit_pos] = True
            side = "BUY"
        else:
            short_entries.iloc[entry_pos] = True
            short_exits.iloc[exit_pos] = True
            side = "SELL"
        audit.append({
            "pair": pair,
            "event_time": row["EVENT_TIME"],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "country": row.get("COUNTRY", ""),
            "event_name": row.get("EVENT_NAME", ""),
            "signal": side,
            "entry_price": entry,
            "exit_price": exitp,
            "pnl_pips": pnl,
        })

    audit = pd.DataFrame(audit)
    pf = vbt.Portfolio.from_signals(
        close=price["Close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=INITIAL_CASH,
        fees=0.0,
        slippage=0.0,
        freq=FREQ,
    )
    pair_dir = OUT_DIR / "vectorbt" / pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(pair_dir / "vectorbt_event_audit.csv", index=False)
    pf.value().rename("equity").to_csv(pair_dir / "vectorbt_equity_curve.csv")
    try:
        trades = pf.trades.records_readable
        trades.to_csv(pair_dir / "vectorbt_trades.csv", index=False)
    except Exception:
        trades = pd.DataFrame()
    stats = pf.stats()
    summary = summarize_pnl(
        "vectorbt_event_audit",
        pair,
        audit["pnl_pips"] if not audit.empty else [],
        len(audit),
        int(audit["signal"].eq("BUY").sum()) if not audit.empty else 0,
        int(audit["signal"].eq("SELL").sum()) if not audit.empty else 0,
    )
    vbt_stats = {
        "vectorbt_total_return_pct": stats.get("Total Return [%]", np.nan),
        "vectorbt_max_drawdown_pct": stats.get("Max Drawdown [%]", np.nan),
        "vectorbt_total_trades": stats.get("Total Trades", np.nan),
        "vectorbt_win_rate_pct": stats.get("Win Rate [%]", np.nan),
        "vectorbt_profit_factor": stats.get("Profit Factor", np.nan),
    }
    return audit, summary, vbt_stats


def md_table(df, floatfmt=".4f"):
    if df.empty:
        return "_Tidak ada data._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def aggregate(summary, engine):
    g = summary[summary["engine"].eq(engine)]
    if g.empty:
        return {}
    gross_profit = float(g["gross_profit"].sum())
    gross_loss = float(g["gross_loss"].sum())
    net = float(g["net_pips"].sum())
    trades = int(g["trades"].sum())
    return {
        "model": "catboost_block_inverse",
        "engine": engine,
        "pair": "ALL",
        "status": "ok",
        "trades": trades,
        "buy_count": int(g["buy_count"].sum()),
        "sell_count": int(g["sell_count"].sum()),
        "net_pips": net,
        "avg_pips": net / trades if trades else 0.0,
        "winrate": np.nan,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf,
        "max_drawdown_pips": np.nan,
        "skipped_same_bar_signals": int(g.get("skipped_same_bar_signals", pd.Series(dtype=float)).fillna(0).sum()),
    }


def main():
    print("[PROGRESS] START 51_test_catboost_block_inverse_backtrader_vectorbt.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signals = load_signals()
    pairs = sorted(signals["SYMBOL"].dropna().unique().tolist())
    print(f"[DATA] signals={len(signals)} pairs={pairs}", flush=True)

    all_bt_orders = []
    all_vbt_audit = []
    summary_rows = []
    vbt_rows = []
    for i, pair in enumerate(pairs, start=1):
        print(f"[BACKTRADER] START pair={i}/{len(pairs)} {pair}", flush=True)
        bt_orders, bt_summary = run_backtrader_pair(pair, signals)
        summary_rows.append(bt_summary)
        if not bt_orders.empty:
            all_bt_orders.append(bt_orders)
        print(f"[BACKTRADER] DONE {pair} trades={bt_summary['trades']} net={bt_summary['net_pips']:.1f} pf={bt_summary['profit_factor']:.4f}", flush=True)

        print(f"[VECTORBT] START pair={i}/{len(pairs)} {pair}", flush=True)
        vbt_audit, vbt_summary, vbt_stats = run_vectorbt_pair(pair, signals)
        summary_rows.append(vbt_summary)
        vbt_row = dict(vbt_summary)
        vbt_row.update(vbt_stats)
        vbt_rows.append(vbt_row)
        if not vbt_audit.empty:
            all_vbt_audit.append(vbt_audit)
        print(f"[VECTORBT] DONE {pair} trades={vbt_summary['trades']} net={vbt_summary['net_pips']:.1f} pf={vbt_summary['profit_factor']:.4f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary = pd.concat([summary, pd.DataFrame([aggregate(summary, "backtrader"), aggregate(summary, "vectorbt_event_audit")])], ignore_index=True)
    vbt_stats = pd.DataFrame(vbt_rows)
    bt_orders = pd.concat(all_bt_orders, ignore_index=True) if all_bt_orders else pd.DataFrame()
    vbt_audit = pd.concat(all_vbt_audit, ignore_index=True) if all_vbt_audit else pd.DataFrame()

    summary.to_csv(OUT_DIR / "catboost_block_inverse_backtrader_vectorbt_summary.csv", index=False)
    vbt_stats.to_csv(OUT_DIR / "catboost_block_inverse_vectorbt_portfolio_stats.csv", index=False)
    bt_orders.to_csv(OUT_DIR / "catboost_block_inverse_backtrader_orders.csv", index=False)
    vbt_audit.to_csv(OUT_DIR / "catboost_block_inverse_vectorbt_event_audit.csv", index=False)

    report = "\n".join([
        "# CatBoost Block/Inverse Backtrader and VectorBT Test",
        "",
        f"Signal table: `{SIGNAL_TABLE}`",
        "Period: `2024-01-01` to `2026-08-31`.",
        "",
        "## Summary Per Engine and Pair",
        "",
        md_table(summary[["model", "engine", "pair", "trades", "buy_count", "sell_count", "net_pips", "avg_pips", "winrate", "profit_factor", "max_drawdown_pips", "skipped_same_bar_signals"]], ".4f"),
        "",
        "## VectorBT Portfolio Stats",
        "",
        md_table(vbt_stats[["model", "pair", "trades", "net_pips", "profit_factor", "vectorbt_total_return_pct", "vectorbt_max_drawdown_pct", "vectorbt_total_trades", "vectorbt_win_rate_pct", "vectorbt_profit_factor"]], ".4f"),
        "",
        "Notes:",
        "",
        "- `vectorbt_event_audit` calculates direct event-entry/exit pips for every signal.",
        "- VectorBT portfolio stats can differ because overlapping entries on the same pair are represented as a portfolio signal series.",
        "- Backtrader sequential allows one active position per pair and can skip clustered same-bar signals.",
    ]) + "\n"
    (OUT_DIR / "LAPORAN_CATBOOST_BLOCK_INVERSE_BACKTRADER_VECTORBT.md").write_text(report, encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_CATBOOST_BLOCK_INVERSE_BACKTRADER_VECTORBT.md'}", flush=True)
    print("[PROGRESS] DONE 51_test_catboost_block_inverse_backtrader_vectorbt.py", flush=True)


if __name__ == "__main__":
    main()
