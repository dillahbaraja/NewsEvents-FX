from common import *

import math
import os

import backtrader as bt


DATATEST = DATA / "datatest"
PHASE7_DIR = Path(os.environ.get("PHASE7_POLICY_DIR", str(ARTIFACTS / "phase7_selected_two_stage_policy")))
SIGNALS_IN = PHASE7_DIR / "selected_two_stage_policy_orders.csv"
PHASE7_SUMMARY_IN = PHASE7_DIR / "selected_two_stage_policy_summary.csv"
OUT_DIR = Path(os.environ.get("PHASE7_BACKTRADER_OUT_DIR", str(ARTIFACTS / "backtrader_phase7_selected_policy_comparison")))

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-08-31 23:59:59")
INITIAL_CASH = 10000.0


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


class Phase7SequentialStrategy(bt.Strategy):
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
        self.active_trade = {
            "model": row.get("model", ""),
            "pair": self.p.pair,
            "runtime": "backtrader_phase7_sequential_close",
            "event_time": row.get("event_time"),
            "entry_time": now,
            "exit_time": now + pd.Timedelta(hours=1),
            "country": row.get("CountryCode", ""),
            "event_name": row.get("Name", ""),
            "signal": row.get("final_signal"),
            "entry_price": close_price,
            "prob_trade": row.get("prob_trade", math.nan),
            "prob_sell": row.get("prob_sell", math.nan),
            "prob_buy": row.get("prob_buy", math.nan),
            "direction_confidence": row.get("direction_confidence", math.nan),
            "policy": row.get("policy", ""),
            "selection_status": row.get("selection_status", ""),
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
    data = read_prices(path)
    data = data[(data["time"] >= START) & (data["time"] <= END)].copy()
    if data.empty:
        return None
    return data.set_index("time")


def max_drawdown_pips(values):
    if len(values) == 0:
        return 0.0
    curve = pd.Series(values, dtype=float).cumsum()
    drawdown = curve.cummax() - curve
    return float(drawdown.max()) if len(drawdown) else 0.0


def summarize_orders(pair, orders, status):
    if orders.empty:
        return {
            "pair": pair,
            "runtime": "backtrader_phase7_sequential_close",
            "status": status,
            "trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "win_rate": 0.0,
            "net_pips": 0.0,
            "avg_pips_per_trade": 0.0,
            "gross_profit_pips": 0.0,
            "gross_loss_pips": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pips": 0.0,
            "skipped_same_bar_signals": 0,
        }
    pnl = pd.to_numeric(orders["pnl_pips"], errors="coerce").fillna(0.0)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    net = float(pnl.sum())
    return {
        "pair": pair,
        "runtime": "backtrader_phase7_sequential_close",
        "status": status,
        "trades": int(len(orders)),
        "buy_trades": int(orders["signal"].eq("BUY").sum()),
        "sell_trades": int(orders["signal"].eq("SELL").sum()),
        "win_rate": float((pnl > 0).mean()),
        "net_pips": net,
        "avg_pips_per_trade": net / len(orders),
        "gross_profit_pips": gross_profit,
        "gross_loss_pips": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else (math.inf if gross_profit > 0 else 0.0),
        "max_drawdown_pips": max_drawdown_pips(pnl),
        "skipped_same_bar_signals": int(pd.to_numeric(orders.get("skipped_same_bar_signals", 0), errors="coerce").fillna(0).sum()),
    }


def make_signal_map(signals):
    out = {}
    for _, row in signals.sort_values("time").iterrows():
        out.setdefault(pd.Timestamp(row["time"]), []).append(row.to_dict())
    return out


def run_one(pair, signals):
    price = read_price(pair)
    if price is None:
        return pd.DataFrame(), summarize_orders(pair, pd.DataFrame(), "missing_price")
    active = signals[signals["final_signal"].isin(["BUY", "SELL"])].copy()
    if active.empty:
        return pd.DataFrame(), summarize_orders(pair, pd.DataFrame(), "no_trade_signals")

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.adddata(PandasH1(dataname=price))
    cerebro.addstrategy(Phase7SequentialStrategy, signals=make_signal_map(active), pair=pair)
    result = cerebro.run()
    orders = pd.DataFrame(result[0].orders_out)
    return orders, summarize_orders(pair, orders, "ok")


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


def aggregate_summary(summary):
    gross_profit = float(summary["gross_profit_pips"].sum())
    gross_loss = float(summary["gross_loss_pips"].sum())
    net = float(summary["net_pips"].sum())
    trades = int(summary["trades"].sum())
    return pd.DataFrame([{
        "engine": "backtrader_phase7_sequential_close",
        "pairs": int(summary["pair"].nunique()),
        "positive_pairs": int((summary["net_pips"] > 0).sum()),
        "trades": trades,
        "net_pips": net,
        "avg_pips_per_trade": net / trades if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf,
        "gross_profit_pips": gross_profit,
        "gross_loss_pips": gross_loss,
    }])


def main():
    print("[PROGRESS] START 48_backtrader_compare_phase7_selected_policy.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    signals = pd.read_csv(SIGNALS_IN, low_memory=False)
    signals["event_time"] = pd.to_datetime(signals["event_time"], errors="coerce")
    signals = signals[signals["split"].eq("oos_2024_2026")].copy()
    signals = signals[signals["final_signal"].isin(["BUY", "SELL"])].copy()
    signals = signals[(signals["event_time"] >= START) & (signals["event_time"] <= END)].copy()
    signals["time"] = signals["event_time"].dt.floor("h")
    print(f"[DATA] phase7_oos_signals={len(signals)} pairs={sorted(signals['pair'].dropna().unique())}", flush=True)

    all_orders = []
    summary_rows = []
    for idx, pair in enumerate(sorted(signals["pair"].dropna().unique()), start=1):
        pair_signals = signals[signals["pair"].eq(pair)].copy()
        print(f"[BACKTRADER] START pair={idx}/{signals['pair'].nunique()} {pair} signals={len(pair_signals)}", flush=True)
        orders, row = run_one(pair, pair_signals)
        summary_rows.append(row)
        if not orders.empty:
            all_orders.append(orders)
        print(f"[BACKTRADER] DONE {pair} trades={row['trades']} net_pips={row['net_pips']:.1f} pf={row['profit_factor']:.4f}", flush=True)

    orders = pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values("net_pips", ascending=False)
    aggregate = aggregate_summary(summary)

    phase7_eval = pd.read_csv(PHASE7_SUMMARY_IN)
    phase7_eval = phase7_eval[phase7_eval["split"].eq("oos_2024_2026")].copy()
    phase7_eval = phase7_eval.rename(columns={
        "trade_count": "trades",
        "avg_pips": "avg_pips_per_trade",
        "winrate": "win_rate",
    })
    phase7_eval["engine"] = "phase7_vectorized_event_eval"
    phase7_total = phase7_eval[phase7_eval["pair"].eq("ALL")].copy()
    phase7_pairs = phase7_eval[~phase7_eval["pair"].eq("ALL")].copy().sort_values("net_pips", ascending=False)

    report_lines = [
        "# Backtrader vs Phase 7 Vectorized OOS Comparison",
        "",
        "Laporan ini menggantikan laporan Backtrader/VectorBT lama untuk konteks Phase 7 terbaru.",
        "",
        "Sumber sinyal: `phase7_selected_two_stage_policy/selected_two_stage_policy_orders.csv`.",
        "Periode OOS: `2024-01-01` sampai `2026-08-31`.",
        "",
        "Catatan metodologi:",
        "",
        "- `phase7_vectorized_event_eval` adalah evaluasi event-level/vectorized dari pipeline Phase 7.",
        "- `backtrader_phase7_sequential_close` menjalankan sinyal yang sama secara sequential: satu posisi aktif per pair, entry pada close candle sinyal, exit 1H setelah entry.",
        "- Perbedaan trade count normal terjadi jika banyak event berada pada candle yang sama atau posisi masih aktif.",
        "",
        "## Ringkasan Engine",
        "",
        md_table(pd.concat([
            phase7_total[["engine", "trades", "net_pips", "avg_pips_per_trade", "win_rate", "profit_factor"]],
            aggregate[["engine", "trades", "net_pips", "avg_pips_per_trade", "profit_factor"]],
        ], ignore_index=True), ".4f"),
        "",
        "## Phase 7 OOS Per Pair",
        "",
        md_table(phase7_pairs[["pair", "selected_model", "trades", "buy_count", "sell_count", "net_pips", "avg_pips_per_trade", "win_rate", "profit_factor"]], ".4f"),
        "",
        "## Backtrader Per Pair",
        "",
        md_table(summary[["pair", "trades", "buy_trades", "sell_trades", "net_pips", "avg_pips_per_trade", "win_rate", "profit_factor", "max_drawdown_pips", "skipped_same_bar_signals"]], ".4f"),
        "",
        "## Phase 7 Pair Selection Gate",
        "",
        md_table(pd.read_csv(PHASE7_DIR / "selected_two_stage_pair_model_selection.csv"), ".4f"),
        "",
        "## Interpretasi",
        "",
        "- Laporan Phase 7 tetap menjadi hasil utama untuk kandidat deployment karena policy dipilih dari validation gate.",
        "- Laporan Backtrader ini dipakai sebagai audit eksekusi Python sequential, bukan pengganti Strategy Tester MT5.",
        "- Jika hasil Backtrader lebih kecil dari Phase 7, penyebab yang paling mungkin adalah overlapping event dan pembatasan satu posisi aktif per pair.",
    ]

    orders.to_csv(OUT_DIR / "backtrader_phase7_selected_policy_orders.csv", index=False)
    summary.to_csv(OUT_DIR / "backtrader_phase7_selected_policy_summary_by_pair.csv", index=False)
    aggregate.to_csv(OUT_DIR / "backtrader_phase7_selected_policy_summary.csv", index=False)
    report = "\n".join(report_lines) + "\n"
    (OUT_DIR / "LAPORAN_BACKTRADER_VECTORBT_PHASE7_FINAL.md").write_text(report, encoding="utf-8")
    (PHASE7_DIR / "LAPORAN_BACKTRADER_VECTORBT_PHASE7_FINAL.md").write_text(report, encoding="utf-8")
    print(f"[OUTPUT] wrote={OUT_DIR / 'LAPORAN_BACKTRADER_VECTORBT_PHASE7_FINAL.md'}", flush=True)
    print("[PROGRESS] DONE 48_backtrader_compare_phase7_selected_policy.py", flush=True)


if __name__ == "__main__":
    main()
