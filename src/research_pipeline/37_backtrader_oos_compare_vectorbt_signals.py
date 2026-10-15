from common import *
import math
from pathlib import Path

import backtrader as bt


DATATEST = DATA / "datatest"
VBT_DIR = ARTIFACTS / "vectorbt_oos_model_comparison"
PREDICTIONS_IN = VBT_DIR / "vectorbt_oos_predictions_all.csv"
VECTORBT_SUMMARY_IN = VBT_DIR / "vectorbt_oos_summary_by_model_pair.csv"
OUT_DIR = ARTIFACTS / "backtrader_oos_vectorbt_signal_comparison"

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-08-31 23:59:59")
INITIAL_CASH = 10000.0
TRADE_SIZE = 1.0


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


class EventSignalSequentialStrategy(bt.Strategy):
    params = (
        ("signals", None),
        ("pair", ""),
        ("model", ""),
    )

    def __init__(self):
        self.signal_map = self.p.signals if self.p.signals is not None else {}
        self.active_trade = None
        self.orders_out = []
        self.pending_queue = []

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
        holding_bars = int(row.get("holding_bars", 1) or 1)
        exit_time = now + pd.Timedelta(hours=holding_bars)
        self.active_trade = {
            "model": self.p.model,
            "pair": self.p.pair,
            "runtime": "backtrader_sequential_close",
            "event_time": row.get("event_time"),
            "entry_time": now,
            "planned_exit_time": exit_time,
            "exit_time": exit_time,
            "country": row.get("CountryCode", row.get("country", "")),
            "event_name": row.get("Name", row.get("event_name", "")),
            "signal": row.get("signal"),
            "entry_price": close_price,
            "confidence": row.get("confidence", math.nan),
            "threshold": row.get("threshold", math.nan),
            "signal_reason": row.get("signal_reason", ""),
            "source_model": row.get("source_model", ""),
            "holding_bars": holding_bars,
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
    if d.empty:
        return None
    d = d.set_index("time")
    return d


def max_drawdown_pips(values):
    if len(values) == 0:
        return 0.0
    curve = pd.Series(values, dtype=float).cumsum()
    dd = curve.cummax() - curve
    return float(dd.max()) if len(dd) else 0.0


def summarize_orders(model, pair, orders, status):
    if orders.empty:
        return {
            "model": model,
            "pair": pair,
            "runtime": "backtrader_sequential_close",
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
    wins = int((pnl > 0).sum())
    net = float(pnl.sum())
    return {
        "model": model,
        "pair": pair,
        "runtime": "backtrader_sequential_close",
        "status": status,
        "trades": int(len(orders)),
        "buy_trades": int(orders["signal"].eq("BUY").sum()),
        "sell_trades": int(orders["signal"].eq("SELL").sum()),
        "win_rate": wins / len(orders),
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
        t = pd.Timestamp(row["time"])
        out.setdefault(t, []).append(row.to_dict())
    return out


def run_one(model, pair, signals):
    price = read_price(pair)
    if price is None:
        return pd.DataFrame(), summarize_orders(model, pair, pd.DataFrame(), "missing_price")
    active = signals[signals["signal"].isin(["BUY", "SELL"])].copy()
    if active.empty:
        return pd.DataFrame(), summarize_orders(model, pair, pd.DataFrame(), "no_trade_signals")

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(INITIAL_CASH)
    data = PandasH1(dataname=price)
    cerebro.adddata(data)
    cerebro.addstrategy(
        EventSignalSequentialStrategy,
        signals=make_signal_map(active),
        pair=pair,
        model=model,
    )
    results = cerebro.run()
    strategy = results[0]
    orders = pd.DataFrame(strategy.orders_out)
    summary = summarize_orders(model, pair, orders, "ok")
    return orders, summary


def comparison_report(summary, merged):
    def md_table(df, floatfmt=".4f"):
        if df.empty:
            return "_Tidak ada data._"
        view = df.copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
        cols = list(view.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# Backtrader vs VectorBT OOS Comparison",
        "",
        "Perbandingan ini memakai sinyal/prediksi yang sama dari `vectorbt_oos_predictions_all.csv`.",
        "Backtrader dijalankan sebagai eksekusi sequential per `model + pair`: satu posisi aktif per pair-model, entry pada close candle sinyal, exit setelah `holding_bars`.",
        "VectorBT sebelumnya dipakai sebagai screening cepat dan dapat menampung sinyal lebih paralel; karena itu jumlah trade bisa berbeda jika ada banyak event berdekatan.",
        "",
        "## Ringkasan Per Model",
        "",
    ]
    if not summary.empty:
        by_model = summary.groupby("model", as_index=False).agg(
            trades=("trades", "sum"),
            net_pips=("net_pips", "sum"),
            gross_profit_pips=("gross_profit_pips", "sum"),
            gross_loss_pips=("gross_loss_pips", "sum"),
            positive_pairs=("net_pips", lambda s: int((s > 0).sum())),
        )
        by_model["profit_factor"] = by_model.apply(
            lambda r: r["gross_profit_pips"] / r["gross_loss_pips"] if r["gross_loss_pips"] else math.inf,
            axis=1,
        )
        by_model = by_model.sort_values("net_pips", ascending=False)
        lines.append(md_table(by_model, ".4f"))
    lines += ["", "## Selisih Dengan VectorBT", ""]
    if not merged.empty:
        cols = [
            "model", "pair", "bt_trades", "vbt_trades", "trade_diff",
            "bt_net_pips", "vbt_net_pips", "net_pips_diff",
            "bt_profit_factor", "vbt_profit_factor",
        ]
        available = [c for c in cols if c in merged.columns]
        lines.append(md_table(merged.sort_values("net_pips_diff", ascending=False)[available], ".4f"))
    lines += [
        "",
        "## Interpretasi",
        "",
        "- Jika `trade_diff` besar, penyebab utama biasanya overlapping event: VectorBT bisa menghitung banyak sinyal, sedangkan Backtrader sequential hanya memegang satu posisi aktif.",
        "- Jika `trade_diff` kecil tetapi `net_pips_diff` besar, penyebab biasanya aturan entry/exit candle atau holding time.",
        "- File ini bukan pengganti Strategy Tester MT5; ini audit tambahan untuk melihat sensitivitas hasil terhadap engine backtest Python.",
    ]
    return "\n".join(lines) + "\n"


def main():
    print("[PROGRESS] START 37_backtrader_oos_compare_vectorbt_signals.py", flush=True)
    if not PREDICTIONS_IN.exists():
        raise FileNotFoundError(PREDICTIONS_IN)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[DATA] read predictions {PREDICTIONS_IN}", flush=True)
    preds = pd.read_csv(PREDICTIONS_IN)
    preds["time"] = pd.to_datetime(preds["time"], errors="coerce")
    preds["event_time"] = pd.to_datetime(preds["event_time"], errors="coerce")
    preds = preds.dropna(subset=["time"]).copy()
    preds = preds[(preds["time"] >= START) & (preds["time"] <= END)].copy()
    preds["pair"] = preds["pair"].astype(str).str.upper()
    preds["model"] = preds["model"].astype(str)
    print(f"[DATA] rows={len(preds)} active_signals={int(preds['signal'].isin(['BUY','SELL']).sum())}", flush=True)

    all_orders = []
    summaries = []
    groups = list(preds.groupby(["model", "pair"], sort=True))
    for idx, ((model, pair), group) in enumerate(groups, start=1):
        print(f"[BACKTRADER] START {idx}/{len(groups)} model={model} pair={pair} rows={len(group)} signals={int(group['signal'].isin(['BUY','SELL']).sum())}", flush=True)
        orders, summary = run_one(model, pair, group)
        summaries.append(summary)
        if not orders.empty:
            all_orders.append(orders)
        print(
            f"[BACKTRADER] DONE model={model} pair={pair} trades={summary['trades']} "
            f"winrate={summary['win_rate']:.4f} net_pips={summary['net_pips']:.1f} pf={summary['profit_factor']:.4f}",
            flush=True,
        )

    orders_all = pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    orders_all.to_csv(OUT_DIR / "backtrader_oos_orders_all.csv", index=False)
    summary.to_csv(OUT_DIR / "backtrader_oos_summary_by_model_pair.csv", index=False)

    merged = pd.DataFrame()
    if VECTORBT_SUMMARY_IN.exists():
        vbt = pd.read_csv(VECTORBT_SUMMARY_IN)
        vbt = vbt.rename(columns={
            "trades": "vbt_trades",
            "net_pips": "vbt_net_pips",
            "profit_factor": "vbt_profit_factor",
            "win_rate": "vbt_win_rate",
        })
        bt_sum = summary.rename(columns={
            "trades": "bt_trades",
            "net_pips": "bt_net_pips",
            "profit_factor": "bt_profit_factor",
            "win_rate": "bt_win_rate",
        })
        merged = bt_sum.merge(
            vbt[["model", "pair", "vbt_trades", "vbt_net_pips", "vbt_profit_factor", "vbt_win_rate"]],
            on=["model", "pair"],
            how="left",
        )
        merged["trade_diff"] = merged["bt_trades"] - merged["vbt_trades"]
        merged["net_pips_diff"] = merged["bt_net_pips"] - merged["vbt_net_pips"]
        merged["win_rate_diff"] = merged["bt_win_rate"] - merged["vbt_win_rate"]
        merged.to_csv(OUT_DIR / "backtrader_vs_vectorbt_model_pair_comparison.csv", index=False)

    report = comparison_report(summary, merged)
    (OUT_DIR / "LAPORAN_BACKTRADER_VS_VECTORBT_OOS.md").write_text(report, encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'backtrader_oos_summary_by_model_pair.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'backtrader_vs_vectorbt_model_pair_comparison.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_BACKTRADER_VS_VECTORBT_OOS.md'}", flush=True)
    print("[PROGRESS] DONE 37_backtrader_oos_compare_vectorbt_signals.py", flush=True)


if __name__ == "__main__":
    main()
