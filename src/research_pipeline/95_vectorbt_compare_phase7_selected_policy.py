from common import *

import json
import math
import os

try:
    import vectorbt as vbt
except Exception as exc:
    raise SystemExit(f"vectorbt import failed: {exc}")


DATATEST = DATA / "datatest"
PHASE7_DIR = Path(os.environ.get("PHASE7_POLICY_DIR", str(ARTIFACTS / "phase7_selected_two_stage_policy")))
SIGNALS_IN = PHASE7_DIR / "selected_two_stage_policy_orders.csv"
OUT_DIR = Path(os.environ.get("PHASE7_VECTORBT_OUT_DIR", str(ARTIFACTS / "vectorbt_phase7_selected_policy_comparison")))

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-08-31 23:59:59")
INITIAL_CASH = 10000.0


def read_price(pair):
    path = DATATEST / pair_file(pair)
    if not path.exists():
        return None
    data = read_prices(path)
    data = data[(data["time"] >= START) & (data["time"] <= END)].copy()
    if data.empty:
        return None
    return data.set_index("time")


def profit_factor(values):
    s = pd.Series(values, dtype=float).dropna()
    gp = float(s[s > 0].sum())
    gl = float(-s[s < 0].sum())
    if gl == 0:
        return math.inf if gp > 0 else 0.0
    return gp / gl


def scalar_stat(stats, key):
    try:
        value = stats.get(key, np.nan)
        if hasattr(value, "iloc"):
            value = value.iloc[0]
        return float(value)
    except Exception:
        return np.nan


def run_pair(pair, signals):
    price = read_price(pair)
    if price is None:
        return pd.DataFrame(), {"pair": pair, "status": "missing_price", "trades": 0}
    active = signals[signals["final_signal"].isin(["BUY", "SELL"])].copy()
    if active.empty:
        return pd.DataFrame(), {"pair": pair, "status": "no_trade_signals", "trades": 0}

    close = price["Close"].astype(float)
    long_entries = pd.Series(False, index=close.index)
    long_exits = pd.Series(False, index=close.index)
    short_entries = pd.Series(False, index=close.index)
    short_exits = pd.Series(False, index=close.index)

    audit_rows = []
    skipped_same_bar = 0
    for entry_time, g in active.sort_values("event_time").groupby("time"):
        if entry_time not in close.index:
            continue
        row = g.iloc[0]
        skipped_same_bar += max(len(g) - 1, 0)
        exit_time = entry_time + pd.Timedelta(hours=1)
        if exit_time not in close.index:
            later = close.index[close.index > entry_time]
            if len(later) == 0:
                continue
            exit_time = later[0]
        signal = row["final_signal"]
        if signal == "BUY":
            long_entries.loc[entry_time] = True
            long_exits.loc[exit_time] = True
        else:
            short_entries.loc[entry_time] = True
            short_exits.loc[exit_time] = True
        direction = 1.0 if signal == "BUY" else -1.0
        pnl_pips = ((float(close.loc[exit_time]) - float(close.loc[entry_time])) / pip_size(pair)) * direction
        audit_rows.append({
            "model": row.get("model", ""),
            "pair": pair,
            "runtime": "vectorbt_phase7_selected_policy",
            "event_time": row.get("event_time"),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "country": row.get("CountryCode", ""),
            "event_name": row.get("Name", ""),
            "signal": signal,
            "entry_price": float(close.loc[entry_time]),
            "exit_price": float(close.loc[exit_time]),
            "pnl_pips": pnl_pips,
            "prob_trade": row.get("prob_trade", np.nan),
            "prob_sell": row.get("prob_sell", np.nan),
            "prob_buy": row.get("prob_buy", np.nan),
            "direction_confidence": row.get("direction_confidence", np.nan),
            "policy": row.get("policy", ""),
            "selection_status": row.get("selection_status", ""),
            "skipped_same_bar_signals": max(len(g) - 1, 0),
        })

    audit = pd.DataFrame(audit_rows)
    try:
        pf = vbt.Portfolio.from_signals(
            close,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=INITIAL_CASH,
            fees=0.0,
            freq="1h",
        )
        stats = pf.stats()
        vbt_stats = {
            "vectorbt_total_return_pct": scalar_stat(stats, "Total Return [%]"),
            "vectorbt_max_drawdown_pct": scalar_stat(stats, "Max Drawdown [%]"),
            "vectorbt_total_trades": scalar_stat(stats, "Total Trades"),
            "vectorbt_win_rate_pct": scalar_stat(stats, "Win Rate [%]"),
            "vectorbt_profit_factor": scalar_stat(stats, "Profit Factor"),
            "vectorbt_expectancy": scalar_stat(stats, "Expectancy"),
        }
    except Exception as exc:
        vbt_stats = {"vectorbt_error": str(exc)}

    if audit.empty:
        summary = {"pair": pair, "status": "no_executable_events", "trades": 0}
    else:
        pnl = pd.to_numeric(audit["pnl_pips"], errors="coerce").fillna(0.0)
        summary = {
            "pair": pair,
            "runtime": "vectorbt_phase7_selected_policy",
            "status": "ok",
            "model": ",".join(sorted(audit["model"].dropna().astype(str).unique())),
            "trades": int(len(audit)),
            "buy_trades": int(audit["signal"].eq("BUY").sum()),
            "sell_trades": int(audit["signal"].eq("SELL").sum()),
            "win_rate": float((pnl > 0).mean()),
            "net_pips": float(pnl.sum()),
            "avg_pips_per_trade": float(pnl.mean()),
            "gross_profit_pips": float(pnl[pnl > 0].sum()),
            "gross_loss_pips": float(-pnl[pnl < 0].sum()),
            "profit_factor": profit_factor(pnl),
            "skipped_same_bar_signals": int(skipped_same_bar),
        }
    summary.update(vbt_stats)
    return audit, summary


def md_table(df, floatfmt=".4f"):
    if df.empty:
        return "_Tidak ada data._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else ("inf" if math.isinf(float(x)) else format(float(x), floatfmt)))
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def aggregate(summary):
    ok = summary[summary["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame([{"runtime": "vectorbt_phase7_selected_policy", "pairs": 0, "trades": 0, "net_pips": 0.0}])
    gp = float(ok["gross_profit_pips"].sum())
    gl = float(ok["gross_loss_pips"].sum())
    net = float(ok["net_pips"].sum())
    trades = int(ok["trades"].sum())
    return pd.DataFrame([{
        "runtime": "vectorbt_phase7_selected_policy",
        "pairs": int(ok["pair"].nunique()),
        "positive_pairs": int((ok["net_pips"] > 0).sum()),
        "trades": trades,
        "net_pips": net,
        "avg_pips_per_trade": net / trades if trades else 0.0,
        "profit_factor": gp / gl if gl else math.inf,
        "gross_profit_pips": gp,
        "gross_loss_pips": gl,
    }])


def main():
    print("[PROGRESS] START 95_vectorbt_compare_phase7_selected_policy.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signals = pd.read_csv(SIGNALS_IN, low_memory=False)
    signals["event_time"] = pd.to_datetime(signals["event_time"], errors="coerce")
    signals = signals[signals["split"].eq("oos_2024_2026")].copy()
    signals = signals[signals["final_signal"].isin(["BUY", "SELL"])].copy()
    signals = signals[(signals["event_time"] >= START) & (signals["event_time"] <= END)].copy()
    signals["time"] = signals["event_time"].dt.floor("h")
    print(f"[DATA] signals={len(signals)} pairs={sorted(signals['pair'].dropna().unique())}", flush=True)

    audits = []
    summaries = []
    for idx, pair in enumerate(sorted(signals["pair"].dropna().unique()), start=1):
        pair_signals = signals[signals["pair"].eq(pair)].copy()
        print(f"[VECTORBT] START pair={idx}/{signals['pair'].nunique()} {pair} signals={len(pair_signals)}", flush=True)
        audit, summary = run_pair(pair, pair_signals)
        if not audit.empty:
            audits.append(audit)
        summaries.append(summary)
        print(f"[VECTORBT] DONE {pair} trades={summary.get('trades', 0)} net_pips={summary.get('net_pips', 0.0):.1f} pf={summary.get('profit_factor', 0.0):.4f}", flush=True)

    audit_all = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    summary = pd.DataFrame(summaries).sort_values("net_pips", ascending=False)
    agg = aggregate(summary)

    audit_all.to_csv(OUT_DIR / "vectorbt_phase7_selected_policy_event_audit.csv", index=False)
    summary.to_csv(OUT_DIR / "vectorbt_phase7_selected_policy_summary_by_pair.csv", index=False)
    agg.to_csv(OUT_DIR / "vectorbt_phase7_selected_policy_summary.csv", index=False)
    metadata = {
        "signals_in": str(SIGNALS_IN),
        "out_dir": str(OUT_DIR),
        "period": "2024-01-01 to 2026-08-31",
        "created_at": pd.Timestamp.now().isoformat(),
    }
    (OUT_DIR / "vectorbt_phase7_selected_policy_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = [
        "# VectorBT Phase 7 Selected Policy OOS",
        "",
        f"Sumber sinyal: `{SIGNALS_IN}`",
        "",
        "## Aggregate",
        "",
        md_table(agg),
        "",
        "## Per Pair",
        "",
        md_table(summary),
    ]
    (OUT_DIR / "LAPORAN_VECTORBT_PHASE7_SELECTED_POLICY.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_VECTORBT_PHASE7_SELECTED_POLICY.md'}", flush=True)
    print("[PROGRESS] DONE 95_vectorbt_compare_phase7_selected_policy.py", flush=True)


if __name__ == "__main__":
    main()
