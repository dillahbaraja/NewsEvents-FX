from common import *
from pathlib import Path
import json
import math
import os
import shutil

import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader

bt10 = SourceFileLoader("bt10", str(ROOT / "10_backtrader_news_events.py")).load_module()

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
MODEL_PATH = Path(os.environ.get("MANUAL_BLOCK_INVERSE_MODEL_PATH", str(MANUAL_DIR / "model_block_inverse.csv")))
BACKTRADER_ROOT = DATA / "backtrader"
MODEL_NAME = os.environ.get("MANUAL_BLOCK_INVERSE_MODEL_NAME", "manual_block_inverse")
BLOCK_UNKNOWN = True


def normalized_event_name(value):
    return " ".join(str(value).strip().lower().split())


INVERSE_KEYWORDS = ["unemployment", "jobless", "claimant", "claims", "inventories", "crude oil inventories"]


def macro_surprise_direction(row):
    actual = row.get("Actual")
    forecast = row.get("Forecast")
    previous = row.get("Previous")
    if pd.notna(actual) and pd.notna(forecast) and float(forecast) != 0:
        diff = float(actual) - float(forecast)
    elif pd.notna(actual) and pd.notna(previous) and float(previous) != 0:
        diff = float(actual) - float(previous)
    elif pd.notna(actual) and pd.notna(forecast):
        diff = float(actual) - float(forecast)
    elif pd.notna(actual) and pd.notna(previous):
        diff = float(actual) - float(previous)
    else:
        return 0
    if diff > 0:
        direction = 1
    elif diff < 0:
        direction = -1
    else:
        direction = 0
    name = str(row.get("Name", "")).lower()
    if direction and any(k in name for k in INVERSE_KEYWORDS):
        direction *= -1
    return direction


def max_drawdown(equity):
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak:
            worst = min(worst, (value - peak) / peak)
    return worst


def summarize_orders(pair, decisions, orders, start, end):
    orders_df = pd.DataFrame(orders)
    trade_count = int(len(orders_df))
    wins = int((orders_df["pnl_pips"] > 0).sum()) if not orders_df.empty else 0
    gross_profit = float(orders_df.loc[orders_df["pnl_pips"] > 0, "pnl_pips"].sum()) if not orders_df.empty else 0.0
    gross_loss = float(-orders_df.loc[orders_df["pnl_pips"] < 0, "pnl_pips"].sum()) if not orders_df.empty else 0.0
    net_pips = float(orders_df["pnl_pips"].sum()) if not orders_df.empty else 0.0
    pnl = float(orders_df["pnl"].sum()) if not orders_df.empty else 0.0
    equity = [10000.0]
    for value in orders_df["pnl"].tolist() if not orders_df.empty else []:
        equity.append(equity[-1] + value)
    return {
        "pair": pair,
        "model": MODEL_NAME,
        "period_start": start,
        "period_end": end,
        "trades": trade_count,
        "buy_signals": int((decisions["final_signal"] == "BUY").sum()) if not decisions.empty else 0,
        "sell_signals": int((decisions["final_signal"] == "SELL").sum()) if not decisions.empty else 0,
        "no_trade_signals": int((~decisions["final_signal"].isin(["BUY", "SELL"])).sum()) if not decisions.empty else 0,
        "blocked_signals": int(decisions["action"].isin(["BLOCK", "BLOCK_UNKNOWN"]).sum()) if not decisions.empty else 0,
        "inversed_signals": int(decisions["action"].eq("INVERSE").sum()) if not decisions.empty else 0,
        "win_rate": wins / trade_count if trade_count else 0.0,
        "net_profit": pnl,
        "net_pips": net_pips,
        "avg_pips_per_trade": net_pips / max(1, trade_count),
        "net_return_pct": pnl / 10000.0,
        "normalized_return": net_pips / max(1, trade_count),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf if gross_profit > 0 else 0.0,
        "threshold": "",
        "signal_mode": "manual_block_inverse",
        "events": int(len(decisions)),
        "status": "ok" if not decisions.empty else "no_events_or_no_price_coverage",
    }


def run_pair(pair, model, start, end):
    out_dir = BACKTRADER_ROOT / pair / MODEL_NAME
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = bt10.prepare_pair_events(pair, bt10.DEFAULT_NEWS_TEST, start, end, 1, 2)
    if events.empty:
        pd.DataFrame().to_csv(out_dir / "backtrader_predictions.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "backtrader_orders.csv", index=False)
        summary = summarize_orders(pair, pd.DataFrame(), [], start, end)
        pd.DataFrame([summary]).to_csv(out_dir / "backtrader_summary.csv", index=False)
        pd.DataFrame([summary]).to_csv(out_dir / "backtrader_normalized_summary.csv", index=False)
        (out_dir / "backtrader_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    events = events.copy()
    events["event_name"] = events["Name"].astype(str)
    events["event_name_norm"] = events["event_name"].map(normalized_event_name)
    events["initial_signal_num"] = events.apply(macro_surprise_direction, axis=1).astype("int8")
    events = events[events["initial_signal_num"].ne(0)].copy()
    if events.empty:
        summary = summarize_orders(pair, events, [], start, end)
        pd.DataFrame([summary]).to_csv(out_dir / "backtrader_summary.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "backtrader_orders.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "backtrader_predictions.csv", index=False)
        return summary

    joined = events.merge(
        model,
        left_on=["pair", "CountryCode", "event_name_norm"],
        right_on=["pair", "country", "event_name_norm"],
        how="left",
        suffixes=("", "_model"),
    )
    joined["action"] = joined["action"].fillna("BLOCK_UNKNOWN" if BLOCK_UNKNOWN else "ALLOW")
    joined["action_reason"] = joined["reason"].fillna("unknown_event_blocked" if BLOCK_UNKNOWN else "unknown_event_allowed")
    joined["initial_signal"] = np.where(joined["initial_signal_num"].eq(1), "BUY", "SELL")
    final_num = joined["initial_signal_num"].copy()
    final_num = np.where(joined["action"].eq("INVERSE"), -final_num, final_num)
    final_num = np.where(joined["action"].isin(["BLOCK", "BLOCK_UNKNOWN"]), 0, final_num)
    joined["final_signal_num"] = final_num.astype("int8")
    joined["final_signal"] = np.select(
        [joined["final_signal_num"].eq(1), joined["final_signal_num"].eq(-1)],
        ["BUY", "SELL"],
        default="NO_TRADE",
    )
    joined["model"] = MODEL_NAME

    pred_cols = [
        "event_time", "pair", "CountryCode", "currency", "event_name", "Actual", "Forecast", "Previous",
        "Importance", "Impact", "initial_signal", "action", "action_reason", "final_signal",
        "Close", "close_fwd", "domestic_return_pips_1h", "trade_count", "net_pips",
        "profit_factor", "hit_rate", "inverse_net_pips", "inverse_profit_factor", "inverse_hit_rate",
    ]
    for col in pred_cols:
        if col not in joined.columns:
            joined[col] = np.nan
    joined[pred_cols].rename(columns={"Close": "close"}).to_csv(out_dir / "backtrader_predictions.csv", index=False)

    orders = []
    for _, row in joined[joined["final_signal"].isin(["BUY", "SELL"])].iterrows():
        domestic_direction = 1.0 if row["final_signal"] == "BUY" else -1.0
        raw_direction = domestic_direction if row["orientation"] == "base" else -domestic_direction
        pnl = (row["close_fwd"] - row["Close"]) * raw_direction
        pnl_pips = row["domestic_return_pips_1h"] * domestic_direction
        orders.append({
            "entry_time": row["time"],
            "exit_time": row["exit_time"],
            "pair": pair,
            "model": MODEL_NAME,
            "event_name": row["event_name"],
            "country": row["CountryCode"],
            "initial_signal": row["initial_signal"],
            "action": row["action"],
            "signal": row["final_signal"],
            "entry_price": row["Close"],
            "exit_price": row["close_fwd"],
            "size": 1.0,
            "pnl": pnl,
            "pnl_pips": pnl_pips,
            "return_pct": pnl / 10000.0,
            "reason": "manual_block_inverse_holding_1h",
        })
    orders_df = pd.DataFrame(orders)
    orders_df.to_csv(out_dir / "backtrader_orders.csv", index=False)
    summary = summarize_orders(pair, joined, orders, start, end)
    pd.DataFrame([summary]).to_csv(out_dir / "backtrader_summary.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "backtrader_normalized_summary.csv", index=False)
    (out_dir / "backtrader_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_monthly(orders):
    orders_df = pd.DataFrame(orders)
    if orders_df.empty:
        monthly = pd.DataFrame(columns=["month", "pair", "model", "monthly_pips", "monthly_profit", "trades", "cumulative_pips"])
    else:
        orders_df["entry_time"] = pd.to_datetime(orders_df["entry_time"], errors="coerce")
        orders_df["month"] = orders_df["entry_time"].dt.to_period("M").dt.to_timestamp()
        monthly = orders_df.groupby(["pair", "model", "month"], as_index=False).agg(
            monthly_pips=("pnl_pips", "sum"),
            monthly_profit=("pnl", "sum"),
            trades=("pnl_pips", "size"),
        )
        monthly = monthly.sort_values(["pair", "month"])
        monthly["cumulative_pips"] = monthly.groupby(["pair", "model"])["monthly_pips"].cumsum()
    monthly.to_csv(MANUAL_DIR / "manual_block_inverse_monthly_cumulative.csv", index=False)
    return monthly


def plot_monthly(monthly):
    out = MANUAL_DIR / "manual_block_inverse_monthly_cumulative_by_pair.png"
    if monthly.empty:
        return None
    plt.figure(figsize=(13, 7))
    for pair in sorted(monthly["pair"].unique()):
        g = monthly[monthly["pair"].eq(pair)].sort_values("month")
        plt.plot(g["month"], g["cumulative_pips"], marker="o", linewidth=1.8, markersize=3, label=pair)
    plt.axhline(0, color="#444444", linewidth=0.9)
    plt.title("Manual Block/Inverse - Monthly Cumulative Profit by Pair (pips)")
    plt.xlabel("Month")
    plt.ylabel("Cumulative profit (pips)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def write_report(summary, monthly):
    s = pd.DataFrame(summary)
    active = s[pd.to_numeric(s["trades"], errors="coerce").fillna(0).gt(0)].copy()
    total_trades = int(active["trades"].sum()) if not active.empty else 0
    total_pips = float(active["net_pips"].sum()) if not active.empty else 0.0
    wins = (active["trades"] * active["win_rate"]).sum() if not active.empty else 0.0
    win_rate = wins / total_trades if total_trades else 0.0
    def markdown_table(df):
        if df.empty:
            return "_Tidak ada data._"
        view = df.copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        cols = list(view.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# Laporan Manual Block/Inverse",
        "",
        "Periode kalibrasi: 2010-2023.",
        "Periode test: 2024-2026.",
        "Unknown events pada test diblok secara default.",
        "",
        "## Ringkasan Test",
        f"- Total trades: {total_trades}",
        f"- Winrate: {win_rate:.2%}",
        f"- Net pips: {total_pips:.2f}",
        f"- Positive pairs: {int((active['net_pips'] > 0).sum()) if not active.empty else 0}",
        "",
        "## Per Pair",
        markdown_table(s),
        "",
        "## Output",
        f"- `{MODEL_PATH}`",
        f"- `{MANUAL_DIR / 'manual_block_inverse_test_orders_2024_2026.csv'}`",
        f"- `{MANUAL_DIR / 'manual_block_inverse_test_summary.csv'}`",
        f"- `{MANUAL_DIR / 'manual_block_inverse_monthly_cumulative.csv'}`",
        f"- `{MANUAL_DIR / 'manual_block_inverse_monthly_cumulative_by_pair.png'}`",
    ]
    (MANUAL_DIR / "LAPORAN_MANUAL_BLOCK_INVERSE.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("[PROGRESS] START 16_backtrader_manual_block_inverse_test.py", flush=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing manual model: {MODEL_PATH}")
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    model = pd.read_csv(MODEL_PATH)
    pairs = available_pairs()
    print(f"[DATA] pairs={pairs}", flush=True)
    summaries = []
    all_orders = []
    for pair in pairs:
        print(f"[BACKTRADER] START pair={pair} model={MODEL_NAME}", flush=True)
        summary = run_pair(pair, model, "2024-01-01", "2026-12-31")
        summaries.append(summary)
        orders_path = BACKTRADER_ROOT / pair / MODEL_NAME / "backtrader_orders.csv"
        if orders_path.exists():
            try:
                od = pd.read_csv(orders_path)
                if not od.empty:
                    all_orders.append(od)
            except pd.errors.EmptyDataError:
                pass
        print(f"[BACKTRADER] DONE pair={pair} trades={summary.get('trades', 0)} status={summary.get('status')}", flush=True)

    orders = pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame()
    orders.to_csv(MANUAL_DIR / "manual_block_inverse_test_orders_2024_2026.csv", index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(MANUAL_DIR / "manual_block_inverse_test_summary.csv", index=False)
    monthly = write_monthly(orders)
    chart = plot_monthly(monthly)
    write_report(summaries, monthly)

    summary_path = BACKTRADER_ROOT / "backtrader_all_summary.csv"
    if summary_path.exists():
        prev = pd.read_csv(summary_path)
        if {"pair", "model"}.issubset(prev.columns):
            prev = prev[~prev["model"].astype(str).eq(MODEL_NAME)]
            merged = pd.concat([prev, summary_df], ignore_index=True, sort=False)
        else:
            merged = summary_df
    else:
        merged = summary_df
    merged.to_csv(summary_path, index=False)
    merged.to_csv(BACKTRADER_ROOT / "backtrader_normalized_summary.csv", index=False)
    print(f"[OUTPUT] {MANUAL_DIR / 'manual_block_inverse_test_orders_2024_2026.csv'}", flush=True)
    print(f"[OUTPUT] {MANUAL_DIR / 'manual_block_inverse_test_summary.csv'}", flush=True)
    print(f"[OUTPUT] {MANUAL_DIR / 'manual_block_inverse_monthly_cumulative.csv'}", flush=True)
    if chart:
        print(f"[OUTPUT] {chart}", flush=True)
    print(f"[OUTPUT] {MANUAL_DIR / 'LAPORAN_MANUAL_BLOCK_INVERSE.md'}", flush=True)
    print(f"[OUTPUT] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
