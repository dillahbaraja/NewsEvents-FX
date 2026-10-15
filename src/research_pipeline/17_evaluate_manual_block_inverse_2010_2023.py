from common import *
import matplotlib.pyplot as plt

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
MODEL_PATH = MANUAL_DIR / "model_block_inverse.csv"
TRAIN_TRADES = MANUAL_DIR / "manual_block_inverse_train_trades_2010_2023.csv"
ORDERS_OUT = MANUAL_DIR / "manual_block_inverse_insample_orders_2010_2023.csv"
SUMMARY_OUT = MANUAL_DIR / "manual_block_inverse_insample_summary_2010_2023.csv"
MONTHLY_OUT = MANUAL_DIR / "manual_block_inverse_insample_monthly_cumulative_2010_2023.csv"
CHART_OUT = MANUAL_DIR / "manual_block_inverse_insample_monthly_cumulative_by_pair.png"
REPORT_OUT = MANUAL_DIR / "LAPORAN_MANUAL_BLOCK_INVERSE_INSAMPLE_2010_2023.md"


def profit_factor(values):
    s = pd.Series(values).dropna()
    gross_profit = float(s[s > 0].sum())
    gross_loss = float(-s[s < 0].sum())
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def summarize(orders, decisions):
    rows = []
    for pair, g_dec in decisions.groupby("pair", sort=False):
        g = orders[orders["pair"].eq(pair)]
        trades = int(len(g))
        wins = int((g["pnl_pips"] > 0).sum()) if trades else 0
        rows.append({
            "pair": pair,
            "model": "manual_block_inverse",
            "period_start": "2010-01-01",
            "period_end": "2023-12-31",
            "trades": trades,
            "buy_signals": int(g["signal"].eq("BUY").sum()) if trades else 0,
            "sell_signals": int(g["signal"].eq("SELL").sum()) if trades else 0,
            "no_trade_signals": int((~g_dec["final_signal"].isin(["BUY", "SELL"])).sum()),
            "blocked_signals": int(g_dec["action"].isin(["BLOCK", "BLOCK_UNKNOWN"]).sum()),
            "inversed_signals": int(g_dec["action"].eq("INVERSE").sum()),
            "win_rate": wins / trades if trades else 0.0,
            "net_pips": float(g["pnl_pips"].sum()) if trades else 0.0,
            "avg_pips_per_trade": float(g["pnl_pips"].mean()) if trades else 0.0,
            "profit_factor": profit_factor(g["pnl_pips"]) if trades else 0.0,
            "events": int(len(g_dec)),
            "status": "ok",
        })
    return pd.DataFrame(rows)


def build_monthly(orders):
    if orders.empty:
        return pd.DataFrame()
    d = orders.copy()
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d["month"] = d["event_time"].dt.to_period("M").dt.to_timestamp()
    monthly = d.groupby(["pair", "month"], as_index=False).agg(
        monthly_pips=("pnl_pips", "sum"),
        trades=("pnl_pips", "size"),
    )
    monthly["model"] = "manual_block_inverse"
    monthly = monthly.sort_values(["pair", "month"])
    monthly["cumulative_pips"] = monthly.groupby("pair")["monthly_pips"].cumsum()
    return monthly


def plot_monthly(monthly):
    if monthly.empty:
        return
    plt.figure(figsize=(13, 7))
    for pair in sorted(monthly["pair"].unique()):
        g = monthly[monthly["pair"].eq(pair)]
        plt.plot(g["month"], g["cumulative_pips"], marker="o", linewidth=1.4, markersize=2.5, label=pair)
    plt.axhline(0, color="#444444", linewidth=0.9)
    plt.title("Manual Block/Inverse In-Sample 2010-2023 - Monthly Cumulative Profit by Pair")
    plt.xlabel("Month")
    plt.ylabel("Cumulative profit (pips)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(CHART_OUT, dpi=160)
    plt.close()


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


def main():
    print("[PROGRESS] START 17_evaluate_manual_block_inverse_2010_2023.py", flush=True)
    model = pd.read_csv(MODEL_PATH)
    trades = pd.read_csv(TRAIN_TRADES, low_memory=False)
    decisions = trades.merge(
        model[["pair", "country", "event_name_norm", "action", "reason"]].rename(
            columns={"action": "model_action", "reason": "model_reason"}
        ),
        left_on=["pair", "CountryCode", "event_name_norm"],
        right_on=["pair", "country", "event_name_norm"],
        how="left",
    )
    decisions["action"] = decisions["model_action"].fillna("BLOCK_UNKNOWN")
    decisions["reason"] = decisions["model_reason"].fillna("unknown_event_blocked")
    final_num = np.where(decisions["initial_signal"].eq("BUY"), 1, -1)
    final_num = np.where(decisions["action"].eq("INVERSE"), -final_num, final_num)
    final_num = np.where(decisions["action"].isin(["BLOCK", "BLOCK_UNKNOWN"]), 0, final_num)
    decisions["final_signal_num"] = final_num.astype("int8")
    decisions["final_signal"] = np.select(
        [decisions["final_signal_num"].eq(1), decisions["final_signal_num"].eq(-1)],
        ["BUY", "SELL"],
        default="NO_TRADE",
    )
    decisions["final_pnl_pips"] = np.where(
        decisions["action"].eq("INVERSE"),
        decisions["inverse_pnl_pips"],
        np.where(decisions["action"].eq("ALLOW"), decisions["pnl_pips"], 0.0),
    )
    orders = decisions[decisions["final_signal"].isin(["BUY", "SELL"])].copy()
    orders["pnl_pips"] = orders["final_pnl_pips"]
    orders["model"] = "manual_block_inverse"
    orders["signal"] = orders["final_signal"]
    orders["country"] = orders["CountryCode"]
    orders_cols = [
        "event_time", "pair", "model", "event_name", "country", "initial_signal", "action",
        "signal", "pnl_pips", "reason",
    ]
    orders[orders_cols].to_csv(ORDERS_OUT, index=False)
    summary = summarize(orders, decisions)
    summary.to_csv(SUMMARY_OUT, index=False)
    monthly = build_monthly(orders)
    monthly.to_csv(MONTHLY_OUT, index=False)
    plot_monthly(monthly)

    active = summary[summary["trades"].gt(0)]
    total_trades = int(active["trades"].sum()) if not active.empty else 0
    total_pips = float(active["net_pips"].sum()) if not active.empty else 0.0
    weighted_wins = float((active["trades"] * active["win_rate"]).sum()) if not active.empty else 0.0
    report = [
        "# Laporan Manual Block/Inverse In-Sample 2010-2023",
        "",
        "Catatan: hasil ini in-sample karena rules block/inverse juga dibentuk dari periode 2010-2023.",
        "",
        "## Ringkasan",
        f"- Total trades: {total_trades}",
        f"- Winrate: {weighted_wins / total_trades:.2%}" if total_trades else "- Winrate: 0.00%",
        f"- Net pips: {total_pips:.2f}",
        f"- Positive pairs: {int((active['net_pips'] > 0).sum()) if not active.empty else 0}",
        "",
        "## Per Pair",
        markdown_table(summary),
        "",
        "## Output",
        f"- `{ORDERS_OUT}`",
        f"- `{SUMMARY_OUT}`",
        f"- `{MONTHLY_OUT}`",
        f"- `{CHART_OUT}`",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")
    print(f"[OUTPUT] {ORDERS_OUT}", flush=True)
    print(f"[OUTPUT] {SUMMARY_OUT}", flush=True)
    print(f"[OUTPUT] {MONTHLY_OUT}", flush=True)
    print(f"[OUTPUT] {CHART_OUT}", flush=True)
    print(f"[OUTPUT] {REPORT_OUT}", flush=True)


if __name__ == "__main__":
    main()
