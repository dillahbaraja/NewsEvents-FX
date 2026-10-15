from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BACKTRADER_ROOT = Path(r"C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Dataset\backtrader")
OUT_DIR = BACKTRADER_ROOT / "monthly_cumulative_profit"
MODELS = ["catboost", "elastic_net", "random_forest", "ebm"]


def load_orders():
    rows = []
    for path in BACKTRADER_ROOT.glob("*/*/backtrader_orders.csv"):
        pair = path.parent.parent.name
        model = path.parent.name
        if model not in MODELS:
            continue
        try:
            d = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if d.empty:
            continue
        d["pair"] = d.get("pair", pair)
        d["model"] = d.get("model", model)
        d["entry_time"] = pd.to_datetime(d["entry_time"], errors="coerce")
        d["pnl_pips"] = pd.to_numeric(d["pnl_pips"], errors="coerce").fillna(0.0)
        d["pnl"] = pd.to_numeric(d.get("pnl", 0.0), errors="coerce").fillna(0.0)
        d = d.dropna(subset=["entry_time"])
        rows.append(d[["entry_time", "pair", "model", "pnl", "pnl_pips"]])
    if not rows:
        return pd.DataFrame(columns=["entry_time", "pair", "model", "pnl", "pnl_pips"])
    return pd.concat(rows, ignore_index=True)


def build_monthly(orders):
    orders = orders.copy()
    orders["month"] = orders["entry_time"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        orders.groupby(["model", "pair", "month"], as_index=False)
        .agg(monthly_pips=("pnl_pips", "sum"), monthly_profit=("pnl", "sum"), trades=("pnl_pips", "size"))
    )
    if monthly.empty:
        return monthly

    months = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    full_rows = []
    for model in MODELS:
        model_pairs = sorted(monthly.loc[monthly["model"].eq(model), "pair"].unique())
        for pair in model_pairs:
            base = pd.DataFrame({"month": months})
            base["model"] = model
            base["pair"] = pair
            full_rows.append(base)
    full = pd.concat(full_rows, ignore_index=True) if full_rows else monthly[["model", "pair", "month"]]
    monthly = full.merge(monthly, on=["model", "pair", "month"], how="left")
    monthly[["monthly_pips", "monthly_profit", "trades"]] = monthly[["monthly_pips", "monthly_profit", "trades"]].fillna(0)
    monthly["cumulative_pips"] = monthly.groupby(["model", "pair"])["monthly_pips"].cumsum()
    monthly["cumulative_profit"] = monthly.groupby(["model", "pair"])["monthly_profit"].cumsum()
    return monthly


def plot_model(monthly, model):
    d = monthly[monthly["model"].eq(model)].copy()
    if d.empty:
        return None

    plt.figure(figsize=(13, 7))
    pairs = sorted(d["pair"].unique())
    for pair in pairs:
        g = d[d["pair"].eq(pair)].sort_values("month")
        if g["trades"].sum() == 0:
            continue
        plt.plot(g["month"], g["cumulative_pips"], marker="o", linewidth=1.8, markersize=3, label=pair)

    plt.axhline(0, color="#444444", linewidth=0.9)
    plt.title(f"Monthly Cumulative Profit - {model} (pips)")
    plt.xlabel("Month")
    plt.ylabel("Cumulative profit (pips)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    out = OUT_DIR / f"monthly_cumulative_profit_{model}.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def main():
    print("[PROGRESS] START 13_make_monthly_cumulative_profit_charts.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orders = load_orders()
    print(f"[DATA] orders={len(orders)} models={sorted(orders['model'].unique()) if not orders.empty else []}", flush=True)
    monthly = build_monthly(orders)
    monthly_path = OUT_DIR / "monthly_cumulative_profit_by_model_pair.csv"
    monthly.to_csv(monthly_path, index=False)
    print(f"[OUTPUT] {monthly_path}", flush=True)

    summary = []
    for model in MODELS:
        d = monthly[monthly["model"].eq(model)]
        total_trades = int(d["trades"].sum()) if not d.empty else 0
        final_by_pair = d.sort_values("month").groupby("pair").tail(1) if not d.empty else pd.DataFrame()
        summary.append({
            "model": model,
            "pairs": int(final_by_pair["pair"].nunique()) if not final_by_pair.empty else 0,
            "trades": total_trades,
            "final_total_cumulative_pips": float(final_by_pair["cumulative_pips"].sum()) if not final_by_pair.empty else 0.0,
            "positive_pairs": int((final_by_pair["cumulative_pips"] > 0).sum()) if not final_by_pair.empty else 0,
        })
        out = plot_model(monthly, model)
        if out:
            print(f"[OUTPUT] {out}", flush=True)

    summary_path = OUT_DIR / "monthly_cumulative_profit_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"[OUTPUT] {summary_path}", flush=True)
    print("[PROGRESS] DONE 13_make_monthly_cumulative_profit_charts.py", flush=True)


if __name__ == "__main__":
    main()
