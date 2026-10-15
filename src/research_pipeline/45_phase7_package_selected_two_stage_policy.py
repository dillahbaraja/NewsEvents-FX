from common import *

import math
import shutil


PHASE6_DIR = ARTIFACTS / "phase6_validation_stability_gate"
OUT_DIR = ARTIFACTS / "phase7_selected_two_stage_policy"
SELECTED_POLICY = "stability_2of3_pf110_avg020"


def profit_factor(values):
    s = pd.Series(values).dropna()
    gp = s[s > 0].sum()
    gl = -s[s < 0].sum()
    if gl == 0:
        return math.inf if gp > 0 else 0.0
    return gp / gl


def df_to_markdown(df):
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = "inf" if math.isinf(value) else f"{value:.4f}"
            else:
                value = "" if pd.isna(value) else str(value)
            vals.append(value.replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def summarize(orders):
    rows = []
    for split_name, gsplit in orders.groupby("split", dropna=False):
        for pair, g in gsplit.groupby("pair", dropna=False):
            rows.append({
                "split": split_name,
                "pair": pair,
                "selected_model": ",".join(sorted(g["model"].dropna().unique())),
                "trade_count": int(len(g)),
                "buy_count": int((g["final_signal"] == "BUY").sum()),
                "sell_count": int((g["final_signal"] == "SELL").sum()),
                "net_pips": float(g["pnl_pips"].sum()),
                "avg_pips": float(g["pnl_pips"].mean()),
                "winrate": float((g["pnl_pips"] > 0).mean()),
                "profit_factor": profit_factor(g["pnl_pips"]),
            })
        rows.append({
            "split": split_name,
            "pair": "ALL",
            "selected_model": "pair_specific_stability",
            "trade_count": int(len(gsplit)),
            "buy_count": int((gsplit["final_signal"] == "BUY").sum()),
            "sell_count": int((gsplit["final_signal"] == "SELL").sum()),
            "net_pips": float(gsplit["pnl_pips"].sum()),
            "avg_pips": float(gsplit["pnl_pips"].mean()),
            "winrate": float((gsplit["pnl_pips"] > 0).mean()),
            "profit_factor": profit_factor(gsplit["pnl_pips"]),
        })
    return pd.DataFrame(rows)


def main():
    print("[PROGRESS] START 45_phase7_package_selected_two_stage_policy.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orders = pd.read_csv(PHASE6_DIR / "phase6_stability_orders.csv", low_memory=False)
    selected = pd.read_csv(PHASE6_DIR / "phase6_stability_pair_model_selection.csv")
    policy_orders = orders[orders["policy"].eq(SELECTED_POLICY)].copy()
    policy_selection = selected[selected["policy"].eq(SELECTED_POLICY)].copy()

    summary = summarize(policy_orders)
    summary.to_csv(OUT_DIR / "selected_two_stage_policy_summary.csv", index=False)
    policy_selection.to_csv(OUT_DIR / "selected_two_stage_pair_model_selection.csv", index=False)
    policy_orders.to_csv(OUT_DIR / "selected_two_stage_policy_orders.csv", index=False)

    oos = policy_orders[policy_orders["split"].eq("oos_2024_2026")].copy()
    signal_table = oos[[
        "event_time", "pair", "model", "CountryCode", "Name", "final_signal",
        "prob_trade", "prob_sell", "prob_buy", "direction_confidence",
        "stage1_threshold", "stage2_threshold",
    ]].copy()
    signal_table = signal_table.rename(columns={
        "event_time": "EVENT_TIME",
        "pair": "PAIR",
        "model": "MODEL",
        "CountryCode": "COUNTRY",
        "Name": "EVENT_NAME",
        "final_signal": "SIGNAL",
        "prob_trade": "PROB_TRADE",
        "prob_sell": "PROB_SELL",
        "prob_buy": "PROB_BUY",
        "direction_confidence": "DIRECTION_CONFIDENCE",
        "stage1_threshold": "STAGE1_THRESHOLD",
        "stage2_threshold": "STAGE2_THRESHOLD",
    })
    signal_table["POLICY"] = SELECTED_POLICY
    signal_table["EVENT_TIME"] = pd.to_datetime(signal_table["EVENT_TIME"], errors="coerce").dt.strftime("%Y.%m.%d %H:%M:%S")
    signal_table.to_csv(OUT_DIR / "selected_two_stage_mt5_signal_table.csv", index=False)

    monthly = policy_orders.copy()
    monthly["month"] = pd.to_datetime(monthly["event_time"], errors="coerce").dt.to_period("M").astype(str)
    monthly = monthly.groupby(["split", "pair", "month"], as_index=False)["pnl_pips"].sum()
    monthly["cumulative_pips"] = monthly.groupby(["split", "pair"])["pnl_pips"].cumsum()
    monthly.to_csv(OUT_DIR / "selected_two_stage_monthly_cumulative.csv", index=False)

    # Preserve the upstream report beside the packaged policy for audit traceability.
    src_report = PHASE6_DIR / "LAPORAN_PHASE6_VALIDATION_STABILITY_GATE.md"
    if src_report.exists():
        shutil.copy2(src_report, OUT_DIR / "SOURCE_LAPORAN_PHASE6_VALIDATION_STABILITY_GATE.md")

    oos_all = summary[(summary["split"].eq("oos_2024_2026")) & (summary["pair"].eq("ALL"))]
    val_all = summary[(summary["split"].eq("validation_2021_2023")) & (summary["pair"].eq("ALL"))]
    oos_pair = summary[(summary["split"].eq("oos_2024_2026")) & (~summary["pair"].eq("ALL"))].sort_values("net_pips", ascending=False)
    report = [
        "# Phase 7 - Selected Two-Stage Policy Package",
        "",
        f"Selected policy: `{SELECTED_POLICY}`.",
        "",
        "Policy ini dipilih sebagai kandidat deployment karena memakai gate validasi yang lebih konservatif:",
        "",
        "- minimal validation profit factor 1.10",
        "- minimal average pips 0.20",
        "- minimal 2 dari 3 tahun validation positif",
        "- OOS 2024+ tetap hanya dipakai sebagai evaluasi final, bukan pemilihan model.",
        "",
        "## OOS Summary",
        "",
        df_to_markdown(oos_all[["trade_count", "buy_count", "sell_count", "net_pips", "avg_pips", "winrate", "profit_factor"]]),
        "",
        "## Validation Summary",
        "",
        df_to_markdown(val_all[["trade_count", "buy_count", "sell_count", "net_pips", "avg_pips", "winrate", "profit_factor"]]),
        "",
        "## OOS Per Pair",
        "",
        df_to_markdown(oos_pair[["pair", "selected_model", "trade_count", "net_pips", "avg_pips", "winrate", "profit_factor"]]),
        "",
        "## Pair Selection",
        "",
        df_to_markdown(policy_selection[["pair", "selected_model", "selection_status", "validation_trade_count", "validation_net_pips", "validation_profit_factor", "positive_years", "worst_year_pips"]]),
        "",
        "## Output Files",
        "",
        f"- `{OUT_DIR / 'selected_two_stage_policy_summary.csv'}`",
        f"- `{OUT_DIR / 'selected_two_stage_policy_orders.csv'}`",
        f"- `{OUT_DIR / 'selected_two_stage_pair_model_selection.csv'}`",
        f"- `{OUT_DIR / 'selected_two_stage_mt5_signal_table.csv'}`",
        f"- `{OUT_DIR / 'selected_two_stage_monthly_cumulative.csv'}`",
    ]
    (OUT_DIR / "LAPORAN_PHASE7_SELECTED_TWO_STAGE_POLICY.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'selected_two_stage_mt5_signal_table.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_PHASE7_SELECTED_TWO_STAGE_POLICY.md'}", flush=True)
    print("[PROGRESS] DONE 45_phase7_package_selected_two_stage_policy.py", flush=True)


if __name__ == "__main__":
    main()
