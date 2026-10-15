from common import *

import math


PHASE4_DIR = ARTIFACTS / "phase4_two_stage_evaluation"
OUT_DIR = ARTIFACTS / "phase6_validation_stability_gate"
SUMMARY = PHASE4_DIR / "phase4_two_stage_summary.csv"
ORDERS = PHASE4_DIR / "phase4_two_stage_orders.csv"


POLICIES = {
    "stability_2of3_pf103_avg010": {
        "min_trades": 1000,
        "min_pf": 1.03,
        "min_avg_pips": 0.10,
        "min_positive_years": 2,
        "max_worst_year_loss": -1500.0,
    },
    "stability_3of3_pf100_avg000": {
        "min_trades": 1000,
        "min_pf": 1.00,
        "min_avg_pips": 0.00,
        "min_positive_years": 3,
        "max_worst_year_loss": 0.0,
    },
    "stability_2of3_pf110_avg020": {
        "min_trades": 1000,
        "min_pf": 1.10,
        "min_avg_pips": 0.20,
        "min_positive_years": 2,
        "max_worst_year_loss": -1000.0,
    },
}


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


def summarize_orders(orders, split_name):
    rows = []
    if orders.empty:
        return pd.DataFrame()
    for (policy, pair), g in orders.groupby(["policy", "pair"], dropna=False):
        rows.append({
            "policy": policy,
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
    for policy, g in orders.groupby("policy", dropna=False):
        rows.append({
            "policy": policy,
            "split": split_name,
            "pair": "ALL",
            "selected_model": "pair_specific_stability",
            "trade_count": int(len(g)),
            "buy_count": int((g["final_signal"] == "BUY").sum()),
            "sell_count": int((g["final_signal"] == "SELL").sum()),
            "net_pips": float(g["pnl_pips"].sum()),
            "avg_pips": float(g["pnl_pips"].mean()),
            "winrate": float((g["pnl_pips"] > 0).mean()),
            "profit_factor": profit_factor(g["pnl_pips"]),
        })
    return pd.DataFrame(rows)


def build_yearly_stats(orders):
    val = orders[orders["split"].eq("validation_2021_2023")].copy()
    val["year"] = pd.to_datetime(val["event_time"], errors="coerce").dt.year
    yearly = val.groupby(["model", "pair", "year"], as_index=False)["pnl_pips"].sum()
    rows = []
    for (model, pair), g in yearly.groupby(["model", "pair"]):
        year_pips = {int(r["year"]): float(r["pnl_pips"]) for _, r in g.iterrows()}
        values = [year_pips.get(y, 0.0) for y in [2021, 2022, 2023]]
        rows.append({
            "model": model,
            "pair": pair,
            "net_2021": values[0],
            "net_2022": values[1],
            "net_2023": values[2],
            "positive_years": int(sum(v > 0 for v in values)),
            "worst_year_pips": float(min(values)),
            "best_year_pips": float(max(values)),
        })
    return pd.DataFrame(rows)


def select_models(summary, yearly):
    validation = summary[(summary["split"].eq("validation_2021_2023")) & (~summary["pair"].eq("ALL"))].copy()
    validation = validation.merge(yearly, on=["model", "pair"], how="left")
    selected_rows = []
    for policy_name, rule in POLICIES.items():
        for pair, g in validation.groupby("pair", sort=True):
            candidates = g[
                (g["trade_count"] >= rule["min_trades"])
                & (g["profit_factor"] >= rule["min_pf"])
                & (g["avg_pips"] >= rule["min_avg_pips"])
                & (g["net_pips"] > 0)
                & (g["positive_years"] >= rule["min_positive_years"])
                & (g["worst_year_pips"] >= rule["max_worst_year_loss"])
            ].copy()
            if candidates.empty:
                selected_rows.append({
                    "policy": policy_name,
                    "pair": pair,
                    "selected_model": "BLOCK_PAIR",
                    "selection_status": "blocked_no_stable_validation_candidate",
                    "validation_trade_count": 0,
                    "validation_net_pips": 0.0,
                    "validation_profit_factor": 0.0,
                    "positive_years": 0,
                    "worst_year_pips": 0.0,
                    "selection_score": 0.0,
                })
                continue
            candidates["selection_score"] = (
                candidates["net_pips"]
                + 1000.0 * candidates["positive_years"]
                + 2500.0 * (candidates["profit_factor"] - 1.0)
                + 0.25 * candidates["worst_year_pips"]
            )
            best = candidates.sort_values(["selection_score", "net_pips"], ascending=False).iloc[0]
            selected_rows.append({
                "policy": policy_name,
                "pair": pair,
                "selected_model": best["model"],
                "selection_status": "selected_from_validation_stability",
                "validation_trade_count": int(best["trade_count"]),
                "validation_net_pips": float(best["net_pips"]),
                "validation_profit_factor": float(best["profit_factor"]),
                "positive_years": int(best["positive_years"]),
                "worst_year_pips": float(best["worst_year_pips"]),
                "selection_score": float(best["selection_score"]),
            })
    return pd.DataFrame(selected_rows), validation


def apply_selection(orders, selected):
    frames = []
    for _, sel in selected.iterrows():
        if sel["selected_model"] == "BLOCK_PAIR":
            continue
        part = orders[
            orders["split"].isin(["validation_2021_2023", "oos_2024_2026"])
            & orders["pair"].eq(sel["pair"])
            & orders["model"].eq(sel["selected_model"])
        ].copy()
        if part.empty:
            continue
        part["policy"] = sel["policy"]
        part["selection_status"] = sel["selection_status"]
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    print("[PROGRESS] START 44_phase6_validation_stability_gate.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY)
    orders = pd.read_csv(ORDERS, low_memory=False)
    print(f"[DATA] summary_rows={len(summary)} orders_rows={len(orders)}", flush=True)
    yearly = build_yearly_stats(orders)
    selected, candidate_audit = select_models(summary, yearly)
    selected_orders = apply_selection(orders, selected)
    val_summary = summarize_orders(selected_orders[selected_orders["split"].eq("validation_2021_2023")], "validation_2021_2023")
    oos_summary = summarize_orders(selected_orders[selected_orders["split"].eq("oos_2024_2026")], "oos_2024_2026")
    combined = pd.concat([val_summary, oos_summary], ignore_index=True) if not selected_orders.empty else pd.DataFrame()

    selected.to_csv(OUT_DIR / "phase6_stability_pair_model_selection.csv", index=False)
    yearly.to_csv(OUT_DIR / "phase6_validation_yearly_model_pair_stats.csv", index=False)
    candidate_audit.to_csv(OUT_DIR / "phase6_candidate_audit.csv", index=False)
    selected_orders.to_csv(OUT_DIR / "phase6_stability_orders.csv", index=False)
    combined.to_csv(OUT_DIR / "phase6_stability_summary.csv", index=False)

    if not selected_orders.empty:
        monthly = selected_orders.copy()
        monthly["month"] = pd.to_datetime(monthly["event_time"], errors="coerce").dt.to_period("M").astype(str)
        monthly = monthly.groupby(["policy", "split", "pair", "month"], as_index=False)["pnl_pips"].sum()
        monthly["cumulative_pips"] = monthly.groupby(["policy", "split", "pair"])["pnl_pips"].cumsum()
        monthly.to_csv(OUT_DIR / "phase6_stability_monthly_cumulative.csv", index=False)

    oos_all = combined[(combined["split"].eq("oos_2024_2026")) & (combined["pair"].eq("ALL"))].sort_values("net_pips", ascending=False)
    val_all = combined[(combined["split"].eq("validation_2021_2023")) & (combined["pair"].eq("ALL"))].sort_values("net_pips", ascending=False)
    oos_pair = combined[(combined["split"].eq("oos_2024_2026")) & (~combined["pair"].eq("ALL"))].sort_values(["policy", "net_pips"], ascending=[True, False])

    report = [
        "# Phase 6 - Validation Stability Gate",
        "",
        "Fase ini menambahkan yearly stability gate berbasis validation 2021-2023. OOS 2024+ tetap tidak dipakai untuk memilih model.",
        "",
        "## OOS Ranking - Policy Level",
        "",
        df_to_markdown(oos_all[["policy", "trade_count", "buy_count", "sell_count", "net_pips", "avg_pips", "winrate", "profit_factor"]]),
        "",
        "## Validation Ranking - Policy Level",
        "",
        df_to_markdown(val_all[["policy", "trade_count", "buy_count", "sell_count", "net_pips", "avg_pips", "winrate", "profit_factor"]]),
        "",
        "## Pair Model Selection",
        "",
        df_to_markdown(selected[["policy", "pair", "selected_model", "selection_status", "validation_trade_count", "validation_net_pips", "validation_profit_factor", "positive_years", "worst_year_pips"]]),
        "",
        "## OOS Per Pair",
        "",
        df_to_markdown(oos_pair[["policy", "pair", "selected_model", "trade_count", "net_pips", "winrate", "profit_factor"]]),
        "",
        "## Output Files",
        "",
        f"- `{OUT_DIR / 'phase6_stability_summary.csv'}`",
        f"- `{OUT_DIR / 'phase6_stability_orders.csv'}`",
        f"- `{OUT_DIR / 'phase6_stability_pair_model_selection.csv'}`",
        f"- `{OUT_DIR / 'phase6_validation_yearly_model_pair_stats.csv'}`",
    ]
    (OUT_DIR / "LAPORAN_PHASE6_VALIDATION_STABILITY_GATE.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'phase6_stability_summary.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_PHASE6_VALIDATION_STABILITY_GATE.md'}", flush=True)
    print("[PROGRESS] DONE 44_phase6_validation_stability_gate.py", flush=True)


if __name__ == "__main__":
    main()
