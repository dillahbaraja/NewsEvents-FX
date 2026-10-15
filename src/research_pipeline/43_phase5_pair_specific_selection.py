from common import *

import math


PHASE4_DIR = ARTIFACTS / "phase4_two_stage_evaluation"
OUT_DIR = ARTIFACTS / "phase5_pair_specific_selection"
SUMMARY = PHASE4_DIR / "phase4_two_stage_summary.csv"
ORDERS = PHASE4_DIR / "phase4_two_stage_orders.csv"


POLICIES = {
    "best_validation_net_min500_pf100": {
        "min_trades": 500,
        "min_pf": 1.00,
        "min_avg_pips": 0.00,
        "score": "net_pips",
    },
    "balanced_validation_pf105_avg010": {
        "min_trades": 1000,
        "min_pf": 1.05,
        "min_avg_pips": 0.10,
        "score": "net_pips",
    },
    "conservative_validation_pf110_avg020": {
        "min_trades": 1000,
        "min_pf": 1.10,
        "min_avg_pips": 0.20,
        "score": "profit_quality",
    },
}


def profit_factor(values):
    s = pd.Series(values).dropna()
    gp = s[s > 0].sum()
    gl = -s[s < 0].sum()
    if gl == 0:
        return math.inf if gp > 0 else 0.0
    return gp / gl


def summarize_orders(orders, split_name):
    rows = []
    if orders.empty:
        return pd.DataFrame()
    for (policy, pair), g in orders.groupby(["policy", "pair"], dropna=False):
        rows.append({
            "policy": policy,
            "split": split_name,
            "pair": pair,
            "trade_count": int(len(g)),
            "buy_count": int((g["final_signal"] == "BUY").sum()),
            "sell_count": int((g["final_signal"] == "SELL").sum()),
            "net_pips": float(g["pnl_pips"].sum()),
            "avg_pips": float(g["pnl_pips"].mean()),
            "winrate": float((g["pnl_pips"] > 0).mean()),
            "profit_factor": profit_factor(g["pnl_pips"]),
            "selected_model": ",".join(sorted(g["model"].dropna().unique())),
        })
    for policy, g in orders.groupby("policy", dropna=False):
        rows.append({
            "policy": policy,
            "split": split_name,
            "pair": "ALL",
            "trade_count": int(len(g)),
            "buy_count": int((g["final_signal"] == "BUY").sum()),
            "sell_count": int((g["final_signal"] == "SELL").sum()),
            "net_pips": float(g["pnl_pips"].sum()),
            "avg_pips": float(g["pnl_pips"].mean()),
            "winrate": float((g["pnl_pips"] > 0).mean()),
            "profit_factor": profit_factor(g["pnl_pips"]),
            "selected_model": "pair_specific",
        })
    return pd.DataFrame(rows)


def df_to_markdown(df):
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = "inf" if math.isinf(value) else f"{value:.4f}"
            else:
                value = "" if pd.isna(value) else str(value)
            vals.append(value.replace("|", "\\|"))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def select_models(summary):
    validation = summary[(summary["split"].eq("validation_2021_2023")) & (~summary["pair"].eq("ALL"))].copy()
    selected_rows = []
    audit_rows = []
    for policy_name, rule in POLICIES.items():
        for pair, g in validation.groupby("pair", sort=True):
            candidates = g[
                (g["trade_count"] >= rule["min_trades"])
                & (g["profit_factor"] >= rule["min_pf"])
                & (g["avg_pips"] >= rule["min_avg_pips"])
                & (g["net_pips"] > 0)
            ].copy()
            if candidates.empty:
                selected_rows.append({
                    "policy": policy_name,
                    "pair": pair,
                    "selected_model": "BLOCK_PAIR",
                    "selection_status": "blocked_no_validation_candidate",
                    "validation_trade_count": 0,
                    "validation_net_pips": 0.0,
                    "validation_profit_factor": 0.0,
                    "validation_avg_pips": 0.0,
                    "selection_score": 0.0,
                })
                for _, row in g.iterrows():
                    audit = row.to_dict()
                    audit["policy"] = policy_name
                    audit["candidate_pass"] = False
                    audit_rows.append(audit)
                continue

            if rule["score"] == "profit_quality":
                candidates["selection_score"] = (
                    candidates["net_pips"]
                    + 5000.0 * (candidates["profit_factor"] - 1.0)
                    + 1000.0 * candidates["avg_pips"]
                )
            else:
                candidates["selection_score"] = candidates["net_pips"]
            best = candidates.sort_values(["selection_score", "profit_factor", "trade_count"], ascending=False).iloc[0]
            selected_rows.append({
                "policy": policy_name,
                "pair": pair,
                "selected_model": best["model"],
                "selection_status": "selected_from_validation",
                "validation_trade_count": int(best["trade_count"]),
                "validation_net_pips": float(best["net_pips"]),
                "validation_profit_factor": float(best["profit_factor"]),
                "validation_avg_pips": float(best["avg_pips"]),
                "selection_score": float(best["selection_score"]),
            })
            for _, row in g.iterrows():
                audit = row.to_dict()
                audit["policy"] = policy_name
                audit["candidate_pass"] = row["model"] in set(candidates["model"])
                audit_rows.append(audit)
    return pd.DataFrame(selected_rows), pd.DataFrame(audit_rows)


def apply_selection(orders, selected):
    frames = []
    for _, sel in selected.iterrows():
        if sel["selected_model"] == "BLOCK_PAIR":
            continue
        part = orders[
            (orders["split"].isin(["validation_2021_2023", "oos_2024_2026"]))
            & (orders["pair"].eq(sel["pair"]))
            & (orders["model"].eq(sel["selected_model"]))
        ].copy()
        if part.empty:
            continue
        part["policy"] = sel["policy"]
        part["selection_status"] = sel["selection_status"]
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    print("[PROGRESS] START 43_phase5_pair_specific_selection.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY)
    orders = pd.read_csv(ORDERS, low_memory=False)
    print(f"[DATA] summary_rows={len(summary)} orders_rows={len(orders)}", flush=True)

    selected, audit = select_models(summary)
    selected_orders = apply_selection(orders, selected)
    val_summary = summarize_orders(selected_orders[selected_orders["split"].eq("validation_2021_2023")], "validation_2021_2023")
    oos_summary = summarize_orders(selected_orders[selected_orders["split"].eq("oos_2024_2026")], "oos_2024_2026")
    combined_summary = pd.concat([val_summary, oos_summary], ignore_index=True) if not selected_orders.empty else pd.DataFrame()

    selected.to_csv(OUT_DIR / "phase5_pair_model_selection.csv", index=False)
    audit.to_csv(OUT_DIR / "phase5_pair_model_selection_audit.csv", index=False)
    selected_orders.to_csv(OUT_DIR / "phase5_pair_specific_orders.csv", index=False)
    combined_summary.to_csv(OUT_DIR / "phase5_pair_specific_summary.csv", index=False)

    if not selected_orders.empty:
        monthly = selected_orders.copy()
        monthly["month"] = pd.to_datetime(monthly["event_time"], errors="coerce").dt.to_period("M").astype(str)
        monthly = monthly.groupby(["policy", "split", "pair", "month"], as_index=False)["pnl_pips"].sum()
        monthly["cumulative_pips"] = monthly.groupby(["policy", "split", "pair"])["pnl_pips"].cumsum()
        monthly.to_csv(OUT_DIR / "phase5_pair_specific_monthly_cumulative.csv", index=False)

    oos_all = combined_summary[(combined_summary["split"].eq("oos_2024_2026")) & (combined_summary["pair"].eq("ALL"))].copy()
    oos_all = oos_all.sort_values("net_pips", ascending=False)
    val_all = combined_summary[(combined_summary["split"].eq("validation_2021_2023")) & (combined_summary["pair"].eq("ALL"))].copy()
    val_all = val_all.sort_values("net_pips", ascending=False)
    oos_pair = combined_summary[(combined_summary["split"].eq("oos_2024_2026")) & (~combined_summary["pair"].eq("ALL"))].copy()
    oos_pair = oos_pair.sort_values(["policy", "net_pips"], ascending=[True, False])

    report = [
        "# Phase 5 - Pair-Specific Two-Stage Selection",
        "",
        "Fase ini memilih model terbaik per pair berdasarkan validation 2021-2023 saja, lalu menguji hasilnya pada OOS 2024+.",
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
        df_to_markdown(selected[["policy", "pair", "selected_model", "selection_status", "validation_trade_count", "validation_net_pips", "validation_profit_factor"]]),
        "",
        "## OOS Per Pair",
        "",
        df_to_markdown(oos_pair[["policy", "pair", "selected_model", "trade_count", "net_pips", "winrate", "profit_factor"]]),
        "",
        "## Output Files",
        "",
        f"- `{OUT_DIR / 'phase5_pair_model_selection.csv'}`",
        f"- `{OUT_DIR / 'phase5_pair_specific_summary.csv'}`",
        f"- `{OUT_DIR / 'phase5_pair_specific_orders.csv'}`",
        f"- `{OUT_DIR / 'phase5_pair_specific_monthly_cumulative.csv'}`",
    ]
    (OUT_DIR / "LAPORAN_PHASE5_PAIR_SPECIFIC_SELECTION.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'phase5_pair_specific_summary.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_PHASE5_PAIR_SPECIFIC_SELECTION.md'}", flush=True)
    print("[PROGRESS] DONE 43_phase5_pair_specific_selection.py", flush=True)


if __name__ == "__main__":
    main()
