from common import *

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
TRADES_IN = MANUAL_DIR / "manual_block_inverse_train_trades_2010_2023.csv"

MODEL_OUT = MANUAL_DIR / "model_block_inverse_pair_country_event_try_both_2010_2023.csv"
TRAIN_SUMMARY_OUT = MANUAL_DIR / "manual_block_inverse_try_both_train_summary_2010_2023.csv"
TEST_SUMMARY_COPY_OUT = MANUAL_DIR / "manual_block_inverse_try_both_test_summary_2024_2026.csv"
TEST_ORDERS_COPY_OUT = MANUAL_DIR / "manual_block_inverse_try_both_orders_2024_2026.csv"
REPORT_OUT = MANUAL_DIR / "LAPORAN_MANUAL_BLOCK_INVERSE_TRY_BOTH.md"

MIN_TRADE_COUNT = 10
MIN_ACTIVE_YEARS = 2
WEAK_DOMINANT_YEAR_SHARE = 0.75


def profit_factor(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(-values[values < 0].sum())
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def yearly_stats(group, col):
    yearly = group.groupby("event_year", as_index=False).agg(
        yearly_pips=(col, "sum"),
        yearly_trades=(col, "size"),
    )
    active_years = int((yearly["yearly_trades"] > 0).sum())
    positive_years = int((yearly["yearly_pips"] > 0).sum())
    abs_total = float(yearly["yearly_pips"].abs().sum())
    dominant_share = float(yearly["yearly_pips"].abs().max() / abs_total) if abs_total > 0 else 1.0
    return active_years, positive_years, dominant_share


def direction_metrics(group, col):
    values = group[col].to_numpy(dtype=float)
    active_years, positive_years, dominant_share = yearly_stats(group, col)
    net = float(values.sum())
    trade_count = int(values.size)
    return {
        "net_pips": net,
        "gross_profit": float(values[values > 0].sum()),
        "gross_loss": float(-values[values < 0].sum()),
        "profit_factor": profit_factor(values),
        "avg_pips": net / max(1, trade_count),
        "active_years": active_years,
        "positive_years": positive_years,
        "positive_year_ratio": positive_years / max(1, active_years),
        "dominant_year_share": dominant_share,
    }


def direction_ok(metrics, trade_count):
    return (
        trade_count >= MIN_TRADE_COUNT
        and metrics["active_years"] >= MIN_ACTIVE_YEARS
        and metrics["net_pips"] > 0
        and metrics["profit_factor"] > 1.0
        and metrics["avg_pips"] > 0
    )


def choose_action(original, inverse, trade_count):
    original_ok = direction_ok(original, trade_count)
    inverse_ok = direction_ok(inverse, trade_count)
    weak = False
    if original_ok and inverse_ok:
        if inverse["net_pips"] > original["net_pips"]:
            action = "INVERSE"
            selected = inverse
            reason = "try_both_inverse_better_net"
        elif original["net_pips"] > inverse["net_pips"]:
            action = "ALLOW"
            selected = original
            reason = "try_both_allow_better_net"
        elif inverse["profit_factor"] > original["profit_factor"]:
            action = "INVERSE"
            selected = inverse
            reason = "try_both_inverse_better_pf"
        else:
            action = "ALLOW"
            selected = original
            reason = "try_both_allow_better_pf"
    elif inverse_ok:
        action = "INVERSE"
        selected = inverse
        reason = "try_both_original_bad_inverse_profit"
    elif original_ok:
        action = "ALLOW"
        selected = original
        reason = "try_both_original_profit"
    else:
        return "BLOCK", "try_both_both_directions_bad", False

    if selected["dominant_year_share"] > WEAK_DOMINANT_YEAR_SHARE:
        weak = True
        action = "WEAK_" + action
        reason += "_weak_year_dominance"
    return action, reason, weak


def summarize_orders(orders, label):
    rows = []
    for pair, g in orders.groupby("pair", sort=False):
        pnl = g["final_pnl_pips"].to_numpy(dtype=float)
        rows.append({
            "sample": label,
            "pair": pair,
            "trades": int(len(g)),
            "win_rate": float((pnl > 0).mean()) if len(g) else 0.0,
            "net_pips": float(pnl.sum()) if len(g) else 0.0,
            "profit_factor": profit_factor(pnl) if len(g) else 0.0,
            "allow_trades": int(g["action"].isin(["ALLOW", "WEAK_ALLOW"]).sum()),
            "inverse_trades": int(g["action"].isin(["INVERSE", "WEAK_INVERSE"]).sum()),
        })
    return pd.DataFrame(rows).sort_values("net_pips", ascending=False)


def apply_model(data, model):
    d = data.merge(
        model[["pair", "country", "event_name_norm", "action", "reason"]].rename(
            columns={"action": "model_action", "reason": "model_reason"}
        ),
        left_on=["pair", "CountryCode", "event_name_norm"],
        right_on=["pair", "country", "event_name_norm"],
        how="left",
    )
    d["action"] = d["model_action"].fillna("BLOCK_UNKNOWN")
    d["reason"] = d["model_reason"].fillna("unknown_event_blocked")
    d["final_signal"] = "NO_TRADE"
    allow = d["action"].isin(["ALLOW", "WEAK_ALLOW"])
    inverse = d["action"].isin(["INVERSE", "WEAK_INVERSE"])
    d.loc[allow, "final_signal"] = d.loc[allow, "initial_signal"]
    d.loc[inverse, "final_signal"] = np.where(d.loc[inverse, "initial_signal"].eq("BUY"), "SELL", "BUY")
    d["final_pnl_pips"] = np.where(
        inverse,
        d["inverse_pnl_pips"],
        np.where(allow, d["pnl_pips"], 0.0),
    )
    return d, d[d["final_signal"].isin(["BUY", "SELL"])].copy()


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
    print("[PROGRESS] START 24_manual_block_inverse_try_both.py", flush=True)
    if not TRADES_IN.exists():
        raise FileNotFoundError(f"Missing trades input: {TRADES_IN}")
    data = pd.read_csv(TRADES_IN, low_memory=False)
    data["event_year"] = pd.to_numeric(data["event_year"], errors="coerce")
    data["event_time"] = pd.to_datetime(data["event_time"], errors="coerce")
    for col in ["pnl_pips", "inverse_pnl_pips"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    data = data.dropna(subset=["event_year", "event_time", "pair", "CountryCode", "event_name_norm"]).copy()
    data = data.sort_values(["event_time", "pair"]).reset_index(drop=True)
    print(f"[DATA] train_trades={len(data)}", flush=True)

    rows = []
    for (pair, country, event_norm), g in data.groupby(["pair", "CountryCode", "event_name_norm"], sort=False):
        original = direction_metrics(g, "pnl_pips")
        inverse = direction_metrics(g, "inverse_pnl_pips")
        action, reason, weak = choose_action(original, inverse, int(len(g)))
        event_name = str(g["event_name"].mode().iloc[0]) if not g["event_name"].mode().empty else event_norm
        rows.append({
            "pair": pair,
            "country": country,
            "event_name": event_name,
            "event_name_norm": event_norm,
            "action": action,
            "reason": reason,
            "selection_period": "try_both_train_2010_2023",
            "trade_count": int(len(g)),
            "original_net_pips": original["net_pips"],
            "original_profit_factor": original["profit_factor"],
            "original_avg_pips": original["avg_pips"],
            "original_active_years": original["active_years"],
            "original_positive_year_ratio": original["positive_year_ratio"],
            "original_dominant_year_share": original["dominant_year_share"],
            "inverse_net_pips": inverse["net_pips"],
            "inverse_profit_factor": inverse["profit_factor"],
            "inverse_avg_pips": inverse["avg_pips"],
            "inverse_active_years": inverse["active_years"],
            "inverse_positive_year_ratio": inverse["positive_year_ratio"],
            "inverse_dominant_year_share": inverse["dominant_year_share"],
            "weak_flag": weak,
        })
    model = pd.DataFrame(rows)
    model.to_csv(MODEL_OUT, index=False)

    _, train_orders = apply_model(data, model)
    train_summary = summarize_orders(train_orders, "try_both_train_2010_2023")
    train_summary.to_csv(TRAIN_SUMMARY_OUT, index=False)

    report = [
        "# Laporan Manual Block/Inverse Try Both",
        "",
        "Model ini memakai unit pair + country + event_name. Setiap unit diuji arah original dan inverse pada 2010-2023.",
        "",
        "## Action Count",
        markdown_table(model.groupby("action", as_index=False).size().rename(columns={"size": "count"})),
        "",
        "## Train Summary 2010-2023",
        markdown_table(train_summary),
        "",
        "## Output",
        f"- `{MODEL_OUT}`",
        f"- `{TRAIN_SUMMARY_OUT}`",
        f"- `{TEST_SUMMARY_COPY_OUT}`",
        f"- `{TEST_ORDERS_COPY_OUT}`",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")
    print(f"[OUTPUT] {MODEL_OUT}", flush=True)
    print(f"[OUTPUT] {TRAIN_SUMMARY_OUT}", flush=True)
    print(f"[OUTPUT] {REPORT_OUT}", flush=True)
    print(f"[DATA] actions={model['action'].value_counts().to_dict()}", flush=True)
    print(
        f"[RESULT] train_trades={len(train_orders)} train_net_pips={train_orders['final_pnl_pips'].sum():.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
