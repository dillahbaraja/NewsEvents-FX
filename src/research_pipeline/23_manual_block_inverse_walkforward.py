from common import *

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
TRADES_IN = MANUAL_DIR / "manual_block_inverse_train_trades_2010_2023.csv"

MODEL_OUT = MANUAL_DIR / "model_block_inverse_walkforward_2010_2023.csv"
VALIDATION_OUT = MANUAL_DIR / "manual_block_inverse_walkforward_validation_2018_2023.csv"
EVENT_SCORES_OUT = MANUAL_DIR / "manual_block_inverse_walkforward_event_scores.csv"
FINAL_ORDERS_OUT = MANUAL_DIR / "manual_block_inverse_walkforward_orders_2010_2023.csv"
TEST_SUMMARY_OUT = MANUAL_DIR / "manual_block_inverse_walkforward_test_2024_2026_summary.csv"
REPORT_OUT = MANUAL_DIR / "LAPORAN_MANUAL_BLOCK_INVERSE_WALKFORWARD.md"

MIN_TRADE_COUNT = 20
MIN_ACTIVE_YEARS = 3
MAX_DOMINANT_YEAR_SHARE = 0.60
VALIDATION_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]


def profit_factor(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(-values[values < 0].sum())
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_drawdown(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    equity = np.cumsum(values)
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def direction_stats(group, col):
    values = group[col].to_numpy(dtype=float)
    yearly = group.groupby("event_year", as_index=False).agg(
        yearly_pips=(col, "sum"),
        yearly_trades=(col, "size"),
    )
    trade_count = int(values.size)
    active_years = int((yearly["yearly_trades"] > 0).sum())
    positive_years = int((yearly["yearly_pips"] > 0).sum())
    positive_year_ratio = positive_years / max(1, active_years)
    abs_total = float(yearly["yearly_pips"].abs().sum())
    dominant_year_share = float(yearly["yearly_pips"].abs().max() / abs_total) if abs_total > 0 else 1.0
    net_pips = float(values.sum())
    avg_pips = net_pips / max(1, trade_count)
    pf = profit_factor(values)
    stability_score = positive_year_ratio * (1.0 - dominant_year_share)
    reliability_score = net_pips * np.log1p(trade_count) * stability_score
    return {
        "trade_count": trade_count,
        "active_years": active_years,
        "positive_years": positive_years,
        "positive_year_ratio": positive_year_ratio,
        "dominant_year_share": dominant_year_share,
        "net_pips": net_pips,
        "avg_pips": avg_pips,
        "profit_factor": pf,
        "stability_score": stability_score,
        "reliability_score": reliability_score,
    }


def choose_action(original, inverse):
    def passes(stats):
        return (
            stats["trade_count"] >= MIN_TRADE_COUNT
            and stats["active_years"] >= MIN_ACTIVE_YEARS
            and stats["dominant_year_share"] <= MAX_DOMINANT_YEAR_SHARE
            and stats["avg_pips"] > 0
            and stats["profit_factor"] > 1.0
            and stats["reliability_score"] > 0
        )

    original_ok = passes(original)
    inverse_ok = passes(inverse)
    if original_ok and (not inverse_ok or original["reliability_score"] >= inverse["reliability_score"]):
        return "ALLOW", "walkforward_allow_reliability"
    if inverse_ok and (not original_ok or inverse["reliability_score"] > original["reliability_score"]):
        return "INVERSE", "walkforward_inverse_reliability"
    return "BLOCK", "walkforward_block_low_reliability"


def build_model(train, selection_period):
    rows = []
    for (pair, country, event_norm), g in train.groupby(["pair", "CountryCode", "event_name_norm"], sort=False):
        event_name = str(g["event_name"].mode().iloc[0]) if not g["event_name"].mode().empty else event_norm
        original = direction_stats(g, "pnl_pips")
        inverse = direction_stats(g, "inverse_pnl_pips")
        action, reason = choose_action(original, inverse)
        rows.append({
            "pair": pair,
            "country": country,
            "event_name": event_name,
            "event_name_norm": event_norm,
            "action": action,
            "reason": reason,
            "selection_period": selection_period,
            "trade_count": original["trade_count"],
            "active_years": original["active_years"],
            "original_net_pips": original["net_pips"],
            "original_profit_factor": original["profit_factor"],
            "original_avg_pips": original["avg_pips"],
            "original_positive_year_ratio": original["positive_year_ratio"],
            "original_dominant_year_share": original["dominant_year_share"],
            "original_reliability_score": original["reliability_score"],
            "inverse_net_pips": inverse["net_pips"],
            "inverse_profit_factor": inverse["profit_factor"],
            "inverse_avg_pips": inverse["avg_pips"],
            "inverse_positive_year_ratio": inverse["positive_year_ratio"],
            "inverse_dominant_year_share": inverse["dominant_year_share"],
            "inverse_reliability_score": inverse["reliability_score"],
        })
    return pd.DataFrame(rows)


def apply_model(data, model):
    decisions = data.merge(
        model[["pair", "country", "event_name_norm", "action", "reason"]].rename(
            columns={"action": "model_action", "reason": "model_reason"}
        ),
        left_on=["pair", "CountryCode", "event_name_norm"],
        right_on=["pair", "country", "event_name_norm"],
        how="left",
    )
    decisions["action"] = decisions["model_action"].fillna("BLOCK_UNKNOWN")
    decisions["reason"] = decisions["model_reason"].fillna("unknown_event_blocked")
    decisions["final_signal"] = "NO_TRADE"
    decisions.loc[decisions["action"].eq("ALLOW"), "final_signal"] = decisions.loc[decisions["action"].eq("ALLOW"), "initial_signal"]
    inv = decisions["action"].eq("INVERSE")
    decisions.loc[inv, "final_signal"] = np.where(decisions.loc[inv, "initial_signal"].eq("BUY"), "SELL", "BUY")
    decisions["final_pnl_pips"] = np.where(
        decisions["action"].eq("INVERSE"),
        decisions["inverse_pnl_pips"],
        np.where(decisions["action"].eq("ALLOW"), decisions["pnl_pips"], 0.0),
    )
    return decisions, decisions[decisions["final_signal"].isin(["BUY", "SELL"])].copy()


def summarize_orders(orders, label, year=None):
    rows = []
    if orders.empty:
        return pd.DataFrame([{
            "sample": label,
            "year": year,
            "pair": "ALL",
            "trades": 0,
            "win_rate": 0.0,
            "net_pips": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pips": 0.0,
        }])
    for pair, g in orders.groupby("pair", sort=False):
        pnl = g["final_pnl_pips"].to_numpy(dtype=float)
        rows.append({
            "sample": label,
            "year": year,
            "pair": pair,
            "trades": int(len(g)),
            "win_rate": float((pnl > 0).mean()),
            "net_pips": float(pnl.sum()),
            "profit_factor": profit_factor(pnl),
            "max_drawdown_pips": max_drawdown(pnl),
            "allow_trades": int(g["action"].eq("ALLOW").sum()),
            "inverse_trades": int(g["action"].eq("INVERSE").sum()),
        })
    return pd.DataFrame(rows)


def aggregate_summary(summary, sample_label):
    active = summary[summary["trades"].gt(0)].copy()
    if active.empty:
        return {
            "sample": sample_label,
            "pairs": 0,
            "trades": 0,
            "net_pips": 0.0,
            "positive_pairs": 0,
            "weighted_win_rate": 0.0,
        }
    return {
        "sample": sample_label,
        "pairs": int(len(active)),
        "trades": int(active["trades"].sum()),
        "net_pips": float(active["net_pips"].sum()),
        "positive_pairs": int((active["net_pips"] > 0).sum()),
        "weighted_win_rate": float((active["trades"] * active["win_rate"]).sum() / active["trades"].sum()),
    }


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
    print("[PROGRESS] START 23_manual_block_inverse_walkforward.py", flush=True)
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

    validation_rows = []
    score_rows = []
    for year in VALIDATION_YEARS:
        train = data[(data["event_year"] >= 2010) & (data["event_year"] < year)].copy()
        valid = data[data["event_year"].eq(year)].copy()
        print(f"[WF] year={year} train={len(train)} valid={len(valid)}", flush=True)
        model = build_model(train, f"walkforward_train_2010_{year-1}_validate_{year}")
        model["validation_year"] = year
        score_rows.append(model)
        _, orders = apply_model(valid, model)
        validation_rows.append(summarize_orders(orders, "walkforward_validation", year))

    validation = pd.concat(validation_rows, ignore_index=True) if validation_rows else pd.DataFrame()
    validation.to_csv(VALIDATION_OUT, index=False)
    event_scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    event_scores.to_csv(EVENT_SCORES_OUT, index=False)

    # Pair deployment filter from walk-forward validation: block pairs that are non-positive over 2018-2023.
    pair_validation = validation.groupby("pair", as_index=False).agg(
        validation_net_pips=("net_pips", "sum"),
        validation_trades=("trades", "sum"),
        validation_positive_years=("net_pips", lambda s: int((s > 0).sum())),
    )
    deploy_pairs = set(pair_validation[pair_validation["validation_net_pips"].gt(0)]["pair"])
    print(f"[WF] deploy_pairs={sorted(deploy_pairs)}", flush=True)

    final_train = data[(data["event_year"] >= 2010) & (data["event_year"] <= 2023)].copy()
    final_model = build_model(final_train, "walkforward_final_train_2010_2023")
    if deploy_pairs:
        final_model.loc[~final_model["pair"].isin(deploy_pairs), "action"] = "BLOCK"
        final_model.loc[~final_model["pair"].isin(deploy_pairs), "reason"] = "walkforward_pair_blocked_by_validation"
    final_model.to_csv(MODEL_OUT, index=False)

    _, final_orders = apply_model(final_train, final_model)
    final_orders.to_csv(FINAL_ORDERS_OUT, index=False)

    validation_agg = pd.DataFrame([aggregate_summary(validation, "walkforward_validation_2018_2023")])
    validation_pair = validation.groupby("pair", as_index=False).agg(
        trades=("trades", "sum"),
        net_pips=("net_pips", "sum"),
        positive_years=("net_pips", lambda s: int((s > 0).sum())),
    ).sort_values("net_pips", ascending=False)
    report = [
        "# Laporan Manual Block/Inverse Walk-Forward",
        "",
        "Model ini memakai unit adaptif pair + country + event_name dan keputusan tiap validasi hanya memakai data masa lalu.",
        "",
        "## Validasi Walk-Forward 2018-2023",
        markdown_table(validation_agg),
        "",
        "## Pair Validation Summary",
        markdown_table(validation_pair),
        "",
        "## Deployment Pair Filter",
        f"- Deploy pairs: {', '.join(sorted(deploy_pairs)) if deploy_pairs else 'NONE'}",
        "",
        "## Output",
        f"- `{MODEL_OUT}`",
        f"- `{VALIDATION_OUT}`",
        f"- `{EVENT_SCORES_OUT}`",
        f"- `{FINAL_ORDERS_OUT}`",
        f"- `{TEST_SUMMARY_OUT}`",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")

    print(f"[OUTPUT] {MODEL_OUT}", flush=True)
    print(f"[OUTPUT] {VALIDATION_OUT}", flush=True)
    print(f"[OUTPUT] {EVENT_SCORES_OUT}", flush=True)
    print(f"[OUTPUT] {FINAL_ORDERS_OUT}", flush=True)
    print(f"[OUTPUT] {REPORT_OUT}", flush=True)
    print(
        f"[RESULT] validation_trades={int(validation_agg['trades'].iloc[0])} "
        f"validation_net_pips={float(validation_agg['net_pips'].iloc[0]):.2f} "
        f"positive_pairs={int(validation_agg['positive_pairs'].iloc[0])}/{int(validation_agg['pairs'].iloc[0])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
