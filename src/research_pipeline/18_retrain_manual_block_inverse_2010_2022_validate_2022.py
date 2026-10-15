from common import *
import matplotlib.pyplot as plt

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"

ALL_TRADES_OUT = MANUAL_DIR / "manual_block_inverse_train_trades_2010_2022_reset.csv"
CALIB_TRADES_OUT = MANUAL_DIR / "manual_block_inverse_calibration_trades_2010_2021_reset.csv"
MODEL_OUT = MANUAL_DIR / "model_block_inverse_2010_2021_validate_2022.csv"
STATS_OUT = MANUAL_DIR / "manual_block_inverse_event_stats_2010_2021_reset.csv"
YEARLY_OUT = MANUAL_DIR / "manual_block_inverse_yearly_stability_2010_2021_reset.csv"
VALIDATION_ORDERS_OUT = MANUAL_DIR / "manual_block_inverse_validation_orders_2022.csv"
VALIDATION_SUMMARY_OUT = MANUAL_DIR / "manual_block_inverse_validation_summary_2022.csv"
VALIDATION_MONTHLY_OUT = MANUAL_DIR / "manual_block_inverse_validation_monthly_2022.csv"
VALIDATION_CHART_OUT = MANUAL_DIR / "manual_block_inverse_validation_2022_cumulative_by_pair.png"
REPORT_OUT = MANUAL_DIR / "LAPORAN_MANUAL_BLOCK_INVERSE_RESET_2010_2022_VALIDASI_2022.md"

START_YEAR = 2010
ALL_END_YEAR = 2022
CALIB_END_YEAR = 2021
VALIDATION_YEAR = 2022

MIN_EVENTS = 20
MIN_ACTIVE_YEARS = 4
MAX_DOMINANT_YEAR_SHARE = 0.75

INVERSE_KEYWORDS = [
    "unemployment",
    "jobless",
    "claimant",
    "claims",
    "inventories",
    "crude oil inventories",
]


def normalized_event_name(value):
    return " ".join(str(value).strip().lower().split())


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
    direction = 1 if diff > 0 else -1 if diff < 0 else 0
    name = str(row.get("Name", "")).lower()
    if direction and any(k in name for k in INVERSE_KEYWORDS):
        direction *= -1
    return direction


def profit_factor(values):
    s = pd.Series(values).dropna()
    gross_profit = float(s[s > 0].sum())
    gross_loss = float(-s[s < 0].sum())
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def hit_rate(values):
    s = pd.Series(values).dropna()
    return float((s > 0).mean()) if len(s) else 0.0


def stability(group, col):
    yearly = group.groupby("event_year", as_index=False).agg(
        yearly_pips=(col, "sum"),
        yearly_trades=(col, "size"),
    )
    active_years = int((yearly["yearly_trades"] > 0).sum())
    positive_years = int((yearly["yearly_pips"] > 0).sum())
    abs_total = float(yearly["yearly_pips"].abs().sum())
    dominant_share = float(yearly["yearly_pips"].abs().max() / abs_total) if abs_total > 0 else 1.0
    return yearly, active_years, positive_years, dominant_share


def classify(row):
    if row["trade_count"] < MIN_EVENTS:
        return "BLOCK", "reset_trade_count_lt_20"
    if row["active_years"] < MIN_ACTIVE_YEARS:
        return "BLOCK", "reset_active_years_lt_4"
    if row["dominant_year_share"] > MAX_DOMINANT_YEAR_SHARE:
        return "BLOCK", "reset_dominant_year_share_gt_0.75"

    inverse_ok = (
        row["net_pips"] < 0
        and row["inverse_net_pips"] > 0
        and row["inverse_profit_factor"] >= 1.10
        and row["inverse_hit_rate"] >= 0.52
        and row["inverse_active_years"] >= MIN_ACTIVE_YEARS
        and row["inverse_dominant_year_share"] <= MAX_DOMINANT_YEAR_SHARE
    )
    if inverse_ok:
        return "INVERSE", "reset_inverse_pf_hit_rate_stable"

    allow_ok = (
        row["net_pips"] > 0
        and row["profit_factor"] >= 1.05
        and row["hit_rate"] >= 0.50
    )
    if allow_ok:
        return "ALLOW", "reset_positive_pf_hit_rate_stable"

    reasons = []
    if row["profit_factor"] < 0.90:
        reasons.append("reset_profit_factor_lt_0.90")
    if row["net_pips"] <= 0:
        reasons.append("reset_net_pips_le_0")
    if row["hit_rate"] < 0.50:
        reasons.append("reset_hit_rate_lt_0.50")
    return "BLOCK", ";".join(reasons) if reasons else "reset_not_meeting_allow_or_inverse"


def load_baseline_trades():
    usecols = [
        "event_time", "Datetime", "Name", "CountryCode", "Actual", "Forecast", "Previous",
        "Importance", "Impact", "pair", "currency", "orientation", "event_year", "Time_y",
        "Close", "close_fwd_1h", "pip_size", "domestic_return_pips_1h",
    ]
    d = pd.read_csv(
        ROOT / "modeling_dataset_multi_currency.csv",
        usecols=lambda c: c in set(usecols),
        low_memory=False,
    )
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d["event_year"] = pd.to_numeric(d["event_year"], errors="coerce")
    d = d[(d["event_year"] >= START_YEAR) & (d["event_year"] <= ALL_END_YEAR)].copy()
    d = d.dropna(subset=["event_time", "Name", "CountryCode", "pair", "orientation", "domestic_return_pips_1h"])
    d["initial_domestic_signal_num"] = d.apply(macro_surprise_direction, axis=1).astype("int8")
    d = d[d["initial_domestic_signal_num"].ne(0)].copy()
    d["initial_signal"] = np.where(d["initial_domestic_signal_num"].eq(1), "BUY", "SELL")
    d["final_signal"] = d["initial_signal"]
    d["action"] = "BASELINE"
    d["event_name"] = d["Name"].astype(str)
    d["event_name_norm"] = d["event_name"].map(normalized_event_name)
    d["pnl_pips"] = pd.to_numeric(d["domestic_return_pips_1h"], errors="coerce") * d["initial_domestic_signal_num"]
    d["inverse_pnl_pips"] = -d["pnl_pips"]
    d["win"] = d["pnl_pips"] > 0
    d["inverse_win"] = d["inverse_pnl_pips"] > 0
    d["selection_period"] = f"baseline_{START_YEAR}_{ALL_END_YEAR}"
    out_cols = [
        "event_time", "event_year", "pair", "CountryCode", "currency", "orientation",
        "event_name", "event_name_norm", "Actual", "Forecast", "Previous", "Importance", "Impact",
        "initial_signal", "final_signal", "action", "Close", "close_fwd_1h",
        "domestic_return_pips_1h", "pnl_pips", "inverse_pnl_pips", "win", "inverse_win",
        "selection_period",
    ]
    d = d[out_cols].dropna(subset=["pnl_pips", "inverse_pnl_pips"]).copy()
    d["event_year"] = pd.to_numeric(d["event_year"], errors="coerce").astype("Int64")
    return d


def build_model(calib):
    model_rows = []
    yearly_rows = []
    for key, g in calib.groupby(["pair", "CountryCode", "event_name_norm"], sort=False):
        pair, country, event_norm = key
        event_name = str(g["event_name"].mode().iloc[0]) if not g["event_name"].mode().empty else event_norm
        yearly, active_years, positive_years, dominant_share = stability(g, "pnl_pips")
        inv_yearly, inv_active_years, inv_positive_years, inv_dominant_share = stability(g, "inverse_pnl_pips")
        for _, y in yearly.iterrows():
            inv = inv_yearly[inv_yearly["event_year"].eq(y["event_year"])]
            yearly_rows.append({
                "pair": pair,
                "country": country,
                "event_name": event_name,
                "event_name_norm": event_norm,
                "event_year": int(y["event_year"]),
                "yearly_trades": int(y["yearly_trades"]),
                "yearly_pips": float(y["yearly_pips"]),
                "inverse_yearly_pips": float(inv["yearly_pips"].iloc[0]) if not inv.empty else 0.0,
            })
        row = {
            "pair": pair,
            "country": country,
            "event_name": event_name,
            "event_name_norm": event_norm,
            "trade_count": int(len(g)),
            "active_years": active_years,
            "positive_years": positive_years,
            "dominant_year_share": dominant_share,
            "net_pips": float(g["pnl_pips"].sum()),
            "profit_factor": profit_factor(g["pnl_pips"]),
            "hit_rate": hit_rate(g["pnl_pips"]),
            "inverse_net_pips": float(g["inverse_pnl_pips"].sum()),
            "inverse_profit_factor": profit_factor(g["inverse_pnl_pips"]),
            "inverse_hit_rate": hit_rate(g["inverse_pnl_pips"]),
            "inverse_active_years": inv_active_years,
            "inverse_positive_years": inv_positive_years,
            "inverse_dominant_year_share": inv_dominant_share,
            "selection_period": f"calibration_{START_YEAR}_{CALIB_END_YEAR}_validate_{VALIDATION_YEAR}",
        }
        action, reason = classify(row)
        row["action"] = action
        row["reason"] = reason
        model_rows.append(row)
    return pd.DataFrame(model_rows), pd.DataFrame(yearly_rows)


def apply_model(validation, model):
    decisions = validation.merge(
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
    orders["model"] = "manual_block_inverse_reset"
    orders["signal"] = orders["final_signal"]
    orders["country"] = orders["CountryCode"]
    return decisions, orders


def summarize(orders, decisions):
    rows = []
    for pair, g_dec in decisions.groupby("pair", sort=False):
        g = orders[orders["pair"].eq(pair)]
        trades = int(len(g))
        wins = int((g["pnl_pips"] > 0).sum()) if trades else 0
        rows.append({
            "pair": pair,
            "model": "manual_block_inverse_reset",
            "period_start": f"{VALIDATION_YEAR}-01-01",
            "period_end": f"{VALIDATION_YEAR}-12-31",
            "events": int(len(g_dec)),
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
    monthly["model"] = "manual_block_inverse_reset"
    monthly = monthly.sort_values(["pair", "month"])
    monthly["cumulative_pips"] = monthly.groupby("pair")["monthly_pips"].cumsum()
    return monthly


def plot_monthly(monthly):
    if monthly.empty:
        return
    plt.figure(figsize=(13, 7))
    for pair in sorted(monthly["pair"].unique()):
        g = monthly[monthly["pair"].eq(pair)]
        plt.plot(g["month"], g["cumulative_pips"], marker="o", linewidth=1.4, markersize=3, label=pair)
    plt.axhline(0, color="#444444", linewidth=0.9)
    plt.title("Manual Block/Inverse Reset - Validation 2022 Monthly Cumulative Profit by Pair")
    plt.xlabel("Month")
    plt.ylabel("Cumulative profit (pips)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(VALIDATION_CHART_OUT, dpi=160)
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


def write_report(summary, model):
    active = summary[summary["trades"].gt(0)].copy()
    total_trades = int(active["trades"].sum()) if not active.empty else 0
    total_pips = float(active["net_pips"].sum()) if not active.empty else 0.0
    weighted_wins = float((active["trades"] * active["win_rate"]).sum()) if not active.empty else 0.0
    positive_pairs = int((active["net_pips"] > 0).sum()) if not active.empty else 0
    active_pairs = int(len(active))
    positive_pair_ratio = positive_pairs / active_pairs if active_pairs else 0.0
    pass_60 = positive_pair_ratio >= 0.60
    action_counts = model["action"].value_counts().to_dict() if not model.empty else {}
    report = [
        "# Laporan Reset Manual Block/Inverse 2010-2022 dan Validasi 2022",
        "",
        "## Desain",
        f"- Baseline event-study dibangun ulang dari kosong untuk periode {START_YEAR}-{ALL_END_YEAR}.",
        f"- Model block/inverse dibentuk dari periode kalibrasi {START_YEAR}-{CALIB_END_YEAR}.",
        f"- Validasi bersih dilakukan pada tahun {VALIDATION_YEAR}; data {VALIDATION_YEAR} tidak dipakai untuk memilih rule model.",
        "- Sinyal dasar berasal dari surprise makro Actual vs Forecast, fallback Actual vs Previous, lalu disesuaikan arah domestic currency.",
        "",
        "## Ringkasan Validasi 2022",
        f"- Total trades: {total_trades}",
        f"- Winrate tertimbang: {weighted_wins / total_trades:.2%}" if total_trades else "- Winrate tertimbang: 0.00%",
        f"- Net pips total: {total_pips:.2f}",
        f"- Positive pairs: {positive_pairs}/{active_pairs} ({positive_pair_ratio:.2%})",
        f"- Target profit di atas 60% pair positif: {'TERCAPAI' if pass_60 else 'BELUM TERCAPAI'}",
        f"- Distribusi action model: {action_counts}",
        "",
        "## Hasil Per Pair",
        markdown_table(summary.sort_values("net_pips", ascending=False)),
        "",
        "## Output",
        f"- `{ALL_TRADES_OUT}`",
        f"- `{CALIB_TRADES_OUT}`",
        f"- `{MODEL_OUT}`",
        f"- `{STATS_OUT}`",
        f"- `{YEARLY_OUT}`",
        f"- `{VALIDATION_ORDERS_OUT}`",
        f"- `{VALIDATION_SUMMARY_OUT}`",
        f"- `{VALIDATION_MONTHLY_OUT}`",
        f"- `{VALIDATION_CHART_OUT}`",
    ]
    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")


def main():
    print("[PROGRESS] START 18_retrain_manual_block_inverse_2010_2022_validate_2022.py", flush=True)
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    trades = load_baseline_trades()
    trades.to_csv(ALL_TRADES_OUT, index=False)
    print(f"[OUTPUT] {ALL_TRADES_OUT}", flush=True)
    print(f"[DATA] baseline_trades_2010_2022={len(trades)}", flush=True)

    calib = trades[(trades["event_year"] >= START_YEAR) & (trades["event_year"] <= CALIB_END_YEAR)].copy()
    validation = trades[trades["event_year"].eq(VALIDATION_YEAR)].copy()
    calib.to_csv(CALIB_TRADES_OUT, index=False)
    print(f"[OUTPUT] {CALIB_TRADES_OUT}", flush=True)
    print(f"[DATA] calibration_trades_2010_2021={len(calib)} validation_candidates_2022={len(validation)}", flush=True)

    model, yearly = build_model(calib)
    model.to_csv(MODEL_OUT, index=False)
    model.to_csv(STATS_OUT, index=False)
    yearly.to_csv(YEARLY_OUT, index=False)
    print(f"[OUTPUT] {MODEL_OUT}", flush=True)
    print(f"[OUTPUT] {STATS_OUT}", flush=True)
    print(f"[OUTPUT] {YEARLY_OUT}", flush=True)
    print(f"[DATA] model_actions={model['action'].value_counts().to_dict()}", flush=True)

    decisions, orders = apply_model(validation, model)
    order_cols = [
        "event_time", "pair", "model", "event_name", "country", "initial_signal", "action",
        "signal", "pnl_pips", "reason",
    ]
    orders[order_cols].to_csv(VALIDATION_ORDERS_OUT, index=False)
    summary = summarize(orders, decisions).sort_values("net_pips", ascending=False)
    summary.to_csv(VALIDATION_SUMMARY_OUT, index=False)
    monthly = build_monthly(orders)
    monthly.to_csv(VALIDATION_MONTHLY_OUT, index=False)
    plot_monthly(monthly)
    write_report(summary, model)
    print(f"[OUTPUT] {VALIDATION_ORDERS_OUT}", flush=True)
    print(f"[OUTPUT] {VALIDATION_SUMMARY_OUT}", flush=True)
    print(f"[OUTPUT] {VALIDATION_MONTHLY_OUT}", flush=True)
    print(f"[OUTPUT] {VALIDATION_CHART_OUT}", flush=True)
    print(f"[OUTPUT] {REPORT_OUT}", flush=True)

    active = summary[summary["trades"].gt(0)]
    total_trades = int(active["trades"].sum()) if not active.empty else 0
    total_pips = float(active["net_pips"].sum()) if not active.empty else 0.0
    positive_pairs = int((active["net_pips"] > 0).sum()) if not active.empty else 0
    print(f"[RESULT] validation_2022_trades={total_trades} net_pips={total_pips:.2f} positive_pairs={positive_pairs}/{len(active)}", flush=True)


if __name__ == "__main__":
    main()
