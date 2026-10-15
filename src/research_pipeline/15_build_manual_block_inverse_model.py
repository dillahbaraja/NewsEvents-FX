from common import *

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
TRAIN_TRADES = MANUAL_DIR / "manual_block_inverse_train_trades_2010_2023.csv"
MODEL_OUT = MANUAL_DIR / "model_block_inverse.csv"
STATS_OUT = MANUAL_DIR / "manual_block_inverse_event_stats.csv"
YEARLY_OUT = MANUAL_DIR / "manual_block_inverse_yearly_stability.csv"

MIN_EVENTS = 20
MIN_ACTIVE_YEARS = 4


def pf(values):
    s = pd.Series(values).dropna()
    gp = float(s[s > 0].sum())
    gl = float(-s[s < 0].sum())
    if gl == 0:
        return np.inf if gp > 0 else 0.0
    return gp / gl


def hit(values):
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
    stable = active_years >= MIN_ACTIVE_YEARS and dominant_share <= 0.75
    return yearly, active_years, positive_years, dominant_share, stable


def classify(row):
    reasons = []
    if row["trade_count"] < MIN_EVENTS:
        return "BLOCK", "trade_count_lt_20"
    if row["active_years"] < MIN_ACTIVE_YEARS:
        return "BLOCK", "active_years_lt_4"
    if row["dominant_year_share"] > 0.75:
        return "BLOCK", "dominant_single_year_profit"

    inverse_ok = (
        row["net_pips"] < 0
        and row["inverse_net_pips"] > 0
        and row["inverse_profit_factor"] >= 1.10
        and row["inverse_hit_rate"] >= 0.52
        and row["inverse_active_years"] >= MIN_ACTIVE_YEARS
        and row["inverse_dominant_year_share"] <= 0.75
    )
    if inverse_ok:
        return "INVERSE", "inverse_pf_hit_rate_stable"

    if row["profit_factor"] < 0.90:
        reasons.append("profit_factor_lt_0.90")
    if row["net_pips"] <= 0 and row["hit_rate"] < 0.50:
        reasons.append("net_pips_le_0_and_hit_rate_lt_0.50")
    if reasons:
        return "BLOCK", ";".join(reasons)

    allow_ok = row["net_pips"] > 0 and row["profit_factor"] >= 1.05 and row["hit_rate"] >= 0.50
    if allow_ok:
        return "ALLOW", "positive_pf_hit_rate_stable"
    return "BLOCK", "not_meeting_allow_or_inverse"


def main():
    print("[PROGRESS] START 15_build_manual_block_inverse_model.py", flush=True)
    if not TRAIN_TRADES.exists():
        raise FileNotFoundError(f"Missing train trades: {TRAIN_TRADES}")
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(TRAIN_TRADES, low_memory=False)
    d["event_year"] = pd.to_numeric(d["event_year"], errors="coerce").astype("Int64")
    d["pnl_pips"] = pd.to_numeric(d["pnl_pips"], errors="coerce")
    d["inverse_pnl_pips"] = pd.to_numeric(d["inverse_pnl_pips"], errors="coerce")
    d = d.dropna(subset=["event_year", "pair", "CountryCode", "event_name_norm", "pnl_pips", "inverse_pnl_pips"]).copy()
    print(f"[DATA] train_trades={len(d)}", flush=True)

    model_rows = []
    yearly_rows = []
    for key, g in d.groupby(["pair", "CountryCode", "event_name_norm"], sort=False):
        pair, country, event_norm = key
        event_name = str(g["event_name"].mode().iloc[0]) if not g["event_name"].mode().empty else event_norm
        yearly, active_years, positive_years, dominant_share, stable = stability(g, "pnl_pips")
        inv_yearly, inv_active_years, inv_positive_years, inv_dominant_share, inv_stable = stability(g, "inverse_pnl_pips")
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
            "action": "",
            "trade_count": int(len(g)),
            "active_years": active_years,
            "positive_years": positive_years,
            "dominant_year_share": dominant_share,
            "net_pips": float(g["pnl_pips"].sum()),
            "profit_factor": pf(g["pnl_pips"]),
            "hit_rate": hit(g["pnl_pips"]),
            "inverse_net_pips": float(g["inverse_pnl_pips"].sum()),
            "inverse_profit_factor": pf(g["inverse_pnl_pips"]),
            "inverse_hit_rate": hit(g["inverse_pnl_pips"]),
            "inverse_active_years": inv_active_years,
            "inverse_positive_years": inv_positive_years,
            "inverse_dominant_year_share": inv_dominant_share,
            "reason": "",
            "selection_period": "calibration_2010_2023",
        }
        action, reason = classify(row)
        row["action"] = action
        row["reason"] = reason
        model_rows.append(row)

    model = pd.DataFrame(model_rows)
    model.to_csv(MODEL_OUT, index=False)
    model.to_csv(STATS_OUT, index=False)
    pd.DataFrame(yearly_rows).to_csv(YEARLY_OUT, index=False)
    print(f"[OUTPUT] {MODEL_OUT}", flush=True)
    print(f"[OUTPUT] {STATS_OUT}", flush=True)
    print(f"[OUTPUT] {YEARLY_OUT}", flush=True)
    print(f"[DATA] actions={model['action'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
