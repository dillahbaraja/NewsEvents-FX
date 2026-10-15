from common import *
from pathlib import Path
import json

ART_DIR = ARTIFACTS / "event_study_v3"
SUMMARY = ART_DIR / "event_study_v3_best_oos_by_pair.csv"
SIGNAL_TABLE = ARTIFACTS / "event_study_v3_signal_table.csv"
MODEL_NAME = "event_study_v3"


def mt5_time(value):
    return pd.to_datetime(value).strftime("%Y.%m.%d %H:%M")


def domestic_signal_from_trade_signal(signal, orientation):
    if pd.isna(signal) or str(signal).upper() == "NO_TRADE":
        return 0
    trade_signal = 1 if str(signal).upper() == "BUY" else -1
    return trade_signal if str(orientation).lower() == "base" else -trade_signal


def main():
    print("[PROGRESS] START 32_export_event_study_v3_mt5_signal_table.py", flush=True)
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing summary: {SUMMARY}")

    best = pd.read_csv(SUMMARY)
    best = best[(pd.to_numeric(best["test_net_pips"], errors="coerce") > 0) &
                (pd.to_numeric(best["test_trades"], errors="coerce") >= 100)].copy()

    rows = []
    status_rows = []
    for _, row in best.iterrows():
        pair = str(row["pair"]).upper()
        horizon = int(row["horizon"])
        pred_path = ART_DIR / pair / f"h{horizon}" / "test_predictions_2024_2026.csv"
        print(f"[EXPORT] pair={pair} horizon={horizon} model={row['model']} source={pred_path}", flush=True)
        if not pred_path.exists():
            status_rows.append({"pair": pair, "horizon": horizon, "status": "missing_predictions"})
            continue

        d = pd.read_csv(pred_path)
        d = d[d["signal"].isin(["BUY", "SELL"])].copy()
        if d.empty:
            status_rows.append({"pair": pair, "horizon": horizon, "status": "no_signals"})
            continue

        if "orientation" not in d.columns:
            mapping = available_currency_pair_mapping()[["pair", "CountryCode", "orientation"]].drop_duplicates()
            d = d.merge(mapping, on=["pair", "CountryCode"], how="left")

        out = pd.DataFrame({
            "EVENT_TIME": pd.to_datetime(d["event_time"]).map(mt5_time),
            "SYMBOL": d["pair"].astype(str).str.upper(),
            "COUNTRY": d["CountryCode"].astype(str).str.upper(),
            "EVENT_NAME": d["Name"].astype(str),
            "PROBABILITY": pd.to_numeric(d["confidence"], errors="coerce").fillna(1.0),
            "SIGNAL": [domestic_signal_from_trade_signal(sig, ori) for sig, ori in zip(d["signal"], d["orientation"])],
            "MODEL": MODEL_NAME,
            "HORIZON_HOURS": horizon,
            "SOURCE_MODEL": str(row["model"]),
        })
        rows.append(out)
        status_rows.append({
            "pair": pair,
            "horizon": horizon,
            "status": "exported",
            "signals": len(out),
            "source_model": row["model"],
            "test_net_pips": row["test_net_pips"],
            "test_win_rate": row["test_win_rate"],
            "test_profit_factor": row["test_profit_factor"],
        })

    signal_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["EVENT_TIME", "SYMBOL", "COUNTRY", "EVENT_NAME", "PROBABILITY", "SIGNAL", "MODEL", "HORIZON_HOURS", "SOURCE_MODEL"]
    )
    signal_table = signal_table.drop_duplicates(subset=["EVENT_TIME", "SYMBOL", "COUNTRY", "EVENT_NAME", "MODEL"], keep="last")
    signal_table.to_csv(SIGNAL_TABLE, index=False)

    status = pd.DataFrame(status_rows)
    status.to_csv(ARTIFACTS / "event_study_v3_mt5_export_status.csv", index=False)
    metadata = {
        "model_name": MODEL_NAME,
        "signal_table": str(SIGNAL_TABLE),
        "exit_mode": "EXIT_AFTER_HOURS",
        "holding_hours": 2,
        "note": "CSV signal table for positive OOS Event Study V3 candidates. Use NewsEventsModelTester with NEWS_MODEL_EVENT_STUDY_V3_CSV.",
        "pairs": status_rows,
    }
    (ARTIFACTS / "event_study_v3_mt5_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OUTPUT] {SIGNAL_TABLE} rows={len(signal_table)}", flush=True)
    print(f"[OUTPUT] {ARTIFACTS / 'event_study_v3_mt5_export_status.csv'}", flush=True)
    print(f"[OUTPUT] {ARTIFACTS / 'event_study_v3_mt5_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
