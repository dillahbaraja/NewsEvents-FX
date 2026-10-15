from common import *

MANUAL_DIR = ARTIFACTS / "manual_block_inverse"
OUT = MANUAL_DIR / "manual_block_inverse_train_trades_2010_2023.csv"

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
    if diff > 0:
        direction = 1
    elif diff < 0:
        direction = -1
    else:
        direction = 0
    name = str(row.get("Name", "")).lower()
    if direction and any(k in name for k in INVERSE_KEYWORDS):
        direction *= -1
    return direction


def main():
    print("[PROGRESS] START 14_backtrader_rule_baseline_2010_2023.py", flush=True)
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    usecols = [
        "event_time", "Datetime", "Name", "CountryCode", "Actual", "Forecast", "Previous",
        "Importance", "Impact", "pair", "currency", "orientation", "event_year", "Time_y",
        "Close", "close_fwd_1h", "pip_size", "domestic_return_pips_1h",
    ]
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv", usecols=lambda c: c in set(usecols), low_memory=False)
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d["event_year"] = pd.to_numeric(d["event_year"], errors="coerce")
    d = d[(d["event_year"] >= 2010) & (d["event_year"] <= 2023)].copy()
    d = d.dropna(subset=["event_time", "Name", "CountryCode", "pair", "orientation", "domestic_return_pips_1h"])
    print(f"[DATA] candidate_events={len(d)}", flush=True)

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
    d["selection_period"] = "calibration_2010_2023"

    out_cols = [
        "event_time", "event_year", "pair", "CountryCode", "currency", "orientation",
        "event_name", "event_name_norm", "Actual", "Forecast", "Previous", "Importance", "Impact",
        "initial_signal", "final_signal", "action", "Close", "close_fwd_1h",
        "domestic_return_pips_1h", "pnl_pips", "inverse_pnl_pips", "win", "inverse_win",
        "selection_period",
    ]
    d[out_cols].to_csv(OUT, index=False)
    print(f"[OUTPUT] {OUT}", flush=True)
    print(f"[DATA] baseline_trades={len(d)}", flush=True)


if __name__ == "__main__":
    main()
