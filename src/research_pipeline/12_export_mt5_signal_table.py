from common import *
from pathlib import Path


BACKTRADER_ROOT = DATA / "backtrader"
OUTPUT_FILE = ARTIFACTS / "mt5_signal_table.csv"


def mt5_time(value):
    ts = pd.to_datetime(value)
    return ts.strftime("%Y.%m.%d %H:%M")


def domestic_signal_from_trade_signal(signal, orientation):
    if signal == "NO_TRADE" or pd.isna(signal):
        return 0
    trade_signal = 1 if str(signal).upper() == "BUY" else -1
    return trade_signal if str(orientation).lower() == "base" else -trade_signal


def event_name_column(df):
    if "Name" in df.columns:
        return "Name"
    if "event_name" in df.columns:
        return "event_name"
    return ""


def signal_column(df):
    if "signal" in df.columns:
        return "signal"
    if "final_signal" in df.columns:
        return "final_signal"
    return ""


def main():
    print("[PROGRESS] START 12_export_mt5_signal_table.py", flush=True)
    mapping = available_currency_pair_mapping()[["pair", "CountryCode", "orientation"]].copy()
    rows = []
    if not BACKTRADER_ROOT.exists():
        raise FileNotFoundError(f"Backtrader output folder not found: {BACKTRADER_ROOT}")

    files = sorted(BACKTRADER_ROOT.glob("*/*/backtrader_predictions.csv"))
    print(f"[DATA] prediction_files={len(files)}", flush=True)
    for idx, path in enumerate(files, start=1):
        print(f"[EXPORT] {idx}/{len(files)} {path}", flush=True)
        if path.stat().st_size == 0:
            continue
        try:
            d = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if d.empty:
            continue
        name_col = event_name_column(d)
        sig_col = signal_column(d)
        required = {"event_time", "pair", "CountryCode"}
        missing = required.difference(d.columns)
        if missing or not name_col or not sig_col:
            if not name_col:
                missing.add("Name/event_name")
            if not sig_col:
                missing.add("signal/final_signal")
            print(f"[SKIP] missing columns {sorted(missing)} in {path}", flush=True)
            continue
        if "model" not in d.columns:
            d["model"] = path.parent.name
        if "confidence" not in d.columns:
            d["confidence"] = 1.0
        d["__signal"] = d[sig_col].astype(str).str.upper()
        d = d[d["__signal"].isin(["BUY", "SELL"])].copy()
        if d.empty:
            continue
        d = d.merge(mapping, on=["pair", "CountryCode"], how="left")
        d = d[d["orientation"].isin(["base", "quote"])].copy()
        if d.empty:
            continue
        d["EVENT_TIME"] = d["event_time"].map(mt5_time)
        d["SYMBOL"] = d["pair"].astype(str)
        d["COUNTRY"] = d["CountryCode"].astype(str)
        d["EVENT_NAME"] = d[name_col].astype(str)
        d["PROBABILITY"] = pd.to_numeric(d["confidence"], errors="coerce").fillna(0.0)
        d["SIGNAL"] = [
            domestic_signal_from_trade_signal(sig, ori)
            for sig, ori in zip(d["__signal"], d["orientation"])
        ]
        d["MODEL"] = d["model"].astype(str)
        rows.append(d[["EVENT_TIME", "SYMBOL", "COUNTRY", "EVENT_NAME", "PROBABILITY", "SIGNAL", "MODEL"]])

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out = out.drop_duplicates(subset=["EVENT_TIME", "SYMBOL", "COUNTRY", "EVENT_NAME", "MODEL"], keep="last")
    else:
        out = pd.DataFrame(columns=["EVENT_TIME", "SYMBOL", "COUNTRY", "EVENT_NAME", "PROBABILITY", "SIGNAL", "MODEL"])

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"[OUTPUT] {OUTPUT_FILE}", flush=True)
    print(f"[OUTPUT] rows={len(out)} models={sorted(out['MODEL'].dropna().unique().tolist()) if len(out) else []}", flush=True)


if __name__ == "__main__":
    main()
