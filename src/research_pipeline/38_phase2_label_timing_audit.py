"""Phase 2 audit for the macro-news FX learning contract.

This is intentionally an audit/preparation step. It does not train models or
overwrite existing artifacts. It measures timing alignment, price coverage,
surprise availability, event overlap, and the current three-class label
contract before the two-stage retraining is started.
"""
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd

from common import ROOT, DATA, ARTIFACTS, TRAIN_END_YEAR, VALIDATION_END_YEAR


OUT = ARTIFACTS / "phase2_label_timing_audit"
MODEL_DATA = ROOT / "modeling_dataset_multi_currency.csv"


def num(v):
    return pd.to_numeric(v, errors="coerce")


def safe_pf(values):
    s = pd.Series(values).dropna()
    gp = float(s[s > 0].sum())
    gl = float(-s[s < 0].sum())
    if gl <= 0:
        return math.inf if gp > 0 else 0.0
    return gp / gl


def write_csv(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("[PROGRESS] START Phase 2 label/timing audit", flush=True)
    print(f"[DATA] {MODEL_DATA}", flush=True)

    if not MODEL_DATA.exists():
        raise FileNotFoundError(MODEL_DATA)

    usecols = [
        "event_time", "Time_y", "Name", "CountryCode", "Importance", "Impact",
        "Actual", "Forecast", "Previous", "pair", "orientation", "Close",
        "close_fwd_1h", "domestic_return_pips_1h", "trade_label_1h",
        "domestic_return_1h", "event_year", "split", "category",
    ]
    chunks = []
    total = 0
    for i, chunk in enumerate(pd.read_csv(MODEL_DATA, usecols=usecols, chunksize=100_000, low_memory=False), 1):
        chunks.append(chunk)
        total += len(chunk)
        print(f"[READ] chunks={i} rows={total}", flush=True)
    d = pd.concat(chunks, ignore_index=True)
    del chunks
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d["price_time"] = pd.to_datetime(d["Time_y"], errors="coerce")
    for c in ["Actual", "Forecast", "Previous", "Importance", "Impact", "Close", "close_fwd_1h", "domestic_return_pips_1h", "trade_label_1h", "domestic_return_1h", "event_year"]:
        d[c] = num(d[c])
    d["timing_delta_minutes"] = (d["price_time"] - d["event_time"]).dt.total_seconds() / 60.0
    d["has_price"] = d["Close"].notna() & d["close_fwd_1h"].notna()
    d["has_surprise_fc"] = d["Actual"].notna() & d["Forecast"].notna()
    d["has_surprise_prev"] = d["Actual"].notna() & d["Previous"].notna()
    d["has_any_surprise"] = d["has_surprise_fc"] | d["has_surprise_prev"]
    d["event_name_norm"] = d["Name"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()

    print("[REPORT] coverage and timing", flush=True)
    coverage = []
    for (pair, country), g in d.groupby(["pair", "CountryCode"], dropna=False):
        valid_delta = g.loc[g["has_price"], "timing_delta_minutes"]
        coverage.append({
            "pair": pair, "country": country, "rows": int(len(g)),
            "price_rows": int(g["has_price"].sum()),
            "price_coverage": float(g["has_price"].mean()),
            "surprise_fc_coverage": float(g["has_surprise_fc"].mean()),
            "surprise_prev_coverage": float(g["has_surprise_prev"].mean()),
            "any_surprise_coverage": float(g["has_any_surprise"].mean()),
            "median_event_to_price_minutes": float(valid_delta.median()) if len(valid_delta) else np.nan,
            "p95_event_to_price_minutes": float(valid_delta.quantile(.95)) if len(valid_delta) else np.nan,
            "same_bar_or_forward_price_pct": float((valid_delta >= 0).mean()) if len(valid_delta) else np.nan,
        })
    write_csv(coverage, "phase2_pair_country_coverage_timing.csv")

    print("[REPORT] label distribution", flush=True)
    labels = []
    for (pair, split), g in d.groupby(["pair", "split"], dropna=False):
        vc = g["trade_label_1h"].value_counts(dropna=False).to_dict()
        labels.append({
            "pair": pair, "split": split, "rows": int(len(g)),
            "sell": int(vc.get(-1, 0)), "no_trade": int(vc.get(0, 0)), "buy": int(vc.get(1, 0)),
            "sell_ratio": float(vc.get(-1, 0) / len(g)) if len(g) else 0.0,
            "no_trade_ratio": float(vc.get(0, 0) / len(g)) if len(g) else 0.0,
            "buy_ratio": float(vc.get(1, 0) / len(g)) if len(g) else 0.0,
        })
    write_csv(labels, "phase2_label_distribution.csv")

    print("[REPORT] overlap and duplicate event pressure", flush=True)
    d["event_hour_bucket"] = d["event_time"].dt.floor("h")
    overlap = d.groupby(["pair", "event_hour_bucket"], dropna=False).size().reset_index(name="events_in_hour")
    overlap_summary = overlap.groupby("pair")["events_in_hour"].agg(
        hour_buckets="size", mean_events_per_hour="mean", p95_events_per_hour=lambda x: x.quantile(.95),
        max_events_per_hour="max"
    ).reset_index()
    overlap_summary.to_csv(OUT / "phase2_event_overlap_summary.csv", index=False)
    d.groupby(["pair", "event_hour_bucket"], dropna=False).size().reset_index(name="events_in_hour").to_csv(
        OUT / "phase2_event_overlap_by_hour.csv", index=False
    )

    print("[REPORT] event-level reliability diagnostics", flush=True)
    event_rows = []
    usable = d[d["has_price"] & d["has_any_surprise"] & d["domestic_return_pips_1h"].notna()].copy()
    usable["raw_surprise_direction"] = np.where(
        usable["has_surprise_fc"], np.sign(usable["Actual"] - usable["Forecast"]), np.sign(usable["Actual"] - usable["Previous"])
    )
    usable["raw_pnl_pips"] = usable["domestic_return_pips_1h"] * usable["raw_surprise_direction"]
    for (pair, country, name), g in usable.groupby(["pair", "CountryCode", "event_name_norm"], dropna=False):
        p = g["raw_pnl_pips"]
        years = g["event_year"].dropna().astype(int)
        yearly = g.groupby("event_year")["raw_pnl_pips"].sum()
        event_rows.append({
            "pair": pair, "country": country, "event_name_norm": name,
            "trade_count": int(len(g)), "net_pips_original": float(p.sum()),
            "profit_factor_original": safe_pf(p), "avg_pips_original": float(p.mean()),
            "win_rate_original": float((p > 0).mean()), "active_years": int(years.nunique()),
            "positive_year_ratio": float((yearly > 0).mean()) if len(yearly) else 0.0,
            "price_coverage": float(g["has_price"].mean()),
        })
    pd.DataFrame(event_rows).sort_values(["pair", "country", "net_pips_original"], ascending=[True, True, False]).to_csv(
        OUT / "phase2_event_reliability_diagnostics.csv", index=False
    )

    feature_leakage = []
    future_tokens = ("return", "close_fwd", "label", "pnl", "profit", "exit", "future")
    for col in usecols:
        if any(token in col.lower() for token in future_tokens):
            feature_leakage.append({"column": col, "classification": "target_or_future_column", "action": "exclude_from_runtime_features"})
    for col in ["event_time", "Time_y", "Name", "CountryCode", "Importance", "Impact", "Actual", "Forecast", "Previous", "pair", "orientation", "category"]:
        if col in usecols:
            feature_leakage.append({"column": col, "classification": "raw_or_context_column", "action": "encode_only_if_available_at_release"})
    pd.DataFrame(feature_leakage).drop_duplicates().to_csv(OUT / "phase2_feature_leakage_inventory.csv", index=False)

    summary = {
        "rows": int(len(d)),
        "pairs": sorted(d["pair"].dropna().astype(str).unique().tolist()),
        "date_min": str(d["event_time"].min()), "date_max": str(d["event_time"].max()),
        "price_coverage": float(d["has_price"].mean()),
        "any_surprise_coverage": float(d["has_any_surprise"].mean()),
        "same_bar_or_forward_price_pct": float(d.loc[d["has_price"], "timing_delta_minutes"].ge(0).mean()) if d["has_price"].any() else None,
        "overlap_hours_with_multiple_events": int((overlap["events_in_hour"] > 1).sum()),
        "event_units": int(len(event_rows)),
        "created_at": pd.Timestamp.now().isoformat(),
        "next_step": "build two-stage train/validation artifacts only after reviewing this audit",
    }
    (OUT / "phase2_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# Fase 2 - Audit Label, Timing, dan Event Reliability

Tanggal: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}

## Status

Ini adalah audit dan persiapan kontrak data. Belum ada training ulang dan tidak ada artefak model lama yang ditimpa.

## Ringkasan Data

- Rows: {summary['rows']:,}
- Periode event: {summary['date_min']} sampai {summary['date_max']}
- Pair: {', '.join(summary['pairs'])}
- Price coverage: {summary['price_coverage']:.2%}
- Actual plus Forecast/Previous coverage: {summary['any_surprise_coverage']:.2%}
- Harga same-bar/forward terhadap event: {summary['same_bar_or_forward_price_pct']:.2%}
- Jam dengan lebih dari satu event: {summary['overlap_hours_with_multiple_events']:,}
- Unit pair-country-event yang dapat dihitung: {summary['event_units']:,}

## Perubahan Kontrak Yang Disiapkan

1. Harga dan event harus dipasangkan secara eksplisit menggunakan bar setelah rilis, bukan asumsi waktu dari baris CSV.
2. Tahap pertama model berikutnya akan memprediksi `TRADE` versus `NO_TRADE`.
3. Tahap kedua hanya memprediksi arah `BUY` versus `SELL` untuk event yang lolos tahap pertama.
4. Event reliability dihitung per `pair + country + event_name`, menggunakan data train saja.
5. Validation 2021-2023 dipakai untuk threshold dan pemilihan model; periode 2024+ tetap murni OOS.
6. Event yang overlap akan dicatat dan aturan posisi/holding dibuat sama untuk vectorbt, Backtrader, dan MT5.

## File Output

- `phase2_pair_country_coverage_timing.csv`
- `phase2_label_distribution.csv`
- `phase2_event_overlap_summary.csv`
- `phase2_event_overlap_by_hour.csv`
- `phase2_event_reliability_diagnostics.csv`
- `phase2_feature_leakage_inventory.csv`
- `phase2_audit_summary.json`

## Catatan Penting

Audit ini tidak membuktikan performa trading. Bila price coverage rendah atau timing delta tidak konsisten, model tidak boleh langsung dibandingkan dengan Strategy Tester karena kontrak datanya belum setara.
"""
    (OUT / "PHASE2_LABEL_TIMING_AUDIT.md").write_text(report, encoding="utf-8")
    print(f"[DONE] outputs={OUT} elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
