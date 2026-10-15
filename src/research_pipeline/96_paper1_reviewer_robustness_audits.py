from pathlib import Path
import json
import numpy as np
import pandas as pd

from common import ARTIFACTS, progress_done, progress_start, progress_step


SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "paper1_reviewer_robustness_audits"
FINAL_DIR = ARTIFACTS / "paper1_final_results_and_engine_comparison"

SELECTED = FINAL_DIR / "paper1_final_selected_model_per_pair.csv"

PREDICTION_FILES = {
    "baseline_news_only": ARTIFACTS / "paper1_baseline" / "paper1_oos_loop_predictions.csv",
    "v1_standardized_surprise": None,
    "v2_volatility_regime": ARTIFACTS / "paper1_v2_volatility_regime" / "v2_oos_predictions.csv",
    "v3_sparse_event_selection": ARTIFACTS / "paper1_v3_sparse_event_selection" / "v3_oos_predictions.csv",
    "v4_jump_volatility_labeling": ARTIFACTS / "paper1_v4_jump_volatility_labeling" / "v4_oos_predictions.csv",
    "v5_event_congestion_filter": ARTIFACTS / "paper1_v5_event_congestion_filter" / "v5_oos_predictions.csv",
}

SVM_EBM_PREDICTIONS = {
    "svm_rbf": ARTIFACTS / "paper1_svm_ebm_base_and_v2_methods" / "svm_ebm_base_v2_oos_predictions.csv",
    "ebm": ARTIFACTS / "paper1_ebm_v2_methods" / "svm_ebm_base_v2_oos_predictions.csv",
}


def pf(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)
    gp = s[s > 0].sum()
    gl = -s[s < 0].sum()
    if gl == 0:
        return float("inf") if gp > 0 else 0.0
    return float(gp / gl)


def norm_variant(v):
    return str(v).strip()


def normalize_predictions(d):
    d = d.copy()
    if "domestic_return_pips_1h" not in d.columns and "domestic_return_pips" in d.columns:
        d["domestic_return_pips_1h"] = d["domestic_return_pips"]
    d["event_time"] = pd.to_datetime(d["event_time"], errors="coerce")
    d["year"] = pd.to_numeric(d["year"], errors="coerce").fillna(d["event_time"].dt.year).astype(int)
    d["signal"] = pd.to_numeric(d["signal"], errors="coerce").fillna(0).astype(int)
    d["strategy_pips"] = pd.to_numeric(d["strategy_pips"], errors="coerce").fillna(0.0)
    d["pair"] = d["pair"].astype(str).str.upper()
    d["model"] = d["model"].astype(str)
    d["variant"] = d["variant"].astype(str)
    return d


def load_selected_predictions():
    selected = pd.read_csv(SELECTED)
    parts = []
    cache = {}
    for _, row in selected.iterrows():
        pair = str(row["pair"]).upper()
        model = str(row["model"])
        variant = norm_variant(row["variant"])
        if model in SVM_EBM_PREDICTIONS:
            path = SVM_EBM_PREDICTIONS[model]
        else:
            path = PREDICTION_FILES.get(variant)
        if path is None or not path.exists():
            progress_step("PREDICTION_FILE_MISSING", pair=pair, model=model, variant=variant, path=str(path))
            continue
        if str(path) not in cache:
            progress_step("READ_PREDICTIONS", path=str(path))
            cache[str(path)] = normalize_predictions(pd.read_csv(path))
        d = cache[str(path)]
        x = d[(d["pair"].eq(pair)) & (d["model"].eq(model)) & (d["variant"].eq(variant))].copy()
        x["selected_family"] = row.get("family", "")
        parts.append(x)
        progress_step("SELECTED_PREDICTIONS", pair=pair, model=model, variant=variant, rows=len(x))
    if not parts:
        raise RuntimeError("No selected prediction rows were loaded.")
    return pd.concat(parts, ignore_index=True), selected


def summarize_by_period(d, value_col="strategy_pips"):
    rows = []
    for period, g in [
        ("oos_dev_2024_2025", d[d["year"].between(2024, 2025)]),
        ("final_oos_2026", d[d["year"].eq(2026)]),
        ("combined_2024_2026", d[d["year"].between(2024, 2026)]),
    ]:
        active = g[pd.to_numeric(g["signal"], errors="coerce").ne(0)]
        rows.append({
            "period": period,
            "events": int(len(g)),
            "trades": int(len(active)),
            "net_pips": round(float(active[value_col].sum()), 4),
            "profit_factor": round(pf(active[value_col]), 6),
            "winrate": round(float((active[value_col] > 0).mean()), 6) if len(active) else 0.0,
            "positive_pairs": int((active.groupby("pair")[value_col].sum() > 0).sum()) if len(active) else 0,
        })
    return pd.DataFrame(rows)


def transaction_cost_sensitivity(d):
    rows = []
    for cost in [0.0, 0.5, 1.0, 2.0, 3.0]:
        x = d.copy()
        x["cost_pips"] = np.where(x["signal"].ne(0), cost, 0.0)
        x["net_after_cost"] = x["strategy_pips"] - x["cost_pips"]
        for (period, pair), g in pd.concat([
            x[x["year"].between(2024, 2025)].assign(period="oos_dev_2024_2025"),
            x[x["year"].eq(2026)].assign(period="final_oos_2026"),
        ]).groupby(["period", "pair"]):
            active = g[g["signal"].ne(0)]
            rows.append({
                "period": period,
                "pair": pair,
                "cost_pips_per_trade": cost,
                "trades": int(len(active)),
                "net_pips_after_cost": round(float(active["net_after_cost"].sum()), 4),
                "profit_factor_after_cost": round(pf(active["net_after_cost"]), 6),
                "winrate_after_cost": round(float((active["net_after_cost"] > 0).mean()), 6) if len(active) else 0.0,
            })
    out = pd.DataFrame(rows)
    agg = out.groupby(["period", "cost_pips_per_trade"], as_index=False).agg(
        trades=("trades", "sum"),
        net_pips_after_cost=("net_pips_after_cost", "sum"),
        positive_pairs=("net_pips_after_cost", lambda s: int((s > 0).sum())),
    )
    agg["net_pips_after_cost"] = agg["net_pips_after_cost"].round(4)
    return out, agg


def signal_distribution(d):
    rows = []
    for (period, pair, model, variant), g in pd.concat([
        d[d["year"].between(2024, 2025)].assign(period="oos_dev_2024_2025"),
        d[d["year"].eq(2026)].assign(period="final_oos_2026"),
    ]).groupby(["period", "pair", "model", "variant"]):
        buy = int((g["signal"] > 0).sum())
        sell = int((g["signal"] < 0).sum())
        no_trade = int((g["signal"] == 0).sum())
        total = len(g)
        trades = buy + sell
        buy_ratio = buy / total if total else 0
        sell_ratio = sell / total if total else 0
        no_trade_ratio = no_trade / total if total else 0
        flags = []
        if buy_ratio > 0.90:
            flags.append("buy_collapse")
        if sell_ratio > 0.90:
            flags.append("sell_collapse")
        if no_trade_ratio > 0.98:
            flags.append("no_trade_collapse")
        if trades < 20:
            flags.append("too_few_trades")
        rows.append({
            "period": period,
            "pair": pair,
            "model": model,
            "variant": variant,
            "events": total,
            "buy_count": buy,
            "sell_count": sell,
            "no_trade_count": no_trade,
            "buy_ratio": round(buy_ratio, 6),
            "sell_ratio": round(sell_ratio, 6),
            "no_trade_ratio": round(no_trade_ratio, 6),
            "collapse_flag": ";".join(flags),
        })
    return pd.DataFrame(rows)


def timestamp_alignment_sample(d):
    x = d.copy()
    x["candle_h1_open"] = x["event_time"].dt.floor("h")
    x["minutes_after_candle_open"] = (x["event_time"] - x["candle_h1_open"]).dt.total_seconds() / 60.0
    x["horizon_1h_close"] = x["candle_h1_open"] + pd.Timedelta(hours=1)
    cols = [
        "event_time", "candle_h1_open", "minutes_after_candle_open", "horizon_1h_close",
        "pair", "CountryCode", "Name", "model", "variant", "signal", "domestic_return_pips_1h", "strategy_pips",
    ]
    sample = (
        x.sort_values(["pair", "event_time"])
        .groupby("pair", group_keys=False)
        .head(25)
        [cols]
    )
    summary = x.groupby("pair").agg(
        events=("event_time", "count"),
        min_minutes_after_candle_open=("minutes_after_candle_open", "min"),
        max_minutes_after_candle_open=("minutes_after_candle_open", "max"),
        median_minutes_after_candle_open=("minutes_after_candle_open", "median"),
    ).reset_index()
    return sample, summary


def drift_audit(d):
    rows = []
    for pair, g in d.groupby("pair"):
        dev = g[g["year"].between(2024, 2025) & g["signal"].ne(0)]
        audit = g[g["year"].eq(2026) & g["signal"].ne(0)]
        dev_net = float(dev["strategy_pips"].sum())
        audit_net = float(audit["strategy_pips"].sum())
        rows.append({
            "pair": pair,
            "oos_dev_trades": int(len(dev)),
            "oos_dev_net_pips": round(dev_net, 4),
            "oos_dev_pf": round(pf(dev["strategy_pips"]), 6),
            "final_2026_trades": int(len(audit)),
            "final_2026_net_pips": round(audit_net, 4),
            "final_2026_pf": round(pf(audit["strategy_pips"]), 6),
            "delta_2026_minus_dev": round(audit_net - dev_net, 4),
            "drift_flag": "sign_flip_or_large_drop" if (dev_net > 0 and audit_net < 0) or (audit_net < dev_net * 0.25) else "",
        })
    return pd.DataFrame(rows).sort_values("delta_2026_minus_dev")


def benchmark_comparison(d):
    rows = []
    # no-trade baseline
    for period, g in [("oos_dev_2024_2025", d[d["year"].between(2024, 2025)]), ("final_oos_2026", d[d["year"].eq(2026)])]:
        rows.append({"benchmark": "no_trade", "period": period, "trades": 0, "net_pips": 0.0, "profit_factor": 0.0, "winrate": 0.0})
        active = g[g["signal"].ne(0)]
        rows.append({
            "benchmark": "final_selected_models",
            "period": period,
            "trades": int(len(active)),
            "net_pips": round(float(active["strategy_pips"].sum()), 4),
            "profit_factor": round(pf(active["strategy_pips"]), 6),
            "winrate": round(float((active["strategy_pips"] > 0).mean()), 6) if len(active) else 0.0,
        })
    return pd.DataFrame(rows)


def write_protocol_doc(outputs):
    md = f"""# Paper 1 - Reviewer Robustness, Bias Control, dan Validity Threats

Dokumen ini dibuat untuk mengantisipasi pertanyaan reviewer/editor terkait kelemahan penelitian Paper 1.

## Tujuan

Tujuan audit ini adalah mengurangi risiko kritik terhadap:

- data snooping dan multiple testing,
- biaya transaksi,
- benchmark yang terlalu lemah,
- timestamp alignment,
- signal collapse,
- drift antara 2024-2025 dan 2026,
- klaim kausalitas yang berlebihan.

## Prinsip Validasi

| Tahap | Periode | Fungsi |
| --- | --- | --- |
| Training | 2010-2020 | Melatih model |
| Validation | 2021-2023 | Pengembangan model dan threshold |
| OOS development | 2024-2025 | Seleksi final model per pair |
| Final audit | 2026 | Audit tambahan, tidak dipakai memilih model |

## Bias-Control Protocol

- Semua kandidat model disimpan dalam file all-candidates.
- Final model dipilih per pair berdasarkan OOS development 2024-2025.
- Tahun 2026 tidak dipakai untuk memilih model.
- Pair yang melemah pada 2026 ditandai sebagai drift-sensitive, bukan dihapus diam-diam.
- Hasil negatif tetap dilaporkan, terutama AUDUSD, NZDUSD, dan USDJPY audit.

## Transaction Cost Sensitivity

File:

`{outputs['cost']}`

Skenario biaya:

- 0 pip
- 0.5 pip
- 1 pip
- 2 pip
- 3 pip

Interpretasi untuk paper:

- Jika hasil melemah setelah biaya, klaim profit harus dikurangi.
- Jika masih positif pada biaya konservatif, robustness meningkat.

## Benchmark Comparison

File:

`{outputs['benchmark']}`

Benchmark minimal:

- no-trade baseline,
- final selected models.

Catatan:

Benchmark naive surprise rule dapat ditambahkan jika file OOS yang memuat `Actual`, `Forecast`, dan `Previous` untuk 2024-2026 tersedia dalam format yang sama dengan event-level prediction table.

## Timestamp Alignment Audit

File:

`{outputs['timestamp_sample']}`

Audit ini menunjukkan:

- waktu event,
- candle H1 yang dipakai,
- posisi menit event dalam candle,
- horizon 1H,
- pair,
- country,
- signal,
- return pips.

Ini penting untuk menjawab apakah event time dan OHLCV H1 sudah sinkron.

## Signal Distribution Audit

File:

`{outputs['signal']}`

Audit ini mendeteksi:

- dominasi BUY,
- dominasi SELL,
- dominasi NO_TRADE,
- trade count terlalu kecil.

## Drift Audit 2026

File:

`{outputs['drift']}`

Audit ini membandingkan OOS development 2024-2025 dengan final audit 2026. Pair dengan sign flip atau penurunan besar harus dibahas sebagai indikasi drift/regime change.

## Klaim Yang Aman

Penelitian ini sebaiknya diklaim sebagai:

> event-conditioned predictive modeling, bukan causal macroeconomic inference.

Klaim yang aman:

- model terbaik berbeda antar pair,
- event-character-aware methods membantu seleksi model,
- 2026 menunjukkan audit robustness yang tidak selalu stabil,
- hasil mendukung pair-specific modeling.

Klaim yang harus dihindari:

- model pasti profit live,
- berita terbukti secara kausal menyebabkan profit,
- satu model unggul universal,
- hasil 2024-2025 otomatis berlaku untuk semua kondisi pasar.
"""
    (OUT / "PAPER1_SELECTION_PROTOCOL_AND_BIAS_CONTROL.md").write_text(md, encoding="utf-8")


def main():
    started = progress_start(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    d, selected = load_selected_predictions()
    d.to_csv(OUT / "paper1_final_selected_predictions_2024_2026.csv", index=False)

    summary = summarize_by_period(d)
    summary.to_csv(OUT / "paper1_final_selected_period_summary.csv", index=False)

    cost, cost_agg = transaction_cost_sensitivity(d)
    cost.to_csv(OUT / "paper1_transaction_cost_sensitivity_by_pair.csv", index=False)
    cost_agg.to_csv(OUT / "paper1_transaction_cost_sensitivity.csv", index=False)

    signal = signal_distribution(d)
    signal.to_csv(OUT / "paper1_signal_distribution_audit.csv", index=False)

    ts_sample, ts_summary = timestamp_alignment_sample(d)
    ts_sample.to_csv(OUT / "paper1_timestamp_alignment_sample.csv", index=False)
    ts_summary.to_csv(OUT / "paper1_timestamp_alignment_summary.csv", index=False)

    drift = drift_audit(d)
    drift.to_csv(OUT / "paper1_drift_audit_2026.csv", index=False)

    benchmark = benchmark_comparison(d)
    benchmark.to_csv(OUT / "paper1_benchmark_comparison.csv", index=False)

    metadata = {
        "script": SCRIPT,
        "inputs": {
            "selected": str(SELECTED),
            "artifact_root": str(ARTIFACTS),
        },
        "outputs": [p.name for p in OUT.iterdir() if p.is_file()],
        "note": "Reviewer robustness audit; no model retraining.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_protocol_doc({
        "cost": str(OUT / "paper1_transaction_cost_sensitivity.csv"),
        "benchmark": str(OUT / "paper1_benchmark_comparison.csv"),
        "timestamp_sample": str(OUT / "paper1_timestamp_alignment_sample.csv"),
        "signal": str(OUT / "paper1_signal_distribution_audit.csv"),
        "drift": str(OUT / "paper1_drift_audit_2026.csv"),
    })
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
