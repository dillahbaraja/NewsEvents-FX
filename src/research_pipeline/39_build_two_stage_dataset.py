"""Build a leakage-aware two-stage target dataset without training models."""
from pathlib import Path
import json
import time
import pandas as pd
import numpy as np

from common import ROOT, ARTIFACTS, model_feature_names

INPUT = ROOT / "modeling_dataset_multi_currency.csv"
OUT = ARTIFACTS / "phase2_two_stage"


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        "event_time", "Name", "CountryCode", "pair", "currency", "orientation",
        "category", "event_year", "split", "domestic_return_pips_1h",
        "trade_label_1h", "domestic_return_1h", "domestic_return_abs_threshold_1h",
    ] + model_feature_names()
    print("[PROGRESS] START two-stage dataset", flush=True)
    print(f"[DATA] input={INPUT}", flush=True)
    parts = []
    rows = 0
    for i, chunk in enumerate(pd.read_csv(INPUT, usecols=lambda c: c in set(required), chunksize=100_000, low_memory=False), 1):
        chunk["domestic_return_pips_1h"] = pd.to_numeric(chunk["domestic_return_pips_1h"], errors="coerce")
        chunk["trade_label_1h"] = pd.to_numeric(chunk["trade_label_1h"], errors="coerce")
        chunk["domestic_return_1h"] = pd.to_numeric(chunk["domestic_return_1h"], errors="coerce")
        chunk["has_valid_outcome"] = chunk["domestic_return_pips_1h"].notna().astype("int8")
        chunk["trade_target"] = np.where(
            chunk["has_valid_outcome"].eq(1), (chunk["trade_label_1h"] != 0).astype("int8"), np.nan
        )
        chunk["direction_target"] = np.where(
            chunk["has_valid_outcome"].eq(1) & chunk["trade_label_1h"].isin([-1, 1]),
            chunk["trade_label_1h"], np.nan
        )
        chunk["direction_name"] = chunk["direction_target"].map({-1: "SELL", 1: "BUY"})
        parts.append(chunk)
        rows += len(chunk)
        print(f"[READ] chunks={i} rows={rows}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    del parts
    d.to_csv(OUT / "two_stage_modeling_dataset.csv", index=False)

    summaries = []
    for (pair, split), g in d.groupby(["pair", "split"], dropna=False):
        valid = g[g["has_valid_outcome"].eq(1)]
        direction = valid[valid["direction_target"].isin([-1, 1])]
        summaries.append({
            "pair": pair, "split": split, "rows": int(len(g)),
            "valid_outcome_rows": int(len(valid)),
            "trade_count_target": int(valid["trade_target"].sum()),
            "no_trade_count_target": int((valid["trade_target"] == 0).sum()),
            "buy_count_direction": int((direction["direction_target"] == 1).sum()),
            "sell_count_direction": int((direction["direction_target"] == -1).sum()),
        })
    pd.DataFrame(summaries).to_csv(OUT / "two_stage_target_distribution.csv", index=False)
    metadata = {
        "target_stage_1": "trade_target: 1=TRADE, 0=NO_TRADE",
        "target_stage_2": "direction_target: 1=BUY, -1=SELL; only trained on stage-1 trade rows",
        "threshold_source": "existing train-only domestic_return_abs_threshold_1h",
        "feature_count": len(model_feature_names()),
        "features": model_feature_names(),
        "oos_policy": "2024+ is never used for training or threshold selection",
        "created_at": pd.Timestamp.now().isoformat(),
    }
    (OUT / "two_stage_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = f"""# Fase 2 - Two-Stage Dataset Contract

Dataset ini dibuat tanpa training dan tidak mengganti artefak model lama.

- Rows: {len(d):,}
- Features: {len(model_feature_names())}
- Stage 1: `TRADE` versus `NO_TRADE`
- Stage 2: `BUY` versus `SELL`, hanya pada baris trade yang valid
- Threshold: kolom threshold yang sudah dihitung dari data train
- Periode 2024+ tetap OOS dan tidak boleh dipakai untuk fitting atau pemilihan threshold.

Output:

- `two_stage_modeling_dataset.csv`
- `two_stage_target_distribution.csv`
- `two_stage_metadata.json`
"""
    (OUT / "TWO_STAGE_DATASET_CONTRACT.md").write_text(report, encoding="utf-8")
    print(f"[DONE] output={OUT} rows={len(d)} elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
