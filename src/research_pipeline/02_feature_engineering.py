from common import *


def main():
    print("[PROGRESS] START 02_feature_engineering.py", flush=True)
    d = pd.read_csv(ROOT / "event_level_dataset.csv")
    print(f"[DATA] input rows={len(d)} columns={len(d.columns)}", flush=True)
    d["event_time"] = pd.to_datetime(d.event_time)
    print("[FEATURE] surprise and category features", flush=True)
    d["surprise_fc"] = d.Actual - d.Forecast
    d["surprise_prev"] = d.Actual - d.Previous
    d["surprise_pct_fc"] = d.surprise_fc / (d.Forecast.abs() + 1e-9)
    d["category"] = d.Name.map(event_category)
    d["release_hour"] = d.event_time.dt.hour
    d["event_year"] = d.event_time.dt.year
    print("[FEATURE] pair/country/currency/category/orientation context features", flush=True)
    d = add_context_features(d)

    sign = np.where(d["orientation"] == "base", 1.0, -1.0)
    for h in [1, 2, 4, 24]:
        print(f"[HORIZON] {h}H", flush=True)
        d[f"return_{h}h"] = np.log(d[f"close_fwd_{h}h"] / d.Close)
        d[f"domestic_return_{h}h"] = d[f"return_{h}h"] * sign
        d[f"label_{h}h"] = (d[f"domestic_return_{h}h"] > 0).astype("Int64")

    d["pip_size"] = d["pair"].map(pip_size)
    d["domestic_return_pips_1h"] = ((d["close_fwd_1h"] - d["Close"]) / d["pip_size"]) * sign
    d["domestic_return_pct_1h"] = np.expm1(d["domestic_return_1h"])
    train_mask = research_split(d["event_year"]) == "train"
    thresholds = []
    for pair, g in d.loc[train_mask].groupby("pair"):
        abs_ret = g["domestic_return_1h"].abs().replace([np.inf, -np.inf], np.nan).dropna()
        abs_pips = g["domestic_return_pips_1h"].abs().replace([np.inf, -np.inf], np.nan).dropna()
        ret_threshold = float(abs_ret.quantile(RETURN_SIGNIFICANCE_PERCENTILE)) if len(abs_ret) else 0.0
        pip_threshold = float(abs_pips.quantile(RETURN_SIGNIFICANCE_PERCENTILE)) if len(abs_pips) else 0.0
        thresholds.append({
            "pair": pair,
            "selection_split": "train_2010_2020",
            "percentile": RETURN_SIGNIFICANCE_PERCENTILE,
            "domestic_return_abs_threshold_1h": ret_threshold,
            "domestic_return_pips_abs_threshold_1h": pip_threshold,
            "train_rows": int(len(g)),
        })
    threshold_df = pd.DataFrame(thresholds)
    threshold_df.to_csv(ROOT / "return_significance_thresholds.csv", index=False)
    d = d.merge(threshold_df[["pair", "domestic_return_abs_threshold_1h"]], on="pair", how="left")
    d["domestic_return_norm_1h"] = d["domestic_return_1h"] / (d["domestic_return_abs_threshold_1h"].replace(0, np.nan))
    d["domestic_return_norm_1h"] = d["domestic_return_norm_1h"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["trade_label_1h"] = np.select(
        [
            d["domestic_return_1h"] >= d["domestic_return_abs_threshold_1h"],
            d["domestic_return_1h"] <= -d["domestic_return_abs_threshold_1h"],
        ],
        [1, -1],
        default=0,
    ).astype("int8")
    label_rows = []
    d["split"] = research_split(d["event_year"])
    for (pair, split_name), g in d.groupby(["pair", "split"], dropna=False):
        counts = g["trade_label_1h"].value_counts().to_dict()
        label_rows.append({
            "pair": pair,
            "split": split_name,
            "sell_count": int(counts.get(-1, 0)),
            "no_trade_count": int(counts.get(0, 0)),
            "buy_count": int(counts.get(1, 0)),
            "rows": int(len(g)),
        })
    pd.DataFrame(label_rows).to_csv(ROOT / "trade_label_distribution.csv", index=False)

    d.to_csv(ROOT / "modeling_dataset_multi_currency.csv", index=False)
    print(f"modeled rows {len(d)} features {len(model_feature_names())}", flush=True)


if __name__ == "__main__":
    main()


