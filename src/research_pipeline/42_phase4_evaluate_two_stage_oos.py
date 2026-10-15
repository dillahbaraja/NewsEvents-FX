from common import *

import json
import math
import time
import warnings

import joblib
from sklearn.exceptions import InconsistentVersionWarning


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


PHASE2_DIR = ARTIFACTS / "phase2_two_stage"
MODEL_DIR = ARTIFACTS / "phase3_two_stage_models"
JOBLIB_DIR = MODEL_DIR / "joblib"
OUT_DIR = ARTIFACTS / "phase4_two_stage_evaluation"
DATASET = PHASE2_DIR / "two_stage_modeling_dataset.csv"
METADATA = PHASE2_DIR / "two_stage_metadata.json"
DATATEST = DATA / "datatest"
NEWS_TEST = DATATEST / "NEWS_EVENTS_TEST.csv"

MODEL_FAMILIES = {
    "elastic_net_two_stage": {
        "stage1": "elastic_net_trade_filter.joblib",
        "stage1_name": "elastic_net_trade_filter",
        "stage2": "elastic_net_direction.joblib",
        "stage2_name": "elastic_net_direction",
    },
    "random_forest_two_stage": {
        "stage1": "random_forest_trade_filter.joblib",
        "stage1_name": "random_forest_trade_filter",
        "stage2": "random_forest_direction.joblib",
        "stage2_name": "random_forest_direction",
    },
    "catboost_two_stage": {
        "stage1": "catboost_trade_filter.joblib",
        "stage1_name": "catboost_trade_filter",
        "stage2": "catboost_direction.joblib",
        "stage2_name": "catboost_direction",
    },
}


def load_features():
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    return meta["features"]


def read_news_file(path):
    d = pd.read_csv(path, low_memory=False)
    d["event_time"] = pd.to_datetime(d["Datetime"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
    if d["event_time"].isna().any():
        d["event_time"] = pd.to_datetime(d["Datetime"], errors="coerce")
    for col in ["Actual", "Forecast", "Previous", "Importance", "Impact"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d.dropna(subset=["event_time"]).sort_values("event_time")


def read_price_file(path):
    d = pd.read_csv(path, usecols=lambda c: c in {"Time", "Open", "High", "Low", "Close", "Volume"})
    d["time"] = pd.to_datetime(d["Time"], format="%Y.%m.%d %H:%M", errors="coerce")
    if d["time"].isna().any():
        d["time"] = pd.to_datetime(d["Time"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d.dropna(subset=["time", "Close"]).sort_values("time").drop_duplicates("time")


def build_oos_dataset(features):
    print(f"[OOS] reading news={NEWS_TEST}", flush=True)
    news = read_news_file(NEWS_TEST)
    full_mapping = build_currency_pair_mapping()
    rows = []
    thresholds = pd.read_csv(ROOT / "return_significance_thresholds.csv")

    for pair in sorted({p.stem.replace("_H1_Data", "") for p in DATATEST.glob("*_H1_Data.csv")}):
        pair_path = DATATEST / pair_file(pair)
        pair_map = full_mapping[full_mapping["pair"].eq(pair)].copy()
        pair_map = pair_map[pair_map["orientation"].isin(["base", "quote"])]
        if pair_map.empty or not pair_path.exists():
            continue
        countries = sorted(pair_map["CountryCode"].unique())
        print(f"[OOS] pair={pair} countries={countries}", flush=True)
        px = read_price_file(pair_path).set_index("time")
        px["close_fwd_1h"] = px["Close"].shift(-1)

        n = news[news["CountryCode"].isin(countries)].copy()
        n["pair"] = pair
        n = n.merge(pair_map[["CountryCode", "currency", "role", "orientation"]], on="CountryCode", how="left")
        n = n[n["orientation"].isin(["base", "quote"])].set_index("event_time").sort_index()
        joined = pd.merge_asof(
            n,
            px.sort_index(),
            left_index=True,
            right_index=True,
            direction="forward",
            tolerance=pd.Timedelta("2h"),
        ).reset_index().rename(columns={"index": "event_time"})
        rows.append(joined)

    if not rows:
        return pd.DataFrame()

    d = pd.concat(rows, ignore_index=True)
    print(f"[OOS] joined_rows={len(d)}", flush=True)
    d["surprise_fc"] = d["Actual"] - d["Forecast"]
    d["surprise_prev"] = d["Actual"] - d["Previous"]
    d["surprise_pct_fc"] = d["surprise_fc"] / (d["Forecast"].abs() + 1e-9)
    d["category"] = d["Name"].map(event_category)
    d["release_hour"] = d["event_time"].dt.hour
    d["event_year"] = d["event_time"].dt.year
    d = add_context_features(d)

    sign = np.where(d["orientation"] == "base", 1.0, -1.0)
    d["return_1h"] = np.log(d["close_fwd_1h"] / d["Close"])
    d["domestic_return_1h"] = d["return_1h"] * sign
    d["pip_size"] = d["pair"].map(pip_size)
    d["domestic_return_pips_1h"] = ((d["close_fwd_1h"] - d["Close"]) / d["pip_size"]) * sign
    d = d.merge(thresholds[["pair", "domestic_return_abs_threshold_1h"]], on="pair", how="left")
    d["has_valid_outcome"] = d["domestic_return_pips_1h"].notna().astype("int8")
    d["trade_label_1h"] = np.select(
        [
            d["domestic_return_1h"] >= d["domestic_return_abs_threshold_1h"],
            d["domestic_return_1h"] <= -d["domestic_return_abs_threshold_1h"],
        ],
        [1, -1],
        default=0,
    ).astype("int8")
    d["trade_target"] = np.where(d["has_valid_outcome"].eq(1), (d["trade_label_1h"] != 0).astype("int8"), np.nan)
    d["direction_target"] = np.where(
        d["has_valid_outcome"].eq(1) & pd.Series(d["trade_label_1h"]).isin([-1, 1]),
        d["trade_label_1h"],
        np.nan,
    )
    d["split"] = "oos_2024_2026"
    for col in features:
        if col not in d.columns:
            d[col] = 0
    return d


def predict_proba_with_progress(model, x, label, chunk_size=5000):
    n = len(x)
    if n == 0:
        return np.empty((0, 0))
    frames = []
    started = time.time()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        frames.append(np.asarray(model.predict_proba(x.iloc[start:end])))
        print(
            f"[PREDICT] {label} rows={end}/{n} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    return np.vstack(frames)


def positive_probability(model, x, positive_class=1, label="stage1"):
    probs = predict_proba_with_progress(model, x, label)
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
    idx = int(np.where(classes == positive_class)[0][0])
    return probs[:, idx]


def direction_prediction(model, x, label="stage2"):
    probs = predict_proba_with_progress(model, x, label)
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
    sell_idx = int(np.where(classes == -1)[0][0])
    buy_idx = int(np.where(classes == 1)[0][0])
    sell_prob = probs[:, sell_idx]
    buy_prob = probs[:, buy_idx]
    pred = np.where(buy_prob >= sell_prob, 1, -1)
    confidence = np.maximum(sell_prob, buy_prob)
    return sell_prob, buy_prob, pred, confidence


def load_thresholds():
    s1 = pd.read_csv(MODEL_DIR / "stage1_trade_filter_thresholds.csv")
    s2 = pd.read_csv(MODEL_DIR / "stage2_direction_thresholds.csv")
    stage1 = {(r["model"], r["pair"]): float(r["threshold"]) for _, r in s1.iterrows()}
    stage2 = {(r["model"], r["pair"]): float(r["confidence_threshold"]) for _, r in s2.iterrows()}
    return stage1, stage2


def profit_factor(values):
    s = pd.Series(values).dropna()
    gross_profit = s[s > 0].sum()
    gross_loss = -s[s < 0].sum()
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def summarize_orders(orders):
    rows = []
    if orders.empty:
        return pd.DataFrame()
    for (model, split_name, pair), g in orders.groupby(["model", "split", "pair"], dropna=False):
        rows.append({
            "model": model,
            "split": split_name,
            "pair": pair,
            "trade_count": int(len(g)),
            "buy_count": int((g["final_signal"] == "BUY").sum()),
            "sell_count": int((g["final_signal"] == "SELL").sum()),
            "net_pips": float(g["pnl_pips"].sum()),
            "avg_pips": float(g["pnl_pips"].mean()),
            "winrate": float((g["pnl_pips"] > 0).mean()),
            "profit_factor": profit_factor(g["pnl_pips"]),
        })
    all_rows = []
    for (model, split_name), g in orders.groupby(["model", "split"], dropna=False):
        all_rows.append({
            "model": model,
            "split": split_name,
            "pair": "ALL",
            "trade_count": int(len(g)),
            "buy_count": int((g["final_signal"] == "BUY").sum()),
            "sell_count": int((g["final_signal"] == "SELL").sum()),
            "net_pips": float(g["pnl_pips"].sum()),
            "avg_pips": float(g["pnl_pips"].mean()),
            "winrate": float((g["pnl_pips"] > 0).mean()),
            "profit_factor": profit_factor(g["pnl_pips"]),
        })
    return pd.DataFrame(rows + all_rows)


def df_to_markdown(df):
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = "inf" if math.isinf(value) else f"{value:.4f}"
            else:
                value = "" if pd.isna(value) else str(value)
            vals.append(value.replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def evaluate_split(split_df, split_name, features, stage1_thresholds, stage2_thresholds):
    prediction_frames = []
    order_frames = []
    x = split_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    for family, spec in MODEL_FAMILIES.items():
        print(f"[EVAL] split={split_name} model={family}", flush=True)
        stage1_model = joblib.load(JOBLIB_DIR / spec["stage1"])
        stage2_model = joblib.load(JOBLIB_DIR / spec["stage2"])
        trade_prob = positive_probability(stage1_model, x, 1, f"{split_name} {family} stage1")
        sell_prob, buy_prob, direction_pred, direction_conf = direction_prediction(stage2_model, x, f"{split_name} {family} stage2")

        out = split_df[["event_time", "Name", "CountryCode", "pair", "domestic_return_pips_1h", "trade_target", "direction_target"]].copy()
        out["model"] = family
        out["split"] = split_name
        out["prob_trade"] = trade_prob
        out["prob_sell"] = sell_prob
        out["prob_buy"] = buy_prob
        out["direction_confidence"] = direction_conf
        out["pred_direction"] = direction_pred
        out["stage1_threshold"] = [stage1_thresholds.get((spec["stage1_name"], pair), 0.50) for pair in out["pair"]]
        out["stage2_threshold"] = [stage2_thresholds.get((spec["stage2_name"], pair), 0.50) for pair in out["pair"]]
        out["stage1_trade"] = out["prob_trade"] >= out["stage1_threshold"]
        out["stage2_pass"] = out["direction_confidence"] >= out["stage2_threshold"]
        out["final_signal"] = np.where(
            out["stage1_trade"] & out["stage2_pass"],
            np.where(out["pred_direction"] == 1, "BUY", "SELL"),
            "NO_TRADE",
        )
        out["reason"] = np.select(
            [
                ~out["stage1_trade"],
                out["stage1_trade"] & ~out["stage2_pass"],
                out["final_signal"].isin(["BUY", "SELL"]),
            ],
            ["STAGE1_NO_TRADE", "STAGE2_LOW_CONFIDENCE", "TRADE"],
            default="UNKNOWN",
        )
        out["pnl_pips"] = np.where(
            out["final_signal"].eq("BUY"),
            out["domestic_return_pips_1h"],
            np.where(out["final_signal"].eq("SELL"), -out["domestic_return_pips_1h"], 0.0),
        )
        prediction_frames.append(out)
        order_frames.append(out[out["final_signal"].isin(["BUY", "SELL"])].copy())

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    orders = pd.concat(order_frames, ignore_index=True) if order_frames else pd.DataFrame()
    return predictions, orders


def main():
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    stage1_thresholds, stage2_thresholds = load_thresholds()

    required = [
        "event_time", "Name", "CountryCode", "pair", "split", "domestic_return_pips_1h",
        "trade_target", "direction_target",
    ] + features
    print("[PROGRESS] START 42_phase4_evaluate_two_stage_oos.py", flush=True)
    print("[VALIDATION] reading two-stage dataset", flush=True)
    trainval = pd.read_csv(DATASET, usecols=lambda c: c in set(required), low_memory=False)
    validation = trainval[trainval["split"].eq("validation") & trainval["domestic_return_pips_1h"].notna()].copy()
    validation["event_time"] = pd.to_datetime(validation["event_time"], errors="coerce")
    print(f"[VALIDATION] rows={len(validation)}", flush=True)

    oos = build_oos_dataset(features)
    oos = oos[oos["domestic_return_pips_1h"].notna()].copy()
    print(f"[OOS] rows={len(oos)}", flush=True)

    val_predictions, val_orders = evaluate_split(validation, "validation_2021_2023", features, stage1_thresholds, stage2_thresholds)
    oos_predictions, oos_orders = evaluate_split(oos, "oos_2024_2026", features, stage1_thresholds, stage2_thresholds)

    predictions = pd.concat([val_predictions, oos_predictions], ignore_index=True)
    orders = pd.concat([val_orders, oos_orders], ignore_index=True)
    summary = summarize_orders(orders)

    predictions.to_csv(OUT_DIR / "phase4_two_stage_predictions.csv", index=False)
    orders.to_csv(OUT_DIR / "phase4_two_stage_orders.csv", index=False)
    summary.to_csv(OUT_DIR / "phase4_two_stage_summary.csv", index=False)

    if not orders.empty:
        monthly = orders.copy()
        monthly["month"] = pd.to_datetime(monthly["event_time"], errors="coerce").dt.to_period("M").astype(str)
        monthly = monthly.groupby(["model", "split", "pair", "month"], as_index=False)["pnl_pips"].sum()
        monthly["cumulative_pips"] = monthly.groupby(["model", "split", "pair"])["pnl_pips"].cumsum()
        monthly.to_csv(OUT_DIR / "phase4_two_stage_monthly_cumulative.csv", index=False)

    oos_rank = summary[summary["split"].eq("oos_2024_2026") & summary["pair"].eq("ALL")].copy()
    oos_rank = oos_rank.sort_values("net_pips", ascending=False)
    val_rank = summary[summary["split"].eq("validation_2021_2023") & summary["pair"].eq("ALL")].copy()
    val_rank = val_rank.sort_values("net_pips", ascending=False)
    pair_rank = summary[summary["split"].eq("oos_2024_2026") & ~summary["pair"].eq("ALL")].copy()
    pair_rank = pair_rank.sort_values(["model", "net_pips"], ascending=[True, False])

    report = [
        "# Phase 4 - Two-Stage Evaluation",
        "",
        "Evaluasi ini memakai model Fase 3 dengan urutan `Stage 1 TRADE/NO_TRADE -> Stage 2 BUY/SELL`.",
        "",
        "- Training model: 2010-2020",
        "- Threshold/validation: 2021-2023",
        "- OOS: 2024+ dari `Dataset\\datatest`",
        "- Data 2024+ tidak dipakai untuk training atau threshold.",
        "",
        "## OOS Ranking - ALL Pairs",
        "",
        df_to_markdown(oos_rank[["model", "trade_count", "buy_count", "sell_count", "net_pips", "avg_pips", "winrate", "profit_factor"]]),
        "",
        "## Validation Ranking - ALL Pairs",
        "",
        df_to_markdown(val_rank[["model", "trade_count", "buy_count", "sell_count", "net_pips", "avg_pips", "winrate", "profit_factor"]]),
        "",
        "## OOS Per Pair",
        "",
        df_to_markdown(pair_rank[["model", "pair", "trade_count", "net_pips", "winrate", "profit_factor"]]),
        "",
        "## Output Files",
        "",
        f"- `{OUT_DIR / 'phase4_two_stage_predictions.csv'}`",
        f"- `{OUT_DIR / 'phase4_two_stage_orders.csv'}`",
        f"- `{OUT_DIR / 'phase4_two_stage_summary.csv'}`",
        f"- `{OUT_DIR / 'phase4_two_stage_monthly_cumulative.csv'}`",
        "",
        f"Elapsed seconds: {time.time() - started:.1f}",
    ]
    (OUT_DIR / "LAPORAN_PHASE4_TWO_STAGE_EVALUATION.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[OUTPUT] {OUT_DIR / 'phase4_two_stage_summary.csv'}", flush=True)
    print(f"[OUTPUT] {OUT_DIR / 'LAPORAN_PHASE4_TWO_STAGE_EVALUATION.md'}", flush=True)
    print("[PROGRESS] DONE 42_phase4_evaluate_two_stage_oos.py", flush=True)


if __name__ == "__main__":
    main()
