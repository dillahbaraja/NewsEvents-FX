from common import *

import json
import math
import time
from pathlib import Path

import joblib
from sklearn.metrics import balanced_accuracy_score, f1_score


PHASE2_DIR = ARTIFACTS / "phase2_two_stage"
DATASET = PHASE2_DIR / "two_stage_modeling_dataset.csv"
METADATA = PHASE2_DIR / "two_stage_metadata.json"
OUT_DIR = ARTIFACTS / "catboost_block_inverse"
JOBLIB_DIR = OUT_DIR / "joblib"
DATATEST = DATA / "datatest"
NEWS_TEST = DATATEST / "NEWS_EVENTS_TEST.csv"
OOS_START = pd.Timestamp("2024-01-01")
OOS_END = pd.Timestamp("2026-08-31 23:59:59")
TOLERANCE_HOURS = 2

MODEL_NAME = "catboost_block_inverse"
STAGE1_TARGET = "trade_target"
STAGE2_TARGET = "direction_target"

VARIANTS = [
    {"variant": "v1_cal_2021_2023", "train_end": 2020, "cal_start": 2021, "cal_end": 2023},
    {"variant": "v2_cal_2020_2023", "train_end": 2019, "cal_start": 2020, "cal_end": 2023},
    {"variant": "v3_cal_2019_2023", "train_end": 2018, "cal_start": 2019, "cal_end": 2023},
]

PARAMS_STAGE1 = {
    "iterations": 450,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5,
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "random_seed": 42,
    "verbose": 100,
    "allow_writing_files": False,
    "auto_class_weights": "Balanced",
}

PARAMS_STAGE2 = {
    "iterations": 450,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5,
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "random_seed": 42,
    "verbose": 100,
    "allow_writing_files": False,
    "auto_class_weights": "Balanced",
}

THRESHOLD_GRID = []
for min_trade_count in [10, 20, 30, 50]:
    for min_pf in [1.05, 1.10, 1.20, 1.30]:
        for min_avg in [0.05, 0.10, 0.25, 0.50]:
            for min_pos_year_ratio in [0.40, 0.50, 0.60]:
                for max_dominant_year_share in [0.60, 0.75]:
                    for unknown_policy in ["BLOCK", "ALLOW_IF_HIGH_PROB"]:
                        THRESHOLD_GRID.append({
                            "min_trade_count": min_trade_count,
                            "min_profit_factor": min_pf,
                            "min_avg_pips": min_avg,
                            "min_positive_year_ratio": min_pos_year_ratio,
                            "max_dominant_year_share": max_dominant_year_share,
                            "unknown_policy": unknown_policy,
                        })


def load_features():
    return json.loads(METADATA.read_text(encoding="utf-8"))["features"]


def normalize_event_name(value):
    return " ".join(str(value).strip().lower().split())


def import_catboost():
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        raise RuntimeError("CatBoost package is required for catboost_block_inverse pipeline") from exc
    return CatBoostClassifier


def read_modeling_dataset(features):
    usecols = [
        "event_time", "Name", "CountryCode", "pair", "event_year", "split",
        "domestic_return_pips_1h", STAGE1_TARGET, STAGE2_TARGET,
    ] + features
    print(f"[DATA] reading={DATASET}", flush=True)
    df = pd.read_csv(DATASET, usecols=usecols, low_memory=False)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["event_name_norm"] = df["Name"].map(normalize_event_name)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["event_year"] = pd.to_numeric(df["event_year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["event_time", "event_year", "domestic_return_pips_1h"]).copy()
    return df


def read_test_price(pair):
    path = DATATEST / pair_file(pair)
    if not path.exists():
        return None
    prices = read_prices(path)
    prices = prices[(prices["time"] >= OOS_START) & (prices["time"] <= OOS_END)].copy()
    return prices if not prices.empty else None


def prepare_oos_events(features):
    print(f"[DATA] building_oos_from={DATATEST}", flush=True)
    if not NEWS_TEST.exists():
        raise FileNotFoundError(f"Missing OOS news file: {NEWS_TEST}")

    rows = []
    mapping = available_currency_pair_mapping()
    pairs = sorted(mapping["pair"].dropna().unique().tolist())
    news_all = read_news(NEWS_TEST)
    news_all = news_all[(news_all["event_time"] >= OOS_START) & (news_all["event_time"] <= OOS_END)].copy()

    for idx, pair in enumerate(pairs, start=1):
        pair_map = mapping[mapping["pair"].eq(pair)].copy()
        prices = read_test_price(pair)
        if prices is None or pair_map.empty:
            print(f"[OOS] SKIP {pair} missing_price_or_mapping", flush=True)
            continue

        px = prices.set_index("time").copy()
        px["close_fwd"] = px["Close"].shift(-1)
        px["exit_time"] = px.index.to_series().shift(-1)
        px = px.reset_index()

        news = news_all[news_all["CountryCode"].isin(pair_map["CountryCode"].unique())].copy()
        if news.empty:
            print(f"[OOS] SKIP {pair} no_relevant_news", flush=True)
            continue
        news["pair"] = pair
        news = news.merge(pair_map[["CountryCode", "currency", "role", "orientation"]], on="CountryCode", how="left")
        news = news[news["orientation"].isin(["base", "quote"])].copy()
        news["category"] = news["Name"].map(event_category)
        news["release_hour"] = news["event_time"].dt.hour
        news["surprise_fc"] = news["Actual"] - news["Forecast"]
        news["surprise_prev"] = news["Actual"] - news["Previous"]
        news["surprise_pct_fc"] = news["surprise_fc"] / (news["Forecast"].abs() + 1e-9)
        news = add_context_features(news)

        joined = pd.merge_asof(
            news.sort_values("event_time"),
            px.sort_values("time"),
            left_on="event_time",
            right_on="time",
            direction="forward",
            tolerance=pd.Timedelta(hours=TOLERANCE_HOURS),
        )
        joined = joined.dropna(subset=["Close", "close_fwd", "exit_time"]).copy()
        if joined.empty:
            print(f"[OOS] SKIP {pair} no_price_coverage", flush=True)
            continue
        sign = np.where(joined["orientation"].eq("base"), 1.0, -1.0)
        joined["pip_size"] = joined["pair"].map(pip_size)
        joined["domestic_return_pips_1h"] = ((joined["close_fwd"] - joined["Close"]) / joined["pip_size"]) * sign
        joined["event_year"] = joined["event_time"].dt.year
        joined["split"] = "oos_2024_2026"
        joined["event_name_norm"] = joined["Name"].map(normalize_event_name)
        joined[features] = joined[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        rows.append(joined)
        print(f"[OOS] pair={idx}/{len(pairs)} {pair} rows={len(joined)}", flush=True)

    if not rows:
        raise RuntimeError("No OOS events could be built from Dataset\\datatest")
    out = pd.concat(rows, ignore_index=True)
    print(f"[DATA] oos_rows={len(out)} pairs={sorted(out['pair'].dropna().unique())}", flush=True)
    return out


def predict_stage1(model, x):
    prob = np.asarray(model.predict_proba(x))
    classes = np.asarray(model.classes_).astype(int)
    idx = int(np.where(classes == 1)[0][0])
    return prob[:, idx]


def predict_stage2(model, x):
    prob = np.asarray(model.predict_proba(x))
    classes = np.asarray(model.classes_).astype(int)
    sell_idx = int(np.where(classes == -1)[0][0])
    buy_idx = int(np.where(classes == 1)[0][0])
    sell_prob = prob[:, sell_idx]
    buy_prob = prob[:, buy_idx]
    pred_direction = np.where(buy_prob >= sell_prob, 1, -1)
    confidence = np.maximum(sell_prob, buy_prob)
    return sell_prob, buy_prob, pred_direction, confidence


def add_predictions(df, stage1_model, stage2_model, features):
    out = df.copy()
    out["prob_trade"] = predict_stage1(stage1_model, out[features])
    sell_prob, buy_prob, pred_direction, confidence = predict_stage2(stage2_model, out[features])
    out["prob_sell"] = sell_prob
    out["prob_buy"] = buy_prob
    out["pred_direction"] = pred_direction.astype(int)
    out["direction_confidence"] = confidence
    return out


def summarize_values(values):
    s = pd.Series(values, dtype=float).dropna()
    trade_count = int(len(s))
    gross_profit = float(s[s > 0].sum())
    gross_loss = float(-s[s < 0].sum())
    net = float(s.sum())
    wins = int((s > 0).sum())
    return {
        "trade_count": trade_count,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pips": net,
        "profit_factor": gross_profit / gross_loss if gross_loss else (math.inf if gross_profit > 0 else 0.0),
        "avg_pips": net / trade_count if trade_count else 0.0,
        "hit_rate": wins / trade_count if trade_count else 0.0,
    }


def yearly_stats(group, pnl_col):
    y = group.groupby("event_year")[pnl_col].sum()
    active_years = int((y != 0).sum())
    positive_years = int((y > 0).sum())
    total_abs = float(y.abs().sum())
    dominant = float(y.abs().max() / total_abs) if total_abs else 1.0
    positive_ratio = positive_years / active_years if active_years else 0.0
    return active_years, positive_years, positive_ratio, dominant


def build_event_stats(calibration):
    active = calibration[calibration["base_signal"].isin([1, -1])].copy()
    if active.empty:
        return pd.DataFrame()
    active["original_pnl"] = active["base_signal"] * active["domestic_return_pips_1h"]
    active["inverse_pnl"] = -active["original_pnl"]

    rows = []
    group_cols = ["pair", "CountryCode", "event_name_norm"]
    for (pair, country, event_name_norm), g in active.groupby(group_cols, dropna=False):
        original = summarize_values(g["original_pnl"])
        inverse = summarize_values(g["inverse_pnl"])
        ay_o, py_o, pyr_o, dom_o = yearly_stats(g, "original_pnl")
        ay_i, py_i, pyr_i, dom_i = yearly_stats(g, "inverse_pnl")
        rows.append({
            "pair": pair,
            "country": country,
            "event_name_norm": event_name_norm,
            "event_name_sample": str(g["Name"].iloc[0]),
            "trade_count": original["trade_count"],
            "original_net_pips": original["net_pips"],
            "original_profit_factor": original["profit_factor"],
            "original_avg_pips": original["avg_pips"],
            "original_hit_rate": original["hit_rate"],
            "inverse_net_pips": inverse["net_pips"],
            "inverse_profit_factor": inverse["profit_factor"],
            "inverse_avg_pips": inverse["avg_pips"],
            "inverse_hit_rate": inverse["hit_rate"],
            "original_active_years": ay_o,
            "original_positive_year_ratio": pyr_o,
            "original_dominant_year_share": dom_o,
            "inverse_active_years": ay_i,
            "inverse_positive_year_ratio": pyr_i,
            "inverse_dominant_year_share": dom_i,
        })
    return pd.DataFrame(rows)


def action_for_row(row, cfg):
    original_ok = (
        row["trade_count"] >= cfg["min_trade_count"] and
        row["original_net_pips"] > 0 and
        row["original_profit_factor"] >= cfg["min_profit_factor"] and
        row["original_avg_pips"] >= cfg["min_avg_pips"] and
        row["original_positive_year_ratio"] >= cfg["min_positive_year_ratio"] and
        row["original_dominant_year_share"] <= cfg["max_dominant_year_share"]
    )
    inverse_ok = (
        row["trade_count"] >= cfg["min_trade_count"] and
        row["inverse_net_pips"] > 0 and
        row["inverse_profit_factor"] >= cfg["min_profit_factor"] and
        row["inverse_avg_pips"] >= cfg["min_avg_pips"] and
        row["inverse_positive_year_ratio"] >= cfg["min_positive_year_ratio"] and
        row["inverse_dominant_year_share"] <= cfg["max_dominant_year_share"]
    )
    if original_ok and inverse_ok:
        if row["inverse_net_pips"] > row["original_net_pips"]:
            return "INVERSE", "both_ok_inverse_higher_net"
        return "ALLOW", "both_ok_original_higher_net"
    if original_ok:
        return "ALLOW", "original_pass"
    if inverse_ok:
        return "INVERSE", "inverse_pass"
    return "BLOCK", "both_fail"


def make_event_map(stats, cfg):
    if stats.empty:
        return pd.DataFrame()
    out = stats.copy()
    if "country" in out.columns and "CountryCode" not in out.columns:
        out["CountryCode"] = out["country"]
    actions = out.apply(lambda r: action_for_row(r, cfg), axis=1)
    out["action"] = [a[0] for a in actions]
    out["reason"] = [a[1] for a in actions]
    for key, value in cfg.items():
        out[key] = value
    return out


def apply_event_map(predictions, event_map, cfg, trade_threshold=0.50, direction_threshold=0.50):
    d = predictions.copy()
    d["base_signal"] = np.where(
        (d["prob_trade"] >= trade_threshold) & (d["direction_confidence"] >= direction_threshold),
        d["pred_direction"],
        0,
    )
    key_cols = ["pair", "CountryCode", "event_name_norm"]
    map_cols = key_cols + ["action", "reason"]
    merged = d.merge(event_map[map_cols], on=key_cols, how="left")
    merged["action"] = merged["action"].fillna("UNKNOWN")
    merged["reason"] = merged["reason"].fillna("unknown_event")
    merged["final_signal"] = 0
    allow_mask = merged["action"].eq("ALLOW")
    inverse_mask = merged["action"].eq("INVERSE")
    unknown_allow = merged["action"].eq("UNKNOWN") & (cfg["unknown_policy"] == "ALLOW_IF_HIGH_PROB") & (merged["prob_trade"] >= 0.70) & (merged["direction_confidence"] >= 0.70)
    merged.loc[allow_mask | unknown_allow, "final_signal"] = merged.loc[allow_mask | unknown_allow, "base_signal"]
    merged.loc[inverse_mask, "final_signal"] = -merged.loc[inverse_mask, "base_signal"]
    merged.loc[~merged["final_signal"].isin([1, -1]), "final_signal"] = 0
    merged["pnl_pips"] = merged["final_signal"] * merged["domestic_return_pips_1h"]
    merged["status"] = np.select(
        [
            merged["final_signal"].isin([1, -1]) & merged["action"].eq("ALLOW"),
            merged["final_signal"].isin([1, -1]) & merged["action"].eq("INVERSE"),
            merged["final_signal"].isin([1, -1]) & unknown_allow,
            merged["action"].eq("BLOCK"),
            merged["action"].eq("UNKNOWN"),
            merged["base_signal"].eq(0),
        ],
        ["TRADE_ALLOW", "TRADE_INVERSE", "TRADE_UNKNOWN_HIGH_PROB", "BLOCK", "BLOCK_UNKNOWN", "NO_BASE_SIGNAL"],
        default="NO_TRADE",
    )
    return merged


def summarize_orders(name, split, orders):
    active = orders[orders["final_signal"].isin([1, -1])].copy()
    pnl = pd.to_numeric(active["pnl_pips"], errors="coerce").fillna(0.0)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    trades = int(len(active))
    return {
        "variant": name,
        "split": split,
        "trade_count": trades,
        "buy_count": int((active["final_signal"] == 1).sum()),
        "sell_count": int((active["final_signal"] == -1).sum()),
        "net_pips": float(pnl.sum()),
        "avg_pips": float(pnl.mean()) if trades else 0.0,
        "winrate": float((pnl > 0).mean()) if trades else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else (math.inf if gross_profit > 0 else 0.0),
        "positive_pairs": int(active.groupby("pair")["pnl_pips"].sum().gt(0).sum()) if trades else 0,
        "active_pairs": int(active["pair"].nunique()) if trades else 0,
    }


def score_summary(row):
    if row["trade_count"] < 200:
        return -1e12
    pf_bonus = min(row["profit_factor"], 3.0) * 1000
    pair_bonus = row["positive_pairs"] * 250
    trade_bonus = min(row["trade_count"], 20000) * 0.02
    return row["net_pips"] + pf_bonus + pair_bonus + trade_bonus


def md_table(df, floatfmt=".4f"):
    if df.empty:
        return "_Tidak ada data._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def write_signal_table(orders, path):
    active = orders[orders["final_signal"].isin([1, -1])].copy()
    out = pd.DataFrame({
        "EVENT_TIME": active["event_time"].dt.strftime("%Y.%m.%d %H:%M"),
        "SYMBOL": active["pair"].astype(str),
        "COUNTRY": active["CountryCode"].astype(str),
        "EVENT_NAME": active["Name"].astype(str),
        "PROBABILITY": active[["prob_trade", "direction_confidence"]].min(axis=1),
        "SIGNAL": active["final_signal"].astype(int),
        "MODEL": MODEL_NAME,
    })
    out.to_csv(path, index=False)
    return out


def train_variant(df, oos_df, features, variant_cfg):
    CatBoostClassifier = import_catboost()
    variant = variant_cfg["variant"]
    vdir = OUT_DIR / variant
    vdir.mkdir(parents=True, exist_ok=True)
    train = df[df["event_year"].le(variant_cfg["train_end"])].copy()
    calibration = df[df["event_year"].between(variant_cfg["cal_start"], variant_cfg["cal_end"])].copy()
    oos = oos_df.copy()

    s1_train = train.dropna(subset=[STAGE1_TARGET]).copy()
    s1_cal = calibration.dropna(subset=[STAGE1_TARGET]).copy()
    s1_oos = oos.dropna(subset=["domestic_return_pips_1h"]).copy()
    s2_train = train.dropna(subset=[STAGE2_TARGET]).copy()

    print(f"[VARIANT] {variant} train_end={variant_cfg['train_end']} calibration={variant_cfg['cal_start']}-{variant_cfg['cal_end']}", flush=True)
    print(f"[DATA] {variant} stage1_train={len(s1_train)} calibration={len(s1_cal)} oos={len(s1_oos)} stage2_train={len(s2_train)}", flush=True)

    stage1 = CatBoostClassifier(**PARAMS_STAGE1)
    stage2 = CatBoostClassifier(**PARAMS_STAGE2)
    started = time.perf_counter()
    print(f"[MODEL] START {variant} stage1_trade_filter", flush=True)
    stage1.fit(s1_train[features], s1_train[STAGE1_TARGET].astype(int))
    print(f"[MODEL] DONE {variant} stage1 seconds={time.perf_counter() - started:.1f}", flush=True)
    started = time.perf_counter()
    print(f"[MODEL] START {variant} stage2_direction", flush=True)
    stage2.fit(s2_train[features], s2_train[STAGE2_TARGET].astype(int))
    print(f"[MODEL] DONE {variant} stage2 seconds={time.perf_counter() - started:.1f}", flush=True)

    joblib.dump(stage1, vdir / "catboost_block_inverse_trade_filter.joblib")
    joblib.dump(stage2, vdir / "catboost_block_inverse_direction.joblib")

    cal_pred = add_predictions(s1_cal, stage1, stage2, features)
    oos_pred = add_predictions(s1_oos, stage1, stage2, features)
    cal_pred["base_signal"] = np.where((cal_pred["prob_trade"] >= 0.50) & (cal_pred["direction_confidence"] >= 0.50), cal_pred["pred_direction"], 0)
    stats = build_event_stats(cal_pred)
    stats.to_csv(vdir / "catboost_block_inverse_event_stats.csv", index=False)

    grid_rows = []
    best = None
    best_orders = None
    best_map = None
    for i, cfg in enumerate(THRESHOLD_GRID, start=1):
        event_map = make_event_map(stats, cfg)
        orders = apply_event_map(cal_pred, event_map, cfg)
        row = summarize_orders(variant, "calibration", orders)
        row.update(cfg)
        row["score"] = score_summary(row)
        row["allow_count"] = int(event_map["action"].eq("ALLOW").sum()) if not event_map.empty else 0
        row["inverse_count"] = int(event_map["action"].eq("INVERSE").sum()) if not event_map.empty else 0
        row["block_count"] = int(event_map["action"].eq("BLOCK").sum()) if not event_map.empty else 0
        grid_rows.append(row)
        if best is None or row["score"] > best["score"]:
            best = row
            best_orders = orders
            best_map = event_map
        if i % 100 == 0:
            print(f"[GRID] {variant} {i}/{len(THRESHOLD_GRID)} best_score={best['score']:.2f} best_net={best['net_pips']:.1f} best_pf={best['profit_factor']:.4f}", flush=True)

    grid = pd.DataFrame(grid_rows).sort_values("score", ascending=False)
    grid.to_csv(vdir / "catboost_block_inverse_grid_results.csv", index=False)
    best_map.to_csv(vdir / "catboost_block_inverse_event_map.csv", index=False)
    best_orders.to_csv(vdir / "catboost_block_inverse_calibration_orders.csv", index=False)

    oos_orders = apply_event_map(oos_pred, best_map, best)
    oos_orders.to_csv(vdir / "catboost_block_inverse_oos_orders.csv", index=False)
    cal_summary = summarize_orders(variant, "calibration", best_orders)
    oos_summary = summarize_orders(variant, "oos_2024_2026", oos_orders)
    for row in [cal_summary, oos_summary]:
        row.update({k: best[k] for k in ["min_trade_count", "min_profit_factor", "min_avg_pips", "min_positive_year_ratio", "max_dominant_year_share", "unknown_policy"]})
        row["allow_count"] = int(best_map["action"].eq("ALLOW").sum()) if not best_map.empty else 0
        row["inverse_count"] = int(best_map["action"].eq("INVERSE").sum()) if not best_map.empty else 0
        row["block_count"] = int(best_map["action"].eq("BLOCK").sum()) if not best_map.empty else 0
    pd.DataFrame([cal_summary, oos_summary]).to_csv(vdir / "catboost_block_inverse_summary.csv", index=False)
    return {
        "variant": variant,
        "config": variant_cfg,
        "best": best,
        "cal_summary": cal_summary,
        "oos_summary": oos_summary,
        "event_map_path": vdir / "catboost_block_inverse_event_map.csv",
        "oos_orders_path": vdir / "catboost_block_inverse_oos_orders.csv",
    }


def make_report(window_summary, best_variant, best_orders, signal_table):
    cal = window_summary[window_summary["split"].eq("calibration")].sort_values("score", ascending=False)
    oos = window_summary[window_summary["split"].eq("oos_2024_2026")].sort_values("net_pips", ascending=False)
    active = best_orders[best_orders["final_signal"].isin([1, -1])].copy()
    by_pair = active.groupby("pair", as_index=False).agg(
        trades=("final_signal", "size"),
        buy_count=("final_signal", lambda s: int((s == 1).sum())),
        sell_count=("final_signal", lambda s: int((s == -1).sum())),
        net_pips=("pnl_pips", "sum"),
        winrate=("pnl_pips", lambda s: float((s > 0).mean())),
    )
    pf_rows = []
    for pair, g in active.groupby("pair"):
        pf_rows.append({"pair": pair, "profit_factor": profit_factor(g["pnl_pips"])})
    if pf_rows:
        by_pair = by_pair.merge(pd.DataFrame(pf_rows), on="pair", how="left").sort_values("net_pips", ascending=False)
    top_events = active.groupby(["pair", "CountryCode", "Name"], as_index=False).agg(
        trades=("final_signal", "size"),
        net_pips=("pnl_pips", "sum"),
    ).sort_values("net_pips", ascending=False).head(20)
    worst_events = active.groupby(["pair", "CountryCode", "Name"], as_index=False).agg(
        trades=("final_signal", "size"),
        net_pips=("pnl_pips", "sum"),
    ).sort_values("net_pips", ascending=True).head(20)

    lines = [
        "# CatBoost Block/Inverse - Macro News FX",
        "",
        "Model ini menggabungkan CatBoost probability dengan event-level block/inverse reliability layer.",
        "",
        "Unit reliability layer: `pair + country + event_name`.",
        "OOS final: `2024-01-01` sampai `2026-08-31`.",
        "",
        "## Window Comparison - Calibration",
        "",
        md_table(cal[["variant", "trade_count", "net_pips", "avg_pips", "winrate", "profit_factor", "positive_pairs", "allow_count", "inverse_count", "block_count", "score", "unknown_policy"]], ".4f"),
        "",
        "## Window Comparison - OOS",
        "",
        md_table(oos[["variant", "trade_count", "net_pips", "avg_pips", "winrate", "profit_factor", "positive_pairs", "allow_count", "inverse_count", "block_count", "unknown_policy"]], ".4f"),
        "",
        f"Selected variant from calibration: `{best_variant}`.",
        "",
        "## Selected Variant OOS Per Pair",
        "",
        md_table(by_pair, ".4f"),
        "",
        "## Top OOS Events",
        "",
        md_table(top_events, ".4f"),
        "",
        "## Worst OOS Events",
        "",
        md_table(worst_events, ".4f"),
        "",
        "## MT5 Export",
        "",
        f"- Signal table: `{signal_table}`",
        f"- Model name for EA CSV filter: `{MODEL_NAME}`",
        "",
        "## Notes",
        "",
        "- Variant selection uses calibration only, not OOS.",
        "- OOS is used only for final evaluation.",
        "- This is a hybrid model, not the same as manual block/inverse.",
    ]
    return "\n".join(lines) + "\n"


def main():
    print("[PROGRESS] START 50_catboost_block_inverse_pipeline.py", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JOBLIB_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    df = read_modeling_dataset(features)
    print(f"[DATA] rows={len(df)} features={len(features)} years={df['event_year'].min()}-{df['event_year'].max()}", flush=True)
    oos_df = prepare_oos_events(features)

    results = []
    for variant_cfg in VARIANTS:
        results.append(train_variant(df, oos_df, features, variant_cfg))

    summary_rows = []
    for r in results:
        cal = dict(r["cal_summary"])
        cal["score"] = score_summary(cal)
        oos = dict(r["oos_summary"])
        oos["score"] = np.nan
        summary_rows.extend([cal, oos])
    window_summary = pd.DataFrame(summary_rows)
    window_summary.to_csv(OUT_DIR / "catboost_block_inverse_window_comparison.csv", index=False)

    best_cal = window_summary[window_summary["split"].eq("calibration")].sort_values("score", ascending=False).iloc[0]
    best_variant = str(best_cal["variant"])
    best_result = next(r for r in results if r["variant"] == best_variant)
    best_dir = OUT_DIR / best_variant
    best_orders = pd.read_csv(best_result["oos_orders_path"], low_memory=False)
    best_orders["event_time"] = pd.to_datetime(best_orders["event_time"], errors="coerce")

    best_map = pd.read_csv(best_result["event_map_path"], low_memory=False)
    best_map.to_csv(OUT_DIR / "catboost_block_inverse_event_map.csv", index=False)
    best_orders.to_csv(OUT_DIR / "catboost_block_inverse_oos_orders.csv", index=False)
    signal_table = OUT_DIR / "catboost_block_inverse_mt5_signal_table.csv"
    signals = write_signal_table(best_orders, signal_table)
    signals.to_csv(ARTIFACTS / "catboost_block_inverse_mt5_signal_table.csv", index=False)

    report = make_report(window_summary, best_variant, best_orders, signal_table)
    (OUT_DIR / "LAPORAN_CATBOOST_BLOCK_INVERSE.md").write_text(report, encoding="utf-8")
    print(f"[SELECTED] variant={best_variant}", flush=True)
    print(f"[OUTPUT] report={OUT_DIR / 'LAPORAN_CATBOOST_BLOCK_INVERSE.md'}", flush=True)
    print(f"[OUTPUT] mt5_signal_table={signal_table}", flush=True)
    print("[PROGRESS] DONE 50_catboost_block_inverse_pipeline.py", flush=True)


if __name__ == "__main__":
    main()
