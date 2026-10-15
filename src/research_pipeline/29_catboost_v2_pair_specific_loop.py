from common import *
from pathlib import Path
import json
import shutil
import time

import joblib
from importlib.machinery import SourceFileLoader

loopmod = SourceFileLoader("catboost_loop_v1", str(ROOT / "28_catboost_loop_backtrader_winrate_search.py")).load_module()

TARGET_MIN_WINRATE = float(os.environ.get("CATBOOST_V2_MIN_WINRATE", "0.55"))
TARGET_ASPIRATIONAL_WINRATE = float(os.environ.get("CATBOOST_V2_ASPIRATIONAL_WINRATE", "0.70"))
MIN_TRADES_PER_PAIR = int(os.environ.get("CATBOOST_V2_MIN_TRADES_PER_PAIR", "30"))
PAIRS = [p.strip().upper() for p in os.environ.get("CATBOOST_V2_PAIRS", ",".join(available_pairs())).split(",") if p.strip()]
OUT_DIR = ARTIFACTS / "catboost_v2_pair_specific_loop"


def pair_attempts():
    # Ordered from balanced to more selective. The loop keeps trying after 55%
    # to look for stronger pair-specific configurations up to the aspirational target.
    return [
        {
            "name": "v2p01_bin_imp1_thr060_d4_lr005",
            "scope": "pair",
            "target_mode": "binary_directional",
            "iterations": 400,
            "depth": 4,
            "learning_rate": 0.05,
            "l2_leaf_reg": 6,
            "threshold": 0.60,
            "importance_min": 1,
            "impact_min": 0,
            "note": "Pair-specific binary, moderate threshold, broad news coverage.",
        },
        {
            "name": "v2p02_bin_imp2_impact1_thr065_d5_lr004",
            "scope": "pair",
            "target_mode": "binary_directional",
            "iterations": 550,
            "depth": 5,
            "learning_rate": 0.04,
            "l2_leaf_reg": 8,
            "threshold": 0.65,
            "importance_min": 2,
            "impact_min": 1,
            "note": "Pair-specific binary, high-impact news and stronger threshold.",
        },
        {
            "name": "v2p03_bin_imp2_impact2_thr070_d5_lr0035",
            "scope": "pair",
            "target_mode": "binary_directional",
            "iterations": 650,
            "depth": 5,
            "learning_rate": 0.035,
            "l2_leaf_reg": 10,
            "threshold": 0.70,
            "importance_min": 2,
            "impact_min": 2,
            "note": "Pair-specific binary, ultra-selective high-impact setup.",
        },
        {
            "name": "v2p04_bin_imp2_impact1_thr070_d6_lr003",
            "scope": "pair",
            "target_mode": "binary_directional",
            "iterations": 750,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 12,
            "threshold": 0.70,
            "importance_min": 2,
            "impact_min": 1,
            "note": "Deeper pair-specific binary model for difficult pairs.",
        },
        {
            "name": "v2p05_bin_imp2_impact2_thr075_d6_lr003",
            "scope": "pair",
            "target_mode": "binary_directional",
            "iterations": 850,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 15,
            "threshold": 0.75,
            "importance_min": 2,
            "impact_min": 2,
            "note": "Very selective pair-specific model for higher winrate.",
        },
        {
            "name": "v2p06_bin_imp2_impact1_thr080_d6_lr0025",
            "scope": "pair",
            "target_mode": "binary_directional",
            "iterations": 950,
            "depth": 6,
            "learning_rate": 0.025,
            "l2_leaf_reg": 18,
            "threshold": 0.80,
            "importance_min": 2,
            "impact_min": 1,
            "note": "High-confidence model; fewer trades are expected.",
        },
        {
            "name": "v2p07_multiclass_imp2_impact1_thr060_d5_lr004",
            "scope": "pair",
            "target_mode": "multiclass",
            "iterations": 650,
            "depth": 5,
            "learning_rate": 0.04,
            "l2_leaf_reg": 10,
            "threshold": 0.60,
            "importance_min": 2,
            "impact_min": 1,
            "class_weights": [1.6, 0.4, 1.6],
            "note": "Pair-specific multiclass with NO_TRADE class retained.",
        },
        {
            "name": "v2p08_multiclass_imp2_impact2_thr065_d6_lr003",
            "scope": "pair",
            "target_mode": "multiclass",
            "iterations": 800,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 12,
            "threshold": 0.65,
            "importance_min": 2,
            "impact_min": 2,
            "class_weights": [1.8, 0.35, 1.8],
            "note": "More selective pair-specific multiclass.",
        },
    ]


def rank_row(row):
    enough_trades = int(row.get("trades", 0)) >= MIN_TRADES_PER_PAIR
    return (
        bool(enough_trades),
        float(row.get("win_rate", 0.0)),
        float(row.get("net_pips", 0.0)),
        int(row.get("trades", 0)),
    )


def copy_best_attempt(attempt_dir, best_dir):
    if best_dir.exists():
        backup_dir = best_dir.parent / f"{best_dir.name}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(best_dir), str(backup_dir))
    shutil.copytree(attempt_dir, best_dir)


def main():
    print("[PROGRESS] START 29_catboost_v2_pair_specific_loop.py", flush=True)
    print(
        f"[CONFIG] min_winrate={TARGET_MIN_WINRATE:.2%} aspirational={TARGET_ASPIRATIONAL_WINRATE:.2%} "
        f"min_trades_per_pair={MIN_TRADES_PER_PAIR} pairs={PAIRS}",
        flush=True,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = loopmod.load_training_frame()
    print(f"[DATA] train_rows={len(train_df)} features={len(loopmod.FEATURES)}", flush=True)

    all_rows = []
    best_rows = []
    attempts = pair_attempts()

    for pair_index, pair in enumerate(PAIRS, start=1):
        print("=" * 72, flush=True)
        print(f"[PAIR] START {pair_index}/{len(PAIRS)} {pair}", flush=True)
        pair_dir = OUT_DIR / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        best = None

        for attempt_index, cfg in enumerate(attempts, start=1):
            started = time.perf_counter()
            attempt_name = f"{attempt_index:02d}_{cfg['name']}"
            attempt_dir = pair_dir / attempt_name
            if attempt_dir.exists():
                backup = pair_dir / f"{attempt_name}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.move(str(attempt_dir), str(backup))
            attempt_dir.mkdir(parents=True, exist_ok=True)
            (attempt_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

            print("-" * 72, flush=True)
            print(f"[ATTEMPT] START pair={pair} {attempt_index}/{len(attempts)} model={cfg['name']}", flush=True)
            print(f"[ATTEMPT] note={cfg['note']}", flush=True)
            print(
                f"[TRAIN] pair={pair} mode={cfg['target_mode']} threshold={cfg['threshold']} "
                f"importance_min={cfg['importance_min']} impact_min={cfg['impact_min']}",
                flush=True,
            )

            model, status = loopmod.train_one_model(train_df, cfg, pair=pair)
            status_row = {"pair": pair, "attempt": attempt_index, "model": cfg["name"], **status}
            pd.DataFrame([status_row]).to_csv(attempt_dir / "training_status.csv", index=False)

            if model is None:
                row = {
                    "pair": pair,
                    "attempt": attempt_index,
                    "model": cfg["name"],
                    "status": "training_failed_or_single_class",
                    "trades": 0,
                    "win_rate": 0.0,
                    "net_pips": 0.0,
                    "seconds": round(time.perf_counter() - started, 2),
                }
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(OUT_DIR / "catboost_v2_pair_loop_all_results.csv", index=False)
                print(f"[DECISION] RETRY pair={pair} reason=no_model status={status}", flush=True)
                continue

            model_path = attempt_dir / f"{cfg['name']}_{pair}.joblib"
            joblib.dump(model, model_path)

            print(f"[TEST] Backtrader start pair={pair} model={cfg['name']}", flush=True)
            result = loopmod.evaluate_pair(pair, cfg, {pair: model}, attempt_dir)
            result.update({
                "attempt": attempt_index,
                "model": cfg["name"],
                "threshold": cfg["threshold"],
                "target_mode": cfg["target_mode"],
                "importance_min": cfg["importance_min"],
                "impact_min": cfg["impact_min"],
                "seconds": round(time.perf_counter() - started, 2),
            })
            pd.DataFrame([result]).to_csv(attempt_dir / "pair_summary.csv", index=False)
            all_rows.append(result)
            pd.DataFrame(all_rows).to_csv(OUT_DIR / "catboost_v2_pair_loop_all_results.csv", index=False)

            print(
                f"[RESULT] pair={pair} model={cfg['name']} trades={result.get('trades', 0)} "
                f"winrate={float(result.get('win_rate', 0.0)):.4f} net_pips={float(result.get('net_pips', 0.0)):.1f} "
                f"profit_factor={float(result.get('profit_factor', 0.0)):.4f}",
                flush=True,
            )

            if best is None or rank_row(result) > best["rank"]:
                best = {"rank": rank_row(result), "result": result, "dir": attempt_dir}
                copy_best_attempt(attempt_dir, pair_dir / "best_model")
                print(f"[BEST] UPDATED pair={pair} model={cfg['name']} winrate={float(result.get('win_rate', 0.0)):.4f}", flush=True)

            enough_trades = int(result.get("trades", 0)) >= MIN_TRADES_PER_PAIR
            winrate = float(result.get("win_rate", 0.0))
            if enough_trades and winrate >= TARGET_ASPIRATIONAL_WINRATE:
                print(f"[DECISION] STOP pair={pair} reason=aspirational_target_reached winrate={winrate:.4f}", flush=True)
                break
            if enough_trades and winrate >= TARGET_MIN_WINRATE:
                print(f"[DECISION] CONTINUE pair={pair} reason=minimum_reached_but_trying_to_improve winrate={winrate:.4f}", flush=True)
            else:
                print(f"[DECISION] RETRY pair={pair} reason=target_not_reached winrate={winrate:.4f} trades={result.get('trades', 0)}", flush=True)

        if best is not None:
            best_result = dict(best["result"])
            best_result["minimum_reached"] = bool(
                int(best_result.get("trades", 0)) >= MIN_TRADES_PER_PAIR and
                float(best_result.get("win_rate", 0.0)) >= TARGET_MIN_WINRATE
            )
            best_result["aspirational_reached"] = bool(
                int(best_result.get("trades", 0)) >= MIN_TRADES_PER_PAIR and
                float(best_result.get("win_rate", 0.0)) >= TARGET_ASPIRATIONAL_WINRATE
            )
            best_rows.append(best_result)
            print(
                f"[PAIR] BEST {pair} model={best_result.get('model')} trades={best_result.get('trades')} "
                f"winrate={float(best_result.get('win_rate', 0.0)):.4f} net_pips={float(best_result.get('net_pips', 0.0)):.1f} "
                f"minimum_reached={best_result['minimum_reached']}",
                flush=True,
            )

    all_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(best_rows)
    all_path = OUT_DIR / "catboost_v2_pair_loop_all_results.csv"
    best_path = OUT_DIR / "catboost_v2_pair_best_results.csv"
    all_df.to_csv(all_path, index=False)
    best_df.to_csv(best_path, index=False)

    lines = [
        "# CatBoost V2 Pair-Specific Loop",
        "",
        f"- Minimum target winrate: `{TARGET_MIN_WINRATE:.2%}`",
        f"- Aspirational target winrate: `{TARGET_ASPIRATIONAL_WINRATE:.2%}`",
        f"- Minimum trades per pair: `{MIN_TRADES_PER_PAIR}`",
        f"- Pairs: `{', '.join(PAIRS)}`",
        "",
        "## Caveat",
        "",
        "This is still a development search because repeated Backtrader test feedback is used to choose the per-pair model. The selected pair models need fresh validation before paper-level claims.",
        "",
        "## Outputs",
        "",
        f"- `{all_path}`",
        f"- `{best_path}`",
        f"- Per-pair best folders: `{OUT_DIR}\\<PAIR>\\best_model`",
    ]
    report_path = OUT_DIR / "LAPORAN_CATBOOST_V2_PAIR_SPECIFIC_LOOP.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("=" * 72, flush=True)
    print(f"[OUTPUT] {all_path}", flush=True)
    print(f"[OUTPUT] {best_path}", flush=True)
    print(f"[OUTPUT] {report_path}", flush=True)


if __name__ == "__main__":
    main()
