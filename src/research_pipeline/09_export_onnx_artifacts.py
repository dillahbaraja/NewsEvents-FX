from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import json
import os
import joblib

ART = ARTIFACTS
FEATURES = model_feature_names()
TARGET = "trade_label_1h"


def load_best_params(model_name):
    path = ART / "en_rf_cb_best_params.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(model_name, {}).get("params", {})


def cleanup_export_outputs(export_mode, compact_export, addon_export):
    targets = [ART / "onnx_export_status.csv"]
    if compact_export:
        targets += [
            ART / "catboost.onnx",
            ART / "catboost_model.cbm",
            ART / "elastic_net.onnx",
            ART / "random_forest.onnx",
            ART / "svm_rbf.onnx",
        ]
    elif addon_export:
        targets += [ART / "svm_rbf.onnx"]
    else:
        targets += [
            ART / "elastic_net.onnx",
            ART / "random_forest.onnx",
            ART / "svm_rbf.onnx",
            ART / "catboost.onnx",
            ART / "catboost_model.cbm",
        ]
    for path in targets:
        if path.exists():
            path.unlink()
    print(f"[CLEANUP] previous export artifacts removed for mode={export_mode}", flush=True)


def main():
    print("[PROGRESS] START 09_export_onnx_artifacts.py", flush=True)
    export_mode = os.environ.get("NEWS_EXPORT_MODE", "full").lower()
    compact_export = export_mode in {"ready", "en_rf_cb"}
    addon_export = export_mode == "ebm_svm"
    print(f"[EXPORT] mode={export_mode}", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "joblib").mkdir(parents=True, exist_ok=True)
    cleanup_export_outputs(export_mode, compact_export, addon_export)
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv", usecols=["event_year", TARGET] + FEATURES)
    d = d.dropna(subset=FEATURES + [TARGET])
    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    tr = d.event_year <= TRAIN_END_YEAR
    X = d.loc[tr, FEATURES].astype("float32")
    y = d.loc[tr, TARGET].astype(int)
    print(f"[DATA] train rows={len(X)} features={len(FEATURES)}", flush=True)
    best_param_path = ART / "en_rf_cb_best_params.json"
    best_params = json.loads(best_param_path.read_text(encoding="utf-8")) if best_param_path.exists() else {}
    meta = {
        "features": FEATURES,
        "input_shape": [None, len(FEATURES)],
        "onnx_outputs": ["label", "probabilities"],
        "target": TARGET,
        "classes": [{"id": -1, "name": "SELL"}, {"id": 0, "name": "NO_TRADE"}, {"id": 1, "name": "BUY"}],
        "train_period": "2010-2020",
        "validation_period": "2021-2023",
        "final_test_period": "2024-2026",
        "threshold_file": "model_pair_thresholds.csv",
        "signal_table_file": "mt5_signal_table.csv",
        "best_params": best_params,
        "asset_universe": "fx_only",
        "excluded_assets": EXCLUDED_ASSETS,
        "onnx_scope": "EN_RF_CB exports CatBoost ONNX for MT5 and keeps Elastic Net, Random Forest, and CatBoost as Python comparison models. EBM_SVM trains additional EBM and SVM-RBF artifacts. EBM is retained as an interpretable joblib artifact.",
    }
    models = {}
    if not compact_export and not addon_export:
        models["elastic_net"] = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            random_state=42,
        ))
    if not compact_export and not addon_export:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=10,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        models["svm_rbf"] = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=False, max_iter=1000, random_state=42))
    elif addon_export:
        models["svm_rbf"] = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=False, max_iter=1000, random_state=42))
    else:
        for stale in ["elastic_net.onnx", "random_forest.onnx", "svm_rbf.onnx"]:
            stale_path = ART / stale
            if stale_path.exists():
                stale_path.unlink()
        export_rows = [
            {
                "model": "elastic_net",
                "status": "skipped_in_en_rf_cb_mode",
                "note": "Sklearn ONNX conversion was slow after context-feature expansion. Use joblib for Python comparison or EBM_SVM batch for ONNX attempt.",
            },
            {
                "model": "random_forest",
                "status": "skipped_in_en_rf_cb_mode",
                "note": "Random Forest ONNX conversion is slow; use EBM_SVM batch for this export. The model is still trained in EN_RF_CB comparison.",
            },
            {
                "model": "svm_rbf",
                "status": "skipped_in_en_rf_cb_mode",
                "note": "SVM-RBF is reserved for EBM_SVM batch due to training cost.",
            },
        ]
    if not compact_export:
        export_rows = []
    for name, model in models.items():
        print(f"[EXPORT] START {name}", flush=True)
        model.fit(X, y)
        print(f"[EXPORT] FIT DONE {name}", flush=True)
        joblib.dump(model, ART / "joblib" / (name + ".joblib"))
        try:
            onx = convert_sklearn(
                model,
                initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
                options={id(model): {"zipmap": False}},
            )
            (ART / (name + ".onnx")).write_bytes(onx.SerializeToString())
            export_rows.append({"model": name, "status": "exported", "note": ""})
        except Exception as exc:
            export_rows.append({"model": name, "status": "onnx_export_failed", "note": str(exc)})

    if not addon_export:
        try:
            from catboost import CatBoostClassifier
            print("[EXPORT] START catboost", flush=True)
            catboost_params = {
                "iterations": 300,
                "depth": 6,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3,
            }
            catboost_params.update(load_best_params("catboost"))
            print(f"[EXPORT] catboost params={catboost_params}", flush=True)
            catboost_model = CatBoostClassifier(
                iterations=catboost_params["iterations"],
                depth=catboost_params["depth"],
                learning_rate=catboost_params["learning_rate"],
                l2_leaf_reg=catboost_params["l2_leaf_reg"],
                loss_function="MultiClass",
                eval_metric="TotalF1",
                verbose=False,
                random_seed=42,
                allow_writing_files=False,
            )
            catboost_model.fit(X, y)
            print("[EXPORT] FIT DONE catboost", flush=True)
            joblib.dump(catboost_model, ART / "joblib" / "catboost.joblib")
            catboost_model.save_model(str(ART / "catboost_model.cbm"))
            export_rows.append({
                "model": "catboost_native",
                "status": "exported",
                "note": "Native CatBoost artifact for audit; MT5 standard runtime should use ONNX unless a custom native bridge is added.",
            })
            try:
                print("[EXPORT] WRITE catboost.onnx", flush=True)
                catboost_model.save_model(
                    str(ART / "catboost.onnx"),
                    format="onnx",
                    export_parameters={
                        "onnx_domain": "ai.catboost",
                        "onnx_model_version": 1,
                        "onnx_doc_string": "macro news fx catboost",
                    },
                )
                export_rows.append({
                    "model": "catboost_onnx",
                    "status": "exported",
                    "note": "Run MT5 OnnxCreate/OnnxRun parity test before final Strategy Tester use.",
                })
            except Exception as exc:
                export_rows.append({"model": "catboost_onnx", "status": "onnx_export_failed", "note": str(exc)})
        except Exception as exc:
            export_rows.append({"model": "catboost", "status": "dependency_or_training_failed", "note": str(exc)})

    if compact_export:
        export_rows.append({
            "model": "ebm",
            "status": "skipped_in_en_rf_cb_mode",
            "note": "EBM is reserved for EBM_SVM/interpretable analysis.",
        })
    else:
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
            print("[EXPORT] START ebm", flush=True)
            ebm = ExplainableBoostingClassifier(random_state=42)
            ebm.fit(X, y)
            print("[EXPORT] FIT DONE ebm", flush=True)
            joblib.dump(ebm, ART / "joblib" / "ebm.joblib")
            export_rows.append({
                "model": "ebm",
                "status": "joblib_exported",
                "note": "EBM is retained for interpretability; build a score-table/EA interpreter before MT5 runtime testing.",
            })
        except Exception as exc:
            export_rows.append({"model": "ebm", "status": "dependency_or_training_failed", "note": str(exc)})

    (ART / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.DataFrame(export_rows).to_csv(ART / "onnx_export_status.csv", index=False)
    print("artifacts written to", ART)


if __name__ == "__main__":
    main()







