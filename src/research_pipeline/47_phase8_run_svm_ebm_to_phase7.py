import importlib.util
import os
from pathlib import Path

from common import ARTIFACTS, ROOT


PHASE8_MODELS = ARTIFACTS / "phase8_svm_ebm_two_stage_models"
PHASE8_EVAL = ARTIFACTS / "phase8_svm_ebm_phase4_evaluation"
PHASE8_PAIR = ARTIFACTS / "phase8_svm_ebm_phase5_pair_selection"
PHASE8_STABILITY = ARTIFACTS / "phase8_svm_ebm_phase6_stability_gate"
PHASE8_SELECTED = ARTIFACTS / "phase8_svm_ebm_phase7_selected_policy"


def load_module(filename, name):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_training_outputs():
    required = [
        PHASE8_MODELS / "stage1_trade_filter_thresholds.csv",
        PHASE8_MODELS / "stage2_direction_thresholds.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 8 training outputs:\n" + "\n".join(missing))
    exact_ready = (PHASE8_MODELS / "joblib" / "svm_rbf_exact_trade_filter.joblib").exists() and (PHASE8_MODELS / "joblib" / "svm_rbf_exact_direction.joblib").exists()
    approx_ready = (PHASE8_MODELS / "joblib" / "svm_rbf_trade_filter.joblib").exists() and (PHASE8_MODELS / "joblib" / "svm_rbf_direction.joblib").exists()
    if not exact_ready and not approx_ready:
        raise FileNotFoundError("Missing SVM_RBF outputs. Expected exact svm_rbf_exact_*.joblib or approximate svm_rbf_*.joblib.")


def main():
    print("[PROGRESS] START 47_phase8_run_svm_ebm_to_phase7.py", flush=True)
    require_training_outputs()

    p42 = load_module("42_phase4_evaluate_two_stage_oos.py", "phase4_eval")
    families = {}
    if (PHASE8_MODELS / "joblib" / "svm_rbf_exact_trade_filter.joblib").exists() and (PHASE8_MODELS / "joblib" / "svm_rbf_exact_direction.joblib").exists():
        families["svm_rbf_exact_two_stage"] = {
            "stage1": "svm_rbf_exact_trade_filter.joblib",
            "stage1_name": "svm_rbf_exact_trade_filter",
            "stage2": "svm_rbf_exact_direction.joblib",
            "stage2_name": "svm_rbf_exact_direction",
        }
    if (PHASE8_MODELS / "joblib" / "svm_rbf_trade_filter.joblib").exists() and (PHASE8_MODELS / "joblib" / "svm_rbf_direction.joblib").exists():
        families["svm_rbf_approx_two_stage"] = {
            "stage1": "svm_rbf_trade_filter.joblib",
            "stage1_name": "svm_rbf_trade_filter",
            "stage2": "svm_rbf_direction.joblib",
            "stage2_name": "svm_rbf_direction",
        }
    if (PHASE8_MODELS / "joblib" / "ebm_trade_filter.joblib").exists() and (PHASE8_MODELS / "joblib" / "ebm_direction.joblib").exists():
        families["ebm_two_stage"] = {
            "stage1": "ebm_trade_filter.joblib",
            "stage1_name": "ebm_trade_filter",
            "stage2": "ebm_direction.joblib",
            "stage2_name": "ebm_direction",
        }
    selected = os.environ.get("PHASE8_EVAL_MODELS", "").strip()
    if selected:
        allowed = {name.strip() for name in selected.split(",") if name.strip()}
        families = {name: spec for name, spec in families.items() if name in allowed}
        if not families:
            raise ValueError(f"PHASE8_EVAL_MODELS selected no available families: {selected}")
    p42.MODEL_DIR = PHASE8_MODELS
    p42.JOBLIB_DIR = PHASE8_MODELS / "joblib"
    p42.OUT_DIR = PHASE8_EVAL
    p42.MODEL_FAMILIES = families
    print(f"[RUN] phase4 evaluation families={list(families)}", flush=True)
    p42.main()

    p43 = load_module("43_phase5_pair_specific_selection.py", "phase5_pair")
    p43.PHASE4_DIR = PHASE8_EVAL
    p43.OUT_DIR = PHASE8_PAIR
    p43.SUMMARY = PHASE8_EVAL / "phase4_two_stage_summary.csv"
    p43.ORDERS = PHASE8_EVAL / "phase4_two_stage_orders.csv"
    print("[RUN] phase5 pair selection", flush=True)
    p43.main()

    p44 = load_module("44_phase6_validation_stability_gate.py", "phase6_stability")
    p44.PHASE4_DIR = PHASE8_EVAL
    p44.OUT_DIR = PHASE8_STABILITY
    p44.SUMMARY = PHASE8_EVAL / "phase4_two_stage_summary.csv"
    p44.ORDERS = PHASE8_EVAL / "phase4_two_stage_orders.csv"
    print("[RUN] phase6 stability gate", flush=True)
    p44.main()

    p45 = load_module("45_phase7_package_selected_two_stage_policy.py", "phase7_package")
    p45.PHASE6_DIR = PHASE8_STABILITY
    p45.OUT_DIR = PHASE8_SELECTED
    p45.SELECTED_POLICY = "stability_2of3_pf110_avg020"
    print("[RUN] phase7 selected policy package", flush=True)
    p45.main()
    print("[PROGRESS] DONE 47_phase8_run_svm_ebm_to_phase7.py", flush=True)


if __name__ == "__main__":
    main()
