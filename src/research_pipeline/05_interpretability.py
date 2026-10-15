from common import *


def main():
    print("[PROGRESS] START 05_interpretability.py", flush=True)
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv")
    features = model_feature_names()
    print(f"[DATA] interpretability rows={len(d)} features={len(features)}", flush=True)
    out = d[features].corrwith(d["domestic_return_1h"]).rename("correlation_with_domestic_return_1h").reset_index().rename(
        columns={"index": "feature"}
    )
    out["interpretation_role"] = "diagnostic_correlation_not_main_novelty"
    out.to_csv(ROOT / "feature_interpretation.csv", index=False)

    notes = [
        "# Interpretability Notes",
        "",
        "SHAP is an interpretation method, not a standalone predictive model in this study.",
        "Use SHAP mainly for CatBoost and Random Forest when the optional `shap` package is available.",
        "EBM is treated as the interpretable predictive model and can be discussed through its own feature contribution curves.",
        "The correlation CSV is kept only as a diagnostic or appendix artifact, not as the main Q1/Q2 novelty.",
    ]
    (ROOT / "interpretability_notes.md").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    main()

