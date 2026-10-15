from common import *


def main():
    print("[PROGRESS] START 08_make_report_tables.py", flush=True)
    files = [
        "currency_pair_mapping.csv",
        "pair_country_coverage.csv",
        "baseline_results.csv",
        "model_comparison_results.csv",
        "model_comparison_results_ebm_svm.csv",
        "trade_label_distribution.csv",
        "return_significance_thresholds.csv",
        "model_pair_thresholds.csv",
        "model_pair_bias_report.csv",
        "event_effectiveness_fx.csv",
        "currency_robustness_results.csv",
    ]
    out = ["# Research Outputs", "", "FX-only universe. XAUUSD and XTIUSD/XTEUSD are excluded from this paper.", ""]
    for f in files:
        p = ROOT / f
        if p.exists():
            out += ["## " + f, "", pd.read_csv(p).head(20).to_string(index=False), ""]
    (ROOT / "research_report_tables.md").write_text(chr(10).join(out), encoding="utf-8")


if __name__ == "__main__":
    main()

