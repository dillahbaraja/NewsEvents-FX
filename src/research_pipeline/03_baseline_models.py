from common import *
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def main():
    print("[PROGRESS] START 03_baseline_models.py", flush=True)
    d = pd.read_csv(
        ROOT / "modeling_dataset_multi_currency.csv",
        usecols=["event_year", "trade_label_1h", "surprise_fc", "surprise_prev"],
    )
    d = d.dropna(subset=["trade_label_1h"])
    d["split"] = research_split(d.event_year)
    d = d[d["split"].isin(["train", "validation", "final_test"])]
    rows = []
    for name, col in [("forecast_surprise", "surprise_fc"), ("previous_surprise", "surprise_prev")]:
        print(f"[BASELINE] {name}", flush=True)
        q = d[col].notna()
        for sp, g in d[q].groupby("split"):
            y = g.trade_label_1h.astype(int)
            p = np.select([g[col] > 0, g[col] < 0], [1, -1], default=0)
            rows.append({
                "model": name,
                "target": "trade_label_1h",
                "split": sp,
                "n": len(g),
                "accuracy": accuracy_score(y, p),
                "balanced_accuracy": balanced_accuracy_score(y, p),
                "macro_f1": f1_score(y, p, average="macro", zero_division=0),
            })
    pd.DataFrame(rows).to_csv(ROOT / "baseline_results.csv", index=False)


if __name__ == "__main__":
    main()


