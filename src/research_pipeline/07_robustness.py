from common import *


def main():
    print("[PROGRESS] START 07_robustness.py", flush=True)
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv")
    d["direction"] = (d.domestic_return_1h > 0).astype(int)
    out = d.groupby(["event_year", "CountryCode", "currency", "pair", "role", "orientation"]).agg(
        n=("direction", "size"),
        mean_domestic_return=("domestic_return_1h", "mean"),
        positive_rate=("direction", "mean"),
    ).reset_index()
    out.to_csv(ROOT / "currency_robustness_results.csv", index=False)
    out.to_csv(ROOT / "yearly_robustness_results.csv", index=False)


if __name__ == "__main__":
    main()

