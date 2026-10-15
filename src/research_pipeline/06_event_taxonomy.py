from common import *


def main():
    print("[PROGRESS] START 06_event_taxonomy.py", flush=True)
    d = pd.read_csv(ROOT / "modeling_dataset_multi_currency.csv")
    cols = ["domestic_return_1h", "surprise_fc"]
    d["split"] = research_split(d.event_year)
    d = d[d["split"].isin(["train", "validation"])].dropna(subset=cols)
    d["correct"] = np.sign(d.surprise_fc) == np.sign(d.domestic_return_1h)
    print("[TAXONOMY] grouping event/currency categories", flush=True)
    g = d.groupby(["CountryCode", "currency", "Name", "category"], dropna=False).agg(
        n=("domestic_return_1h", "size"),
        mean_domestic_return=("domestic_return_1h", "mean"),
        hit_rate=("correct", "mean"),
    ).reset_index()
    g["classification"] = np.select(
        [g.n < 20, g.hit_rate < 0.45, g.hit_rate > 0.55],
        ["insufficient", "blocked", "effective"],
        default="unstable",
    )
    g["inverse"] = (g.hit_rate < 0.45) & (g.n >= 20)
    g["selection_period"] = "train_2010_2020_plus_validation_2021_2023"
    g.to_csv(ROOT / "event_effectiveness_fx.csv", index=False)
    g[g.classification == "blocked"].to_csv(ROOT / "event_block_list_fx.csv", index=False)
    g[g.inverse].to_csv(ROOT / "event_inverse_list_fx.csv", index=False)
    g[["Name", "category", "CountryCode", "currency"]].drop_duplicates().to_csv(ROOT / "event_category_map.csv", index=False)

    # Backward-compatible copies for older report scripts and notebooks.
    g.to_csv(ROOT / "event_effectiveness.csv", index=False)
    g[g.classification == "blocked"].to_csv(ROOT / "event_block_list.csv", index=False)
    g[g.inverse].to_csv(ROOT / "event_inverse_list.csv", index=False)


if __name__ == "__main__":
    main()


