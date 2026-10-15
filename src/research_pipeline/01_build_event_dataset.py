from common import *


def main():
    print("[PROGRESS] START 01_build_event_dataset.py", flush=True)
    news = read_news()
    full_mapping = build_currency_pair_mapping()
    full_mapping.to_csv(ROOT / "currency_pair_mapping.csv", index=False)
    mapping = full_mapping[full_mapping["status"].eq("available")].copy()
    available_pairs = ", ".join(mapping["pair"].drop_duplicates().tolist())
    print(f"[DATA] available pairs: {available_pairs if available_pairs else '-'}", flush=True)

    frames = []
    coverage = []
    for pair, pair_map in mapping.groupby("pair", sort=False):
        print(f"[PAIR] START {pair}", flush=True)
        fn = pair_file(pair)
        p = DATA / fn
        countries = sorted(pair_map["CountryCode"].unique())
        if not p.exists():
            for _, row in pair_map.iterrows():
                coverage.append({
                    "CountryCode": row.CountryCode,
                    "currency": row.currency,
                    "pair": pair,
                    "file": fn,
                    "role": row.role,
                    "orientation": row.orientation,
                    "status": "missing",
                    "price_rows": 0,
                    "event_rows": 0,
                })
            continue

        px = read_prices(p).set_index("time")
        for h in [1, 2, 4, 24]:
            px[f"close_fwd_{h}h"] = px["Close"].shift(-h)

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
        frames.append(joined)

        counts = joined.groupby("CountryCode").size().to_dict()
        for _, row in pair_map.iterrows():
            coverage.append({
                "CountryCode": row.CountryCode,
                "currency": row.currency,
                "pair": pair,
                "file": fn,
                "role": row.role,
                "orientation": row.orientation,
                "status": "available",
                "price_rows": len(px),
                "event_rows": int(counts.get(row.CountryCode, 0)),
            })

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(ROOT / "event_level_dataset.csv", index=False)
    pd.DataFrame(coverage).to_csv(ROOT / "pair_country_coverage.csv", index=False)
    print("event rows", len(out))


if __name__ == "__main__":
    main()


