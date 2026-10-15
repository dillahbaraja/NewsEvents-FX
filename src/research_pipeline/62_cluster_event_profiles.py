"""Cluster event profiles for characterization, not as a standalone trading model."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from common import ARTIFACTS, progress_done, progress_start, progress_step

SCRIPT = Path(__file__).name
OUT = ARTIFACTS / "event_character_aware"
PROFILE = OUT / "profiles"
CLUSTERS = OUT / "clusters"
INPUT = PROFILE / "event_profiles_best_horizon_train_2010_2020.csv"

FEATURES = [
    "trade_count", "original_net_pips", "inverse_net_pips",
    "original_profit_factor", "inverse_profit_factor", "avg_pips_original",
    "avg_pips_inverse", "winrate_original", "winrate_inverse",
    "volatility_after_event", "max_adverse_excursion", "max_favorable_excursion",
    "positive_year_ratio_original", "positive_year_ratio_inverse",
    "dominant_year_share_original", "original_reliability_score",
    "inverse_reliability_score", "best_event_score",
]


def main():
    started = progress_start(SCRIPT)
    CLUSTERS.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(INPUT, low_memory=False)
    x = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x["original_profit_factor"] = x["original_profit_factor"].clip(0, 10)
    x["inverse_profit_factor"] = x["inverse_profit_factor"].clip(0, 10)
    n_clusters = min(6, max(2, int(np.sqrt(max(len(x), 2) / 2))))
    progress_step("CLUSTER_CONFIG", rows=len(x), features=len(FEATURES), clusters=n_clusters)
    scaled = StandardScaler().fit_transform(x)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = model.fit_predict(scaled)
    out = d.copy()
    out["event_cluster"] = labels
    out.to_csv(CLUSTERS / "event_clusters.csv", index=False)

    summary = out.groupby("event_cluster").agg(
        rows=("event_cluster", "size"),
        avg_trade_count=("trade_count", "mean"),
        avg_original_net_pips=("original_net_pips", "mean"),
        avg_inverse_net_pips=("inverse_net_pips", "mean"),
        avg_original_pf=("original_profit_factor", "mean"),
        avg_inverse_pf=("inverse_profit_factor", "mean"),
        avg_best_score=("best_event_score", "mean"),
        allow_count=("best_direction", lambda s: int((s == "ALLOW").sum())),
        inverse_count=("best_direction", lambda s: int((s == "INVERSE").sum())),
        block_count=("best_direction", lambda s: int((s == "BLOCK").sum())),
    ).reset_index()
    summary.to_csv(CLUSTERS / "event_cluster_summary.csv", index=False)
    (CLUSTERS / "event_cluster_metadata.json").write_text(json.dumps({
        "method": "KMeans",
        "note": "Cluster is used as event-character feature and explanation, not as proof of profitability.",
        "input": str(INPUT),
        "features": FEATURES,
        "n_clusters": int(n_clusters),
    }, indent=2), encoding="utf-8")
    progress_done(SCRIPT, started)


if __name__ == "__main__":
    main()
