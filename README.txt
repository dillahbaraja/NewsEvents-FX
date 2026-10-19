NewsEvents-FX

Repository for the Paper 1 macroeconomic-news FX experiments.

Contents
- src/research_pipeline: Python training, feature construction, evaluation, and reporting scripts.
- src/mt5: MQL5 source files retained for runtime/reference testing.
- scripts: batch and shell launchers.
- data/news: NEWS_EVENTS.csv used for historical event modeling.
- data/ohlcv: H1 OHLCV data for the nine FX pairs.
- data/datatest: out-of-sample news and H1 OHLCV data.
- models: selected ONNX/CBM artifacts and metadata.
- results: Backtrader, VectorBT, benchmark, metrics, and supplementary CSV outputs.
- figures: generated paper figures.
- data/DATA_DICTIONARY.csv: field definitions and units for the released data files.

Data split
- Training: 2010-2020.
- Validation/model selection: 2021-2023.
- OOS development: 2024-2025.
- Final audit: 2026.

The manuscript, editorial notes, and other Markdown documentation remain on the local computer and are intentionally excluded from this repository. The repository contains reproducibility-oriented source, data, model metadata, selected model artifacts, result tables, and figures. Large binary and prediction files use Git LFS.

Supplementary outputs
- results/supplementary: model selection, best/worst cases, failure taxonomy, transaction metrics, and event-response profiles.
- results/backtrader: sequential event-driven evaluation outputs.
- results/vectorbt: vectorized replication and monthly cumulative outputs.
- results/metrics: pair-level and model-level comparison tables.
- figures: paper figures and diagnostic plots, including event-character and formal-inference diagnostics.

Reproduction order
1. Inspect data/DATA_DICTIONARY.csv and the temporal split above.
2. Run the event construction and feature scripts in src/research_pipeline.
3. Train or load the model artifacts using the launchers in scripts.
4. Run Backtrader and VectorBT evaluations.
5. Regenerate metrics and figures from the resulting CSV files.

The released data and artifacts are for research replication. Results should be interpreted under the stated cost assumptions and temporal protocol; they are not a guarantee of live trading profitability.

Run instructions
Check the batch files in scripts and the Python source under src/research_pipeline. Update local data paths if the repository is moved to another computer.
