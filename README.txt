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
- results: Backtrader, VectorBT, metrics, and supplementary CSV outputs.
- figures: generated paper figures.

Data split
- Training: 2010-2020.
- Validation/model selection: 2021-2023.
- OOS development: 2024-2025.
- Final audit: 2026.

The manuscript and other Markdown documentation remain on the local computer and are intentionally excluded from this repository. Large binary and prediction files use Git LFS.

Run instructions
Check the batch files in scripts and the Python source under src/research_pipeline. Update local data paths if the repository is moved to another computer.
