from pathlib import Path
import os
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("NEWS_EVENTS_DATA", r"C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Dataset"))
NEWS = DATA / "NEWS_EVENTS.csv"
ARTIFACTS = DATA / "artifacts"
PAIR_FILTER = [p.strip().upper() for p in os.environ.get("FX_PAIRS", "").split(",") if p.strip()]
PROGRESS_HEARTBEAT_SECONDS = int(os.environ.get("FX_PROGRESS_HEARTBEAT_SECONDS", "60"))

TRAIN_END_YEAR = 2020
VALIDATION_END_YEAR = 2023
FINAL_TEST_START_YEAR = 2024
RETURN_SIGNIFICANCE_PERCENTILE = 0.60
TRADE_LABEL_ORDER = [-1, 0, 1]
TRADE_LABEL_NAME = {-1: "SELL", 0: "NO_TRADE", 1: "BUY"}

CURRENCY = {
    "US": "USD",
    "EU": "EUR",
    "GB": "GBP",
    "JP": "JPY",
    "AU": "AUD",
    "CA": "CAD",
    "NZ": "NZD",
    "CH": "CHF",
}

EXCLUDED_ASSETS = {
    "XAUUSD": "reserved_for_gold_macro_news_paper",
    "XTIUSD": "reserved_for_oil_macro_news_paper",
    "XTEUSD": "reserved_for_oil_macro_news_paper",
}

NUMERIC_FEATURES = ["surprise_fc", "surprise_prev", "surprise_pct_fc", "Importance", "Impact", "release_hour"]
CATEGORY_LEVELS = ["inflation", "labor", "growth", "rates", "sentiment", "trade", "housing", "commodities", "other"]
ORIENTATION_LEVELS = ["base", "quote"]

COUNTRY_PAIR_POLICY = {
    "US": {"primary": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"], "comparison": []},
    "EU": {"primary": ["EURUSD"], "comparison": ["EURJPY", "EURGBP", "EURAUD", "EURCAD", "EURNZD", "EURCHF"]},
    "GB": {"primary": ["GBPUSD"], "comparison": ["GBPJPY", "EURGBP", "GBPAUD", "GBPCAD", "GBPNZD", "GBPCHF"]},
    "JP": {"primary": ["USDJPY"], "comparison": ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "NZDJPY", "CHFJPY"]},
    "AU": {"primary": ["AUDUSD"], "comparison": ["AUDJPY", "EURAUD", "GBPAUD"]},
    "CA": {"primary": ["USDCAD"], "comparison": ["CADJPY", "EURCAD", "GBPCAD"]},
    "NZ": {"primary": ["NZDUSD"], "comparison": ["NZDJPY", "EURNZD", "GBPNZD"]},
    "CH": {"primary": ["USDCHF"], "comparison": ["EURCHF", "GBPCHF", "CHFJPY"]},
}


def pair_file(pair):
    return f"{pair}_H1_Data.csv"


def split_pair(pair):
    return pair[:3], pair[3:]


def pair_currency(pair, country):
    cur = CURRENCY.get(str(country).upper(), "")
    if not cur or pair in EXCLUDED_ASSETS:
        return "other"
    base, quote = split_pair(pair)
    if base == cur:
        return "base"
    if quote == cur:
        return "quote"
    return "other"


def domestic_return(raw_return, orientation):
    return raw_return if orientation == "base" else -raw_return


def pip_size(pair):
    return 0.01 if str(pair).upper().endswith("JPY") else 0.0001


def signal_from_trade_label(label):
    return TRADE_LABEL_NAME.get(int(label), "NO_TRADE")


def collapse_reason(buy_count, sell_count, no_trade_count, min_trades=20):
    total = buy_count + sell_count + no_trade_count
    trade_count = buy_count + sell_count
    if total <= 0:
        return "no_predictions"
    if trade_count < min_trades:
        return "too_few_trades"
    buy_ratio = buy_count / total
    sell_ratio = sell_count / total
    no_trade_ratio = no_trade_count / total
    if buy_ratio > 0.90:
        return "buy_ratio_gt_0.90"
    if sell_ratio > 0.90:
        return "sell_ratio_gt_0.90"
    if no_trade_ratio > 0.98:
        return "no_trade_ratio_gt_0.98"
    return ""


def profit_factor(values):
    s = pd.Series(values).dropna()
    gross_profit = s[s > 0].sum()
    gross_loss = -s[s < 0].sum()
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def research_split(event_year):
    y = pd.to_numeric(event_year, errors="coerce")
    return np.select(
        [y <= TRAIN_END_YEAR, (y > TRAIN_END_YEAR) & (y <= VALIDATION_END_YEAR), y >= FINAL_TEST_START_YEAR],
        ["train", "validation", "final_test"],
        default="out_of_scope",
    )


def build_currency_pair_mapping():
    rows = []
    seen = set()
    for country, policy in COUNTRY_PAIR_POLICY.items():
        currency = CURRENCY[country]
        for role in ["primary", "comparison"]:
            for pair in policy.get(role, []):
                if PAIR_FILTER and pair.upper() not in PAIR_FILTER:
                    continue
                key = (country, pair)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "CountryCode": country,
                    "currency": currency,
                    "pair": pair,
                    "file": pair_file(pair),
                    "role": role,
                    "orientation": pair_currency(pair, country),
                    "status": "available" if (DATA / pair_file(pair)).exists() else "missing",
                })
    return pd.DataFrame(rows)


def available_currency_pair_mapping():
    mapping = build_currency_pair_mapping()
    if mapping.empty:
        return mapping
    return mapping[mapping["status"].eq("available")].copy()


def read_news(path=NEWS):
    d = pd.read_csv(path, low_memory=False)
    d["event_time"] = pd.to_datetime(d["Datetime"], errors="coerce")
    for c in ["Actual", "Forecast", "Previous", "Importance", "Impact"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["event_time"]).sort_values("event_time")


def read_prices(path):
    d = pd.read_csv(path, usecols=lambda c: c in {"Time", "Open", "High", "Low", "Close", "Volume"})
    d["time"] = pd.to_datetime(d["Time"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["time", "Close"]).sort_values("time").drop_duplicates("time")


def event_category(name):
    s = str(name).lower()
    groups = {
        "inflation": ["cpi", "inflation", "pce", "ppi"],
        "labor": ["employment", "payroll", "unemployment", "jobless", "claimant", "wage"],
        "growth": ["gdp", "production", "retail sales", "industrial"],
        "rates": ["interest rate", "rate decision", "fomc", "ecb", "boe", "boj", "central bank"],
        "sentiment": ["pmi", "confidence", "sentiment"],
        "trade": ["trade balance", "exports", "imports"],
        "housing": ["housing", "home sale"],
        "commodities": ["inventories", "oil", "crude"],
    }
    for k, words in groups.items():
        if any(w in s for w in words):
            return k
    return "other"


def available_pairs():
    mapping = available_currency_pair_mapping()
    return mapping["pair"].drop_duplicates().tolist() if not mapping.empty else []


def progress(message, **fields):
    extras = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    print(f"[PROGRESS] {message}" + (f" | {extras}" if extras else ""), flush=True)


def progress_start(script_name):
    progress("START", script=script_name)
    return time.perf_counter()


def progress_done(script_name, started=None):
    elapsed = None if started is None else f"{time.perf_counter() - started:.1f}s"
    progress("DONE", script=script_name, elapsed=elapsed)


def progress_step(step, current=None, total=None, **fields):
    if current is not None and total is not None:
        fields = {"progress": f"{current}/{total}", **fields}
    progress(step, **fields)


def progress_heartbeat(label, started, force=False, state=None, **fields):
    now = time.perf_counter()
    if state is None:
        state = {}
    last = state.get(label, 0.0)
    if force or now - last >= PROGRESS_HEARTBEAT_SECONDS:
        state[label] = now
        progress("HEARTBEAT", task=label, elapsed=f"{now - started:.1f}s", **fields)
    return state


def feature_pairs():
    rows = []
    seen = set()
    for policy in COUNTRY_PAIR_POLICY.values():
        for role in ["primary", "comparison"]:
            for pair in policy.get(role, []):
                if pair in EXCLUDED_ASSETS or pair in seen:
                    continue
                seen.add(pair)
                if (DATA / pair_file(pair)).exists():
                    rows.append(pair)
    return rows


def model_feature_names():
    pairs = feature_pairs()
    countries = list(CURRENCY.keys())
    currencies = list(CURRENCY.values())
    features = list(NUMERIC_FEATURES)
    features += [f"pair_{p}" for p in pairs]
    features += [f"country_{c}" for c in countries]
    features += [f"currency_{c}" for c in currencies]
    features += [f"category_{c}" for c in CATEGORY_LEVELS]
    features += [f"orientation_{c}" for c in ORIENTATION_LEVELS]
    return features


def add_context_features(df):
    out = df.copy()
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for pair in feature_pairs():
        out[f"pair_{pair}"] = (out["pair"].astype(str) == pair).astype("int8")
    for country in CURRENCY.keys():
        out[f"country_{country}"] = (out["CountryCode"].astype(str) == country).astype("int8")
    for currency in CURRENCY.values():
        out[f"currency_{currency}"] = (out["currency"].astype(str) == currency).astype("int8")
    for category in CATEGORY_LEVELS:
        out[f"category_{category}"] = (out["category"].astype(str) == category).astype("int8")
    for orientation in ORIENTATION_LEVELS:
        out[f"orientation_{orientation}"] = (out["orientation"].astype(str) == orientation).astype("int8")
    return out
