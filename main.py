from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import download_crypto_prices, prepare_dataset
from src.evaluation import compute_metrics


def ensure_results_dir() -> None:
    os.makedirs("results", exist_ok=True)


def naive_baseline_predict(df: pd.DataFrame) -> pd.Series:
    return df["target"].shift(1)


def run_for_ticker(ticker: str, start: str, end: str, vol_window: int, horizon: int,):
    prices = download_crypto_prices(ticker, start=start, end=end)
    df = prepare_dataset(prices, vol_window=vol_window, horizon=horizon)

    df["y_true"] = df["target"]

    y_pred = naive_baseline_predict(df)
    df["y_pred_naive"] = y_pred

    df = df[y_pred.notna()]

    metrics = compute_metrics(df["y_true"], df["y_pred_naive"])

    out = df.reset_index()[["Date", "y_true", "y_pred_naive"]]
    out["ticker"] = ticker

    return out, metrics


def plot_predictions(pred_df: pd.DataFrame, out_path: str) -> None:
    plt.figure()
    for ticker, g in pred_df.groupby("ticker"):
        plt.plot(g["Date"], g["y_true"], label=f"{ticker} true")
        plt.plot(g["Date"], g["y_pred_naive"], linestyle="--", label=f"{ticker} naive")

    plt.title("Realized Volatility: True vs Naive Baseline")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ensure_results_dir()

    tickers = ["BTC-USD", "ETH-USD"]
    start = "2022-01-01"
    end = "2025-01-01"
    vol_window = 7
    horizon = 1

    all_preds = []
    metrics_rows = []

    for t in tickers:
        preds, m = run_for_ticker(t, start, end, vol_window, horizon)
        all_preds.append(preds)
        metrics_rows.append({"ticker": t, **m})

    pred_df = pd.concat(all_preds, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)

    pred_df.to_csv("results/predictions.csv", index=False)
    metrics_df.to_csv("results/metrics.csv", index=False)
    plot_predictions(pred_df, "results/plot_predictions.png")

    print("Done")
    print(metrics_df)


if __name__ == "__main__":
    main()
