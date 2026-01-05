from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler  

from src.data_loader import download_crypto_prices, prepare_dataset
from src.evaluation import compute_metrics
from src.models import garch_forecast_sigma_path

from src.models import LSTMConfig, make_sequences, train_lstm_predict


def ensure_results_dir() -> None:
    os.makedirs("results/plots", exist_ok=True)


def make_walk_forward_splits(n: int, initial_train_size: int, test_size: int, step_size: int):
    splits = []
    train_end = initial_train_size

    while True:
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n:
            break
        splits.append((train_end, test_start, test_end))
        train_end += step_size

    return splits


def naive_predict_on_block(df: pd.DataFrame, test_start: int, test_end: int):
    y_true_block = df["target"].iloc[test_start:test_end]
    y_pred_block = df["target"].shift(1).iloc[test_start:test_end]
    return y_true_block, y_pred_block


def run_for_ticker(ticker: str, start: str, end: str, vol_window: int, horizon: int):
    prices = download_crypto_prices(ticker, start=start, end=end)
    df = prepare_dataset(prices, vol_window=vol_window, horizon=horizon)

    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})

    df = df.dropna().reset_index(drop=True)

    feature_cols = ["rv_7", "log_return"]
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}. Available columns: {df.columns.tolist()}")

    n = len(df)

    initial_train_size = int(0.6 * n)
    test_size = 30
    step_size = 30

    splits = make_walk_forward_splits(n, initial_train_size, test_size, step_size)
    if len(splits) == 0:
        raise ValueError("Not enough data for walk-forward splits. Use more data or reduce test_size/step_size.")

    all_pred_rows = []
    fold_metrics = []

    for fold_id, (train_end, test_start, test_end) in enumerate(splits, start=1):
        # NAIVE
        y_true_block, y_pred_block = naive_predict_on_block(df, test_start, test_end)
        m_naive = compute_metrics(y_true_block, y_pred_block)

        # GARCH
        past_returns = df["log_return"].iloc[:test_start]
        horizon_block = test_end - test_start
        garch_path = garch_forecast_sigma_path(past_returns, horizon_block)
        y_pred_garch_block = pd.Series(garch_path, index=y_true_block.index)
        valid = y_pred_garch_block.notna()
        m_garch = compute_metrics(y_true_block[valid], y_pred_garch_block[valid])

        # LSTM (prep)
        cfg = LSTMConfig(lookback=30)

        train_df = df.iloc[:test_start].copy()
        full_df = df.iloc[:test_end].copy()

        X_train_raw = train_df[feature_cols].values
        y_train_raw = train_df["target"].values

        X_full_raw = full_df[feature_cols].values
        y_full_raw = full_df["target"].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_full = scaler.transform(X_full_raw)

        X_train_seq, y_train_seq = make_sequences(X_train, y_train_raw, cfg.lookback)
        X_full_seq, _ = make_sequences(X_full, y_full_raw, cfg.lookback)

        seq_offset = cfg.lookback
        X_test_seq = X_full_seq[test_start - seq_offset : test_end - seq_offset]

        lstm_preds = train_lstm_predict(X_train_seq, y_train_seq, X_test_seq, cfg)
        y_pred_lstm_block = pd.Series(lstm_preds, index=y_true_block.index)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            raise ValueError("LSTM sequences empty. Try smaller lookback or more data.")

        fold_metrics.append({"ticker": ticker, "fold": fold_id, "train_end_index": train_end, "test_start_index": test_start, "test_end_index": test_end, "rmse_naive": m_naive["rmse"], "mae_naive": m_naive["mae"], "rmse_garch": m_garch["rmse"], "mae_garch": m_garch["mae"],})

        block = df.iloc[test_start:test_end][["Date", "target"]].copy()
        block = block.rename(columns={"target": "y_true"})
        block["y_pred_naive"] = y_pred_block.values
        block["y_pred_garch"] = y_pred_garch_block.values
        block["y_pred_lstm"] = y_pred_lstm_block.values
        block["ticker"] = ticker
        block["fold"] = fold_id
        all_pred_rows.append(block)

    pred_df = pd.concat(all_pred_rows, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics)

    overall_naive = compute_metrics(pred_df["y_true"], pred_df["y_pred_naive"])
    valid_all = pred_df["y_pred_garch"].notna()
    overall_garch = compute_metrics(pred_df.loc[valid_all, "y_true"], pred_df.loc[valid_all, "y_pred_garch"])

    overall_df = pd.DataFrame([{"ticker": ticker, "rmse_naive": overall_naive["rmse"], "mae_naive": overall_naive["mae"], "rmse_garch": overall_garch["rmse"], "mae_garch": overall_garch["mae"],}])

    return pred_df, fold_metrics_df, overall_df


def plot_predictions(pred_df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pred_df = pred_df.copy()
    pred_df["Date"] = pd.to_datetime(pred_df["Date"])

    for ticker, g in pred_df.groupby("ticker"):
        g = g.sort_values("Date")

        plt.figure(figsize=(10, 4))
        plt.plot(g["Date"], g["y_true"], label="True")
        plt.plot(g["Date"], g["y_pred_naive"], linestyle="--", label="Naive")
        plt.plot(g["Date"], g["y_pred_garch"], linestyle=":", label="GARCH")
        plt.title(f"Realized Volatility Forecast - {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_{ticker}.png"), dpi=150)
        plt.close()


def main() -> None:
    ensure_results_dir()

    tickers = ["BTC-USD", "ETH-USD"]
    start = "2022-01-01"
    end = "2025-01-01"
    vol_window = 7
    horizon = 1

    all_preds = []
    all_fold_metrics = []
    all_overall_metrics = []

    for t in tickers:
        preds, fold_m, overall_m = run_for_ticker(t, start, end, vol_window, horizon)
        all_preds.append(preds)
        all_fold_metrics.append(fold_m)
        all_overall_metrics.append(overall_m)

    pred_df = pd.concat(all_preds, ignore_index=True)
    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    overall_metrics_df = pd.concat(all_overall_metrics, ignore_index=True)

    pred_df.to_csv("results/predictions_walkforward.csv", index=False)
    fold_metrics_df.to_csv("results/metrics_walkforward_folds.csv", index=False)
    overall_metrics_df.to_csv("results/metrics_walkforward_overall.csv", index=False)

    plot_predictions(pred_df, "results/plots")

    print("Done")
    print(overall_metrics_df)


if __name__ == "__main__":
    main()

