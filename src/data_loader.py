from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def download_crypto_prices(ticker: str, start: str = "2022-01-01", end: str = "2025-01-01",) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False,)

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker={ticker}. Check symbol/dates.")

    df = df.copy()
    df.index.name = "Date"
    return df


def prepare_dataset(prices: pd.DataFrame, vol_window: int = 7, horizon: int = 1, price_col: str = "Adj Close",) -> pd.DataFrame:
    df = prices.copy()

    df["log_price"] = np.log(df[price_col])
    df["log_return"] = df["log_price"].diff()

    vol_col = f"rv_{vol_window}"
    df[vol_col] = df["log_return"].rolling(window=vol_window).std()

    df["target"] = df[vol_col].shift(-horizon)

    df = df.dropna()
    return df
