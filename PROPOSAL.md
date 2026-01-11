# Project Proposal: Forecasting Cryptocurrency Volatility using Econometric and Deep Learning Models

## Research Question: 
Can deep learning models (LSTM) provide meaningful improvements over traditional econometric models (GARCH) and simple persistence-based benchmarks when forecasting short-term realized volatility of major cryptocurrencies such as Bitcoin and Ethereum?

## Motivation
Cryptocurrencies are characterized by high and unpredictable price fluctuations. Accurate volatility forecasting is crucial for risk management and financial decision-making. This project investigates whether the non-linear patterns captured by recurrent neural networks offer practical improvements over classical econometric models and simple benchmark approaches in short-term volatility forecasting.

## Data & Methodology
- **Data Source**: Daily historical price data for BTC/USD and ETH/USD collected using the yfinance Python library.
- **Timeframe**: Data period from 2022 to 2025 to capture various market regimes (bull and bear markets).
- **Target**: Realized short-term volatility, computed as the standard deviation of log returns over a rolling window.
- **Models**:
    - **Baseline**: Naive persistence-based benchmark and a standard GARCH(1,1) model to capture volatility clustering.
    - **Advanced**: LSTM (Long Short-Term Memory) network to model non-linear temporal dependencies.
- **Validation**: Walk-forward validation using an initial training period followed by consecutive non-overlapping test windows, ensuring realistic out-of-sample evaluation and preventing data leakage.
- **Evaluation**: Performance comparison using Root Mean Square Error (RMSE) and Mean Absolute Error (MAE).

## Expected Outcomes
The goal is to provide a comparative analysis of model performances and to build a clean, modular, and reproducible Python codebase for financial time-series forecasting.
