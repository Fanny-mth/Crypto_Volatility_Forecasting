# Project Proposal: Forecasting Cryptocurrency Volatility through Deep Learning

## Research Question: 
Can deep learning models (LSTM) outperform traditional econometric models (GARCH) in predicting the short-term realized volatility of major cryptocurrencies like Bitcoin and Ethereum?

## Motivation
Cryptocurrencies are characterized by high and unpredictable price fluctuations. Accurate volatility forecasting is crucial for risk management and financial decision-making. This project explores whether the non-linear patterns captured by Recurrent Neural Networks offer a significant advantage over classical financial methods.

## Data & Methodology
- **Data Source**: Public data from Binance or Yahoo Finance APIs.
- **Target**: Realized short-term volatility, computed as the standard deviation of log returns over a rolling window.
- **Models**:
    - **Baseline**: GARCH (Generalized Autoregressive Conditional Heteroskedasticity).
    - **Advanced**: LSTM (Long Short-Term Memory) network.
- **Validation**: Walk-forward validation with rolling windows to respect the temporal order of financial series.
- **Evaluation**: Standard metrics including Root Mean Square Error (RMSE) and Mean Absolute Error (MAE).

## Expected Outcomes
The goal is to provide a comparative analysis of model performances and to build a clean, modular, and reproducible Python codebase for financial time-series forecasting.
