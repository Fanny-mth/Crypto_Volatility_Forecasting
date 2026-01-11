# Crypto_Volatility_Forecasting

Forecasting short-term cryptocurrency volatility using naive benchmarks, GARCH models, and LSTM neural networks.

## Project Overview

This project compares different approaches for short-term cryptocurrency volatility forecasting.
It focuses on two major cryptocurrencies, Bitcoin and Ethereum, and evaluates whether more complex
models provide meaningful improvements over simple persistence-based benchmarks.

Three models are considered:
- A naive persistence-based benchmark
- A standard GARCH(1,1) model
- A Long Short-Term Memory (LSTM) neural network

Model performance is evaluated using a walk-forward validation framework and standard error metrics
(RMSE and MAE).

## Setup and Usage Instructions

### Requirements
- Python 3.10 or higher
- pip

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Fanny-mth/Crypto_Volatility_Forecasting.git
cd Crypto_Volatility_Forecasting
pip install -r requirements.txt
```

### Running the Project

From the root directory of the repository, run:

```bash
python main.py
```


