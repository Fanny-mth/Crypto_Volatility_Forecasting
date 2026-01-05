from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def garch_one_step_forecast_sigma(returns: pd.Series) -> float:
    r = returns.dropna()

    if len(r) < 200:
        return float("nan")

    r_scaled = 100.0 * r

    model = arch_model(r_scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal",)

    res = model.fit(disp="off")

    forecast = res.forecast(horizon=1, reindex=False)
    variance_next = forecast.variance.iloc[-1, 0]

    sigma_next = np.sqrt(variance_next) / 100.0
    return float(sigma_next)

def garch_forecast_sigma_path(returns: pd.Series, horizon: int) -> np.ndarray:
    r = returns.dropna()
    if len(r) < 200:
        return np.full(horizon, np.nan, dtype=float)

    r_scaled = 100.0 * r

    model = arch_model(r_scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal",)
    res = model.fit(disp="off")

    forecast = res.forecast(horizon=horizon, reindex=False)
    var_path = forecast.variance.values[-1]

    sigma_path = np.sqrt(var_path) / 100.0
    return sigma_path.astype(float)

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VolatilityLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0,)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        y = self.fc(last)
        return y.squeeze(-1)


@dataclass
class LSTMConfig:
    lookback: int = 30
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 30
    patience: int = 5
    seed: int = 42


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for t in range(lookback, len(X)):
        Xs.append(X[t - lookback:t])
        ys.append(y[t])
    return np.asarray(Xs, dtype=float), np.asarray(ys, dtype=float)


def train_lstm_predict(X_train_seq: np.ndarray, y_train_seq: np.ndarray, X_test_seq: np.ndarray, cfg: LSTMConfig,) -> np.ndarray:
    set_seeds(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(X_train_seq)
    if n < 50:
        return np.full(len(X_test_seq), np.nan, dtype=float)

    val_size = max(int(0.1 * n), 1)
    X_tr, y_tr = X_train_seq[:-val_size], y_train_seq[:-val_size]
    X_val, y_val = X_train_seq[-val_size:], y_train_seq[-val_size:]

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=cfg.batch_size, shuffle=False,)

    model = VolatilityLSTM(n_features=X_train_seq.shape[-1], hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, dropout=cfg.dropout,).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    X_te_t = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_te_t).detach().cpu().numpy()

    return preds.astype(float)
