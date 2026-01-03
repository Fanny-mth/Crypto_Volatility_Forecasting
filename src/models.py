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
