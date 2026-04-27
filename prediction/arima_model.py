import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def arima_forecast(df: pd.DataFrame, forecast_days: int = 20) -> dict:
    prices = df["close"].dropna().values
    if len(prices) < 60:
        return _naive_forecast(df, forecast_days)

    best_aic = float("inf")
    best_order = (2, 1, 1)
    candidates = [(1,1,0), (2,1,0), (0,1,1), (1,1,1), (2,1,1), (3,1,0), (0,1,2), (3,1,1)]

    for p, d, q in candidates:
                try:
                    model = ARIMA(prices, order=(p, d, q), enforce_stationarity=False)
                    result = model.fit()
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    try:
        model = ARIMA(prices, order=best_order)
        result = model.fit()
        forecast = result.get_forecast(steps=forecast_days)
        pred = forecast.predicted_mean
        ci = forecast.conf_int(alpha=0.05)

        return {
            "forecast": pred,
            "lower": ci[:, 0],
            "upper": ci[:, 1],
            "order": best_order,
            "aic": best_aic,
            "success": True,
        }
    except Exception:
        return _naive_forecast(df, forecast_days)


def _naive_forecast(df: pd.DataFrame, forecast_days: int) -> dict:
    prices = df["close"].dropna().values
    last_price = prices[-1]

    if len(prices) >= 20:
        drift = (prices[-1] - prices[-20]) / 20
    else:
        drift = 0

    forecast = np.array([last_price + drift * (i + 1) for i in range(forecast_days)])
    std = prices[-min(30, len(prices)):].std()
    lower = forecast - 1.96 * std * np.sqrt(np.arange(1, forecast_days + 1))
    upper = forecast + 1.96 * std * np.sqrt(np.arange(1, forecast_days + 1))

    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "order": "naive",
        "aic": None,
        "success": False,
    }


def generate_forecast_dates(df: pd.DataFrame, forecast_days: int) -> pd.DatetimeIndex:
    last_date = df["date"].iloc[-1]
    # Skip weekends (rough approximation)
    dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    return dates
