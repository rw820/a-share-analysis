import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def ml_forecast(df: pd.DataFrame, forecast_days: int = 20) -> dict:
    df_feat = _build_features(df)
    if df_feat.empty or len(df_feat) < 100:
        return {"success": False, "message": "数据不足（至少需要100个交易日）"}

    feature_cols = [c for c in df_feat.columns if c.startswith("feat_")]
    X = df_feat[feature_cols].values
    y = df_feat["target"].values

    # Split: use last 100 rows as test, rest as train
    train_size = max(len(X) - 100, int(len(X) * 0.8))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    score = model.score(X_test_scaled, y_test)

    # Predict future: iteratively use last known features
    last_row = df_feat[feature_cols].iloc[-1].values.reshape(1, -1)
    predictions = []
    for _ in range(forecast_days):
        pred = model.predict(scaler.transform(last_row))[0]
        predictions.append(pred)
        # Simple update: shift features (rough approximation)
        last_row = np.roll(last_row, -1)
        last_row[0, -1] = pred

    last_close = df["close"].iloc[-1]
    pred_prices = [last_close * (1 + p) for p in predictions]

    # Direction prediction
    avg_pred = np.mean(predictions[:5])
    direction = "看涨" if avg_pred > 0 else "看跌"
    confidence = min(abs(avg_pred) * 100, 100)

    return {
        "success": True,
        "predictions": pred_prices,
        "direction": direction,
        "confidence": confidence,
        "r2_score": score,
        "5day_return": f"{predictions[4]:.2%}" if len(predictions) >= 5 else "N/A",
        "10day_return": f"{predictions[9]:.2%}" if len(predictions) >= 10 else "N/A",
        "20day_return": f"{predictions[-1]:.2%}",
    }


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df) < 60:
        return pd.DataFrame()

    # Returns
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_10d"] = df["close"].pct_change(10)
    df["ret_20d"] = df["close"].pct_change(20)

    # Volume ratio
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Price position relative to MA
    if "MA5" in df.columns:
        df["price_ma5_ratio"] = df["close"] / df["MA5"] - 1
    if "MA20" in df.columns:
        df["price_ma20_ratio"] = df["close"] / df["MA20"] - 1

    # RSI value
    if "RSI" in df.columns:
        df["rsi_norm"] = df["RSI"] / 100

    # MACD histogram
    if "MACD_hist" in df.columns:
        df["macd_hist_norm"] = df["MACD_hist"] / df["close"]

    # BB position
    if "BB_upper" in df.columns:
        bb_width = df["BB_upper"] - df["BB_lower"]
        df["bb_position"] = (df["close"] - df["BB_lower"]) / bb_width.replace(0, np.nan)

    # Target: future N-day return (shifted back)
    df["target"] = df["close"].pct_change(5).shift(-5)

    feature_cols = [c for c in df.columns if c in [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "vol_ratio", "price_ma5_ratio", "price_ma20_ratio",
        "rsi_norm", "macd_hist_norm", "bb_position",
    ]]
    df_features = df[feature_cols].copy()
    df_features.columns = [f"feat_{c}" for c in df_features.columns]
    df_features["target"] = df["target"]

    df_features = df_features.dropna()
    return df_features
