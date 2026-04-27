import pandas as pd
import ta
from config import MA_PERIODS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BB_PERIOD, BB_STD


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = compute_ma(df)
    df = compute_macd(df)
    df = compute_rsi(df)
    df = compute_bollinger(df)
    return df


def compute_ma(df: pd.DataFrame, periods=None) -> pd.DataFrame:
    if periods is None:
        periods = MA_PERIODS
    for p in periods:
        df[f"MA{p}"] = ta.trend.sma_indicator(df["close"], window=p)
    return df


def compute_ema(df: pd.DataFrame, periods=None) -> pd.DataFrame:
    if periods is None:
        periods = [12, 26]
    for p in periods:
        df[f"EMA{p}"] = ta.trend.ema_indicator(df["close"], window=p)
    return df


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    macd = ta.trend.MACD(
        df["close"],
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGNAL,
    )
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    return df


def compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df["RSI"] = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    return df


def compute_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    bb = ta.volatility.BollingerBands(
        df["close"], window=BB_PERIOD, window_dev=BB_STD
    )
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_mid"] = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()
    return df
