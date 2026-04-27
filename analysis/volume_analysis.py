import pandas as pd
import numpy as np


def analyze_volume(df: pd.DataFrame) -> dict:
    if df.empty or "volume" not in df.columns:
        return {}

    last_vol = df.iloc[-1]["volume"]
    result = {"最新成交量": f"{last_vol:,.0f}"}

    if len(df) >= 5:
        avg5 = df.tail(5)["volume"].mean()
        result["5日均量"] = f"{avg5:,.0f}"
        result["量比(5日)"] = f"{last_vol / avg5:.2f}" if avg5 > 0 else "N/A"

    if len(df) >= 10:
        avg10 = df.tail(10)["volume"].mean()
        result["10日均量"] = f"{avg10:,.0f}"

    if len(df) >= 20:
        avg20 = df.tail(20)["volume"].mean()
        result["20日均量"] = f"{avg20:,.0f}"

    # Volume trend
    if len(df) >= 10:
        vol_5 = df.tail(5)["volume"].mean()
        vol_10 = df.tail(10)["volume"].mean()
        if vol_5 > vol_10 * 1.3:
            result["量能趋势"] = "放量"
        elif vol_5 < vol_10 * 0.7:
            result["量能趋势"] = "缩量"
        else:
            result["量能趋势"] = "平稳"

    # Volume spike detection
    if len(df) >= 20:
        avg20 = df.tail(20)["volume"].mean()
        std20 = df.tail(20)["volume"].std()
        if std20 > 0 and last_vol > avg20 + 2 * std20:
            result["异常放量"] = True

    return result


def detect_volume_price_divergence(df: pd.DataFrame, window: int = 20) -> str:
    if len(df) < window:
        return "数据不足"

    recent = df.tail(window)

    price_trend = np.polyfit(range(len(recent)), recent["close"].values, 1)[0]
    vol_trend = np.polyfit(range(len(recent)), recent["volume"].values, 1)[0]

    if price_trend > 0 and vol_trend < 0:
        return "量价背离（价涨量缩，注意风险）"
    elif price_trend < 0 and vol_trend > 0:
        return "放量下跌（卖压较大）"
    elif price_trend > 0 and vol_trend > 0:
        return "量价齐升（健康上涨）"
    elif price_trend < 0 and vol_trend < 0:
        return "缩量下跌（跌势可能趋缓）"
    else:
        return "量价正常"
