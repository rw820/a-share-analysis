import pandas as pd
import numpy as np


def detect_trend(df: pd.DataFrame) -> dict:
    if df.empty or "MA5" not in df.columns:
        return {"trend": "数据不足", "description": ""}

    last = df.iloc[-1]
    ma5, ma10, ma20, ma60 = last.get("MA5"), last.get("MA10"), last.get("MA20"), last.get("MA60")

    if any(pd.isna(x) for x in [ma5, ma10, ma20, ma60]):
        return {"trend": "数据不足", "description": "均线数据不完整"}

    if ma5 > ma10 > ma20 > ma60:
        trend = "多头排列（看多）"
    elif ma5 < ma10 < ma20 < ma60:
        trend = "空头排列（看空）"
    else:
        trend = "震荡整理"

    price = last["close"]
    desc = f"MA5={ma5:.2f}, MA10={ma10:.2f}, MA20={ma20:.2f}, MA60={ma60:.2f}"

    return {"trend": trend, "description": desc, "price_vs_ma60": "线上" if price > ma60 else "线下"}


def find_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    if len(df) < window:
        return {"support": [], "resistance": []}

    recent = df.tail(window)
    last_close = df.iloc[-1]["close"]

    highs = recent["high"].values
    lows = recent["low"].values

    supports = []
    resistances = []

    for i in range(1, len(lows) - 1):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            level = round(lows[i], 2)
            if level < last_close:
                supports.append(level)

    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            level = round(highs[i], 2)
            if level > last_close:
                resistances.append(level)

    # Add round number levels near price
    base = 10 ** (len(str(int(last_close))) - 1)
    for mult in [1, 2, 5, 10, 20, 50]:
        level = base * mult / 10
        if level < last_close:
            supports.append(round(level, 2))
        elif level > last_close:
            resistances.append(round(level, 2))

    supports = sorted(set(supports), reverse=True)[:3]
    resistances = sorted(set(resistances))[:3]

    return {"support": supports, "resistance": resistances}


def get_key_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    last = df.iloc[-1]
    metrics = {
        "最新价": f"{last['close']:.2f}",
        "涨跌幅": f"{last.get('pctChg', 0):.2f}%",
        "换手率": f"{last.get('turn', 0):.2f}%" if pd.notna(last.get("turn")) else "N/A",
    }

    if len(df) >= 20:
        high_20 = df.tail(20)["high"].max()
        low_20 = df.tail(20)["low"].min()
        metrics["20日最高"] = f"{high_20:.2f}"
        metrics["20日最低"] = f"{low_20:.2f}"
        metrics["距20日高"] = f"{(last['close'] / high_20 - 1) * 100:.1f}%"
        metrics["距20日低"] = f"{(last['close'] / low_20 - 1) * 100:.1f}%"

    if "RSI" in df.columns and pd.notna(last.get("RSI")):
        rsi = last["RSI"]
        if rsi > 70:
            metrics["RSI状态"] = f"{rsi:.1f} (超买)"
        elif rsi < 30:
            metrics["RSI状态"] = f"{rsi:.1f} (超卖)"
        else:
            metrics["RSI状态"] = f"{rsi:.1f} (中性)"

    if "MACD_hist" in df.columns and pd.notna(last.get("MACD_hist")):
        hist = last["MACD_hist"]
        metrics["MACD柱"] = f"{hist:.4f} ({'红柱' if hist > 0 else '绿柱'})"

    return metrics
