"""
风险度量：最大回撤、年化波动率、Beta、夏普比率、风险等级
"""
import pandas as pd
import numpy as np
from config import (
    RISK_FREE_RATE, RISK_MD_LOW, RISK_MD_HIGH,
    RISK_VOL_LOW, RISK_VOL_HIGH,
    RISK_BETA_LOW, RISK_BETA_HIGH,
    BETA_REGRESSION_DAYS, BETA_MIN_DAYS,
)


def calc_max_drawdown(prices: np.ndarray, window_days: int | None = None) -> float:
    """计算最大回撤

    从区间内任意峰值到后续最低谷的最大跌幅百分比。

    Args:
        prices: 收盘价序列
        window_days: 限制窗口天数（None=全区间）
    """
    if len(prices) < 2:
        return 0.0

    if window_days is not None:
        prices = prices[-window_days:]

    peak = prices[0]
    max_dd = 0.0
    for p in prices[1:]:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def calc_annualized_volatility(prices: np.ndarray, window: int = 20) -> float:
    """计算年化波动率

    Args:
        prices: 收盘价序列
        window: 计算日收益率标准差的滚动窗口
    """
    if len(prices) < window + 1:
        window = len(prices) - 1
    if window < 2:
        return 0.0

    recent = prices[-window - 1:]
    returns = np.diff(recent) / recent[:-1]
    daily_std = np.std(returns)
    annual_vol = daily_std * np.sqrt(252)
    return float(annual_vol)


def calc_beta(stock_df: pd.DataFrame, index_df: pd.DataFrame) -> dict:
    """计算相对指数的Beta系数

    Args:
        stock_df: 个股K线，含 date, close
        index_df: 指数K线，含 date, close

    Returns:
        {"beta": float|None, "degraded": bool}
    """
    if stock_df.empty or index_df.empty:
        return {"beta": None, "degraded": True}

    stock_df = stock_df.copy()
    index_df = index_df.copy()

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    index_df["date"] = pd.to_datetime(index_df["date"])

    merged = pd.merge(stock_df[["date", "close"]], index_df[["date", "close"]],
                      on="date", suffixes=("_stock", "_index"))
    merged = merged.tail(BETA_REGRESSION_DAYS)

    if len(merged) < BETA_MIN_DAYS:
        return {"beta": None, "degraded": True}

    stock_ret = merged["close_stock"].pct_change().dropna().values
    index_ret = merged["close_index"].pct_change().dropna().values

    min_len = min(len(stock_ret), len(index_ret))
    if min_len < BETA_MIN_DAYS:
        return {"beta": None, "degraded": True}

    stock_ret = stock_ret[-min_len:]
    index_ret = index_ret[-min_len:]

    try:
        cov = np.cov(stock_ret, index_ret)
        beta = cov[0, 1] / cov[1, 1]
        return {"beta": round(float(beta), 4), "degraded": False}
    except (IndexError, ZeroDivisionError):
        return {"beta": None, "degraded": True}


def calc_sharpe(prices: np.ndarray, risk_free_rate: float = RISK_FREE_RATE) -> float | None:
    """计算年化夏普比率"""
    if len(prices) < 30:
        return None

    returns = np.diff(prices) / prices[:-1]
    daily_mean = np.mean(returns)
    daily_std = np.std(returns)

    if daily_std == 0:
        return None

    annual_return = daily_mean * 252
    annual_vol = daily_std * np.sqrt(252)

    if annual_vol == 0:
        return None

    return round(float((annual_return - risk_free_rate) / annual_vol), 4)


def calc_risk_level(md_60: float | None, vol: float | None,
                    beta: float | None) -> str:
    """综合风险等级：取各维度最劣（最高风险）等级"""
    levels = set()

    if md_60 is not None:
        if md_60 < RISK_MD_LOW:
            levels.add("低风险")
        elif md_60 < RISK_MD_HIGH:
            levels.add("中风险")
        else:
            levels.add("高风险")
    else:
        levels.add("中风险")  # 缺失时默认中风险

    if vol is not None:
        if vol < RISK_VOL_LOW:
            levels.add("低风险")
        elif vol < RISK_VOL_HIGH:
            levels.add("中风险")
        else:
            levels.add("高风险")

    if beta is not None:
        if beta < RISK_BETA_LOW:
            levels.add("低风险")
        elif beta < RISK_BETA_HIGH:
            levels.add("中风险")
        else:
            levels.add("高风险")

    if "高风险" in levels:
        return "高风险"
    if "中风险" in levels:
        return "中风险"
    return "低风险"


def get_risk_signal(df: pd.DataFrame, index_df: pd.DataFrame) -> dict:
    """生成风险等级信号（special类型，不参与方向评分）"""
    if df.empty:
        return {
            "name": "风险等级", "value": "不可用", "color": "amber",
            "detail": "K线数据不足", "signal_type": "special", "participate": False,
        }

    prices = df["close"].dropna().values

    md_60 = calc_max_drawdown(prices, window_days=60)
    md_120 = calc_max_drawdown(prices, window_days=120) if len(prices) >= 120 else None
    md_250 = calc_max_drawdown(prices, window_days=250) if len(prices) >= 250 else None

    vol = calc_annualized_volatility(prices)

    beta_info = calc_beta(df, index_df)
    beta = beta_info["beta"]

    sharpe = calc_sharpe(prices)

    risk_level = calc_risk_level(md_60, vol, beta)

    detail_parts = []
    if md_60 is not None:
        detail_parts.append(f"60日最大回撤: {md_60:.1%}")
    if md_120 is not None:
        detail_parts.append(f"120日最大回撤: {md_120:.1%}")
    if md_250 is not None:
        detail_parts.append(f"250日最大回撤: {md_250:.1%}")
    if vol is not None:
        detail_parts.append(f"年化波动率: {vol:.1%}")
    if beta is not None:
        detail_parts.append(f"Beta: {beta:.2f}")
    else:
        detail_parts.append("Beta: N/A")
    if sharpe is not None:
        detail_parts.append(f"夏普比率: {sharpe:.2f}")

    risk_colors = {"低风险": "red", "中风险": "amber", "高风险": "green"}

    return {
        "name": "风险等级",
        "value": risk_level,
        "color": risk_colors.get(risk_level, "amber"),
        "signal_type": "special",
        "detail": "\n".join(detail_parts),
        "participate": False,
        "risk_data": {
            "md_60": md_60, "md_120": md_120, "md_250": md_250,
            "vol": vol, "beta": beta, "sharpe": sharpe,
            "level": risk_level,
        },
    }
