from datetime import datetime, timedelta

# 默认分析参数
DEFAULT_LOOKBACK_DAYS = 365
MA_PERIODS = [5, 10, 20, 60, 120]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

FORECAST_DAYS_DEFAULT = 20
FORECAST_DAYS_MAX = 60
FORECAST_DAYS_MIN = 5

CACHE_TTL_SECONDS = 3600

# K线周期映射
FREQUENCY_MAP = {
    "日线": "d",
    "周线": "w",
    "月线": "m",
}

# 交易所前缀映射
def get_exchange_prefix(code: str) -> str:
    code = code.strip()
    if code.startswith("6"):
        return "sh"
    elif code.startswith("0") or code.startswith("3"):
        return "sz"
    elif code.startswith("68"):
        return "sh"
    else:
        return "sz"

# 默认日期范围
def get_default_date_range():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
