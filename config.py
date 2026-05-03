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

# ============ 新增：估值参数 ============
VALUATION_PERCENTILE_LOW = 25       # 低于此分位=低估
VALUATION_PERCENTILE_HIGH = 75      # 高于此分位=高估
PE_HISTORY_YEARS = 5                # PE分位计算回溯年数
VALUATION_CACHE_TTL = 4 * 3600      # 估值数据缓存4小时

# ============ 新增：财务趋势参数 ============
FINANCIAL_TREND_QUARTERS = 3        # 连续N期增长/下滑判定
FINANCIAL_CRASH_THRESHOLD = -0.50   # YoY跌幅超过此值=业绩大幅下滑
FINANCIAL_SURGE_THRESHOLD = 1.00    # YoY涨幅超过此值=需标记（可能非经常性）
FINANCIAL_CACHE_TTL = 24 * 3600     # 财报数据缓存24小时

# ============ 新增：风险参数 ============
RISK_FREE_RATE = 0.03               # 无风险利率（年化）
RISK_MD_LOW = 0.10                  # 最大回撤低风险阈值（<此值=低风险）
RISK_MD_HIGH = 0.25                 # 最大回撤高风险阈值（>此值=高风险）
RISK_VOL_LOW = 0.25                 # 波动率低风险阈值
RISK_VOL_HIGH = 0.45                # 波动率高风险阈值
RISK_BETA_LOW = 1.0                 # Beta低风险阈值
RISK_BETA_HIGH = 1.5                # Beta高风险阈值
BETA_REGRESSION_DAYS = 60           # Beta回归所需交易日数
BETA_MIN_DAYS = 40                  # Beta最少有效数据点数

# ============ 新增：资金流向参数 ============
FUND_FLOW_CACHE_TTL = 3600          # 资金流向缓存1小时

# ============ 新增：融资融券参数 ============
MARGIN_TREND_DAYS = 10              # 融资余额趋势计算天数
MARGIN_HIGH_LEVERAGE_RATIO = 0.05   # 融资金额/流通市值 > 此值=高杠杆
MARGIN_CACHE_TTL = 2 * 3600         # 融资融券缓存2小时

# ============ 新增：分红参数 ============
DIVIDEND_HIGH_YIELD = 0.02          # 高股息阈值（2%）
DIVIDEND_CONSECUTIVE_YEARS = 3      # 连续分红年数阈值
DIVIDEND_CACHE_TTL = 6 * 3600       # 分红/公告缓存6小时

# ============ 新增：综合评分权重 ============
SCORE_WEIGHT_TECHNICAL = 0.35
SCORE_WEIGHT_PREDICTION = 0.20
SCORE_WEIGHT_FUNDAMENTAL = 0.25
SCORE_WEIGHT_SENTIMENT = 0.20
SCORE_BULL_THRESHOLD = 0.15
SCORE_BEAR_THRESHOLD = -0.15

# ============ 新增：数据获取超时 ============
DATA_FETCH_TIMEOUT = 15             # 单个数据源请求超时（秒）
NETWORK_TIMEOUT = 25                # 单次网络请求超时（秒），cloud 环境需更长

# ============ 新增：akshare 字段映射表 ============
FINANCIAL_FIELD_MAP = {
    "营业总收入": ["营业总收入", "营业收入", "revenue", "total_revenue", "营业总收入(元)"],
    "归母净利润": ["归母净利润", "净利润", "parent_net_profit", "net_profit", "归属母公司股东的净利润", "归属母公司股东的净利润(元)"],
    "ROE": ["净资产收益率", "ROE", "加权净资产收益率", "roe_weighted", "净资产收益率(%)"],
    "毛利率": ["毛利率", "gross_margin", "gross_profit_margin", "销售毛利率", "销售毛利率(%)"],
    "净利率": ["净利率", "net_margin", "net_profit_margin", "销售净利率", "销售净利率(%)"],
    "资产负债率": ["资产负债率", "debt_ratio", "asset_liability_ratio", "资产负债率(%)"],
    "经营活动现金流净额": ["经营活动现金流净额", "经营活动产生的现金流量净额", "operating_cash_flow", "经营活动产生的现金流量净额(元)"],
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
