import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from config import (
    FREQUENCY_MAP, get_default_date_range,
    FORECAST_DAYS_DEFAULT, FORECAST_DAYS_MIN, FORECAST_DAYS_MAX,
    SCORE_WEIGHT_TECHNICAL, SCORE_WEIGHT_PREDICTION,
    SCORE_WEIGHT_FUNDAMENTAL, SCORE_WEIGHT_SENTIMENT,
    SCORE_BULL_THRESHOLD, SCORE_BEAR_THRESHOLD,
)
from data.stock_search import search_stocks
from data.market_data import get_kline_data
from data.fundamental_data import get_valuation_data, get_pe_history, get_financial_data, get_index_kline, get_dividend_history, get_industry_info
from data.fund_flow_data import get_fund_flow, get_margin_data
from data.notice_data import get_recent_notices
from analysis.technical_indicators import compute_all_indicators
from analysis.trend_analysis import detect_trend, find_support_resistance, get_key_metrics
from analysis.volume_analysis import analyze_volume, detect_volume_price_divergence
from analysis.valuation import get_valuation_signal
from analysis.financial_health import get_financial_signal
from analysis.risk_metrics import get_risk_signal
from analysis.fund_flow_analysis import get_fund_flow_signal, get_margin_signal
from analysis.event_analysis import get_event_signal, get_dividend_signal
from visualization.charts import create_main_chart, create_forecast_chart
from visualization.fundamental_charts import (
    create_pe_history_chart, create_financial_chart, create_fund_flow_chart,
)
from prediction.arima_model import arima_forecast, generate_forecast_dates
from prediction.ml_model import ml_forecast

st.set_page_config(page_title="A股个股分析", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
[data-testid="stMetricValue"] { white-space: normal !important; font-size: 1rem !important; overflow: visible !important; text-overflow: unset !important; }
[data-testid="stMetricLabel"] { white-space: normal !important; font-size: 0.8rem !important; }
[data-testid="stMetricDelta"] { white-space: normal !important; font-size: 0.85rem !important; }

.signal-card {
    padding: 10px 14px; border-radius: 8px; margin-bottom: 4px; border-left: 4px solid; background: #fafafa;
}
.signal-card .label { font-size: 11px; color: #888; margin-bottom: 2px; }
.signal-card .sublabel { font-size: 10px; color: #aaa; }
.signal-card .value { font-size: 15px; font-weight: 600; }
.signal-bull { border-left-color: #ef4444; background: #fef2f2; }
.signal-bull .value { color: #dc2626; }
.signal-bear { border-left-color: #22c55e; background: #f0fdf4; }
.signal-bear .value { color: #16a34a; }
.signal-neutral { border-left-color: #f59e0b; background: #fffbeb; }
.signal-neutral .value { color: #d97706; }
.signal-special { border-left-color: #3b82f6; background: #eff6ff; }
.signal-special .value { color: #2563eb; }

.verdict {
    text-align: center; padding: 12px 20px; border-radius: 10px;
    font-size: 18px; font-weight: 700; margin: 10px 0;
}
.verdict-bull { background: linear-gradient(135deg, #fef2f2, #fee2e2); color: #dc2626; border: 2px solid #fecaca; }
.verdict-bear { background: linear-gradient(135deg, #f0fdf4, #dcfce7); color: #16a34a; border: 2px solid #bbf7d0; }
.verdict-neutral { background: linear-gradient(135deg, #fffbeb, #fef3c7); color: #d97706; border: 2px solid #fde68a; }

@media (max-width: 768px) {
    .block-container { padding-top: 0.5rem; padding-bottom: 1rem; padding-left: 0.8rem; padding-right: 0.8rem; }
    h1 { font-size: 1.3rem !important; }
    .verdict { font-size: 15px; padding: 10px 14px; }
    .signal-card .value { font-size: 14px; }
}
</style>
""", unsafe_allow_html=True)

# ==================== Search bar ====================
st.title("A股个股分析")

search_col1, search_col2 = st.columns([5, 1])
with search_col1:
    query = st.text_input("输入股票代码或名称", placeholder="如: 600519 或 贵州茅台", label_visibility="collapsed", key="search_input")
with search_col2:
    search_clicked = st.button("搜索", use_container_width=True)

# Sidebar
with st.sidebar:
    st.header("分析参数")
    st.caption("点击左上角 > 展开侧边栏")
    start_default, end_default = get_default_date_range()
    start_date = st.date_input("开始日期", pd.to_datetime(start_default))
    end_date = st.date_input("结束日期", pd.to_datetime(end_default))
    freq_label = st.selectbox("K线周期", list(FREQUENCY_MAP.keys()))
    freq = FREQUENCY_MAP[freq_label]
    forecast_days = st.slider("预测天数", FORECAST_DAYS_MIN, FORECAST_DAYS_MAX, FORECAST_DAYS_DEFAULT)

    with st.expander("指标说明"):
        st.markdown("""
**K线图**：红蜡烛=上涨，绿蜡烛=下跌，细线=当日最高/最低价

**均线（MA）**：跟踪股价趋势的辅助线
- 价格在均线上方=偏强，下方=偏弱
- MA5>MA10>MA20>MA60 = "多头排列"（看涨信号）

**MACD**：DIF上穿DEA=金叉（买入信号），下穿=死叉（卖出信号）

**RSI**：>70超买（可能过高），<30超卖（可能过低）

**布林带**：灰色区域是正常波动范围，触及上下轨可能反转

**资金流向**：红柱=主力净流入，绿柱=主力净流出

**风险等级**：蓝框卡片，仅提示风险不判断方向
""")

# ==================== Search logic ====================
selected_stock = None
if query or search_clicked:
    effective_query = query.strip() if query else ""
    if effective_query:
        with st.spinner("搜索中..."):
            results = search_stocks(effective_query)
        if results:
            if len(results) == 1:
                selected_stock = results[0]
            else:
                options = [f"{r['code']} - {r['name']}" for r in results]
                choice = st.selectbox("选择股票", options, key="stock_select")
                idx = options.index(choice)
                selected_stock = results[idx]
            st.success(f"已选择: {selected_stock['name']} ({selected_stock['code']})")
        else:
            st.warning("未找到匹配的股票，请检查输入")

if not selected_stock:
    st.info("在上方输入股票代码或名称，点击【搜索】开始分析")
    st.markdown("*本工具仅供学习研究，不构成任何投资建议。*")
    st.stop()

# ==================== Data Fetching ====================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kline(bs_code, start, end, freq):
    return get_kline_data(bs_code, str(start), str(end), freq)

@st.cache_data(ttl=4 * 3600, show_spinner=False)
def fetch_valuation(bs_code):
    return get_valuation_data(bs_code)

@st.cache_data(ttl=4 * 3600, show_spinner=False)
def fetch_pe_history(bs_code):
    return get_pe_history(bs_code)

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_financial(bs_code):
    return get_financial_data(bs_code)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fund_flow(bs_code):
    return get_fund_flow(bs_code)

@st.cache_data(ttl=2 * 3600, show_spinner=False)
def fetch_margin(bs_code):
    return get_margin_data(bs_code)

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_notices(bs_code, days=30):
    return get_recent_notices(bs_code, days)

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_dividends(bs_code):
    return get_dividend_history(bs_code)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_index_kline(start, end, freq="d"):
    return get_index_kline("sz.399300", str(start), str(end), freq)

bs_code = selected_stock["bs_code"]

with st.status("正在分析中，网络请求较多请耐心等待...", expanded=False) as status:
    st.write("⏳ 获取K线数据...")
    df = fetch_kline(bs_code, start_date, end_date, freq)
    if df.empty:
        status.update(label="数据获取失败", state="error")
        st.error("无法获取该股票数据，请检查股票代码或日期范围")
        st.stop()

    st.write("⏳ 获取估值数据...")
    valuation_data = fetch_valuation(bs_code)
    st.write("⏳ 获取PE历史...")
    pe_history_data = fetch_pe_history(bs_code)
    st.write("⏳ 获取财务数据...")
    financial_data = fetch_financial(bs_code)
    st.write("⏳ 获取资金流向...")
    fund_flow_data = fetch_fund_flow(bs_code)
    st.write("⏳ 获取融资融券...")
    margin_data = fetch_margin(bs_code)
    st.write("⏳ 获取近期公告...")
    notice_data = fetch_notices(bs_code, 30)
    st.write("⏳ 获取分红记录...")
    dividend_data = fetch_dividends(bs_code)
    st.write("⏳ 获取指数数据...")
    index_df = fetch_index_kline(start_date, end_date, freq)
    st.write("⏳ 计算技术指标...")
    df = compute_all_indicators(df)
    trend_info = detect_trend(df)
    sr = find_support_resistance(df)
    vol_info = analyze_volume(df)
    divergence = detect_volume_price_divergence(df)
    metrics = get_key_metrics(df)

    st.write("⏳ 生成预测...")
    arima_result = arima_forecast(df, forecast_days)
    ml_result = ml_forecast(df, forecast_days)

    st.write("⏳ 评估基本面...")
    pe_history = pe_history_data.get("data", {}).get("pe_values") if pe_history_data.get("success") else None
    valuation_signal = get_valuation_signal(valuation_data, pe_history)
    financial_signal = get_financial_signal(financial_data)

    st.write("⏳ 分析资金面...")
    fund_flow_signal = get_fund_flow_signal(fund_flow_data)
    float_mv = valuation_data.get("data", {}).get("float_mv") if valuation_data.get("success") else None
    margin_signal = get_margin_signal(margin_data, float_mv)

    st.write("⏳ 汇总消息面...")
    event_signal = get_event_signal(notice_data)
    dividend_info = get_dividend_signal(dividend_data)

    st.write("⏳ 评估风险...")
    risk_signal = get_risk_signal(df, index_df)

    status.update(label="分析完成！", state="complete", expanded=False)

# ==================== Signals ====================
last_close = df["close"].iloc[-1]
last_pct = df["pctChg"].iloc[-1] if "pctChg" in df.columns else 0

SIGNAL_META = {
    "趋势": {
        "cn": "趋势判断", "category": "technical",
        "explain": "根据MA5/MA10/MA20/MA60四条均线的排列关系判断。\n"
                   "多头排列(MA5>MA10>MA20>MA60)=短期买盘积极，看涨；\n"
                   "空头排列(反过来)=卖压大，看空；交织=震荡，方向不明。",
    },
    "RSI": {
        "cn": "RSI（相对强弱指标）", "category": "technical",
        "explain": "相对强弱指标，衡量近期涨跌力度。\n"
                   "RSI>70说明近期涨幅过大，可能超买（要回调）；\n"
                   "RSI<30说明近期跌幅过大，可能超卖（要反弹）；\n"
                   "30-70为正常区间。",
    },
    "MACD": {
        "cn": "MACD（指数平滑异同均线）", "category": "technical",
        "explain": "MACD柱状图反映多空力量对比。\n"
                   "红柱(>0)说明多头力量占优；绿柱(<0)说明空头力量占优。\n"
                   "DIF线上穿DEA线叫'金叉'，是买入信号。",
    },
    "量价": {
        "cn": "量价关系", "category": "technical",
        "explain": "成交量与价格走势的关系。\n"
                   "量价齐升=健康上涨；价涨量缩=上涨动力不足，注意风险；\n"
                   "放量下跌=恐慌抛售，卖压大。",
    },
    "ML预测": {
        "cn": "ML机器学习预测", "category": "prediction",
        "explain": "使用梯度提升机器学习模型，综合10+个技术特征，\n"
                   "在近500个交易日的历史数据上训练，预测未来涨跌方向和幅度。\n置信度>60%有参考价值。",
    },
    "ARIMA": {
        "cn": "ARIMA（时间序列预测）", "category": "prediction",
        "explain": "ARIMA是经典时间序列预测方法，根据历史价格的自相关规律推算未来走势。\n"
                   "自动选择最优参数组合，提供预测价格和95%置信区间。",
    },
    "估值水平": {
        "cn": "估值水平（PE/PB分位）", "category": "fundamental",
        "explain": "基于PE-TTM在近3-5年的历史分位数判断。\n"
                   "分位<25%=低估(看多)，25%-75%=合理(中性)，>75%=高估(看空)。\n"
                   "亏损企业（PE为负）不参与判断，以蓝色特殊卡片展示。",
    },
    "盈利增长": {
        "cn": "盈利增长（财报趋势）", "category": "fundamental",
        "explain": "基于近几期财报的营收和净利润同比增长率判断。\n"
                   "连续增长且YoY>0=增长(看多)；连续下滑且YoY<0=下滑(看空)。\n"
                   "YoY<-50%=业绩大幅下滑(警报)，YoY>100%=可能非经常性损益。",
    },
    "资金流向": {
        "cn": "资金流向（主力动向）", "category": "sentiment",
        "explain": "统计近5日/20日主力资金（超大单+大单）累计净流入。\n"
                   "持续净流入=主力看多；持续净流出=主力看空；\n"
                   "5日和20日方向不一致=分歧信号。",
    },
    "融资融券": {
        "cn": "融资融券（杠杆资金）", "category": "sentiment",
        "explain": "融资余额变化反映杠杆资金的做多意愿。\n"
                   "融资余额持续增加=杠杆资金看多；融资余额持续减少=看空。\n"
                   "融资/流通市值>5%=高杠杆风险标记。",
    },
    "事件提醒": {
        "cn": "事件提醒（公告分析）", "category": "sentiment",
        "explain": "分析近30天公告内容：业绩预告方向、股东增减持、重大合同/诉讼。\n"
                   "综合评判生成偏多/偏空/中性信号。",
    },
    "风险等级": {
        "cn": "风险等级（波动/回撤/Beta）", "category": "risk",
        "explain": "基于最大回撤、年化波动率、Beta系数综合评估风险。\n"
                   "蓝色特殊卡片，**不判断涨跌方向**，仅提醒风险程度。\n"
                   "高风险≠看空，只是波动较大，需注意仓位管理。",
    },
}


def classify(color):
    return "bull" if color == "red" else ("bear" if color == "green" else ("special" if color == "blue" else "neutral"))


signals = []

# Trend
if "多" in trend_info["trend"]:
    signals.append({"name": "趋势", "value": "看多", "color": "red", "detail": trend_info.get("description", ""),
                    "signal_type": "bull", "participate": True, "category": "technical"})
elif "空" in trend_info["trend"]:
    signals.append({"name": "趋势", "value": "看空", "color": "green", "detail": trend_info.get("description", ""),
                    "signal_type": "bear", "participate": True, "category": "technical"})
else:
    signals.append({"name": "趋势", "value": "震荡", "color": "amber", "detail": trend_info.get("description", ""),
                    "signal_type": "neutral", "participate": True, "category": "technical"})

# RSI
if "RSI" in df.columns:
    rsi_val = df["RSI"].iloc[-1]
    if pd.notna(rsi_val):
        if rsi_val > 70:
            signals.append({"name": "RSI", "value": f"超买 {rsi_val:.0f}", "color": "green",
                            "detail": f"RSI={rsi_val:.1f}，超过70警戒线", "signal_type": "bear",
                            "participate": True, "category": "technical"})
        elif rsi_val < 30:
            signals.append({"name": "RSI", "value": f"超卖 {rsi_val:.0f}", "color": "red",
                            "detail": f"RSI={rsi_val:.1f}，低于30警戒线", "signal_type": "bull",
                            "participate": True, "category": "technical"})
        else:
            signals.append({"name": "RSI", "value": f"中性 {rsi_val:.0f}", "color": "amber",
                            "detail": f"RSI={rsi_val:.1f}，处于30-70正常区间", "signal_type": "neutral",
                            "participate": True, "category": "technical"})

# MACD
if "MACD_hist" in df.columns:
    hist = df["MACD_hist"].iloc[-1]
    if pd.notna(hist):
        if hist > 0:
            signals.append({"name": "MACD", "value": "多头", "color": "red",
                            "detail": f"MACD柱={hist:.4f}，多头力量占优", "signal_type": "bull",
                            "participate": True, "category": "technical"})
        else:
            signals.append({"name": "MACD", "value": "空头", "color": "green",
                            "detail": f"MACD柱={hist:.4f}，空头力量占优", "signal_type": "bear",
                            "participate": True, "category": "technical"})

# Volume-price
if "量价齐升" in divergence:
    signals.append({"name": "量价", "value": "齐升", "color": "red",
                    "detail": "量价齐升，健康上涨形态", "signal_type": "bull",
                    "participate": True, "category": "technical"})
elif "价涨量缩" in divergence or "背离" in divergence:
    signals.append({"name": "量价", "value": "注意", "color": "amber",
                    "detail": f"当前: {divergence}", "signal_type": "neutral",
                    "participate": True, "category": "technical"})
elif "放量下跌" in divergence:
    signals.append({"name": "量价", "value": "卖压", "color": "green",
                    "detail": "放量下跌，抛售压力大", "signal_type": "bear",
                    "participate": True, "category": "technical"})
else:
    signals.append({"name": "量价", "value": "正常", "color": "amber",
                    "detail": f"当前: {divergence}", "signal_type": "neutral",
                    "participate": True, "category": "technical"})

# ML
if ml_result.get("success"):
    ml_dir = ml_result["direction"]
    ml_conf = ml_result["confidence"]
    ml_color = "red" if ml_dir == "看涨" else "green"
    ml_signal_type = "bull" if ml_dir == "看涨" else "bear"
    signals.append({"name": "ML预测", "value": f"{ml_dir} {ml_conf:.0f}%", "color": ml_color,
                    "detail": f"ML预测: {ml_dir}\n置信度: {ml_conf:.1f}%\n"
                              f"5日: {ml_result.get('5day_return','N/A')}, "
                              f"10日: {ml_result.get('10day_return','N/A')}, "
                              f"{forecast_days}日: {ml_result.get('20day_return','N/A')}",
                    "signal_type": ml_signal_type, "participate": True, "category": "prediction"})

# ARIMA
if arima_result.get("forecast") is not None:
    pred_last = arima_result["forecast"][-1]
    arima_pct = (pred_last - last_close) / last_close * 100
    arima_color = "red" if arima_pct > 0 else "green"
    arima_signal_type = "bull" if arima_pct > 0 else "bear"
    signals.append({"name": "ARIMA", "value": f"{'看涨' if arima_pct > 0 else '看跌'} {arima_pct:+.1f}%",
                    "color": arima_color,
                    "detail": f"ARIMA预测{forecast_days}日后价格: {pred_last:.2f}（当前 {last_close:.2f}）\n"
                              f"涨跌幅: {arima_pct:+.2f}%\n"
                              f"波动范围: {arima_result['lower'][-1]:.2f} ~ {arima_result['upper'][-1]:.2f}",
                    "signal_type": arima_signal_type, "participate": True, "category": "prediction"})

# 基本面信号
signals.append({**valuation_signal, "category": "fundamental"})
signals.append({**financial_signal, "category": "fundamental"})

# 资金面+消息面信号
signals.append({**fund_flow_signal, "category": "sentiment"})
signals.append({**margin_signal, "category": "sentiment"})
signals.append({**event_signal, "category": "sentiment"})

# 风险信号
signals.append({**risk_signal, "category": "risk"})

# ==================== Weighted Scoring ====================
category_weights = {
    "technical": SCORE_WEIGHT_TECHNICAL,
    "prediction": SCORE_WEIGHT_PREDICTION,
    "fundamental": SCORE_WEIGHT_FUNDAMENTAL,
    "sentiment": SCORE_WEIGHT_SENTIMENT,
    "risk": 0.0,  # Never counts
}

def calculate_weighted_score(signals_list):
    """加权综合评分"""
    categories = {}
    for sig in signals_list:
        cat = sig.get("category", "technical")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(sig)

    total_score = 0.0
    total_weight_used = 0.0

    for cat, cat_sigs in categories.items():
        cat_weight = category_weights.get(cat, 0.0)
        if cat_weight == 0:
            continue

        participating = [s for s in cat_sigs if s.get("participate", True)]
        if not participating:
            continue

        dir_map = {"bull": 1, "bear": -1, "neutral": 0}
        cat_score = sum(dir_map.get(s.get("signal_type", "neutral"), 0) for s in participating)
        cat_score_normalized = cat_score / len(participating)

        total_score += cat_score_normalized * cat_weight
        total_weight_used += cat_weight

    if total_weight_used > 0:
        total_score = total_score / total_weight_used

    return total_score

weighted_score = calculate_weighted_score(signals)

if weighted_score > SCORE_BULL_THRESHOLD:
    overall_text = f"偏多（评分 {weighted_score:+.2f}）— 综合信号偏乐观"
    overall_cls = "bull"
elif weighted_score < SCORE_BEAR_THRESHOLD:
    overall_text = f"偏空（评分 {weighted_score:+.2f}）— 综合信号偏谨慎"
    overall_cls = "bear"
else:
    overall_text = f"多空交织（评分 {weighted_score:+.2f}）— 建议观望"
    overall_cls = "neutral"

# 检查是否有模块不可用
unavailable_modules = [s["name"] for s in signals if not s.get("participate", True) and s.get("signal_type", "neutral") not in ["special"]]
non_tech_unavailable = [m for m in unavailable_modules if m not in ["趋势", "RSI", "MACD", "量价", "ML预测", "ARIMA"]]

# ===================== PAGE LAYOUT =====================

# Stock header
stock_name = selected_stock.get("name", selected_stock["code"])
stock_code = selected_stock["code"]
st.markdown(f"## {stock_name}（{stock_code}）")

# Price info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("最新价", f"{last_close:.2f}")
with col2:
    st.metric("涨跌幅", f"{last_pct:+.2f}%", delta_color="inverse" if last_pct < 0 else "normal")
with col3:
    if "turn" in df.columns and pd.notna(df["turn"].iloc[-1]):
        st.metric("换手率", f"{df['turn'].iloc[-1]:.2f}%")

# Overall verdict
st.markdown(f'<div class="verdict verdict-{overall_cls}">{overall_text}</div>', unsafe_allow_html=True)
if non_tech_unavailable:
    st.caption(f"部分模块暂时不可用: {', '.join(non_tech_unavailable)}（当前评分仅基于可用模块）")

# Signal cards (4 rows x 3 cols = 12)
st.markdown("**信号分析**（点击「查看详解」了解指标含义和当前数据来源）")
signal_rows_list = [signals[i:i + 3] for i in range(0, len(signals), 3)]
for row in signal_rows_list:
    cols = st.columns(3)
    for i, sig in enumerate(row):
        with cols[i]:
            css_class = classify(sig["color"])
            meta = SIGNAL_META.get(sig["name"], {})
            cn_name = meta.get("cn", sig["name"])
            explain = meta.get("explain", "")
            st.markdown(f"""
            <div class="signal-card signal-{css_class}">
                <div class="label">{cn_name}</div>
                <div class="sublabel">{sig['name']}</div>
                <div class="value">{sig['value']}</div>
            </div>
            """, unsafe_allow_html=True)
            with st.popover("查看详解"):
                st.markdown(f"**{cn_name}**\n\n{explain}\n\n---\n**当前数据：**\n{sig['detail']}")

# K-line chart
st.divider()
st.subheader("K线图")
st.caption("红蜡烛=上涨，绿蜡烛=下跌。均线+布林带叠加。支持鼠标滚轮/双指缩放。")
fig = create_main_chart(df)
st.plotly_chart(fig, use_container_width=True, config={
    "scrollZoom": True, "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
})

# Recent data
st.subheader("近期行情")
display_cols = ["date", "open", "high", "low", "close", "volume", "pctChg"]
show_cols = [c for c in display_cols if c in df.columns]
recent = df[show_cols].tail(20).sort_values("date", ascending=False).copy()
recent = recent.rename(columns={
    "date": "日期", "open": "开盘价", "high": "最高价",
    "low": "最低价", "close": "收盘价", "volume": "成交量", "pctChg": "涨跌幅",
})
if "涨跌幅" in recent.columns:
    recent["涨跌幅"] = recent["涨跌幅"].apply(lambda x: f"{x:.2f}%")
if "成交量" in recent.columns:
    recent["成交量"] = recent["成交量"].apply(lambda x: f"{x:,.0f}")
st.dataframe(recent, use_container_width=True, hide_index=True)

# Key indicators
st.subheader("关键指标")
st.caption("各技术指标当前数值，快速了解股票技术面状态。")
metric_cols = st.columns(min(len(metrics), 4))
for i, (k, v) in enumerate(metrics.items()):
    metric_cols[i % len(metric_cols)].metric(k, v)

# ==================== Expandable Sections ====================
st.divider()

# 估值详情
with st.expander("估值详情"):
    st.caption("PE(市盈率)=股价÷每股收益，越低回本越快；PB(市净率)=股价÷每股净资产，<1可能低估；PS(市销率)=总市值÷营收，适用于轻资产/成长型公司。PE走势图反映历史估值中枢变化。")
    if valuation_data.get("success"):
        data = valuation_data["data"]
        vc1, vc2, vc3 = st.columns(3)
        pe = data.get("pe")
        vc1.metric("PE(TTM)", f"{pe:.2f}" if pe and pe > 0 else ("亏损" if pe and pe < 0 else "N/A"))
        pb = data.get("pb")
        vc2.metric("PB", f"{pb:.2f}" if pb else "N/A")
        ps = data.get("ps")
        vc3.metric("PS", f"{ps:.2f}" if ps else "N/A")
        st.caption(f"数据来源: {valuation_data.get('source', 'unknown')}")
        pe_chart = create_pe_history_chart(valuation_data)
        if pe_chart:
            st.plotly_chart(pe_chart, use_container_width=True, config={
                "scrollZoom": True, "displayModeBar": False,
            })
        else:
            st.info("PE历史数据不足，无法生成走势图")
    else:
        st.info(f"估值数据不可用: {valuation_data.get('message', '')}")

# 财报概览
with st.expander("财报概览"):
    st.caption("ROE(净资产收益率)衡量股东回报率，>15%较优秀；毛利率/净利率反映产品竞争力与费用控制；资产负债率>60%为高杠杆；经营现金流应持续为正且匹配净利润，否则利润质量存疑。")
    if financial_data.get("success"):
        quarters = financial_data["data"]["quarters"]
        if quarters:
            last_q = quarters[-1]
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("营业总收入", f"{last_q.get('营业总收入', 0)/1e8:.2f}亿" if last_q.get("营业总收入") else "N/A")
            fc2.metric("归母净利润", f"{last_q.get('归母净利润', 0)/1e8:.2f}亿" if last_q.get("归母净利润") else "N/A")
            roe = last_q.get("ROE")
            fc3.metric("ROE", f"{roe:.2f}%" if roe else "N/A")
            gpm = last_q.get("毛利率")
            fc4.metric("毛利率", f"{gpm:.2f}%" if gpm else "N/A")

            fc5, fc6, fc7, fc8 = st.columns(4)
            npm = last_q.get("净利率")
            fc5.metric("净利率", f"{npm:.2f}%" if npm else "N/A")
            debt = last_q.get("资产负债率")
            fc6.metric("资产负债率", f"{debt:.2f}%" if debt else "N/A")
            ocf = last_q.get("经营活动现金流净额")
            fc7.metric("经营现金流", f"{ocf/1e8:.2f}亿" if ocf else "N/A")

            fin_chart = create_financial_chart(financial_data)
            if fin_chart:
                st.plotly_chart(fin_chart, use_container_width=True, config={
                    "scrollZoom": True, "displayModeBar": False,
                })
        else:
            st.info("财务数据为空")
    else:
        st.info(f"财务数据不可用: {financial_data.get('message', '')}")

# 资金流向
with st.expander("资金流向"):
    st.caption("主力=大单(>10万手)资金净额，持续主力净流入通常看多，持续流出看空；散户资金方向常与主力反向。5日/20日累计反映中期资金态度。")
    if fund_flow_data.get("success"):
        d = fund_flow_data["data"]
        ff1, ff2, ff3 = st.columns(3)
        from analysis.fund_flow_analysis import format_fund_amount
        ff1.metric("主力日净流入", format_fund_amount(d.get("main_net_inflow", 0)))
        ff2.metric("主力5日累计", format_fund_amount(d.get("main_5d_cum", 0)))
        ff3.metric("主力20日累计", format_fund_amount(d.get("main_20d_cum", 0)))
        fund_chart = create_fund_flow_chart(fund_flow_data)
        if fund_chart:
            st.plotly_chart(fund_chart, use_container_width=True, config={
                "scrollZoom": True, "displayModeBar": False,
            })
    else:
        st.info(f"资金流向数据不可用: {fund_flow_data.get('message', '')}")

# 风险度量
with st.expander("风险度量"):
    st.caption("最大回撤=历史最大亏损幅度，越小越抗跌；年化波动率越高价格波动越剧烈；Beta>1比大盘波动大(进攻型)，<1比大盘稳(防御型)；夏普比率>1较好，衡量每单位风险换来的超额收益。")
    risk_data = risk_signal.get("risk_data", {})
    if risk_data:
        r1, r2, r3, r4 = st.columns(4)
        md60 = risk_data.get("md_60")
        r1.metric("60日最大回撤", f"{md60:.1%}" if md60 is not None else "N/A")
        vol = risk_data.get("vol")
        r2.metric("年化波动率", f"{vol:.1%}" if vol is not None else "N/A")
        beta = risk_data.get("beta")
        r3.metric("Beta(沪深300)", f"{beta:.2f}" if beta is not None else "N/A")
        sharpe = risk_data.get("sharpe")
        r4.metric("夏普比率", f"{sharpe:.2f}" if sharpe is not None else "N/A")

        if beta is None:
            st.caption("Beta无法计算：指数数据不足或获取失败")
    else:
        st.info("风险指标不可用")

# 支撑位/压力位
with st.expander("支撑位 / 压力位"):
    st.caption("支撑位=近期低点/均线密集区，价格跌至此处买盘可能介入止跌；压力位=近期高点/密集成交区，价格涨至此处套牢盘可能卖出形成阻力。跌破支撑转空，突破压力转多。")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**支撑位**（下跌可能止跌的位置）")
        if sr["support"]:
            for s in sr["support"]:
                st.write(f"- {s:.2f}")
        else:
            st.write("- 无明显支撑")
    with c2:
        st.markdown("**压力位**（上涨可能受阻的位置）")
        if sr["resistance"]:
            for r in sr["resistance"]:
                st.write(f"- {r:.2f}")
        else:
            st.write("- 无明显压力")

# 量能分析
with st.expander("量能分析"):
    st.caption("成交量是价格变动的\"燃料\"：放量上涨=强趋势，缩量上涨=动能不足；放量下跌=恐慌抛售，缩量下跌=卖压减弱。价涨量增为健康，价跌量缩可能见底。")
    vol_cols = st.columns(min(len(vol_info), 3))
    for i, (k, v) in enumerate(vol_info.items()):
        vol_cols[i % len(vol_cols)].metric(k, str(v))
    st.info(f"量价关系: {divergence}")

# 融资融券
with st.expander("融资融券"):
    st.caption("融资余额=投资者借钱买入的未偿还金额，余额持续增长看多情绪浓；融资买入额=当日新增融资规模；融券余量=借股卖出的未还数量，越大做空力量越强。融资余额占流通市值>5%需警惕杠杆风险。")
    if margin_data.get("success"):
        d = margin_data["data"]
        m1, m2 = st.columns(2)
        mb = d.get("margin_balance")
        m1.metric("融资余额", f"{mb/1e8:.2f}亿" if mb else "N/A")
        mbuy = d.get("margin_buy")
        m2.metric("融资买入额", f"{mbuy/1e8:.2f}亿" if mbuy else "N/A")
        if float_mv and float_mv > 0 and mb:
            ratio = mb / float_mv
            st.caption(f"融资余额占流通市值: {ratio:.2%}" +
                       ("（高杠杆风险）" if ratio > 0.05 else ""))
    elif "非两融" in str(margin_data.get("reason", "")):
        st.info("该股票非两融标的")
    else:
        st.info(f"融资融券数据不可用: {margin_data.get('reason', '')}")

# 近期公告
with st.expander("近期公告"):
    st.caption("公告是公司信息的第一手来源：业绩预告/快报影响短期估值；分红预案反映回馈股东意愿；增发/配股可能稀释每股收益；股东增减持反映内部人态度；重大事件(诉讼/重组/并购)可能改变基本面。")
    if notice_data.get("success"):
        notices = notice_data["data"].get("notices", [])
        if notices:
            type_names = {
                "earnings_forecast": "业绩预告", "earnings_report": "业绩报告",
                "dividend": "分红预案", "refinance": "增发预案",
                "shareholder_trade": "股东增减持", "major_event": "重大事件",
                "other": "其他",
            }
            for n in notices[:15]:
                t = type_names.get(n.get("type", "other"), n.get("type", ""))
                st.caption(f"[{t}] {n.get('date', '')} - {n.get('title', '')}")
        else:
            st.info("近期无公告")
    else:
        st.info(f"公告数据不可用: {notice_data.get('message', '')}")

# 分红记录
with st.expander("分红记录"):
    st.caption("股息率=每股分红÷股价，>3%属于高股息；连续多年稳定分红是现金流良好的信号。但高股息也可能因为股价大跌，需结合利润增速判断分红持续性。")
    if dividend_info.get("has_data"):
        if dividend_info.get("is_aristocrat"):
            st.success(dividend_info["message"])
        else:
            st.info(dividend_info["message"])
        dividends = dividend_info.get("dividends", [])
        if dividends:
            for d in dividends:
                ex = d.get("ex_date", "")
                dps = d.get("div_per_share")
                dy = d.get("div_yield")
                st.caption(f"{ex} | 每股{dps:.4f}元 | 股息率{dy:.2%}" if dps and dy else f"{ex} | 每股{dps}")
    else:
        st.info(dividend_info.get("message", "分红数据不可用"))

# 行业对比
with st.expander("行业对比"):
    st.caption("判断个股估值在行业中的相对位置：PE低于行业中位数可能被低估(也可能是基本面有问题)；高于行业均值可能溢价合理(龙头溢价)或高估。需结合ROE、成长性综合判断。")
    industry_info = get_industry_info(bs_code)
    if industry_info.get("success") and industry_info.get("data"):
        ind_data = industry_info["data"]
        industry = ind_data.get("industry", "")
        ind_class = ind_data.get("industry_classification", "")
        st.metric("所属行业", industry if industry else "未知")
        if ind_class:
            st.caption(f"行业分类: {ind_class}")
        stock_pe = valuation_data.get("data", {}).get("pe") if valuation_data.get("success") else None
        if stock_pe is not None and stock_pe > 0:
            st.caption(f"个股PE: {stock_pe:.2f}（行业PE中位数需通过更多数据源获取）")
        else:
            st.caption("行业PE对比暂时不可用")
    else:
        st.info("行业分类数据不可用")

# 预测详情
with st.expander("预测详情"):
    st.caption("**ARIMA**=基于历史价格时间序列的统计建模，适合短期趋势延续预测，不考虑基本面；**ML**=基于技术指标+资金流向等多因子的机器学习方向预测。两者仅供参考，不可作为交易决策唯一依据。")
    st.markdown("**ARIMA（时间序列预测）**")
    if arima_result.get("success"):
        st.caption(f"模型参数: ARIMA{arima_result['order']}")
    else:
        st.caption("ARIMA拟合失败或数据不足，使用朴素预测法")

    forecast_dates = generate_forecast_dates(df, forecast_days)
    fig_forecast = create_forecast_chart(
        df, forecast_dates,
        arima_result["forecast"], arima_result["lower"], arima_result["upper"],
    )
    st.plotly_chart(fig_forecast, use_container_width=True, config={
        "scrollZoom": True, "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    })

    pred_last = arima_result["forecast"][-1]
    arima_pct = (pred_last - last_close) / last_close * 100
    st.metric(f"ARIMA {forecast_days}日预测", f"{pred_last:.2f}", f"{arima_pct:+.2f}%")
    st.caption(f"当前价 {last_close:.2f} → 预测价 {pred_last:.2f}")

    st.divider()

    st.markdown("**ML（机器学习预测）**")
    if ml_result.get("success"):
        direction = ml_result["direction"]
        icon = "📈" if direction == "看涨" else "📉"
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("方向", f"{icon} {direction}")
        mc2.metric("置信度", f"{ml_result['confidence']:.1f}%")
        mc3.metric("R²", f"{ml_result['r2_score']:.4f}")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("5日收益", ml_result.get("5day_return", "N/A"))
        rc2.metric("10日收益", ml_result.get("10day_return", "N/A"))
        rc3.metric(f"{forecast_days}日收益", ml_result.get("20day_return", "N/A"))
    else:
        st.warning(ml_result.get("message", "数据不足"))

st.markdown("---")
st.markdown("*免责声明：本工具仅供学习研究，不构成任何投资建议。股市有风险，投资需谨慎。*")
