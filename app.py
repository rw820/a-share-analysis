import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from config import FREQUENCY_MAP, get_default_date_range, FORECAST_DAYS_DEFAULT, FORECAST_DAYS_MIN, FORECAST_DAYS_MAX
from data.stock_search import search_stocks
from data.market_data import get_kline_data
from analysis.technical_indicators import compute_all_indicators
from analysis.trend_analysis import detect_trend, find_support_resistance, get_key_metrics
from analysis.volume_analysis import analyze_volume, detect_volume_price_divergence
from visualization.charts import create_main_chart, create_forecast_chart
from prediction.arima_model import arima_forecast, generate_forecast_dates
from prediction.ml_model import ml_forecast

st.set_page_config(page_title="A股个股分析", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
/* Force metric values to wrap and not truncate */
[data-testid="stMetricValue"] { white-space: normal !important; font-size: 1rem !important; overflow: visible !important; text-overflow: unset !important; }
[data-testid="stMetricLabel"] { white-space: normal !important; font-size: 0.8rem !important; }
[data-testid="stMetricDelta"] { white-space: normal !important; font-size: 0.85rem !important; }

/* Signal card styling */
.signal-card {
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 8px;
    border-left: 4px solid;
    background: #fafafa;
}
.signal-card .label { font-size: 12px; color: #888; margin-bottom: 2px; }
.signal-card .value { font-size: 15px; font-weight: 600; }
.signal-bull { border-left-color: #ef4444; background: #fef2f2; }
.signal-bull .value { color: #dc2626; }
.signal-bear { border-left-color: #22c55e; background: #f0fdf4; }
.signal-bear .value { color: #16a34a; }
.signal-neutral { border-left-color: #f59e0b; background: #fffbeb; }
.signal-neutral .value { color: #d97706; }

/* Overall verdict banner */
.verdict {
    text-align: center;
    padding: 12px 20px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 700;
    margin: 10px 0;
}
.verdict-bull { background: linear-gradient(135deg, #fef2f2, #fee2e2); color: #dc2626; border: 2px solid #fecaca; }
.verdict-bear { background: linear-gradient(135deg, #f0fdf4, #dcfce7); color: #16a34a; border: 2px solid #bbf7d0; }
.verdict-neutral { background: linear-gradient(135deg, #fffbeb, #fef3c7); color: #d97706; border: 2px solid #fde68a; }

/* Mobile */
@media (max-width: 768px) {
    .block-container { padding-top: 0.5rem; padding-bottom: 1rem; padding-left: 1rem; padding-right: 1rem; }
    h1 { font-size: 1.3rem !important; }
    .verdict { font-size: 15px; padding: 10px 14px; }
    .signal-card .value { font-size: 14px; }
}
</style>
""", unsafe_allow_html=True)

# === Search bar ===
st.title("A股个股分析")

search_col1, search_col2 = st.columns([5, 1])
with search_col1:
    query = st.text_input("输入股票代码或名称", placeholder="如: 600519 或 贵州茅台", label_visibility="collapsed")
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

**支撑位/压力位**：下跌可能止跌/上涨可能受阻的价格位置
""")

# Search logic
selected_stock = None
if query or search_clicked:
    effective_query = query.strip()
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

# === Fetch & compute ===
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(bs_code, start, end, freq):
    return get_kline_data(bs_code, str(start), str(end), freq)

with st.spinner("正在分析中..."):
    df = fetch_data(selected_stock["bs_code"], start_date, end_date, freq)

if df.empty:
    st.error("无法获取该股票数据，请检查股票代码或日期范围")
    st.stop()

df = compute_all_indicators(df)
trend_info = detect_trend(df)
sr = find_support_resistance(df)
vol_info = analyze_volume(df)
divergence = detect_volume_price_divergence(df)
metrics = get_key_metrics(df)
arima_result = arima_forecast(df, forecast_days)
ml_result = ml_forecast(df, forecast_days)

# === Analyze signals ===
last_close = df["close"].iloc[-1]
last_pct = df["pctChg"].iloc[-1] if "pctChg" in df.columns else 0

SIGNAL_EXPLANATIONS = {
    "趋势": "根据MA5/MA10/MA20/MA60四条均线的排列关系判断。\n多头排列(MA5>MA10>MA20>MA60)=短期买盘积极，看涨；\n空头排列(反过来)=卖压大，看空；\n交织=震荡，方向不明。",
    "RSI": "相对强弱指标，反映近期涨跌力度。\nRSI>70说明近期涨幅过大，可能超买（要回调）；\nRSI<30说明近期跌幅过大，可能超卖（要反弹）；\n30-70为正常区间。",
    "MACD": "MACD柱状图反映多空力量对比。\n红柱(>0)说明多头力量占优；\n绿柱(<0)说明空头力量占优。\nDIF线上穿DEA线叫'金叉'，是买入信号。",
    "量价": "成交量与价格走势的关系。\n量价齐升=健康上涨；\n价涨量缩=上涨动力不足，注意风险；\n放量下跌=恐慌抛售，卖压大。",
    "ML预测": "使用梯度提升机器学习模型，综合MA斜率、RSI、MACD、成交量比等10+个技术特征，\n在近500个交易日的历史数据上训练，预测未来涨跌方向和幅度。\n置信度>60%有参考价值。",
    "ARIMA": "ARIMA是经典时间序列预测方法，根据历史价格的自相关规律推算未来走势。\n自动选择最优参数组合，提供预测价格和95%置信区间（波动范围）。",
}

def classify(color):
    return "bull" if color == "green" else ("bear" if color == "red" else "neutral")

# Build signal list: (name, value, color, detail_text)
signals = []

# Trend
if "多" in trend_info["trend"]:
    signals.append(("趋势", "看多", "red", trend_info.get("description", "")))
elif "空" in trend_info["trend"]:
    signals.append(("趋势", "看空", "green", trend_info.get("description", "")))
else:
    signals.append(("趋势", "震荡", "amber", trend_info.get("description", "")))

# RSI
rsi_val = None
if "RSI" in df.columns:
    rsi_val = df["RSI"].iloc[-1]
    if pd.notna(rsi_val):
        if rsi_val > 70:
            signals.append(("RSI", f"超买 {rsi_val:.0f}", "green", f"RSI={rsi_val:.1f}，超过70警戒线，短期可能过热回调"))
        elif rsi_val < 30:
            signals.append(("RSI", f"超卖 {rsi_val:.0f}", "red", f"RSI={rsi_val:.1f}，低于30警戒线，短期可能超跌反弹"))
        else:
            signals.append(("RSI", f"中性 {rsi_val:.0f}", "amber", f"RSI={rsi_val:.1f}，处于30-70正常区间"))

# MACD
if "MACD_hist" in df.columns:
    hist = df["MACD_hist"].iloc[-1]
    if pd.notna(hist):
        if hist > 0:
            signals.append(("MACD", "多头", "red", f"MACD柱={hist:.4f}，多头力量占优"))
        else:
            signals.append(("MACD", "空头", "green", f"MACD柱={hist:.4f}，空头力量占优"))

# Volume-price
if "量价齐升" in divergence:
    signals.append(("量价", "齐升", "red", "价格上涨且成交量放大，健康上涨形态"))
elif "价涨量缩" in divergence or "背离" in divergence:
    signals.append(("量价", "注意", "amber", f"当前: {divergence}"))
elif "放量下跌" in divergence:
    signals.append(("量价", "卖压", "green", "价格下跌且成交量放大，抛售压力大"))
else:
    signals.append(("量价", "正常", "amber", f"当前: {divergence}"))

# ML
if ml_result.get("success"):
    ml_dir = ml_result["direction"]
    ml_conf = ml_result["confidence"]
    ml_color = "red" if ml_dir == "看涨" else "green"
    signals.append(("ML预测", f"{ml_dir} {ml_conf:.0f}%", ml_color,
        f"机器学习模型预测方向: {ml_dir}\n置信度: {ml_conf:.1f}%\n"
        f"预测收益 - 5日: {ml_result.get('5day_return','N/A')}, 10日: {ml_result.get('10day_return','N/A')}, "
        f"{forecast_days}日: {ml_result.get('20day_return','N/A')}"))

# ARIMA
if arima_result.get("forecast") is not None:
    pred_last = arima_result["forecast"][-1]
    arima_pct = (pred_last - last_close) / last_close * 100
    arima_color = "red" if arima_pct > 0 else "green"
    arima_label = f"{'看涨' if arima_pct > 0 else '看跌'} {arima_pct:+.1f}%"
    signals.append(("ARIMA", arima_label, arima_color,
        f"ARIMA预测{forecast_days}日后价格: {pred_last:.2f}（当前 {last_close:.2f}）\n"
        f"预测涨跌幅: {arima_pct:+.2f}%\n"
        f"波动范围: {arima_result['lower'][-1]:.2f} ~ {arima_result['upper'][-1]:.2f}"))

# Overall score
bull_count = sum(1 for _, _, c, _ in signals if c == "red")
bear_count = sum(1 for _, _, c, _ in signals if c == "green")
total = len(signals)
# Note: in Chinese stock market convention, red=bullish, green=bearish

if bull_count >= total * 0.6:
    overall_text = f"偏多 — {bull_count}/{total}个信号看涨，短期走势乐观"
    overall_cls = "bull"
elif bear_count >= total * 0.6:
    overall_text = f"偏空 — {bear_count}/{total}个信号看跌，短期注意风险"
    overall_cls = "bear"
else:
    overall_text = f"多空交织 — 看涨{bull_count}个/看跌{bear_count}个，建议观望"
    overall_cls = "neutral"

# === Page Layout ===

# Stock name
st.markdown(f"## {selected_stock['name']}（{selected_stock['code']}）")

# Key price info — 2 columns only to avoid truncation
col1, col2 = st.columns(2)
with col1:
    st.metric("最新价", f"{last_close:.2f}")
with col2:
    st.metric("涨跌幅", f"{last_pct:+.2f}%", delta_color="inverse" if last_pct < 0 else "normal")

# Overall verdict banner
st.markdown(f'<div class="verdict verdict-{overall_cls}">{overall_text}</div>', unsafe_allow_html=True)

# Signal cards — 3 columns, 2 rows
st.markdown("**信号分析**（点击卡片查看详解）")

# Build signal rows
signal_rows = [signals[i:i+3] for i in range(0, len(signals), 3)]
for row in signal_rows:
    cols = st.columns(3)
    for i, (name, val, color, detail) in enumerate(row):
        with cols[i]:
            css_class = classify(color)
            st.markdown(f"""
            <div class="signal-card signal-{css_class}">
                <div class="label">{name}</div>
                <div class="value">{val}</div>
            </div>
            """, unsafe_allow_html=True)
            with st.popover("查看详解"):
                st.markdown(f"**{name}指标说明：**\n\n{SIGNAL_EXPLANATIONS.get(name, '')}\n\n---\n**当前数据：**\n{detail}")

# === K-line chart ===
st.divider()
fig = create_main_chart(df)
st.plotly_chart(fig, use_container_width=True, config={
    "scrollZoom": True,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
})

# === Expandable details ===
with st.expander("支撑位 / 压力位"):
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

with st.expander("量能分析"):
    vol_cols = st.columns(min(len(vol_info), 3))
    for i, (k, v) in enumerate(vol_info.items()):
        vol_cols[i % len(vol_cols)].metric(k, str(v))
    st.info(f"量价关系: {divergence}")

with st.expander("预测详情"):
    st.markdown("**ARIMA 时间序列预测**")
    if arima_result.get("success"):
        st.caption(f"模型参数: ARIMA{arima_result['order']}")
    else:
        st.caption("ARIMA拟合失败，使用朴素预测法")

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

    st.markdown("**ML 机器学习预测**")
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

with st.expander("近期行情数据"):
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

st.markdown("---")
st.markdown("*免责声明：本工具仅供学习研究，不构成任何投资建议。股市有风险，投资需谨慎。*")
