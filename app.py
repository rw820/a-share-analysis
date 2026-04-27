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

st.markdown("""
<style>
@media (max-width: 768px) {
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stMetric > div { font-size: 0.9rem; }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.1rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.3rem 0.6rem; font-size: 0.85rem; }
}
</style>
""", unsafe_allow_html=True)

# === Search bar ===
st.title("A股个股分析")

search_col1, search_col2 = st.columns([4, 1])
with search_col1:
    query = st.text_input("输入股票代码或名称", placeholder="如: 600519 或 贵州茅台", label_visibility="collapsed")
with search_col2:
    search_clicked = st.button("搜索", use_container_width=True)

# Sidebar: params + guide
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

# === Generate summary conclusion ===
last_close = df["close"].iloc[-1]
last_pct = df["pctChg"].iloc[-1] if "pctChg" in df.columns else 0

# Collect signals
signals = []
# Trend signal
if "多" in trend_info["trend"]:
    signals.append(("趋势", "看多", "🟢"))
elif "空" in trend_info["trend"]:
    signals.append(("趋势", "看空", "🔴"))
else:
    signals.append(("趋势", "震荡", "🟡"))

# RSI signal
rsi_val = None
if "RSI" in df.columns:
    rsi_val = df["RSI"].iloc[-1]
    if pd.notna(rsi_val):
        if rsi_val > 70:
            signals.append(("RSI", f"超买({rsi_val:.0f})", "🔴"))
        elif rsi_val < 30:
            signals.append(("RSI", f"超卖({rsi_val:.0f})", "🟢"))
        else:
            signals.append(("RSI", f"中性({rsi_val:.0f})", "🟡"))

# MACD signal
if "MACD_hist" in df.columns:
    hist = df["MACD_hist"].iloc[-1]
    if pd.notna(hist):
        if hist > 0:
            signals.append(("MACD", "红柱(多头)", "🟢"))
        else:
            signals.append(("MACD", "绿柱(空头)", "🔴"))

# Volume-price signal
if "量价齐升" in divergence:
    signals.append(("量价", "齐升", "🟢"))
elif "价涨量缩" in divergence or "背离" in divergence:
    signals.append(("量价", "注意风险", "🟡"))
elif "放量下跌" in divergence:
    signals.append(("量价", "卖压大", "🔴"))
else:
    signals.append(("量价", "正常", "🟡"))

# Prediction signal
if ml_result.get("success"):
    ml_dir = ml_result["direction"]
    ml_conf = ml_result["confidence"]
    if ml_dir == "看涨":
        signals.append(("ML预测", f"看涨({ml_conf:.0f}%)", "🟢"))
    else:
        signals.append(("ML预测", f"看跌({ml_conf:.0f}%)", "🔴"))

if arima_result.get("forecast") is not None:
    pred_last = arima_result["forecast"][-1]
    arima_pct = (pred_last - last_close) / last_close * 100
    if arima_pct > 0:
        signals.append(("ARIMA", f"看涨({arima_pct:+.1f}%)", "🟢"))
    else:
        signals.append(("ARIMA", f"看跌({arima_pct:+.1f}%)", "🔴"))

# Overall score
bull_count = sum(1 for _, _, icon in signals if icon == "🟢")
bear_count = sum(1 for _, _, icon in signals if icon == "🔴")
total = len(signals)

if bull_count >= total * 0.6:
    overall = ("偏多，短期走势乐观", "🟢")
elif bear_count >= total * 0.6:
    overall = ("偏空，短期注意风险", "🔴")
else:
    overall = ("多空交织，建议观望", "🟡")

# === Page layout ===

# Stock header
st.markdown(f"## {selected_stock['name']} ({selected_stock['code']})")

# Key price info
price_col1, price_col2, price_col3, price_col4 = st.columns(4)
with price_col1:
    st.metric("最新价", f"{last_close:.2f}")
with price_col2:
    st.metric("涨跌幅", f"{last_pct:+.2f}%", delta_color="inverse" if last_pct < 0 else "normal")
with price_col3:
    st.metric("趋势", trend_info["trend"].replace("多头排列（", "").replace("空头排列（", "").replace("）", ""))
with price_col4:
    st.metric("综合判断", f"{overall[1]} {overall[0]}")

# Signal cards
st.markdown("**信号一览**")
sig_cols = st.columns(len(signals))
for i, (name, val, icon) in enumerate(signals):
    sig_cols[i % len(sig_cols)].metric(name, f"{icon} {val}")

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
    vol_cols = st.columns(min(len(vol_info), 4))
    for i, (k, v) in enumerate(vol_info.items()):
        vol_cols[i % len(vol_cols)].metric(k, str(v))
    st.info(f"量价关系: {divergence}")

with st.expander("预测详情"):
    # ARIMA
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
    st.metric(
        f"ARIMA {forecast_days}日预测",
        f"{pred_last:.2f}",
        f"{arima_pct:+.2f}%",
    )
    st.caption(f"当前价 {last_close:.2f} → 预测价 {pred_last:.2f}")

    st.divider()

    # ML
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

# Disclaimer
st.markdown("---")
st.markdown("*免责声明：本工具仅供学习研究，不构成任何投资建议。股市有风险，投资需谨慎。*")
