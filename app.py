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

st.set_page_config(page_title="A股个股分析", layout="wide")
st.title("A股个股分析系统")

# Sidebar
with st.sidebar:
    st.header("股票搜索")
    query = st.text_input("输入股票代码或名称", placeholder="如: 600519 或 贵州茅台")

    selected_stock = None
    if query:
        with st.spinner("搜索中..."):
            results = search_stocks(query)
        if results:
            options = [f"{r['code']} - {r['name']}" for r in results]
            choice = st.selectbox("选择股票", options)
            idx = options.index(choice)
            selected_stock = results[idx]
            st.success(f"已选择: {selected_stock['name']} ({selected_stock['code']})")
        else:
            st.warning("未找到匹配的股票")

    if selected_stock:
        st.divider()
        st.header("分析参数")
        start_default, end_default = get_default_date_range()
        start_date = st.date_input("开始日期", pd.to_datetime(start_default))
        end_date = st.date_input("结束日期", pd.to_datetime(end_default))

        freq_label = st.selectbox("K线周期", list(FREQUENCY_MAP.keys()))
        freq = FREQUENCY_MAP[freq_label]

        forecast_days = st.slider("预测天数", FORECAST_DAYS_MIN, FORECAST_DAYS_MAX, FORECAST_DAYS_DEFAULT)

# Main content
if not selected_stock:
    st.info("请在左侧输入股票代码或名称开始分析")
    st.markdown("---")
    st.markdown("*本工具仅供学习研究，不构成任何投资建议。股市有风险，投资需谨慎。*")
    st.stop()

# Fetch data
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(bs_code, start, end, freq):
    return get_kline_data(bs_code, str(start), str(end), freq)

with st.spinner("正在获取行情数据..."):
    df = fetch_data(
        selected_stock["bs_code"],
        start_date,
        end_date,
        freq,
    )

if df.empty:
    st.error("无法获取该股票数据，请检查股票代码或日期范围")
    st.stop()

# Compute indicators
df = compute_all_indicators(df)

# Tabs
tab1, tab2, tab3 = st.tabs(["K线图 & 技术分析", "趋势分析", "走势预测"])

with tab1:
    st.subheader(f"{selected_stock['name']} ({selected_stock['code']}) K线图")
    fig = create_main_chart(df)
    st.plotly_chart(fig, use_container_width=True)

    # Recent data table
    st.subheader("近期行情")
    display_cols = ["date", "open", "high", "low", "close", "volume", "pctChg"]
    show_cols = [c for c in display_cols if c in df.columns]
    recent = df[show_cols].tail(20).sort_values("date", ascending=False)
    recent = recent.copy()
    if "pctChg" in recent.columns:
        recent["pctChg"] = recent["pctChg"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(recent, use_container_width=True, hide_index=True)

with tab2:
    st.subheader(f"{selected_stock['name']} 趋势分析")

    # Trend
    trend_info = detect_trend(df)
    col1, col2 = st.columns(2)
    with col1:
        trend_color = "🟢" if "多" in trend_info["trend"] else ("🔴" if "空" in trend_info["trend"] else "🟡")
        st.metric("趋势判断", f"{trend_color} {trend_info['trend']}")
    with col2:
        st.metric("MA60位置", trend_info.get("price_vs_ma60", "N/A"))
    if trend_info["description"]:
        st.caption(trend_info["description"])

    st.divider()

    # Support & Resistance
    sr = find_support_resistance(df)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("支撑位")
        if sr["support"]:
            for s in sr["support"]:
                st.write(f"  {s:.2f}")
        else:
            st.write("无明显支撑")
    with col2:
        st.subheader("压力位")
        if sr["resistance"]:
            for r in sr["resistance"]:
                st.write(f"  {r:.2f}")
        else:
            st.write("无明显压力")

    st.divider()

    # Volume analysis
    st.subheader("量能分析")
    vol_info = analyze_volume(df)
    vol_cols = st.columns(min(len(vol_info), 4))
    for i, (k, v) in enumerate(vol_info.items()):
        vol_cols[i % len(vol_cols)].metric(k, str(v))

    # Volume-price divergence
    divergence = detect_volume_price_divergence(df)
    st.info(f"量价关系: {divergence}")

    st.divider()

    # Key metrics
    st.subheader("关键指标汇总")
    metrics = get_key_metrics(df)
    metric_cols = st.columns(min(len(metrics), 4))
    for i, (k, v) in enumerate(metrics.items()):
        metric_cols[i % len(metric_cols)].metric(k, v)

with tab3:
    st.subheader(f"{selected_stock['name']} 走势预测")

    col_arima, col_ml = st.columns(2)

    with col_arima:
        st.markdown("### ARIMA 时间序列预测")
        with st.spinner("ARIMA模型拟合中..."):
            arima_result = arima_forecast(df, forecast_days)

        if arima_result.get("success", False):
            st.caption(f"最优参数: ARIMA{arima_result['order']}, AIC={arima_result['aic']:.1f}")
        else:
            st.caption("ARIMA拟合失败，使用朴素预测法")

        forecast_dates = generate_forecast_dates(df, forecast_days)
        fig_forecast = create_forecast_chart(
            df,
            forecast_dates,
            arima_result["forecast"],
            arima_result["lower"],
            arima_result["upper"],
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        last_close = df["close"].iloc[-1]
        pred_last = arima_result["forecast"][-1]
        change_pct = (pred_last - last_close) / last_close * 100
        st.metric(
            f"ARIMA {forecast_days}日预测",
            f"{pred_last:.2f}",
            f"{change_pct:+.2f}%",
        )

    with col_ml:
        st.markdown("### ML 机器学习预测")
        with st.spinner("ML模型训练中..."):
            ml_result = ml_forecast(df, forecast_days)

        if ml_result.get("success", False):
            direction = ml_result["direction"]
            direction_icon = "📈" if direction == "看涨" else "📉"
            st.metric("方向预测", f"{direction_icon} {direction}")
            st.metric("置信度", f"{ml_result['confidence']:.1f}%")
            st.metric("模型R²", f"{ml_result['r2_score']:.4f}")
            st.write("---")
            st.write("**预测收益率:**")
            st.write(f"- 5日: {ml_result.get('5day_return', 'N/A')}")
            st.write(f"- 10日: {ml_result.get('10day_return', 'N/A')}")
            st.write(f"- {forecast_days}日: {ml_result.get('20day_return', 'N/A')}")
        else:
            st.warning(ml_result.get("message", "预测失败，数据不足"))

# Disclaimer
st.markdown("---")
st.markdown("*免责声明：本工具仅供学习研究，不构成任何投资建议。股市有风险，投资需谨慎。*")
