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

# Inject mobile-friendly CSS
st.markdown("""
<style>
/* Mobile optimizations */
@media (max-width: 768px) {
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stMetric > div { font-size: 0.9rem; }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.1rem !important; }
    /* Make tabs more compact on mobile */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.3rem 0.6rem; font-size: 0.85rem; }
}
/* Search input styling */
.search-row { display: flex; gap: 0.5rem; align-items: flex-end; }
.search-row > div { flex: 1; }
</style>
""", unsafe_allow_html=True)

# === Search bar in main area (not sidebar) for mobile visibility ===
st.title("A股个股分析系统")

search_col1, search_col2 = st.columns([4, 1])
with search_col1:
    query = st.text_input(
        "输入股票代码或名称",
        placeholder="如: 600519 或 贵州茅台",
        label_visibility="collapsed",
    )
with search_col2:
    search_clicked = st.button("搜索", use_container_width=True)

# Also read from sidebar for desktop users who prefer it
with st.sidebar:
    st.header("分析参数")
    st.caption("点击左上角 > 展开侧边栏")

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

if selected_stock:
    with st.sidebar:
        start_default, end_default = get_default_date_range()
        start_date = st.date_input("开始日期", pd.to_datetime(start_default))
        end_date = st.date_input("结束日期", pd.to_datetime(end_default))
        freq_label = st.selectbox("K线周期", list(FREQUENCY_MAP.keys()))
        freq = FREQUENCY_MAP[freq_label]
        forecast_days = st.slider("预测天数", FORECAST_DAYS_MIN, FORECAST_DAYS_MAX, FORECAST_DAYS_DEFAULT)

    with st.sidebar:
        with st.expander("指标说明"):
            st.markdown("""
**K线图**
- **红色蜡烛**：收盘价高于开盘价（上涨）
- **绿色蜡烛**：收盘价低于开盘价（下跌）
- 蜡烛上下细线是当日最高/最低价

**均线（MA）**
- **MA5**（红线）：5日均线，反映短期趋势
- **MA10**（青线）：10日均线
- **MA20**（蓝线）：20日均线，月线级别
- **MA60**（绿线）：60日均线，季线级别
- 价格在均线上方说明走势偏强，下方偏弱
- 短期均线在长期均线上方叫"多头排列"（看涨信号）

**MACD**
- **DIF线**（快线）上穿 **DEA线**（慢线）→ 买入信号（金叉）
- DIF线下穿DEA线 → 卖出信号（死叉）
- **红色柱**：多头力量增强
- **绿色柱**：空头力量增强

**RSI（相对强弱指标）**
- **>70**：超买区，股价可能过高，注意回调风险
- **<30**：超卖区，股价可能过低，有反弹机会
- **30-70之间**：正常区间

**布林带（Bollinger Bands）**
- 灰色阴影区域是价格波动的"正常范围"
- 价格触及上轨 → 可能超涨
- 价格触及下轨 → 可能超跌
- 带宽收窄 → 即将选择方向（大涨或大跌）

**支撑位/压力位**
- 支撑位：股价下跌时容易止跌的价格位置
- 压力位：股价上涨时容易受阻的价格位置

**量能分析**
- 放量：成交量显著增加，说明多空双方分歧加大
- 缩量：成交量减少，说明市场观望情绪浓厚
- 量价齐升：健康上涨形态
- 价涨量缩：上涨动力不足，注意风险
""")
else:
    # Show guide when no stock selected
    with st.sidebar:
        with st.expander("指标说明"):
            st.markdown("选择股票后可查看详细指标说明")

# Main content
if not selected_stock:
    st.info("在上方输入股票代码或名称，点击【搜索】开始分析")
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

# Detect mobile via session state heuristic (screen width check via columns)
# We use a simpler approach: always render mobile-friendly layout
is_mobile = True  # Streamlit doesn't expose viewport, so design for mobile-first

# Tabs
tab1, tab2, tab3 = st.tabs(["K线图 & 技术分析", "趋势分析", "走势预测"])

with tab1:
    st.subheader(f"{selected_stock['name']} ({selected_stock['code']}) K线图")
    st.caption("红蜡烛=上涨，绿蜡烛=下跌。彩色细线是均线，灰色阴影是布林带。双指缩放查看细节。")
    fig = create_main_chart(df)
    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    })

    # Recent data table
    st.subheader("近期行情")
    st.caption("最近20个交易日的行情数据，左滑可查看更多列。")
    display_cols = ["date", "open", "high", "low", "close", "volume", "pctChg"]
    show_cols = [c for c in display_cols if c in df.columns]
    recent = df[show_cols].tail(20).sort_values("date", ascending=False)
    recent = recent.copy()
    recent = recent.rename(columns={
        "date": "日期", "open": "开盘价", "high": "最高价",
        "low": "最低价", "close": "收盘价", "volume": "成交量", "pctChg": "涨跌幅",
    })
    if "涨跌幅" in recent.columns:
        recent["涨跌幅"] = recent["涨跌幅"].apply(lambda x: f"{x:.2f}%")
    if "成交量" in recent.columns:
        recent["成交量"] = recent["成交量"].apply(lambda x: f"{x:,.0f}")
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

    with st.expander("趋势判断怎么看？"):
        st.markdown("""
- **多头排列（看多）**：MA5 > MA10 > MA20 > MA60，短期均线在长期均线上方，上涨趋势明确
- **空头排列（看空）**：MA5 < MA10 < MA20 < MA60，卖压较大，下跌趋势明显
- **震荡整理**：均线互相交织，没有明确方向，建议观望
- **MA60位置**：价格在60日均线上方说明中期趋势偏多，下方偏空
""")

    st.divider()

    # Support & Resistance
    sr = find_support_resistance(df)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("支撑位")
        st.caption("下跌时可能止跌反弹的位置")
        if sr["support"]:
            for s in sr["support"]:
                st.write(f"  **{s:.2f}**")
        else:
            st.write("无明显支撑")
    with col2:
        st.subheader("压力位")
        st.caption("上涨时可能遇阻回落的位置")
        if sr["resistance"]:
            for r in sr["resistance"]:
                st.write(f"  **{r:.2f}**")
        else:
            st.write("无明显压力")

    st.divider()

    # Volume analysis
    st.subheader("量能分析")
    st.caption("成交量反映市场活跃程度，是判断趋势可靠性的重要参考。")
    vol_info = analyze_volume(df)
    vol_cols = st.columns(min(len(vol_info), 4))
    for i, (k, v) in enumerate(vol_info.items()):
        vol_cols[i % len(vol_cols)].metric(k, str(v))

    divergence = detect_volume_price_divergence(df)
    st.info(f"量价关系: {divergence}")

    with st.expander("量能分析怎么看？"):
        st.markdown("""
- **量比**：当日成交量与近期平均的比值。>1.5为放量，<0.5为缩量
- **量价齐升**：价格上涨+成交量放大 → 健康上涨
- **价涨量缩**：上涨动力不足，有回调风险
- **放量下跌**：恐慌性抛售，卖压大
- **缩量下跌**：跌势可能趋缓，接近底部
""")

    st.divider()

    # Key metrics
    st.subheader("关键指标汇总")
    st.caption("快速了解当前最重要的技术指标。")
    metrics = get_key_metrics(df)
    metric_cols = st.columns(min(len(metrics), 4))
    for i, (k, v) in enumerate(metrics.items()):
        metric_cols[i % len(metric_cols)].metric(k, v)

    with st.expander("关键指标怎么看？"):
        st.markdown("""
- **距20日高/低**：当前价格距离近期高/低点的涨跌幅
- **RSI状态**：>70超买，<30超卖，30-70正常
- **MACD柱**：红柱=多头力量，绿柱=空头力量
""")

with tab3:
    st.subheader(f"{selected_stock['name']} 走势预测")
    st.caption("基于历史数据的统计模型预测，仅供参考。")

    # Stack vertically on mobile (always stack since we design mobile-first)
    # ARIMA
    st.markdown("### ARIMA 时间序列预测")
    st.caption("根据历史价格走势推算未来趋势。红色虚线是预测走势，浅红色区域是波动范围。")
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
    st.plotly_chart(fig_forecast, use_container_width=True, config={
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    })

    last_close = df["close"].iloc[-1]
    pred_last = arima_result["forecast"][-1]
    change_pct = (pred_last - last_close) / last_close * 100
    st.metric(
        f"ARIMA {forecast_days}日预测",
        f"{pred_last:.2f}",
        f"{change_pct:+.2f}%",
    )
    st.caption(f"当前价 {last_close:.2f} → 预测价 {pred_last:.2f}，{'上涨' if change_pct > 0 else '下跌'} {abs(change_pct):.2f}%")

    st.divider()

    # ML
    st.markdown("### ML 机器学习预测")
    st.caption("综合技术指标、量能、动量等特征，预测未来涨跌方向。")
    with st.spinner("ML模型训练中..."):
        ml_result = ml_forecast(df, forecast_days)

    if ml_result.get("success", False):
        direction = ml_result["direction"]
        direction_icon = "📈" if direction == "看涨" else "📉"

        ml_col1, ml_col2, ml_col3 = st.columns(3)
        with ml_col1:
            st.metric("方向预测", f"{direction_icon} {direction}")
        with ml_col2:
            st.metric("置信度", f"{ml_result['confidence']:.1f}%")
        with ml_col3:
            st.metric("模型R²", f"{ml_result['r2_score']:.4f}")

        st.write("---")
        st.write("**预测收益率:**")
        ret_col1, ret_col2, ret_col3 = st.columns(3)
        with ret_col1:
            st.metric("5日", ml_result.get("5day_return", "N/A"))
        with ret_col2:
            st.metric("10日", ml_result.get("10day_return", "N/A"))
        with ret_col3:
            st.metric(f"{forecast_days}日", ml_result.get("20day_return", "N/A"))

        with st.expander("ML预测结果怎么看？"):
            st.markdown("""
- **方向预测**：模型认为未来走势是看涨还是看跌
- **置信度**：对预测结果的把握。>60%有参考价值，<40%参考意义较弱
- **模型R²**：历史拟合效果，越接近1越好
- 注意：ML预测基于历史规律，不能保证未来一定如此
""")
    else:
        st.warning(ml_result.get("message", "预测失败，数据不足"))

    st.divider()
    st.info("**两个模型怎么看？** ARIMA和ML都看涨 → 信号一致，参考性更强。结论矛盾 → 信号不明确，谨慎操作。")

# Disclaimer
st.markdown("---")
st.markdown("*免责声明：本工具仅供学习研究，不构成任何投资建议。股市有风险，投资需谨慎。*")
