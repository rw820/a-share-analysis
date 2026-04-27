import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_main_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="K线", increasing_line_color="red", decreasing_line_color="green",
    ), row=1, col=1)

    # Moving Averages
    ma_colors = {"MA5": "#FF6B6B", "MA10": "#4ECDC4", "MA20": "#45B7D1", "MA60": "#96CEB4", "MA120": "#FFEAA7"}
    for ma, color in ma_colors.items():
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[ma], name=ma,
                line=dict(color=color, width=1),
            ), row=1, col=1)

    # Bollinger Bands
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["BB_upper"], name="BB上轨",
            line=dict(color="gray", width=1, dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["BB_lower"], name="BB下轨",
            line=dict(color="gray", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
        ), row=1, col=1)

    # Volume
    colors = ["red" if c >= o else "green" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"], name="成交量",
        marker_color=colors, opacity=0.7,
    ), row=2, col=1)

    # MACD
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["MACD"], name="DIF",
            line=dict(color="#FF6B6B", width=1),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["MACD_signal"], name="DEA",
            line=dict(color="#4ECDC4", width=1),
        ), row=3, col=1)
        hist_colors = ["red" if v >= 0 else "green" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df["date"], y=df["MACD_hist"], name="MACD柱",
            marker_color=hist_colors, opacity=0.7,
        ), row=3, col=1)

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["RSI"], name="RSI",
            line=dict(color="#9B59B6", width=1.5),
        ), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1, opacity=0.5)

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=40, r=20, t=30, b=20),
        dragmode="pan",
    )

    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)

    return fig


def create_forecast_chart(df: pd.DataFrame, forecast_dates, forecast_values, lower_bound, upper_bound) -> go.Figure:
    fig = go.Figure()

    # Historical prices
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"], name="历史价格",
        line=dict(color="#2C3E50", width=2),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values, name="预测价格",
        line=dict(color="#E74C3C", width=2, dash="dash"),
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=upper_bound, name="置信上界",
        line=dict(color="rgba(231,76,60,0.3)", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=lower_bound, name="置信下界",
        line=dict(color="rgba(231,76,60,0.3)", width=1),
        fill="tonexty", fillcolor="rgba(231,76,60,0.1)",
    ))

    fig.update_layout(
        height=400,
        template="plotly_white",
        title="走势预测",
        title_font=dict(size=14),
        xaxis_title="日期",
        yaxis_title="价格",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=40, r=20, t=50, b=20),
        dragmode="pan",
    )

    return fig
