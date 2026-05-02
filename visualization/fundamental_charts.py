"""
基本面可视化：估值走势图、财报趋势图、资金流向图
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_pe_history_chart(valuation_data: dict) -> go.Figure | None:
    """PE历史走势图 + 分位标注线"""
    if not valuation_data.get("success"):
        return None

    data = valuation_data["data"]
    pe_history = data.get("pe_history")

    if pe_history is None or len(pe_history) < 20:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=pe_history, mode="lines",
        name="PE", line=dict(color="#2C3E50", width=1.5),
    ))

    # 25%和75%分位线
    pct_25 = float(pd.Series(pe_history).quantile(0.25))
    pct_75 = float(pd.Series(pe_history).quantile(0.75))

    fig.add_hline(y=pct_25, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text=f"25%分位: {pct_25:.1f}")
    fig.add_hline(y=pct_75, line_dash="dash", line_color="green", opacity=0.5,
                  annotation_text=f"75%分位: {pct_75:.1f}")

    # 标记当前PE
    current_pe = data.get("pe")
    if current_pe is not None:
        fig.add_hline(y=current_pe, line_dash="dot", line_color="#E74C3C", line_width=2,
                      annotation_text=f"当前PE: {current_pe:.1f}")

    fig.update_layout(
        height=300,
        template="plotly_white",
        title="PE历史走势",
        title_font=dict(size=12),
        xaxis_title="交易日序号（从旧到新）",
        yaxis_title="PE",
        margin=dict(l=40, r=20, t=40, b=20),
        showlegend=False,
        dragmode="pan",
    )

    return fig


def create_financial_chart(financial_data: dict) -> go.Figure | None:
    """财报趋势图：柱状图(营收/净利润) + 折线(增长率)"""
    if not financial_data.get("success"):
        return None

    quarters = financial_data["data"].get("quarters", [])
    if len(quarters) < 2:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # X轴标签
    labels = [q.get("report_period", f"Q{i+1}") for i, q in enumerate(quarters)]

    revenues = [q.get("营业总收入") or 0 for q in quarters]
    profits = [q.get("归母净利润") or 0 for q in quarters]

    fig.add_trace(go.Bar(
        x=labels, y=revenues, name="营业收入",
        marker_color="#3498db", opacity=0.7,
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=labels, y=profits, name="归母净利润",
        marker_color="#2ecc71", opacity=0.7,
    ), secondary_y=False)

    fig.update_layout(
        height=300,
        template="plotly_white",
        title="营收与净利润趋势",
        title_font=dict(size=12),
        margin=dict(l=40, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        dragmode="pan",
        barmode="group",
    )
    fig.update_yaxes(title_text="金额（元）", secondary_y=False)

    return fig


def create_fund_flow_chart(fund_data: dict) -> go.Figure | None:
    """资金流向柱状图"""
    if not fund_data.get("success"):
        return None

    daily = fund_data["data"].get("daily_details", [])
    if not daily:
        return None

    dates = [d.get("date", "")[-10:] for d in daily]
    main_net = [d.get("main_net", 0) or 0 for d in daily]

    colors = ["red" if v >= 0 else "green" for v in main_net]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=main_net, marker_color=colors,
        opacity=0.8, name="主力净流入",
    ))

    fig.update_layout(
        height=250,
        template="plotly_white",
        title="主力资金净流入（日）",
        title_font=dict(size=12),
        xaxis_title="日期",
        yaxis_title="净流入（元）",
        margin=dict(l=40, r=20, t=40, b=20),
        dragmode="pan",
        showlegend=False,
    )

    return fig
