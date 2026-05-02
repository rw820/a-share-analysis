"""
财务健康与成长性分析
"""
from config import FINANCIAL_TREND_QUARTERS, FINANCIAL_CRASH_THRESHOLD, FINANCIAL_SURGE_THRESHOLD


def calc_yoy_growth(current: float | None, previous: float | None) -> float | None:
    """计算同比增长率"""
    if current is None or previous is None or previous == 0:
        return None
    try:
        return (current - previous) / abs(previous) * 100.0
    except (ZeroDivisionError, TypeError):
        return None


def judge_financial_trend(quarters: list[dict]) -> dict:
    """判断财务趋势

    Args:
        quarters: 按时间顺序排列的季度财报数据列表，每项含 "归母净利润", "营业总收入" 等

    Returns:
        {"trend": str, "signal": str, "alert": bool, "yoy_revenue": float|None, "yoy_profit": float|None}
    """
    if len(quarters) < 2:
        return {"trend": "数据不足", "signal": "neutral", "alert": False,
                "yoy_revenue": None, "yoy_profit": None}

    # 尝试找到最近的同比增长（同季度比较：Q1 vs Q1-去年）
    latest_profit = quarters[-1].get("归母净利润")
    prev_profit = quarters[0].get("归母净利润")  # 最早的
    yoy_profit = calc_yoy_growth(latest_profit, prev_profit)

    latest_revenue = quarters[-1].get("营业总收入")
    prev_revenue = quarters[0].get("营业总收入")
    yoy_revenue = calc_yoy_growth(latest_revenue, prev_revenue)

    # 如果数据充足，用中间期做同比（假设季度数据 Q1/Q2/Q3/Q4 循环）
    if len(quarters) >= 5:
        prev_year_idx = len(quarters) - 5
        if prev_year_idx >= 0:
            prev_p = quarters[prev_year_idx].get("归母净利润")
            if prev_p is not None and prev_p != 0:
                p = calc_yoy_growth(latest_profit, prev_p)
                if p is not None:
                    yoy_profit = p
            prev_r = quarters[prev_year_idx].get("营业总收入")
            if prev_r is not None and prev_r != 0:
                r = calc_yoy_growth(latest_revenue, prev_r)
                if r is not None:
                    yoy_revenue = r

    # 判断趋势：近N期净利润是否持续增长/下滑
    profits = [q.get("归母净利润") for q in quarters[-FINANCIAL_TREND_QUARTERS:] if q.get("归母净利润") is not None]

    if len(profits) >= FINANCIAL_TREND_QUARTERS:
        all_rising = all(profits[i] < profits[i + 1] for i in range(len(profits) - 1))
        all_falling = all(profits[i] > profits[i + 1] for i in range(len(profits) - 1))

        if all_rising and yoy_profit is not None and yoy_profit > 0:
            trend = "增长"
            signal_type = "bull"
        elif all_falling and yoy_profit is not None and yoy_profit < 0:
            trend = "下滑"
            signal_type = "bear"
        else:
            trend = "波动"
            signal_type = "neutral"
    else:
        trend = "数据不足"
        signal_type = "neutral"

    alert = False
    if yoy_profit is not None and yoy_profit < FINANCIAL_CRASH_THRESHOLD * 100:
        trend = "大幅下滑"
        signal_type = "bear"
        alert = True

    return {
        "trend": trend,
        "signal": signal_type,
        "alert": alert,
        "yoy_revenue": round(yoy_revenue, 1) if yoy_revenue is not None else None,
        "yoy_profit": round(yoy_profit, 1) if yoy_profit is not None else None,
    }


def get_financial_signal(financial_data: dict) -> dict:
    """生成财务健康信号"""
    if not financial_data.get("success"):
        return {
            "name": "盈利增长", "value": "不可用", "color": "amber",
            "detail": "财务数据获取失败", "signal_type": "neutral",
            "participate": False,
        }

    quarters = financial_data["data"]["quarters"]
    trend_info = judge_financial_trend(quarters)

    if trend_info["trend"] == "数据不足":
        return {
            "name": "盈利增长", "value": "数据不足",
            "color": "amber", "signal_type": "neutral",
            "detail": "财报数据不足，无法判断趋势", "participate": False,
        }

    color_map = {"bull": "red", "bear": "green", "neutral": "amber"}
    color = color_map.get(trend_info["signal"], "amber")

    yoy_p = trend_info.get("yoy_profit")
    yoy_r = trend_info.get("yoy_revenue")

    detail_parts = []
    if yoy_r is not None:
        detail_parts.append(f"营收 YoY: {yoy_r:+.1f}%")
    if yoy_p is not None:
        detail_parts.append(f"净利 YoY: {yoy_p:+.1f}%")

    if trend_info["alert"]:
        detail_parts.append(f"业绩大幅下滑（净利YoY={yoy_p:+.1f}%），警报")
        detail_parts.append("可能意味着经营困难或重大一次性亏损")

    is_surge = yoy_p is not None and yoy_p > FINANCIAL_SURGE_THRESHOLD * 100
    if is_surge:
        detail_parts.append(f"净利增长超过{FINANCIAL_SURGE_THRESHOLD*100:.0f}%，可能为非经常性损益")

    return {
        "name": "盈利增长",
        "value": trend_info["trend"],
        "color": color,
        "signal_type": trend_info["signal"],
        "detail": "\n".join(detail_parts) if detail_parts else "暂无详细数据",
        "participate": True,
        "is_surge": is_surge,
    }
