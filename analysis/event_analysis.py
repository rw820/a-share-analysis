"""
消息面分析：事件信号、分红信号
"""
from config import DIVIDEND_HIGH_YIELD, DIVIDEND_CONSECUTIVE_YEARS


def get_event_signal(notice_data: dict) -> dict:
    """从公告列表中生成事件信号"""
    if not notice_data.get("success"):
        return {
            "name": "事件提醒", "value": "不可用", "color": "amber",
            "detail": "公告数据获取失败", "signal_type": "neutral",
            "participate": False,
        }

    notices = notice_data["data"].get("notices", [])
    if not notices:
        return {
            "name": "事件提醒", "value": "无重大事件",
            "color": "amber", "signal_type": "neutral",
            "detail": "近期无重要公告", "participate": True,
        }

    # 统计各类事件
    type_counts = {}
    type_examples = {}
    for n in notices:
        t = n.get("type", "other")
        type_counts[t] = type_counts.get(t, 0) + 1
        if t not in type_examples:
            type_examples[t] = n.get("title", "")

    bullish = 0
    bearish = 0

    # 业绩预告：预增/扭亏=偏多，预减/首亏/续亏=偏空
    if "earnings_forecast" in type_counts:
        for n in notices:
            if n.get("type") == "earnings_forecast":
                title = n.get("title", "")
                if any(w in title for w in ["预增", "扭亏", "大增", "增长"]):
                    bullish += 1
                elif any(w in title for w in ["预减", "首亏", "续亏", "下降", "亏损"]):
                    bearish += 1

    if "shareholder_trade" in type_counts:
        for n in notices:
            if n.get("type") == "shareholder_trade":
                title = n.get("title", "")
                if "减持" in title:
                    bearish += 1
                elif "增持" in title:
                    bullish += 1

    if "major_event" in type_counts:
        for n in notices:
            if n.get("type") == "major_event":
                title = n.get("title", "")
                if any(w in title for w in ["中标", "合同", "签约"]):
                    bullish += 1
                elif any(w in title for w in ["诉讼", "仲裁", "处罚"]):
                    bearish += 1

    # 综合方向
    if bullish > bearish:
        signal_type, color, label = "bull", "red", "偏多"
    elif bearish > bullish:
        signal_type, color, label = "bear", "green", "偏空"
    else:
        signal_type, color, label = "neutral", "amber", "中性"

    # 构建详情
    detail_parts = []
    for t, cnt in type_counts.items():
        type_name = {
            "earnings_forecast": "业绩预告", "earnings_report": "业绩报告",
            "dividend": "分红预案", "refinance": "增发预案",
            "shareholder_trade": "股东增减持", "major_event": "重大事件",
            "other": "其他",
        }.get(t, t)
        detail_parts.append(f"{type_name}: {cnt}条")

    detail = "\n".join(detail_parts) if detail_parts else "近期无重要公告"

    return {
        "name": "事件提醒",
        "value": f"{label}（{len(notices)}条公告）",
        "color": color,
        "signal_type": signal_type,
        "detail": detail,
        "participate": True,
    }


def get_dividend_signal(dividend_data: dict) -> dict:
    """从分红记录生成信号（用于UI展示，不单独参与评分）"""
    if not dividend_data.get("success"):
        return {"has_data": False, "message": "分红数据不可用"}

    dividends = dividend_data["data"].get("dividends", [])
    if not dividends:
        return {"has_data": False, "message": "近3年无分红记录"}

    # 计算年均股息率
    yields = [d.get("div_yield") for d in dividends if d.get("div_yield") is not None]
    avg_yield = sum(yields) / len(yields) if yields else 0

    # 估算连续分红年数
    years = sorted(set(d.get("year") for d in dividends if d.get("year") is not None))
    consecutive = 1
    for i in range(len(years) - 1, 0, -1):
        if years[i] - years[i - 1] == 1:
            consecutive += 1
        else:
            break

    is_aristocrat = avg_yield > DIVIDEND_HIGH_YIELD and consecutive >= DIVIDEND_CONSECUTIVE_YEARS

    return {
        "has_data": True,
        "dividends": dividends,
        "avg_yield": avg_yield,
        "consecutive_years": consecutive,
        "is_aristocrat": is_aristocrat,
        "message": (
            f"⛽ 高股息现金奶牛（年均股息率{avg_yield:.2%}，连续分红{consecutive}年）"
            if is_aristocrat
            else f"年均股息率{avg_yield:.2%}，连续分红{consecutive}年"
        ),
    }
