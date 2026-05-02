"""
资金面分析：主力资金流向信号、融资融券信号
"""
import numpy as np
from config import MARGIN_TREND_DAYS, MARGIN_HIGH_LEVERAGE_RATIO


def format_fund_amount(amount: float) -> str:
    """格式化金额：>=1亿显示"亿"，<1亿显示"万" """
    if abs(amount) >= 1e8:
        return f"{amount / 1e8:.2f}亿"
    else:
        return f"{amount / 1e4:.0f}万"


def get_fund_flow_signal(fund_data: dict) -> dict:
    """生成资金流向信号"""
    if not fund_data.get("success"):
        return {
            "name": "资金流向", "value": "不可用", "color": "amber",
            "detail": "资金流向数据获取失败", "signal_type": "neutral",
            "participate": False,
        }

    d = fund_data["data"]
    main_5d = d.get("main_5d_cum", 0)
    main_20d = d.get("main_20d_cum", 0)

    if main_5d > 0 and main_20d > 0:
        signal_type, color, label = "bull", "red", "主力流入"
    elif main_5d < 0 and main_20d < 0:
        signal_type, color, label = "bear", "green", "主力流出"
    else:
        signal_type, color, label = "neutral", "amber", "分歧"

    detail = (
        f"近5日主力净流入: {format_fund_amount(main_5d)}\n"
        f"近20日主力净流入: {format_fund_amount(main_20d)}"
    )

    return {
        "name": "资金流向",
        "value": label,
        "color": color,
        "signal_type": signal_type,
        "detail": detail,
        "participate": True,
    }


def get_margin_signal(margin_data: dict, float_mv: float | None = None) -> dict:
    """生成融资融券信号"""
    if not margin_data.get("success"):
        reason = margin_data.get("reason", "数据获取失败")
        return {
            "name": "融资融券", "value": "不可用" if "非" not in str(reason) else "非两融",
            "color": "amber", "signal_type": "neutral",
            "detail": str(reason), "participate": False,
        }

    d = margin_data["data"]
    history = d.get("history", [])

    if len(history) < MARGIN_TREND_DAYS:
        return {
            "name": "融资融券", "value": "数据不足",
            "color": "amber", "signal_type": "neutral",
            "detail": f"仅{len(history)}天数据，无法判断趋势",
            "participate": False,
        }

    # 线性拟合斜率 + 近5日连续变化
    recent = history[-MARGIN_TREND_DAYS:]
    balances = [h.get("margin_balance") for h in recent if h.get("margin_balance") is not None]

    if len(balances) < 5:
        return {
            "name": "融资融券", "value": "数据不足",
            "color": "amber", "signal_type": "neutral",
            "detail": "融资余额数据不完整", "participate": False,
        }

    x = np.arange(len(balances))
    y = np.array(balances)
    slope = np.polyfit(x, y, 1)[0]

    consecutive_up = all(balances[i] < balances[i + 1] for i in range(-5, -1)) if len(balances) >= 5 else False
    consecutive_down = all(balances[i] > balances[i + 1] for i in range(-5, -1)) if len(balances) >= 5 else False

    detail = f"融资余额: {balances[-1]/1e8:.2f}亿"
    if float_mv is not None and float_mv > 0:
        margin_ratio = balances[-1] / float_mv
        detail += f"\n占流通市值: {margin_ratio:.1%}"
        if margin_ratio > MARGIN_HIGH_LEVERAGE_RATIO:
            detail += f"（⚠ 融资占比超过{MARGIN_HIGH_LEVERAGE_RATIO:.0%}，高杠杆风险）"

    if slope > 0 and consecutive_up:
        signal_type, color, label = "bull", "red", "持续增加"
    elif slope < 0 and consecutive_down:
        signal_type, color, label = "bear", "green", "持续减少"
    else:
        signal_type, color, label = "neutral", "amber", "震荡"

    return {
        "name": "融资融券",
        "value": label,
        "color": color,
        "signal_type": signal_type,
        "detail": detail,
        "participate": True,
    }
