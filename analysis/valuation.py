"""
估值分析：PE/PB分位计算、估值信号生成
支持 baostock EPS 历史数据（反向计算 PE 分位）
"""
import numpy as np
from config import VALUATION_PERCENTILE_LOW, VALUATION_PERCENTILE_HIGH


def calc_percentile(values: np.ndarray, current_val: float, invert: bool = False,
                    min_points: int = 4) -> dict:
    """计算当前值在历史序列中的分位数

    Args:
        values: 历史数值序列
        current_val: 当前值
        invert: True=反转分位（用于 EPS→PE 转换，高EPS→低PE分位）
        min_points: 最少数据点

    Returns:
        {"pct": float|None, "data_points": int}
    """
    if values is None or len(values) < min_points:
        return {"pct": None, "data_points": len(values) if values is not None else 0}

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < min_points:
        return {"pct": None, "data_points": len(values)}

    pct = float(np.sum(values < current_val) / len(values) * 100)
    if invert:
        pct = 100.0 - pct

    # 限制在 [0, 100]
    pct = max(0.0, min(100.0, pct))

    return {"pct": round(pct, 1), "data_points": len(values)}


def get_valuation_signal(valuation_data: dict, pe_history: np.ndarray | None) -> dict:
    """生成估值信号

    当 pe_history 来自 baostock 时，值为 EPS 而非 PE（需反转向量）。

    Returns signal dict:
        {"name", "value", "color", "detail", "signal_type", "participate"}
    """
    if not valuation_data.get("success"):
        return {
            "name": "估值水平", "value": "不可用", "color": "amber",
            "detail": "估值数据获取失败", "signal_type": "neutral",
            "participate": False,
        }

    data = valuation_data["data"]
    pe = data.get("pe")
    pb = data.get("pb")
    source = valuation_data.get("source", "")

    # 亏损企业：PE 为 None 或负数
    if pe is None or pe <= 0:
        pe_str = f"{pe:.1f}" if pe is not None and pe < 0 else "亏损"
        return {
            "name": "估值水平", "value": pe_str,
            "color": "amber", "signal_type": "special",
            "detail": f"PE不可用(亏损), PB={pb:.2f}" if pb else "企业处于亏损状态",
            "participate": False,
        }

    # baostock 来源: pe_history 实为 EPS 值，需要反向
    invert = (source == "baostock")
    current_val = data.get("eps_ttm") if invert else pe

    pct_info = calc_percentile(pe_history, current_val, invert=invert) if pe_history is not None else {"pct": None, "data_points": 0}
    pct = pct_info.get("pct")
    n_points = pct_info.get("data_points", 0)

    if pct is None:
        detail = f"PE={pe:.2f}"
        if pb is not None:
            detail += f", PB={pb:.2f}"
        detail += f"\n历史数据不足（仅{n_points}个数据点）"
        return {
            "name": "估值水平", "value": "数据不足",
            "color": "amber", "signal_type": "neutral",
            "detail": detail,
            "participate": False,
        }

    # 基于分位判断
    if pct <= VALUATION_PERCENTILE_LOW:
        color, signal_type, label = "red", "bull", "低估"
    elif pct >= VALUATION_PERCENTILE_HIGH:
        color, signal_type, label = "green", "bear", "高估"
    else:
        color, signal_type, label = "amber", "neutral", "合理"

    # 详情
    detail = f"PE={pe:.2f}, 处于历史{pct:.0f}%分位（{label}）"
    if pb is not None:
        detail += f"\nPB={pb:.2f}"
    eps = data.get("eps_ttm")
    if eps is not None:
        detail += f"\nEPS(TTM)={eps:.2f}"
    detail += f"\n数据来源: {source}（{n_points}期历史）"

    return {
        "name": "估值水平",
        "value": f"{label} {pct:.0f}%分位",
        "color": color,
        "signal_type": signal_type,
        "detail": detail,
        "participate": True,
    }
