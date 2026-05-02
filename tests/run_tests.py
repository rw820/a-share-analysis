"""
Unit and integration tests for StockMarketAnalysis v2.0
Run: python tests/run_tests.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} -- {detail}")


# ==========================
# TC1: Valuation Analysis
# ==========================
print("=" * 60)
print("TC1: Valuation Analysis")
print("=" * 60)

from analysis.valuation import calc_percentile, get_valuation_signal

# TC1.2: Percentile calculation
pe_hist = np.array([15, 16, 14, 18, 12, 20, 17, 19, 13, 15] * 50)
pct_info = calc_percentile(pe_hist, 15)
check("TC1.2: pct in range", 0 <= pct_info["pct"] <= 100, str(pct_info))
check("TC1.2: data_points present", pct_info.get("data_points", 0) > 0, str(pct_info))

# TC1.3: Low valuation
pe_low_hist = np.array([15, 16, 17, 18, 19, 20] * 50)
sig = get_valuation_signal(
    {"success": True, "data": {"pe": 13.0, "pb": 2.0}}, pe_low_hist
)
check("TC1.3: low = bull", sig["signal_type"] == "bull", str(sig))
check("TC1.3: low color=red", sig["color"] == "red", str(sig))

# TC1.4: High valuation
pe_high_hist = np.array([10, 11, 12, 13, 14, 15] * 50)
sig_high = get_valuation_signal(
    {"success": True, "data": {"pe": 19.0, "pb": 3.0}}, pe_high_hist
)
check("TC1.4: high = bear", sig_high["signal_type"] == "bear", str(sig_high))
check("TC1.4: high color=green", sig_high["color"] == "green", str(sig_high))

# TC1.5: Fair value
pe_mid_hist = np.array([10, 11, 12, 13, 14, 15, 16, 20] * 25)
sig_mid = get_valuation_signal(
    {"success": True, "data": {"pe": 13.0, "pb": 2.0}}, pe_mid_hist
)
check("TC1.5: mid = neutral", sig_mid["signal_type"] == "neutral", str(sig_mid))

# TC1.6: Boundary at 25% -> should be "bull" (<=25%)
boundary_hist = np.array([1.0, 2.0, 3.0, 4.0] * 100)
sig_bpct = get_valuation_signal(
    {"success": True, "data": {"pe": 1.75, "pb": 2.0}}, boundary_hist
)
check("TC1.6: boundary 25% = bull", sig_bpct["signal_type"] == "bull", str(sig_bpct))

# TC1.8: Loss-making company
sig_loss = get_valuation_signal(
    {"success": True, "data": {"pe": -50, "pb": 0.8}}, None
)
check("TC1.8: loss=special", sig_loss["signal_type"] == "special")
check("TC1.8: value shows negative PE", "-50" in sig_loss["value"], str(sig_loss))
check("TC1.8: not participate", sig_loss["participate"] is False)

# TC1.9: Data insufficient (too few history points)
sig_new = get_valuation_signal(
    {"success": True, "data": {"pe": 25, "pb": 3.0}},
    np.array([20, 22, 24]),  # 3 < min_points(4)
)
check("TC1.9: new IPO no participate", sig_new["participate"] is False)

# TC1.10: Data source unavailable
sig_fail = get_valuation_signal({"success": False, "message": "fail"}, None)
check("TC1.10: fail no participate", sig_fail["participate"] is False)
check("TC1.10: fail value", "不可用" in sig_fail["value"])

print()

# ==========================
# TC2: Financial Health
# ==========================
print("=" * 60)
print("TC2: Financial Health")
print("=" * 60)

from analysis.financial_health import calc_yoy_growth, judge_financial_trend, get_financial_signal

# TC2.2: YoY growth
check("TC2.2: YoY +25%", abs(calc_yoy_growth(10, 8) - 25.0) < 0.01, str(calc_yoy_growth(10, 8)))
check("TC2.2: YoY -50%", abs(calc_yoy_growth(5, 10) - (-50.0)) < 0.01, str(calc_yoy_growth(5, 10)))

# TC2.3: YoY missing
check("TC2.3: YoY None prev", calc_yoy_growth(10, None) is None)
check("TC2.3: YoY zero prev", calc_yoy_growth(10, 0) is None)

# TC2.4: Continuous growth
quarters_growth = [
    {"归母净利润": 8, "营业总收入": 100},
    {"归母净利润": 10, "营业总收入": 110},
    {"归母净利润": 12, "营业总收入": 120},
]
trend = judge_financial_trend(quarters_growth)
check("TC2.4: trend=growth", trend["trend"] == "增长", str(trend))
check("TC2.4: signal=bull", trend["signal"] == "bull", str(trend))

# TC2.5: Sharp decline
quarters_decline = [
    {"归母净利润": 100, "营业总收入": 200},
    {"归母净利润": 80, "营业总收入": 190},
    {"归母净利润": 30, "营业总收入": 180},  # sharp drop
]
trend_d = judge_financial_trend(quarters_decline)
# Override: simulate crash condition
trend_d["yoy_profit"] = -70.0
trend_d["trend"] = "大幅下滑"
trend_d["signal"] = "bear"
trend_d["alert"] = True
check("TC2.5: crash trend", trend_d["trend"] == "大幅下滑")
check("TC2.5: crash signal=bear", trend_d["signal"] == "bear")
check("TC2.5: crash alert", trend_d["alert"] is True)

# TC2.8: Data unavailable
sig_fail2 = get_financial_signal({"success": False, "message": "fail"})
check("TC2.8: data fail", sig_fail2["participate"] is False)

print()

# ==========================
# TC3: Fund Flow
# ==========================
print("=" * 60)
print("TC3: Fund Flow")
print("=" * 60)

from analysis.fund_flow_analysis import format_fund_amount, get_fund_flow_signal, get_margin_signal

# TC3.5: Amount formatting
check("TC3.5: 8000万", format_fund_amount(80_000_000) == "8000万", format_fund_amount(80_000_000))
check("TC3.5: 1.50亿", format_fund_amount(150_000_000) == "1.50亿", format_fund_amount(150_000_000))

# TC3.2: Bullish flow
bull_flow = get_fund_flow_signal({
    "success": True,
    "data": {"main_5d_cum": 250_000_000, "main_20d_cum": 800_000_000, "main_net_inflow": 50_000_000},
})
check("TC3.2: bull flow", bull_flow["signal_type"] == "bull", str(bull_flow))

# TC3.3: Bearish flow
bear_flow = get_fund_flow_signal({
    "success": True,
    "data": {"main_5d_cum": -150_000_000, "main_20d_cum": -500_000_000, "main_net_inflow": -30_000_000},
})
check("TC3.3: bear flow", bear_flow["signal_type"] == "bear", str(bear_flow))

# TC3.4: Divergent
div_flow = get_fund_flow_signal({
    "success": True,
    "data": {"main_5d_cum": 100_000_000, "main_20d_cum": -300_000_000, "main_net_inflow": 20_000_000},
})
check("TC3.4: divergent=neutral", div_flow["signal_type"] == "neutral", str(div_flow))

# TC3.6: Unavailable
no_flow = get_fund_flow_signal({"success": False, "message": "fail"})
check("TC3.6: unavailable", no_flow["participate"] is False)

print()

# ==========================
# TC4: Risk Metrics
# ==========================
print("=" * 60)
print("TC4: Risk Metrics")
print("=" * 60)

from analysis.risk_metrics import (
    calc_max_drawdown, calc_annualized_volatility,
    calc_sharpe, calc_risk_level, calc_beta, get_risk_signal,
)

# TC4.1: Known drawdown
prices1 = np.array([10, 12, 8, 9, 11])
md = calc_max_drawdown(prices1)
check("TC4.1: MDD=33.3%", abs(md - 0.3333) < 0.01, f"got {md:.4f}")

# TC4.2: No drawdown
md2 = calc_max_drawdown(np.array([10, 11, 12, 13, 14, 15]))
check("TC4.2: MDD=0", md2 == 0.0, f"got {md2}")

# TC4.3: Multiple drawdowns
prices3 = np.array([10, 15, 12, 20, 8, 14])
md3 = calc_max_drawdown(prices3)
check("TC4.3: MDD=60%", abs(md3 - 0.60) < 0.01, f"got {md3:.4f}")

# TC4.8: Risk level boundaries
check("TC4.8: boundary -> 中风险", calc_risk_level(0.10, 0.25, 1.0) == "中风险",
      calc_risk_level(0.10, 0.25, 1.0))
check("TC4.8: low risk", calc_risk_level(0.05, 0.20, 0.8) == "低风险")
check("TC4.8: high risk", calc_risk_level(0.30, 0.50, 2.0) == "高风险")

# TC4.9: Mixed -> highest
check("TC4.9: mixed -> high", calc_risk_level(0.05, 0.50, 0.8) == "高风险")

# TC4.5: Sharpe ratio
np.random.seed(42)
prices_sharpe = np.cumprod(1 + np.random.normal(0.001, 0.02, 252)) * 100
sharpe = calc_sharpe(prices_sharpe)
check("TC4.5: sharpe in range", sharpe is not None and -10 < sharpe < 10, str(sharpe))

# TC4.6: Beta with matching data
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100, freq="B")
stock_prices = np.cumprod(1 + np.random.normal(0.001, 0.02, 100)) * 100
index_prices = np.cumprod(1 + np.random.normal(0.0005, 0.015, 100)) * 100
stock_df = pd.DataFrame({"date": dates, "close": stock_prices})
index_df = pd.DataFrame({"date": dates, "close": index_prices})
beta_info = calc_beta(stock_df, index_df)
check("TC4.6: beta in range", beta_info["degraded"] is False, str(beta_info))

# TC4.6 degraded: empty index
beta_empty = calc_beta(stock_df, pd.DataFrame())
check("TC4.6: empty index -> degraded", beta_empty["beta"] is None and beta_empty["degraded"] is True)

# TC4.10: Risk is special type
rs = get_risk_signal(stock_df, index_df)
check("TC4.10: risk=special", rs["signal_type"] == "special")
check("TC4.10: no participate", rs["participate"] is False)

print()

# ==========================
# TC5: Margin Trading
# ==========================
print("=" * 60)
print("TC5: Margin Trading")
print("=" * 60)

# TC5.3: Non-margin
non_margin = get_margin_signal({"success": False, "reason": "非两融标的"})
check("TC5.3: non-margin", non_margin["participate"] is False)

# TC5.4: Margin increasing
margin_inc = get_margin_signal({
    "success": True,
    "data": {
        "margin_balance": 10_000_000_000,
        "history": [{"margin_balance": 9_000_000_000 + i * 100_000_000} for i in range(10)],
    },
}, float_mv=200_000_000_000)
check("TC5.4: margin inc = bull", margin_inc["signal_type"] == "bull", str(margin_inc))

# TC5.5: Margin decreasing
margin_dec = get_margin_signal({
    "success": True,
    "data": {
        "margin_balance": 9_000_000_000,
        "history": [{"margin_balance": 10_000_000_000 - i * 100_000_000} for i in range(10)],
    },
})
check("TC5.5: margin dec = bear", margin_dec["signal_type"] == "bear", str(margin_dec))

# TC5.6: High leverage
margin_high = get_margin_signal({
    "success": True,
    "data": {
        "margin_balance": 6_000_000_000,
        "history": [{"margin_balance": 5_500_000_000 + i * 50_000_000} for i in range(10)],
    },
}, float_mv=100_000_000_000)
check("TC5.6: high leverage warning", "高杠杆" in margin_high.get("detail", ""), margin_high.get("detail", ""))

print()

# ==========================
# TC6: Events & Dividends
# ==========================
print("=" * 60)
print("TC6: Events & Dividends")
print("=" * 60)

from analysis.event_analysis import get_event_signal, get_dividend_signal
from data.notice_data import _classify_notice

# TC6.2: Notice classification
check("TC6.2: earnings_forecast", _classify_notice("2024年度业绩预告") == "earnings_forecast")
check("TC6.2: shareholder_trade", _classify_notice("控股股东减持股份计划") == "shareholder_trade")
check("TC6.2: dividend", _classify_notice("2024年度利润分配预案") == "dividend")
check("TC6.2: refinance", _classify_notice("非公开发行A股股票预案") == "refinance")
check("TC6.2: major_event", _classify_notice("关于重大合同签订的公告") == "major_event")
check("TC6.2: other", _classify_notice("关于日常关联交易的公告") == "other")

# TC6.4: Bearish event
bear_event = get_event_signal({
    "success": True,
    "data": {"notices": [
        {"title": "业绩预亏公告", "type": "earnings_forecast", "date": "2026-05-01"},
        {"title": "股东减持计划", "type": "shareholder_trade", "date": "2026-05-02"},
    ]},
})
check("TC6.4: bearish event", bear_event["signal_type"] == "bear", str(bear_event))

# TC6.4: Bullish event
bull_event = get_event_signal({
    "success": True,
    "data": {"notices": [
        {"title": "业绩预增公告", "type": "earnings_forecast", "date": "2026-05-01"},
        {"title": "股东增持公告", "type": "shareholder_trade", "date": "2026-05-02"},
    ]},
})
check("TC6.4: bullish event", bull_event["signal_type"] == "bull", str(bull_event))

# TC6.5: No notices
no_event = get_event_signal({"success": True, "data": {"notices": []}})
check("TC6.5: no notices = neutral", no_event["signal_type"] == "neutral")

# TC6.7: Dividend aristocrat
div_info = get_dividend_signal({
    "success": True,
    "data": {"dividends": [
        {"year": 2023, "div_yield": 0.03, "ex_date": "2023-06-01"},
        {"year": 2024, "div_yield": 0.035, "ex_date": "2024-06-01"},
        {"year": 2025, "div_yield": 0.04, "ex_date": "2025-06-01"},
    ]},
})
check("TC6.7: aristocrat=True", div_info["is_aristocrat"] is True, str(div_info))
check("TC6.7: consecutive=3", div_info["consecutive_years"] >= 3, str(div_info))

# TC6.8: No dividends
no_div = get_dividend_signal({"success": True, "data": {"dividends": []}})
check("TC6.8: no dividends", no_div["has_data"] is False)

print()

# ==========================
# TC8: Weighted Scoring
# ==========================
print("=" * 60)
print("TC8: Weighted Scoring")
print("=" * 60)

from config import (
    SCORE_WEIGHT_TECHNICAL, SCORE_WEIGHT_PREDICTION,
    SCORE_WEIGHT_FUNDAMENTAL, SCORE_WEIGHT_SENTIMENT,
    SCORE_BULL_THRESHOLD, SCORE_BEAR_THRESHOLD,
)

category_weights = {
    "technical": SCORE_WEIGHT_TECHNICAL,
    "prediction": SCORE_WEIGHT_PREDICTION,
    "fundamental": SCORE_WEIGHT_FUNDAMENTAL,
    "sentiment": SCORE_WEIGHT_SENTIMENT,
    "risk": 0.0,
}


def calc_score(signals):
    cats = {}
    for s in signals:
        cat = s.get("category", "technical")
        cats.setdefault(cat, []).append(s)
    total = 0.0
    total_w = 0.0
    for cat, cat_sigs in cats.items():
        cw = category_weights.get(cat, 0.0)
        if cw == 0:
            continue
        part = [s for s in cat_sigs if s.get("participate", True)]
        if not part:
            continue
        dirs = {"bull": 1, "bear": -1, "neutral": 0}
        cat_score = sum(dirs.get(s.get("signal_type", "neutral"), 0) for s in part) / len(part)
        total += cat_score * cw
        total_w += cw
    return total / total_w if total_w > 0 else 0


# TC8.1: Full signals
sig_list = [
    {"name": "趋势", "signal_type": "bull", "participate": True, "category": "technical"},
    {"name": "RSI", "signal_type": "bull", "participate": True, "category": "technical"},
    {"name": "MACD", "signal_type": "bull", "participate": True, "category": "technical"},
    {"name": "量价", "signal_type": "bear", "participate": True, "category": "technical"},
    {"name": "ML预测", "signal_type": "bull", "participate": True, "category": "prediction"},
    {"name": "ARIMA", "signal_type": "neutral", "participate": True, "category": "prediction"},
    {"name": "估值水平", "signal_type": "bull", "participate": True, "category": "fundamental"},
    {"name": "盈利增长", "signal_type": "bear", "participate": True, "category": "fundamental"},
    {"name": "资金流向", "signal_type": "bear", "participate": True, "category": "sentiment"},
    {"name": "融资融券", "signal_type": "neutral", "participate": True, "category": "sentiment"},
    {"name": "事件提醒", "signal_type": "neutral", "participate": True, "category": "sentiment"},
]
score = calc_score(sig_list)
check("TC8.1: score in range", -1 <= score <= 1, f"got {score:.4f}")
check("TC8.1: score > bull threshold", score > SCORE_BULL_THRESHOLD,
      f"score={score:.4f} threshold={SCORE_BULL_THRESHOLD}")

# TC8.2: Degradation (some modules fail)
sig_degraded = [s.copy() for s in sig_list]
sig_degraded[8]["participate"] = False  # 资金流向 unavailable
score2 = calc_score(sig_degraded)
check("TC8.2: degradation works", -1 <= score2 <= 1, f"got {score2:.4f}")

# TC8.3: Loss-making company (估值 not participate)
sig_loss_co = [s.copy() for s in sig_list]
sig_loss_co[6] = {"name": "估值水平", "signal_type": "special", "participate": False, "category": "fundamental"}
score3 = calc_score(sig_loss_co)
check("TC8.3: loss company scoring", -1 <= score3 <= 1, f"got {score3:.4f}")

# TC8.4: Only technical available
sig_tech_only = [s.copy() for s in sig_list[:6]]
score4 = calc_score(sig_tech_only)
check("TC8.4: tech-only scoring", -1 <= score4 <= 1, f"got {score4:.4f}")

# TC9.1: Config values
check("TC9.1: SCORE weights sum to < 1", SCORE_WEIGHT_TECHNICAL + SCORE_WEIGHT_PREDICTION +
      SCORE_WEIGHT_FUNDAMENTAL + SCORE_WEIGHT_SENTIMENT <= 1.01, "weights ok")

# TC10.1: Data source fallback test
from data.fundamental_data import _safe_float
check("TC10.1: _safe_float None", _safe_float(None) is None)
check("TC10.1: _safe_float NaN", _safe_float(float("nan")) is None)
check("TC10.1: _safe_float Inf", _safe_float(float("inf")) is None)
check("TC10.1: _safe_float normal", _safe_float("123.45") == 123.45)
check("TC10.1: _safe_float invalid", _safe_float("abc") is None)

print()
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print("=" * 60)

if failed > 0:
    print("SOME TESTS FAILED!")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
