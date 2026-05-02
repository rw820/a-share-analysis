"""
资金面数据获取：资金流向、融资融券
"""
import time
import concurrent.futures
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import akshare as ak
from config import FUND_FLOW_CACHE_TTL, MARGIN_CACHE_TTL

try:
    import streamlit as st
    HAS_ST = True
except ImportError:
    HAS_ST = False

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


def _cache_decorator(ttl):
    if HAS_ST:
        return st.cache_data(ttl=ttl, show_spinner=False)
    else:
        return lambda f: f


def _retry_akshare(fn, name: str):
    """Retry akshare calls up to MAX_RETRIES times with exponential backoff."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            return result, None
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
    return None, last_err


# ==================== 资金流向 ====================

@_cache_decorator(FUND_FLOW_CACHE_TTL)
def get_fund_flow(bs_code: str) -> dict:
    """获取个股资金流向日频数据"""
    code = bs_code.split(".")[-1]
    try:
        df, err = _retry_akshare(
            lambda: ak.stock_individual_fund_flow(stock=code, market="sh" if bs_code.startswith("sh") else "sz"),
            "stock_individual_fund_flow"
        )
        if err is not None:
            return {"success": False, "message": f"资金流向获取失败(重试{MAX_RETRIES}次): {str(err)}"}
        if df is None or df.empty:
            return {"success": False, "message": "资金流向数据为空"}

        cols = df.columns.tolist()

        def _find_col(keywords: list[str]) -> str | None:
            for kw in keywords:
                for c in cols:
                    if kw in str(c):
                        return c
            return None

        main_in_col = _find_col(["主力净流入", "主力净额"])
        retail_in_col = _find_col(["散户净流入", "散户净额"])
        date_col = _find_col(["日期", "date"])

        if main_in_col is None:
            return {"success": False, "message": "无法识别资金流字段"}

        def sum_recent(col_name: str, n: int, sign: float = 1.0) -> float:
            vals = df[col_name].tail(n).apply(lambda x: sign * _safe_float(x)).dropna()
            return float(vals.sum()) if len(vals) > 0 else 0.0

        main_daily = sum_recent(main_in_col, 1)
        main_5d = sum_recent(main_in_col, 5)
        main_20d = sum_recent(main_in_col, 20)

        retail_daily = sum_recent(retail_in_col, 1) if retail_in_col else 0.0
        retail_5d = sum_recent(retail_in_col, 5) if retail_in_col else 0.0

        daily_details = []
        for _, row in df.tail(30).iterrows():
            detail = {
                "date": str(row.get(date_col, "")) if date_col else "",
                "main_net": _safe_float(row.get(main_in_col)) if main_in_col else 0,
            }
            if retail_in_col:
                detail["retail_net"] = _safe_float(row.get(retail_in_col))
            daily_details.append(detail)

        return {
            "success": True,
            "data": {
                "main_net_inflow": main_daily,
                "retail_net_inflow": retail_daily,
                "main_5d_cum": main_5d,
                "main_20d_cum": main_20d,
                "retail_5d_cum": retail_5d,
                "daily_details": daily_details,
            },
        }
    except Exception as e:
        return {"success": False, "message": f"资金流向获取失败: {str(e)}"}


# ==================== 融资融券 ====================

MARGIN_API_TIMEOUT = 10  # seconds per margin API call

def _call_with_timeout(fn, timeout: int):
    """Call a function with a timeout using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except (concurrent.futures.TimeoutError, Exception):
            return None

def _get_margin_for_date(date_str: str, code: str, is_sh: bool) -> dict | None:
    """Get margin data for a specific stock on a specific date."""
    def _fetch():
        if is_sh:
            return ak.stock_margin_detail_sse(date=date_str)
        else:
            return ak.stock_margin_detail_szse(date=date_str)

    try:
        df = _call_with_timeout(_fetch, MARGIN_API_TIMEOUT)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Columns: 信用交易日期, 标的证券代码, 标的证券简称, 融资买入额, 融资偿还额, 融资余额, 融券卖出量, 融券偿还量, 融券余量
    cols = df.columns.tolist()
    code_col = cols[1] if len(cols) > 1 else None
    if code_col is None:
        return None

    row = df[df[code_col].astype(str).str.strip() == code]
    if row.empty:
        return None

    r = row.iloc[0]
    return {
        "date": str(r[cols[0]]) if len(cols) > 0 else date_str,
        "margin_balance": _safe_float(r[cols[5]]) if len(cols) > 5 else None,
        "margin_buy": _safe_float(r[cols[3]]) if len(cols) > 3 else None,
        "short_volume": _safe_float(r[cols[8]]) if len(cols) > 8 else None,
    }


@_cache_decorator(MARGIN_CACHE_TTL)
def get_margin_data(bs_code: str) -> dict:
    """获取融资融券数据，自动区分 SSE/SZSE"""
    code = bs_code.split(".")[-1]
    is_sh = bs_code.startswith("sh")
    today = datetime.now()

    try:
        history = []
        latest = None
        for days_back in range(10):
            check_date = (today - timedelta(days=days_back)).strftime("%Y%m%d")
            entry = _get_margin_for_date(check_date, code, is_sh)
            if entry is not None:
                if latest is None:
                    latest = entry
                history.append(entry)
                if len(history) >= 2:
                    break

        if latest is None:
            return {"success": False, "reason": "非两融标的"}

        history.sort(key=lambda x: x["date"])

        return {
            "success": True,
            "data": {
                "margin_balance": latest["margin_balance"],
                "margin_buy": latest["margin_buy"],
                "short_volume": latest["short_volume"],
                "history": history,
            },
            "source": "sse" if is_sh else "szse",
        }
    except Exception as e:
        return {"success": False, "reason": f"获取失败: {str(e)}"}


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (ValueError, TypeError):
        return None
