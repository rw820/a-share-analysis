"""
消息面数据获取：公告
"""
import concurrent.futures
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from config import DIVIDEND_CACHE_TTL, NETWORK_TIMEOUT

try:
    import streamlit as st
    HAS_ST = True
except ImportError:
    HAS_ST = False


def _cache_decorator(ttl):
    if HAS_ST:
        return st.cache_data(ttl=ttl, show_spinner=False)
    else:
        return lambda f: f


# ==================== 公告数据 ====================

def _call_with_timeout(fn, timeout: int):
    """Call a function with a timeout using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except (concurrent.futures.TimeoutError, Exception):
            return None


@_cache_decorator(DIVIDEND_CACHE_TTL)
def get_recent_notices(bs_code: str, days: int = 30) -> dict:
    """获取近期公告列表"""
    code = bs_code.split(".")[-1]
    try:
        # Try stock_individual_notice_report first, fall back to stock_notice_report
        def _fetch_individual():
            return ak.stock_individual_notice_report(security=code)

        df = _call_with_timeout(_fetch_individual, NETWORK_TIMEOUT)
        if df is None:
            # Try fallback
            def _fetch_all():
                result = ak.stock_notice_report()
                if result is not None and not result.empty:
                    col = _find_code_column(result)
                    if col:
                        mask = result[col].astype(str).str.contains(code, na=False)
                        return result[mask]
                return result
            df = _call_with_timeout(_fetch_all, NETWORK_TIMEOUT)

        if df is None or df.empty:
            return {"success": True, "data": {"notices": []}, "message": "近期无公告"}

        col_map = _infer_columns(df)
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        notices = []
        for _, row in df.iterrows():
            title = str(row.get(col_map.get("title", ""), ""))
            date_val = row.get(col_map.get("date", ""))
            url = str(row.get(col_map.get("url", ""), "")) if col_map.get("url") else ""
            if hasattr(date_val, "strftime"):
                date_str = date_val.strftime("%Y-%m-%d")
            else:
                date_str = str(date_val)
            if date_str < cutoff:
                continue
            if title:
                notices.append({
                    "title": title,
                    "date": date_str,
                    "type": _classify_notice(title),
                    "url": url if url and url != "nan" else "",
                })

        return {"success": True, "data": {"notices": notices}}
    except Exception as e:
        return {"success": False, "message": f"公告获取失败: {str(e)}"}


def _find_code_column(df: pd.DataFrame) -> str | None:
    """Find the stock code column in a DataFrame."""
    cols = [str(c).strip() for c in df.columns]
    for c in cols:
        if "代码" in c or "code" in c.lower():
            return c
    return cols[0] if cols else None


def _infer_columns(df: pd.DataFrame) -> dict:
    """推断公告DataFrame的列映射"""
    cols = [str(c).strip() for c in df.columns]
    result = {}
    for c in cols:
        if "公告标题" in c or "标题" in c:
            result["title"] = c
        elif "公告日期" in c or "日期" in c:
            result["date"] = c
        elif "网址" in c or "url" in c.lower() or "地址" in c or "链接" in c:
            result["url"] = c
    if "title" not in result:
        result["title"] = df.columns[2] if len(df.columns) > 2 else df.columns[0]
    if "date" not in result:
        result["date"] = df.columns[4] if len(df.columns) > 4 else df.columns[0]
    if "url" not in result:
        result["url"] = df.columns[5] if len(df.columns) > 5 else None
    return result


# ==================== 公告分类 ====================

NOTICE_KEYWORDS = {
    "earnings_forecast": ["业绩预告", "业绩预增", "业绩预减", "预告", "预计", "业绩修正"],
    "earnings_report": ["年度报告", "季度报告", "半年报", "年报", "季报", "业绩快报", "业绩报告"],
    "dividend": ["分红", "利润分配", "权益分派", "股息", "现金分红", "转增", "送股"],
    "refinance": ["增发", "非公开发行", "配股", "可转债", "定向增发", "发行股票"],
    "shareholder_trade": ["减持", "增持", "股份变动", "股东", "权益变动", "集中竞价"],
    "major_event": ["重大合同", "中标", "诉讼", "仲裁", "重组", "并购", "收购", "重大资产"],
}


def _classify_notice(title: str) -> str:
    """根据标题关键词分类公告"""
    for notice_type, keywords in NOTICE_KEYWORDS.items():
        for kw in keywords:
            if kw in title:
                return notice_type
    return "other"


