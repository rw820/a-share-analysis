"""
消息面数据获取：公告
"""
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from config import DIVIDEND_CACHE_TTL

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

@_cache_decorator(DIVIDEND_CACHE_TTL)
def get_recent_notices(bs_code: str, days: int = 30) -> dict:
    """获取近期公告列表"""
    code = bs_code.split(".")[-1]
    try:
        # Try stock_individual_notice_report first, fall back to stock_notice_report
        try:
            df = ak.stock_individual_notice_report(security=code)
        except (AttributeError, TypeError):
            df = ak.stock_notice_report()
            if df is not None and not df.empty:
                code_col = _find_code_column(df)
                if code_col:
                    mask = df[code_col].astype(str).str.contains(code, na=False)
                    df = df[mask]

        if df is None or df.empty:
            return {"success": True, "data": {"notices": []}, "message": "近期无公告"}

        col_map = _infer_columns(df)
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        notices = []
        for _, row in df.iterrows():
            title = str(row.get(col_map.get("title", ""), ""))
            date_val = row.get(col_map.get("date", ""))
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
    if "title" not in result:
        result["title"] = df.columns[2] if len(df.columns) > 2 else df.columns[0]
    if "date" not in result:
        result["date"] = df.columns[4] if len(df.columns) > 4 else df.columns[0]
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


