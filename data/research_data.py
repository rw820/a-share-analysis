"""
机构研报数据获取
"""
import concurrent.futures
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime
from config import NETWORK_TIMEOUT

try:
    import streamlit as st
    HAS_ST = True
except ImportError:
    HAS_ST = False

RESEARCH_CACHE_TTL = 6 * 3600  # 研报缓存6小时


def _cache_decorator(ttl):
    if HAS_ST:
        return st.cache_data(ttl=ttl, show_spinner=False)
    else:
        return lambda f: f


def _call_with_timeout(fn, timeout: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except (concurrent.futures.TimeoutError, Exception):
            return None


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


@_cache_decorator(RESEARCH_CACHE_TTL)
def get_research_reports(bs_code: str) -> dict:
    """获取机构研报数据"""
    code = bs_code.split(".")[-1]

    try:
        def _fetch():
            return ak.stock_research_report_em(symbol=code)

        df = _call_with_timeout(_fetch, NETWORK_TIMEOUT)
        if df is None or df.empty:
            return {"success": False, "message": "暂无研报数据"}

        # Column positions are stable: 序号=0, 股票代码=1, 股票简称=2, 研报标题=3,
        # 评级=4, 机构=5, 近一个月个股数=6, 2026EPS=7, 2026PE=8,
        # 2027EPS=9, 2027PE=10, 2028EPS=11, 2028PE=12, 行业=13, 日期=14, PDF=15
        cols = df.columns.tolist()
        title_col = cols[3] if len(cols) > 3 else None
        rating_col = cols[4] if len(cols) > 4 else None
        org_col = cols[5] if len(cols) > 5 else None
        date_col = cols[14] if len(cols) > 14 else None
        ind_col = cols[13] if len(cols) > 13 else None
        pdf_col = cols[15] if len(cols) > 15 else None
        eps_cols = {2026: cols[7], 2027: cols[9], 2028: cols[11]} if len(cols) > 11 else {}
        pe_cols = {2026: cols[8], 2027: cols[10], 2028: cols[12]} if len(cols) > 12 else {}

        reports = []
        for _, row in df.head(30).iterrows():
            title = str(row[title_col]) if title_col and pd.notna(row[title_col]) else ""
            rating = str(row[rating_col]) if rating_col and pd.notna(row[rating_col]) else ""
            org = str(row[org_col]) if org_col and pd.notna(row[org_col]) else ""
            ind = str(row[ind_col]) if ind_col and pd.notna(row[ind_col]) else ""
            pdf = str(row[pdf_col]) if pdf_col and pd.notna(row[pdf_col]) else ""

            date_val = row[date_col] if date_col else ""
            if hasattr(date_val, "strftime"):
                date_str = date_val.strftime("%Y-%m-%d")
            else:
                date_str = str(date_val)

            eps_forecast = {}
            pe_forecast = {}
            for y in [2026, 2027, 2028]:
                ec = eps_cols.get(y)
                pc = pe_cols.get(y)
                if ec is not None:
                    eps_forecast[y] = _safe_float(row[ec])
                if pc is not None:
                    pe_forecast[y] = _safe_float(row[pc])

            if title:
                reports.append({
                    "title": title,
                    "rating": rating,
                    "org": org,
                    "date": date_str,
                    "industry": ind,
                    "eps_forecast": eps_forecast,
                    "pe_forecast": pe_forecast,
                    "pdf_url": pdf if pdf and pdf != "nan" else "",
                })

        if not reports:
            return {"success": False, "message": "暂无研报数据"}

        # Rating distribution
        rating_counts = {}
        for r in reports:
            rt = r["rating"]
            if rt:
                rating_counts[rt] = rating_counts.get(rt, 0) + 1
        total = sum(rating_counts.values())

        # Aggregate EPS forecasts from latest reports
        latest_eps = {}
        latest_pe = {}
        for y in [2026, 2027, 2028]:
            eps_vals = [r["eps_forecast"][y] for r in reports if r["eps_forecast"].get(y)]
            pe_vals = [r["pe_forecast"][y] for r in reports if r["pe_forecast"].get(y)]
            if eps_vals:
                latest_eps[y] = {
                    "mean": round(np.mean(eps_vals), 2),
                    "high": round(max(eps_vals), 2),
                    "low": round(min(eps_vals), 2),
                }
            if pe_vals:
                latest_pe[y] = {
                    "mean": round(np.mean(pe_vals), 2),
                    "high": round(max(pe_vals), 2),
                    "low": round(min(pe_vals), 2),
                }

        return {
            "success": True,
            "data": {
                "reports": reports,
                "rating_distribution": rating_counts,
                "rating_total": total,
                "eps_consensus": latest_eps,
                "pe_consensus": latest_pe,
            },
        }
    except Exception as e:
        return {"success": False, "message": f"研报获取失败: {str(e)}"}
