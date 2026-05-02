"""
基本面数据获取：估值、财务、分红、行业、指数K线
完全基于 baostock，不依赖 akshare（避开 eastmoney 网络不稳定问题）
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import (
    DATA_FETCH_TIMEOUT, FINANCIAL_FIELD_MAP,
    VALUATION_CACHE_TTL, FINANCIAL_CACHE_TTL, DIVIDEND_CACHE_TTL,
    PE_HISTORY_YEARS,
)

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


def _bs_login():
    """baostock 登录（确保已安装）"""
    import baostock as bs
    lg = bs.login()
    return bs, lg


# ==================== 估值数据（baostock） ====================

@_cache_decorator(VALUATION_CACHE_TTL)
def get_valuation_data(bs_code: str) -> dict:
    """计算个股估值 PE/PB/PS/市值

    数据来源: baostock query_profit_data (epsTTM, totalShare, roeAvg, netProfit)
    PE = close / epsTTM
    PB = (close * totalShare) / (netProfit / roeAvg)
    """
    bs, lg = _bs_login()
    if lg.error_code != "0":
        return {"success": False, "message": f"baostock 登录失败: {lg.error_msg}"}

    # 获取最新季度利润数据（epsTTM, totalShare等），逐季度回退
    now = datetime.now()
    cur_q = (now.month - 1) // 3 + 1
    cur_y = now.year

    profit_data = None
    rs = None
    for offset in range(4):  # try current + 3 previous quarters
        q = cur_q - offset
        y = cur_y
        if q <= 0:
            q += 4
            y -= 1
        rs = bs.query_profit_data(code=bs_code, year=y, quarter=q)
        if rs.error_code == "0":
            row = rs.get_row_data()
            if row:  # empty list [] when no data
                profit_data = row
                break

    if profit_data is None:
        bs.logout()
        return {"success": False, "message": "无法获取利润数据"}

    field_idx = {f: i for i, f in enumerate(rs.fields)}
    eps_ttm = _safe_float(profit_data[field_idx.get("epsTTM", -1)]) if "epsTTM" in field_idx else None
    total_share = _safe_float(profit_data[field_idx.get("totalShare", -1)]) if "totalShare" in field_idx else None
    roe_avg = _safe_float(profit_data[field_idx.get("roeAvg", -1)]) if "roeAvg" in field_idx else None
    net_profit = _safe_float(profit_data[field_idx.get("netProfit", -1)]) if "netProfit" in field_idx else None
    np_margin = _safe_float(profit_data[field_idx.get("npMargin", -1)]) if "npMargin" in field_idx else None
    gp_margin = _safe_float(profit_data[field_idx.get("gpMargin", -1)]) if "gpMargin" in field_idx else None

    # 获取最新收盘价
    rs_k = bs.query_history_k_data_plus(
        bs_code, "close", start_date=(now - timedelta(days=10)).strftime("%Y-%m-%d"),
        end_date=now.strftime("%Y-%m-%d"), frequency="d", adjustflag="2"
    )
    close_price = None
    if rs_k.error_code == "0":
        k_rows = []
        while (row := rs_k.get_row_data()):
            k_rows.append(row)
            if not rs_k.next():
                break
        if k_rows:
            close_price = _safe_float(k_rows[-1][0])

    if close_price is None:
        bs.logout()
        return {"success": False, "message": "无法获取最新股价"}

    # 计算 PE
    pe = round(close_price / eps_ttm, 2) if eps_ttm and eps_ttm > 0 else None

    # 计算 PB: PB = close / (每股净资产) = close / (netProfit / roeAvg / totalShare)
    # 净资产 = netProfit / roeAvg, 每股净资产 = 净资产 / totalShare
    pb = None
    if net_profit and roe_avg and total_share and roe_avg > 0 and total_share > 0:
        equity = net_profit / roe_avg
        book_per_share = equity / total_share
        if book_per_share > 0:
            pb = round(close_price / book_per_share, 2)

    # 计算市值
    total_mv = round(close_price * total_share, 2) if total_share else None
    float_share = _safe_float(profit_data[field_idx.get("liqaShare", -1)]) if "liqaShare" in field_idx else None
    float_mv = round(close_price * float_share, 2) if float_share and close_price else None

    # PE 历史数据（近5年按季度）— 必须在 logout 前调用
    pe_history = _get_pe_history_bs(bs, bs_code, total_share)

    bs.logout()

    return {
        "success": True,
        "data": {
            "pe": pe,
            "pb": pb,
            "ps": None,  # baostock 不提供营收
            "total_mv": total_mv,
            "float_mv": float_mv,
            "pe_history": pe_history,
            "eps_ttm": eps_ttm,
            "roe": roe_avg * 100 if roe_avg else None,
            "net_profit": net_profit,
            "np_margin": np_margin * 100 if np_margin else None,
            "gp_margin": gp_margin * 100 if gp_margin else None,
            "total_share": total_share,
        },
        "source": "baostock",
    }


def _get_pe_history_bs(bs, bs_code: str, total_share: float | None) -> np.ndarray | None:
    """通过历史季度利润数据计算 PE 历史序列"""
    if total_share is None:
        return None

    now = datetime.now()
    pe_list = []
    for year in range(now.year - PE_HISTORY_YEARS, now.year + 1):
        for quarter in [1, 2, 3, 4]:
            if year == now.year and quarter > (now.month - 1) // 3 + 1:
                continue
            try:
                rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                if rs.error_code != "0":
                    continue
                while (row := rs.get_row_data()):
                    idx = {f: i for i, f in enumerate(rs.fields)}
                    eps = _safe_float(row[idx.get("epsTTM", -1)]) if "epsTTM" in idx else None
                    if eps and eps > 0:
                        pe_list.append(eps)
                    if not rs.next():
                        break
            except Exception:
                continue

    if len(pe_list) < 4:
        return None

    return np.array(pe_list)


@_cache_decorator(VALUATION_CACHE_TTL)
def get_pe_history(bs_code: str) -> dict:
    """获取 PE 历史序列"""
    val = get_valuation_data(bs_code)
    if val.get("success") and val["data"].get("pe_history") is not None:
        return {"success": True, "data": {"pe_values": val["data"]["pe_history"], "dates": None}}
    return {"success": False, "message": "无法生成PE历史序列"}


# ==================== 财务数据（baostock） ====================

@_cache_decorator(FINANCIAL_CACHE_TTL)
def get_financial_data(bs_code: str) -> dict:
    """获取多季度财务指标（baostock profit_data + balance_data + cash_flow_data + growth_data）"""
    bs, lg = _bs_login()
    if lg.error_code != "0":
        return {"success": False, "message": f"baostock 登录失败: {lg.error_msg}"}

    now = datetime.now()
    quarters = []

    for year in range(now.year - 3, now.year + 1):
        for q in [1, 2, 3, 4]:
            if year == now.year and q > (now.month - 1) // 3 + 1:
                continue

            entry = {"report_period": f"{year}Q{q}"}
            has_data = False

            # 利润数据
            try:
                rs = bs.query_profit_data(code=bs_code, year=year, quarter=q)
                if rs.error_code == "0":
                    row = rs.get_row_data()
                    if row:
                        idx = {f: i for i, f in enumerate(rs.fields)}
                        np_val = _safe_float(row[idx["netProfit"]]) if "netProfit" in idx else None
                        npm_val = _safe_float(row[idx["npMargin"]]) if "npMargin" in idx else None
                        rev_val = _safe_float(row[idx["MBRevenue"]]) if "MBRevenue" in idx else None
                        # Derive revenue from profit/margin if MBRevenue is empty
                        if rev_val is None and np_val and npm_val and npm_val > 0:
                            rev_val = np_val / npm_val

                        entry.update({
                            "归母净利润": np_val,
                            "ROE": _safe_float(row[idx["roeAvg"]]) * 100 if "roeAvg" in idx and row[idx["roeAvg"]] else None,
                            "毛利率": _safe_float(row[idx["gpMargin"]]) * 100 if "gpMargin" in idx and row[idx["gpMargin"]] else None,
                            "净利率": npm_val * 100 if npm_val else None,
                            "营业总收入": rev_val,
                            "epsTTM": _safe_float(row[idx["epsTTM"]]) if "epsTTM" in idx else None,
                        })
                        has_data = True
            except Exception:
                pass

            # 资产负债数据
            try:
                rs_b = bs.query_balance_data(code=bs_code, year=year, quarter=q)
                if rs_b.error_code == "0":
                    row = rs_b.get_row_data()
                    if row:
                        bidx = {f: i for i, f in enumerate(rs_b.fields)}
                        lta = _safe_float(row[bidx["liabilityToAsset"]]) if "liabilityToAsset" in bidx else None
                        entry["资产负债率"] = lta * 100 if lta else None
                        has_data = True
            except Exception:
                pass

            # 现金流数据
            try:
                rs_c = bs.query_cash_flow_data(code=bs_code, year=year, quarter=q)
                if rs_c.error_code == "0":
                    row = rs_c.get_row_data()
                    if row:
                        cidx = {f: i for i, f in enumerate(rs_c.fields)}
                        cfo_np = _safe_float(row[cidx["CFOToNP"]]) if "CFOToNP" in cidx else None
                        if cfo_np and entry.get("归母净利润"):
                            entry["经营活动现金流净额"] = cfo_np * entry["归母净利润"]
                        has_data = True
            except Exception:
                pass

            # 增长率数据
            try:
                rs_g = bs.query_growth_data(code=bs_code, year=year, quarter=q)
                if rs_g.error_code == "0":
                    row = rs_g.get_row_data()
                    if row:
                        gidx = {f: i for i, f in enumerate(rs_g.fields)}
                        if "YOYNI" in gidx:
                            entry["净利润同比"] = _safe_float(row[gidx["YOYNI"]]) * 100
                        if "YOYEquity" in gidx:
                            entry["净资产同比"] = _safe_float(row[gidx["YOYEquity"]]) * 100
                        has_data = True
            except Exception:
                pass

            if has_data:
                quarters.append(entry)

    bs.logout()

    if len(quarters) < 2:
        return {"success": False, "message": "财务数据不足（少于2期）"}

    return {
        "success": True,
        "data": {"quarters": quarters},
        "source": "baostock",
    }


# ==================== 分红数据（baostock） ====================

@_cache_decorator(DIVIDEND_CACHE_TTL)
def get_dividend_history(bs_code: str) -> dict:
    """获取分红记录（baostock query_dividend_data）"""
    bs, lg = _bs_login()
    if lg.error_code != "0":
        return {"success": False, "message": f"baostock 登录失败: {lg.error_msg}"}

    now = datetime.now()
    dividends = []

    for year in range(now.year - 3, now.year + 1):
        try:
            rs = bs.query_dividend_data(code=bs_code, year=year)
            if rs.error_code != "0":
                continue
            while (row := rs.get_row_data()):
                idx = {f: i for i, f in enumerate(rs.fields)}
                div_per_share = _safe_float(row[idx["dividCashPsBeforeTax"]]) if "dividCashPsBeforeTax" in idx else None
                div_date = row[idx["dividPayDate"]] if "dividPayDate" in idx else ""
                dividends.append({
                    "ex_date": div_date,
                    "div_per_share": div_per_share,
                    "div_yield": None,  # 需要收盘价才能算
                    "year": year,
                })
                if not rs.next():
                    break
        except Exception:
            continue

    bs.logout()
    return {"success": True, "data": {"dividends": dividends},
            "message": "无分红记录" if not dividends else ""}


# ==================== 行业数据（baostock） ====================

def get_industry_info(bs_code: str) -> dict:
    """获取个股行业分类（baostock query_stock_industry）"""
    bs, lg = _bs_login()
    if lg.error_code != "0":
        return {"success": False, "message": f"baostock 登录失败: {lg.error_msg}"}

    rs = bs.query_stock_industry(code=bs_code)
    industry = ""
    industry_class = ""
    if rs.error_code == "0":
        while (row := rs.get_row_data()):
            idx = {f: i for i, f in enumerate(rs.fields)}
            industry = row[idx["industry"]] if "industry" in idx else ""
            industry_class = row[idx["industryClassification"]] if "industryClassification" in idx else ""
            if not rs.next():
                break

    bs.logout()

    return {
        "success": True if industry else False,
        "data": {
            "industry": industry,
            "industry_classification": industry_class,
        } if industry else {},
        "source": "baostock",
    }


# ==================== 指数数据 ====================

def get_index_kline(index_code: str = "sz.399300", start_date: str = None,
                    end_date: str = None, freq: str = "d") -> pd.DataFrame:
    """获取指数K线数据（baostock，默认沪深300）"""
    try:
        import baostock as bs
    except ImportError:
        return pd.DataFrame()

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        lg = bs.login()
        if lg.error_code != "0":
            return pd.DataFrame()

        rs = bs.query_history_k_data_plus(
            index_code, "date,close,pctChg",
            start_date=start_date, end_date=end_date, frequency=freq, adjustflag="1"
        )
        if rs.error_code != "0":
            bs.logout()
            return pd.DataFrame()

        rows = []
        while (row := rs.get_row_data()):
            rows.append(row)
            if not rs.next():
                break
        bs.logout()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["date", "close", "pctChg"])
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = df["close"].astype(float)
        df["pctChg"] = df["pctChg"].astype(float)
        return df
    except Exception:
        try:
            bs.logout()
        except Exception:
            pass
        return pd.DataFrame()
