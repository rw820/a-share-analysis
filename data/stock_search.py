import pandas as pd
import streamlit as st
from config import get_exchange_prefix
from datetime import datetime, timedelta


@st.cache_data(ttl=86400, show_spinner=False)
def load_stock_list() -> pd.DataFrame:
    # Try akshare first (works well from China)
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        df.columns = ["code", "name"]
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback: use baostock query_all_stock
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            return pd.DataFrame(columns=["code", "name"])
        date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        rs = bs.query_all_stock(date=date)
        data = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            if len(row) >= 3:
                # Fields: tradeStatus, code, code_name
                bs_code = row[1]
                name = row[2]
                if bs_code and name and bs_code.startswith(("sh.6", "sz.0", "sz.3")):
                    code = bs_code.split(".")[1]
                    data.append({"code": code, "name": name})
        bs.logout()
        if data:
            return pd.DataFrame(data)
    except Exception:
        try:
            bs.logout()
        except Exception:
            pass

    return pd.DataFrame(columns=["code", "name"])


@st.cache_data(ttl=86400, show_spinner=False)
def get_name_by_code(code: str) -> str:
    """Get stock Chinese name by code, using baostock as last resort."""
    bs_code = f"{get_exchange_prefix(code)}.{code}"
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            return code
        rs = bs.query_stock_basic(code=bs_code)
        name = code
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            if len(row) > 1 and row[1]:
                name = row[1]
                break
        bs.logout()
        return name
    except Exception:
        try:
            bs.logout()
        except Exception:
            pass
    return code


def search_stocks(query: str) -> list[dict]:
    query = query.strip()
    if not query:
        return []

    stock_list = load_stock_list()

    if not stock_list.empty:
        if query.isdigit() and len(query) == 6:
            matches = stock_list[stock_list["code"] == query]
        else:
            matches = stock_list[stock_list["name"].str.contains(query, na=False)]

        if not matches.empty:
            results = []
            for _, row in matches.head(20).iterrows():
                code = row["code"]
                results.append({
                    "code": code,
                    "name": row["name"],
                    "bs_code": f"{get_exchange_prefix(code)}.{code}",
                })
            return results

    # Last resort: direct code entry, try to get name from baostock
    if query.isdigit() and len(query) == 6:
        bs_code = f"{get_exchange_prefix(query)}.{query}"
        name = get_name_by_code(query)
        return [{"code": query, "name": name, "bs_code": bs_code}]

    return []
