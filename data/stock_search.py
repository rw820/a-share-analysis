import pandas as pd
import streamlit as st
from config import get_exchange_prefix


@st.cache_data(ttl=86400, show_spinner=False)
def load_stock_list_akshare() -> pd.DataFrame:
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        df.columns = ["code", "name"]
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["code", "name"])


@st.cache_data(ttl=86400, show_spinner=False)
def get_name_from_baostock(bs_code: str) -> str:
    """Get Chinese stock name via baostock query_stock_basic."""
    import baostock as bs
    try:
        lg = bs.login()
        if lg.error_code != "0":
            return ""
        rs = bs.query_stock_basic(code=bs_code)
        # Returns: [code, code_name, ipoDate, outDate, type, status]
        name = ""
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
        return ""


def search_stocks(query: str) -> list[dict]:
    query = query.strip()
    if not query:
        return []

    stock_list = load_stock_list_akshare()

    # akshare has data — use it for search
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

    # akshare empty or no match — fallback to baostock for single code
    if query.isdigit() and len(query) == 6:
        bs_code = f"{get_exchange_prefix(query)}.{query}"
        name = get_name_from_baostock(bs_code)
        return [{"code": query, "name": name or query, "bs_code": bs_code}]

    return []
