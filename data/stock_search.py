import pandas as pd
import streamlit as st
from config import get_exchange_prefix


@st.cache_data(ttl=86400, show_spinner=False)
def load_stock_list() -> pd.DataFrame:
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        df.columns = ["code", "name"]
        return df
    except Exception:
        return pd.DataFrame(columns=["code", "name"])


def search_stocks(query: str) -> list[dict]:
    query = query.strip()
    if not query:
        return []

    stock_list = load_stock_list()

    if stock_list.empty:
        # Fallback: allow direct code entry even if akshare fails
        if query.isdigit() and len(query) == 6:
            return [{"code": query, "name": query, "bs_code": f"{get_exchange_prefix(query)}.{query}"}]
        return []

    if query.isdigit() and len(query) == 6:
        matches = stock_list[stock_list["code"] == query]
    else:
        matches = stock_list[stock_list["name"].str.contains(query, na=False)]

    results = []
    for _, row in matches.head(20).iterrows():
        code = row["code"]
        results.append({
            "code": code,
            "name": row["name"],
            "bs_code": f"{get_exchange_prefix(code)}.{code}",
        })
    return results
