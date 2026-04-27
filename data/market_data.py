import pandas as pd
import baostock as bs
import streamlit as st
from config import CACHE_TTL_SECONDS


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_kline_data(code: str, start_date: str, end_date: str, frequency: str = "d") -> pd.DataFrame:
    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
    try:
        lg = bs.login()
        if lg.error_code != "0":
            return pd.DataFrame()
        rs = bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag="2",
        )
        data = []
        while rs.error_code == "0" and rs.next():
            data.append(rs.get_row_data())
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=fields.split(","))
        numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        try:
            bs.logout()
        except Exception:
            pass


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_recent_kline(code: str, days: int = 500, frequency: str = "d") -> pd.DataFrame:
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=days)
    return get_kline_data(code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), frequency)
