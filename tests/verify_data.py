"""Verify baostock K-line data accuracy vs public financial websites"""
import baostock as bs

bs.login()
rs = bs.query_history_k_data_plus("sh.600519",
    "date,open,high,low,close,volume,amount,turn,pctChg",
    start_date="2026-04-30", end_date="2026-04-30", frequency="d", adjustflag="2")
rows = []
while (row := rs.next()):
    rows.append(row)
bs.logout()

r = rows[0]
print("Baostock data for 600519 on 2026-04-30:")
print(f"  open={r[1]}, high={r[2]}, low={r[3]}, close={r[4]}")
print(f"  volume={r[5]}, pctChg={r[7]}")
print()
print("Public data (eastmoney): open=1400.00, high=1401.17, low=1380.00, close=1384.79")
print("Public data (eastmoney): volume=52753, pctChg=-1.17")
print()
print("Match:")
print(f"  open match:   {r[1] == '1400.00'}")
print(f"  high match:   {r[2] == '1401.17'}")
print(f"  low match:    {r[3] == '1380.00'}")
print(f"  close match:  {r[4] == '1384.79'}")
print(f"  volume match: {r[5] == '52753'}")
print(f"  pctChg match: {r[7] == '-1.17'}")

# Also check the last few trading days
print()
print("Last 5 trading days:")
bs.login()
rs2 = bs.query_history_k_data_plus("sh.600519",
    "date,close,pctChg",
    start_date="2026-04-01", end_date="2026-05-03", frequency="d", adjustflag="2")
rows2 = []
while (row := rs2.next()):
    rows2.append(row)
bs.logout()
for r in rows2[-5:]:
    print(f"  {r[0]}: close={r[1]}, pctChg={r[2]}%")
