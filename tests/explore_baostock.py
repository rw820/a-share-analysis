"""Explore baostock financial data APIs"""
import baostock as bs
import sys

lg = bs.login()
if lg.error_code != '0':
    print(f'Login failed: {lg.error_msg}')
    sys.exit(1)
print('Login OK\n')


def show_fields(rs, limit=10):
    """Print fields and first row of a result set"""
    print(f'  Fields: {rs.fields}')
    row = rs.get_row_data()
    if row:
        n = min(limit, len(rs.fields))
        for i in range(n):
            print(f'    {rs.fields[i]}: {row[i]}')
    else:
        print('    (no data)')
    print()


def show_all_rows(rs, max_rows=3):
    """Print up to max_rows of data"""
    print(f'  Fields: {rs.fields}')
    count = 0
    while (row := rs.get_row_data()):
        print(f'  Row {count+1}:')
        for i, f in enumerate(rs.fields):
            print(f'    {f}: {row[i]}')
        count += 1
        if count >= max_rows:
            break
        if not rs.next():
            break
    print()


# 1. Stock basic
print('=== query_stock_basic ===')
rs = bs.query_stock_basic(code='sh.600519')
show_fields(rs)

# 2. Profit data
print('=== query_profit_data (2025 Q1) ===')
try:
    rs = bs.query_profit_data(code='sh.600519', year=2025, quarter=1)
    show_fields(rs)
except Exception as e:
    print(f'  Error: {e}\n')

# 3. Balance data
print('=== query_balance_data (2025 Q1) ===')
try:
    rs = bs.query_balance_data(code='sh.600519', year=2025, quarter=1)
    show_fields(rs)
except Exception as e:
    print(f'  Error: {e}\n')

# 4. Operation data
print('=== query_operation_data (2025 Q1) ===')
try:
    rs = bs.query_operation_data(code='sh.600519', year=2025, quarter=1)
    show_fields(rs)
except Exception as e:
    print(f'  Error: {e}\n')

# 5. Growth data
print('=== query_growth_data (2025 Q1) ===')
try:
    rs = bs.query_growth_data(code='sh.600519', year=2025, quarter=1)
    show_fields(rs)
except Exception as e:
    print(f'  Error: {e}\n')

# 6. DuPont data
print('=== query_dupont_data (2025 Q1) ===')
try:
    rs = bs.query_dupont_data(code='sh.600519', year=2025, quarter=1)
    show_fields(rs)
except Exception as e:
    print(f'  Error: {e}\n')

# 7. Dividend data
print('=== query_dividend_data (2024) ===')
try:
    rs = bs.query_dividend_data(code='sh.600519', year=2024)
    show_all_rows(rs)
except Exception as e:
    print(f'  Error: {e}\n')

# 8. Industry
print('=== query_stock_industry ===')
try:
    rs = bs.query_stock_industry(code='sh.600519')
    show_fields(rs)
except Exception as e:
    print(f'  Error: {e}\n')

# 9. Forecast report
print('=== query_forecast_report (2025 Q1) ===')
try:
    rs = bs.query_forecast_report(code='sh.600519', year=2025, quarter=1)
    show_fields(rs)
except Exception as e:
    print(f'  Error: {e}\n')

# 10. Performance express
print('=== query_performance_express_report (2025 Q1) ===')
try:
    rs = bs.query_performance_express_report(code='sh.600519', year=2025, quarter=1)
    show_fields(rs)
except Exception as e:
    print(f'  Error: {e}\n')

bs.logout()
print('Done!')
