from datetime import datetime, timedelta
import functools
import re
import time

import polars as pl
import numpy as np
import clickhouse_driver

from numba.typed import List


@functools.lru_cache
def get_conn():
    """
    获取ClickHouse连接
    """
    clickhouse_uri = "clickhouse://reader:d9f6ed24@172.16.7.30:9900/joinquant"
    conn = clickhouse_driver.Client.from_url(url=clickhouse_uri)
    return conn


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 调用被装饰的函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算运行时间
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result  # 返回函数的结果

    return wrapper


def get_last_trading_day(date: str) -> str:
    """
    获取上一个交易日
    """
    conn = get_conn()
    sql = f"SELECT toDate('{date}') - INTERVAL 1 DAY"
    last_trading_day = conn.execute(sql)[0][0]
    return last_trading_day.strftime("%Y-%m-%d")


def get_term(code: str) -> int:
    """
    get the term of future
    """

    pattern = r"[a-zA-Z]+(\d{4})\.[a-zA-Z]+"
    match = re.search(pattern, code)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Invalid code format. code may not be a future code.")


def get_exchange(code: str) -> str:
    """
    get the exchange of future
    """
    pattern = r"[a-zA-Z]+(\d{4})\.[a-zA-Z]+"
    match = re.search(pattern, code)
    if match:
        return code.split(".")[-1].upper()
    else:
        raise ValueError("Invalid code format. code may not be a future code.")


def next_term(term: int) -> str:
    """
    get the next term of future
    """
    str_term = str(term)
    date_term = datetime(
        year=int("".join(["20", str_term[:2]])),
        month=int("".join(str_term[2:4])),
        day=1,
    )
    next_term = date_term + timedelta(days=31)
    return "".join([str(next_term.year)[-2:], str(next_term.month).zfill(2)])


def get_last_close_dat(product: str, date: str) -> pl.DataFrame:
    """
    get the last close dat of the product
    """
    pass


def get_last_major(product: str, exchange: str, date: str = "2024-07-19") -> dict:
    """
    获取上一个主力合约
    """
    conn = get_conn()
    fields = "symbol_id,datetime,current,position"
    # 获得日线数据中主力合约和熟练合约的数据
    sql = f"""
            SELECT symbol_id, datetime,close, open_interest
            FROM (
                SELECT *, 
                    COUNT(*) OVER (PARTITION BY open_interest) AS cnt
                FROM (
                    SELECT symbol_id, datetime,close, open_interest
                    FROM jq.`1d`
                    WHERE symbol_id LIKE '{product}____.{exchange.upper()}'
                    AND symbol_id NOT LIKE '%8888%'
                    AND datetime = '{date} 00:00:00'
                ) AS subquery
            ) AS main_query
            WHERE cnt > 1;

    """
    rows = conn.execute(sql)
    for row in rows:
        if "9999" in row[0]:
            last_tick = dict(zip(fields.split(","), row))
        else:
            last_major = dict(zip(fields.split(","), row))

    return last_major, last_tick


def get_last_secondery(product: str, date: str) -> str:
    """
    get the last secondery of the product
    """
    pass


def get_nearest_hour(dt):
    """
    get the nearest hour of the given datetime object
    """
    hour = dt.hour
    minute = dt.minute
    if minute >= 30:
        hour += 1
    nearest_hour = dt.replace(hour=hour % 24, minute=0, second=0, microsecond=0)

    return nearest_hour


def get_all_contracts(product: str, exchange: str, date: str = "2024-07-19") -> list:
    """
    获取所有合约
    """
    conn = get_conn()
    sql = f"select symbol_id from jq.`1d` where datetime = '{date} 00:00:00'  and symbol_id like '{product}____.{exchange.upper()}';"
    rows = conn.execute(sql)
    symbol_ids = [row[0] for row in rows if get_term(row[0]) not in [9999, 8888, 7777]]
    return List(symbol_ids)


def creat_snapshot_array(length: int, columns: int) -> np.ndarray:
    """
    创建快照数组
    """
    # 快照数组里面没有symbol_id和datetime列
    return np.zeros((length, columns), dtype=np.float64)


def get_last_snapshot(symbol_ids: list, date: str = "2024-07-19") -> np.ndarray:
    """
    获取上一个快照数据
    date 需要为交易日。
    """
    conn = get_conn()
    start_datetime = " ".join([date, "15:00:00"])
    end_datetime = " ".join([date, "21:00:00"])
    fields = "symbol_id,datetime,current,position"
    snapshot = creat_snapshot_array(len(symbol_ids), len(fields.split(",")) - 2)
    idx = 0
    for symbol_id in symbol_ids:
        sql = f"select {fields} from jq.`tick` where datetime between '{start_datetime}' and  '{end_datetime}' and symbol_id = '{symbol_id}' order by datetime desc limit 1"
        row = conn.execute(sql)
        if not row:
            continue
        snapshot[idx] = np.array(row[0][2:], dtype=np.float64)
        idx += 1
    return snapshot


if __name__ == "__main__":
    print(get_last_trading_day("2024-07-22"))
