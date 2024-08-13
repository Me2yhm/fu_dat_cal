from datetime import datetime, timedelta
import functools
import json
from pathlib import Path
import re
import sqlite3
import time

import numba
import pandas as pd
import polars as pl
import numpy as np
import clickhouse_driver

from numba.typed import List

parent_dir = Path(__file__).parent / "log"

JSON_FILE = parent_dir / "records.json"


def init_saved_records():
    """
    初始化保存的记录
    """
    conn = get_db_conn("future_1d")
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()
    saved_records = {row[0].split("_")[0]: [] for row in rows}
    with open(JSON_FILE, "w") as json_file:
        json.dump(saved_records, json_file)


def dump_records(records: dict):
    """
    保存记录到文件
    """
    with open(JSON_FILE, "w") as json_file:
        json.dump(records, json_file)


# 从文件读取 JSON 数据并反序列化为字典对象
def load_records() -> dict[str, list]:
    with open(JSON_FILE, "r") as json_file:
        data_from_file = json.load(json_file)
    return data_from_file


@functools.lru_cache
def get_conn():
    """
    获取ClickHouse连接
    """
    clickhouse_uri = "clickhouse://reader:d9f6ed24@172.16.7.30:9900/joinquant"
    conn = clickhouse_driver.Client.from_url(url=clickhouse_uri)
    return conn


@functools.lru_cache(maxsize=None)
def get_db_conn(db_name: str = "future_index") -> sqlite3.Connection:
    """
    获取sqlite数据库连接
    """
    conn = sqlite3.connect(f"C:\\用户\\newf4\\database\\{db_name}.db")
    return conn


def get_product_dict() -> dict[str, str]:
    """
    获得所有产品和交易所组成的字典
    """
    conn = get_db_conn("future_1d")
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()
    product_dict = {}
    for row in rows:
        product_name, exchange_name = row[0].split("_")
        if product_name not in product_dict:
            product_dict[product_name] = exchange_name
    return product_dict


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


def get_pro_dates(product: str, exchange: str) -> list:
    """
    获取指定品种的交易日列表
    """
    conn = get_db_conn("future_1d")
    query = f"SELECT distinct Date(datetime) as date FROM {product}_{exchange} order by datetime asc"
    df = pd.read_sql(query, conn, columns=["datetime"])
    return df["date"].tolist()


def get_last_trading_day(date: str) -> str:
    """
    获取上一个交易日
    """
    conn = get_db_conn("future_1d")
    sql = f"SELECT MAX(DATE(datetime)) as last_trading_day FROM a_DCE WHERE datetime < '{date} 00:00:00'"
    last_trading_day = conn.execute(sql).fetchall()[0][0]
    return last_trading_day


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


def get_last_major(
    product: str,
    exchange: str,
    date: str = "2024-07-19",
    fields: str = "symbol_id,datetime,close,open_interest",
) -> dict:
    """
    获取上一个主力合约
    """
    conn = get_conn()
    # 获得日线数据中主力合约和熟练合约的数据
    sql = f"""
            SELECT symbol_id, datetime,close, open_interest
            FROM (
                SELECT *, 
                    COUNT(*) OVER (PARTITION BY open_interest) AS cnt
                FROM (
                    SELECT {fields}
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


@timeit
def get_all_contracts(
    product: str, exchange: str, date: str = "2024-07-19"
) -> np.ndarray:
    """
    获取所有合约
    """
    conn = get_db_conn("future_1d")
    cursor = conn.cursor()
    sql = f"select symbol_id from {product}_{exchange} where datetime = '{date} 00:00:00';"
    rows = cursor.execute(sql).fetchall()
    symbol_ids = List.empty_list(numba.types.unicode_type)
    for row in rows:
        symbol_ids.append(row[0])
    return symbol_ids


def creat_snapshot_array(length: int, columns: int) -> np.ndarray:
    """
    创建快照数组
    """
    # 快照数组里面没有symbol_id和datetime列
    return np.zeros((length, columns), dtype=np.float64)


@timeit
def get_last_snapshot(
    symbol_ids: list,
    date: str = "2024-07-19",
    fields: str = "symbol_id,datetime,current,position",
) -> np.ndarray:
    """
    获取上一个快照数据
    date 需要为交易日。
    """
    conn = get_conn()
    start_datetime = " ".join([date, "15:00:00"])
    end_datetime = " ".join([date, "21:00:00"])
    snapshot = creat_snapshot_array(len(symbol_ids), len(fields.split(",")) - 2)
    idx = 0
    for symbol_id in symbol_ids:
        sql = f"select {fields} from jq.`tick` where datetime between '{start_datetime}' and  '{end_datetime}' and symbol_id = '{symbol_id}' order by datetime desc limit 1"
        row = conn.execute(sql)
        if not row:
            idx += 1
            continue
        snapshot[idx] = np.array(row[0][2:], dtype=np.float64)
        idx += 1
    return snapshot


if __name__ == "__main__":
    print(get_last_trading_day("2024-07-22"))
