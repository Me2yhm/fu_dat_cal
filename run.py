import time
import polars as pl
import pandas as pd
import numpy as np
import functools
import sqlite3
from datetime import datetime, timedelta
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from numba import jit, njit, types
from numba.typed import Dict, List

from utils import (
    get_all_contracts,
    get_last_snapshot,
    get_last_trading_day,
    get_term,
    get_conn,
    timeit,
)

SYMBOL_IDX: Dict[str, int] = Dict.empty(
    key_type=types.unicode_type, value_type=types.int64
)


def get_tick_dataframe(date: str = "2024-07-19"):
    """
    获取Tick数据
    date 需要为交易日。
    """
    conn = get_conn()
    start_datetime = " ".join([get_last_trading_day(date), "16:00:00"])
    end_datetime = " ".join([date, "16:00:00"])
    fields = "symbol_id,datetime,position,current"
    sql = f"select {fields} from jq.`tick` where datetime between '{start_datetime}' and  '{end_datetime}' and match(symbol_id, '^[a-z]+\\d+.[A-Z]+$')"
    rows = conn.execute(sql, columnar=True)
    df = pl.DataFrame(rows, schema=fields.split(","))
    return df


@timeit
def get_symbols_tick(symbols: list, date: str = "2024-07-19"):
    """
    获取指定若干合约的Tick数据
    """
    conn = get_conn()
    start_datetime = " ".join([get_last_trading_day(date), "21:00:00"])
    end_datetime = " ".join([date, "16:00:00"])
    fields = "symbol_id,datetime,current,position"
    sql = f"select {fields} from jq.`tick` where datetime between '{start_datetime}' and  '{end_datetime}' and symbol_id in {tuple(symbols)} order by datetime asc;"
    rows = conn.execute(sql, columnar=True)
    df = pl.DataFrame(rows, schema=fields.split(","))
    return df


def get_1d_dat(condition: str = ""):
    """
    获取日线数据
    """
    conn = get_conn()
    fields = "symbol_id,datetime"
    sql = f"select {fields} from jq.`1d` {condition}"
    rows = conn.execute(sql, columnar=True)
    df = pl.DataFrame(rows, schema=fields.split(","))
    return df


def save_1d_dat(df: pl.DataFrame):
    """
    保存日线数据到sqlite数据库
    """
    conn = sqlite3.connect("C:\\用户\\newf4\\database\\future.db")
    cur = conn.cursor()
    pandas_df = df.to_pandas()
    pandas_df["datetime"] = pandas_df["datetime"].astype("datetime64[us]")
    pandas_df.to_sql("1d", conn, if_exists="append", index=False)
    conn.close()


@timeit
def get_tick_multithread(offsets: list, batch_size: int = 10000):
    """
    多线程处理Tick数据
    """
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for offset in offsets:
            futures.append(executor.submit(get_tick_dataframe, batch_size, offset))
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error fetching data for offset {offset}: {e}")
    return results


@timeit
def get_tick_multiprocess(total_rows, batch_size):
    offsets = range(0, total_rows, batch_size)
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        offset_chunks = [offsets[i::4] for i in range(4)]
        futures = [
            executor.submit(get_tick_multithread, chunk, batch_size)
            for chunk in offset_chunks
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def extract_future_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    提取期货数据
    """
    future_df = df[df["symbol_id"].str.contains(r"^[a-zA-Z]+\d{4}.[A-Z]+$")]
    print(f"Extracted future data for len {len(future_df)}")
    return future_df


def get_mayjor_contract(
    tick_df: pl.DataFrame, last_mayjor_contract: str
) -> pl.DataFrame:
    """
    获取主力合约tick数据
    """
    major_tick_df = pl.DataFrame(schema=tick_df.schema)
    mayjor_contract = last_mayjor_contract
    timp_gro = tick_df.group_by("datetime")
    for time, df in timp_gro:
        if len(df) > 1:
            df = df.sort("position", reverse=True)
            max_contract = df["symbol_id"][0]
            if max_contract == "" or get_term(max_contract) >= get_term(
                mayjor_contract
            ):
                mayjor_contract = max_contract
            major_tick_df.extend(df.head(1))
        else:
            pass


def group_by_product(df: pl.DataFrame):
    """
    按产品分组
    """
    df = df.with_columns(
        pl.col("symbol_id").str.extract(r"([a-z]+)", 1).alias("letter_part")
    )
    product_gro = df.group_by("letter_part")
    return product_gro


def save_db(df: pl.DataFrame, product_id: str):
    """
    保存数据到sqlite数据库
    """
    conn = sqlite3.connect("C:\\用户\\newf4\\database\\future.db")
    cur = conn.cursor()
    print(product_id)
    pandas_df = df.to_pandas()
    pandas_df.to_sql(product_id, conn, if_exists="append", index=False)
    conn.close()


def process_data(df: pl.DataFrame):
    """
    处理数据
    """
    # 按产品分组
    product_gro = group_by_product(df)
    # 保存数据到sqlite数据库
    for pro, dat in product_gro:
        dat.drop("letter_part")
        prcessed_dat = get_mayjor_contract(dat)
        save_db(prcessed_dat, pro[0])
        print(f"Processed {pro[0]}")


@timeit
def execute_1d_extract():
    df = get_1d_dat()
    # save_1d_dat(df)


@njit(nopython=True)
def creat_value_array(columns: int) -> np.ndarray:
    """
    创建index值数组
    """
    # 注意value数组的columns和snapshot数组的columns不一致
    return np.zeros((100000, columns), dtype=np.float64)


@njit(nopython=True)
def creat_snapshot_array(length: int, columns: int) -> np.ndarray:
    """
    创建快照数组
    """
    # 快照数组里面没有symbol_id和datetime列
    return np.zeros((length, columns), dtype=np.float64)


@timeit
@njit(nopython=True)
def creat_symbol_index(symbol_ids: list, index_dict: Dict[str, int]) -> dict:
    """
    创建symbol_id索引
    """
    for i, symbol_id in enumerate(symbol_ids):
        index_dict[symbol_id] = i
    return index_dict


@njit(nopython=True)
def cal_value(snapshot: np.ndarray, datetime: datetime):
    """
    根据snapshot计算指数value
    """
    value = np.zeros(snapshot.shape[1] + 1)
    value[0] = datetime
    position = snapshot[:, -1]
    value[1:] = (position @ snapshot) / np.sum(position)
    return value


@timeit
@njit(nopython=True)
def execute_single_pro(
    snap_shots: np.ndarray,
    tick_data: np.ndarray,
):
    """
    执行单个产品的处理
    开盘价与同花顺不同，怀疑是集合竞价导致？
    """
    values = creat_value_array(3)
    value_index = 0
    last_datetime = tick_data[0][1]  # datetime 精度为10ms
    # 循环之前几乎占据一般的时间，需要优化
    for row in tick_data:
        symbol_idx = row[0]
        new_datetime = row[1]
        if new_datetime != last_datetime:
            value = cal_value(snap_shots, last_datetime)
            values[value_index] = value
            value_index += 1
            last_datetime = new_datetime
        snap_shots[symbol_idx] = [
            row.current,
            row.position,
        ]  # symbol_idx会不会有缺失值？

    # 更新最后一个tick，但有一个问题，如果最后一个datetime恰好只有一个数据怎么办？
    value = cal_value(snap_shots, last_datetime)
    values[value_index] = value
    value_index += 1
    return values[:value_index]


@timeit
def main(date: str = "2024-07-19", product: str = "ag", exchange: str = "SHFE"):
    """
    主函数
    """
    symbol_ids = get_all_contracts(product, exchange, date)
    symbol_idx = creat_symbol_index(symbol_ids, SYMBOL_IDX)
    snap_shots = get_last_snapshot(symbol_ids, get_last_trading_day(date))
    tick_data = get_symbols_tick(symbol_ids, date).to_pandas(
        use_pyarrow_extension_array=True
    )  # tick_df 的列顺序需要和snapshot保持一致

    @timeit
    def test(tick_data):

        tick_data["symbol_id"] = tick_data["symbol_id"].apply(lambda x: symbol_idx[x])

        return tick_data

    tick_data = test(tick_data)
    tick_data["datetime"] = (
        tick_data["datetime"].apply(lambda x: x.timestamp()).round(1)
    )
    tick_data = tick_data.to_records(index=False, column_dtypes={"symbol_id": "int64"})
    values = execute_single_pro(snap_shots, tick_data)


if __name__ == "__main__":
    main()
    # main("2024-07-18", "pp", "DCE")
    # main("2024-07-17")
