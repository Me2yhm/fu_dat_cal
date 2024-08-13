import polars as pl
import pandas as pd
import numpy as np
import functools
import sqlite3
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from numba import njit, types
from numba.typed import Dict

from utils import (
    dump_records,
    get_all_contracts,
    get_last_snapshot,
    get_last_trading_day,
    get_pro_dates,
    get_product_dict,
    get_term,
    get_conn,
    get_db_conn,
    load_records,
    timeit,
)
from logger import Logger, setup_logger

SYMBOL_IDX: Dict[str, int] = Dict.empty(
    key_type=types.unicode_type, value_type=types.int64
)
RECORDS = load_records()


@functools.lru_cache(maxsize=None)  # 使用 LRU 缓存
def get_symbol_index(symbol: str):
    return SYMBOL_IDX[symbol]


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
def get_symbols_tick(
    symbols: list,
    date: str = "2024-07-19",
    fields: str = "symbol_id,datetime,current,position",
):
    """
    获取指定若干合约的Tick数据
    """
    conn = get_conn()
    start_datetime = " ".join([get_last_trading_day(date), "21:00:00"])
    end_datetime = " ".join([date, "16:00:00"])
    sql = f"select {fields} from jq.`tick` where datetime between '{start_datetime}' and  '{end_datetime}' and symbol_id in {tuple(symbols)};"
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


def save_1d_dat(df: pl.DataFrame, product: str, exchange: str):
    """
    保存日线数据到sqlite数据库
    """
    conn = sqlite3.connect("C:\\用户\\newf4\\database\\future_1d.db")
    cur = conn.cursor()
    pandas_df = df.to_pandas()
    pandas_df["datetime"] = pandas_df["datetime"].astype("datetime64[us]")
    pandas_df.to_sql(f"{product}_{exchange}", conn, if_exists="append", index=False)
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
        pl.col("symbol_id").str.extract(r"(^[a-zA-Z]+)", 1).alias("pro_part")
    )
    df = df.with_columns(
        pl.col("symbol_id").str.extract(r"([A-Z]+)$", 1).alias("exchange")
    )
    product_gro = df.group_by("pro_part")
    return product_gro


@timeit
def save_db(
    df: pl.DataFrame | pd.DataFrame,
    product_id: str,
    date: str = "all",
    db_name: str = "future_index",
):
    """
    保存数据到sqlite数据库
    """
    conn = get_db_conn(db_name)
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    df.to_sql(product_id, conn, if_exists="append", index=False)
    Logger.info(f"saved {product_id} of {date} data to {db_name} successfully")


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
        save_db(prcessed_dat, pro[0], date="all")
        print(f"Processed {pro[0]}")


@timeit
def execute_1d_extract():
    condition = "where match(symbol_id,'^[a-zA-Z]+\\d+.CZCE$') and symbol_id not like '%8888%' and symbol_id not like '%9999%';"
    df = get_1d_dat(condition)
    product_gro = group_by_product(df)
    logg = setup_logger("day", "1d_extract.log")
    for pro, dat in product_gro:
        exchange = dat["exchange"][0]
        dat = dat.drop("pro_part")
        dat = dat.sort("datetime")
        save_1d_dat(dat, pro[0], exchange)
        logg.info(f"Saved {pro[0]} of {exchange} data to future_1d.db successfully")


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
def creat_symbol_index(symbol_ids: np.ndarray, index_dict: Dict[str, int]) -> dict:
    """
    创建symbol_id索引
    """
    for i, symbol_id in enumerate(symbol_ids):
        # symbol_id = f"{symbol_id}"
        index_dict[symbol_id] = i
    return index_dict


@njit(nopython=True)
def cal_value(snapshot: np.ndarray, datetime: float, high: float, low: float):
    """
    根据snapshot计算指数value
    """
    value = np.zeros(snapshot.shape[1] + 4)
    value[1] = datetime
    position = snapshot[:, -1]
    value[2:-2] = (position @ snapshot) / np.sum(position)
    current = value[2]
    if current >= high or high == 0:
        high = current
    if current <= low or low == 0:
        low = current
    value[-2] = high
    value[-1] = low
    return value, high, low


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
    values = creat_value_array(snap_shots.shape[1] + 4)
    value_index = 0
    last_datetime = tick_data[0][1]  # datetime 精度为10ms
    high = low = 0
    # 循环之前几乎占据一般的时间，需要优化
    for row in tick_data:
        symbol_idx = row[0]
        new_datetime = row[1]
        if new_datetime != last_datetime:
            value, high, low = cal_value(snap_shots, last_datetime, high, low)
            values[value_index] = value
            value_index += 1
            last_datetime = new_datetime
        snap_shots[symbol_idx] = [
            row.current,
            row.a1_v,
            row.a1_p,
            row.b1_v,
            row.b1_p,
            row.position,
        ]  # symbol_idx会不会有缺失值？

    # 更新最后一个tick，但有一个问题，如果最后一个datetime恰好只有一个数据怎么办？
    value, high, low = cal_value(snap_shots, last_datetime, high, low)
    values[value_index] = value
    value_index += 1
    return values[:value_index]


@timeit
def save_1d_index(
    date: str = "2024-07-19", product: str = "ag", exchange: str = "SHFE"
):
    """
    主函数
    """
    global SYMBOL_IDX
    if date in RECORDS[product]:
        Logger.warning(f"{date} {product} of {exchange} data has been processed")
        return
    # 新合约的处理：空行填充
    fields = "symbol_id,datetime,current,a1_v,a1_p,b1_v,b1_p,position"
    symbol_ids = get_all_contracts(product, exchange, date)
    tick_data = (
        get_symbols_tick(symbol_ids, date, fields)
        .to_pandas(use_pyarrow_extension_array=True)
        .sort_values(by="datetime")
    )  # tick_df 的列顺序需要和snapshot保持一致
    snap_shots = get_last_snapshot(symbol_ids, get_last_trading_day(date), fields)
    SYMBOL_IDX = creat_symbol_index(symbol_ids, SYMBOL_IDX)
    # 为什么main在第一次调用时很慢？如果symbol_idx不用全局变量会更快？
    tick_data["symbol_id"] = tick_data["symbol_id"].apply(get_symbol_index)
    tick_data["datetime"] = (
        tick_data["datetime"].apply(lambda x: x.timestamp()).round(1)
    )
    tick_data = tick_data.to_records(index=False, column_dtypes={"symbol_id": "int64"})
    values = execute_single_pro(snap_shots, tick_data)
    index_df = pd.DataFrame(
        values,
        columns=["symbol_id", "datetime"] + fields.split(",")[2:] + ["high", "low"],
    )
    index_df["datetime"] = index_df["datetime"].astype("datetime64[s]")
    index_df["symbol_id"] = f"{product}8888.{exchange}"
    # save_db(index_df, product, date)
    # RECORDS[product].append(date)
    # dump_records(RECORDS)


def process_single_product(product: str, exchange: str):
    """
    处理单个产品
    """
    date_lst = get_pro_dates(product, exchange)
    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for date in date_lst:
            futures.append(executor.submit(save_1d_index, date, product, exchange))
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing {product} {exchange} {date}: {e}")
    return results


@timeit
def main():
    """
    主函数
    """
    prodct_dct = get_product_dict()
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_product, product, exchange)
            for product, exchange in prodct_dct.items()
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    return results


if __name__ == "__main__":
    execute_1d_extract()
