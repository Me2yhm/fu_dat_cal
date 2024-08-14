import polars as pl
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from numba import njit, types

from utils import (
    get_all_contracts,
    get_last_snapshot,
    get_last_trading_day,
    get_pro_dates,
    get_product_dict,
    get_symbols_tick,
    insert_records,
    is_processed,
    save_db,
    timeit,
)
from logger import Logger


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
    执行单个产品一天的指数数据的处理
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
    record = {"product": product, "exchange": exchange, "date": date}
    if is_processed(record):
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
    symbol_ids = tick_data["symbol_id"].unique()
    snap_shots = get_last_snapshot(symbol_ids, get_last_trading_day(date), fields)
    tick_data["symbol_id"] = tick_data["symbol_id"].astype("category").cat.codes
    tick_data["datetime"] = tick_data["datetime"].astype("int64").round(-7) / 1e6
    tick_data = tick_data.to_records(
        index=False,
        column_dtypes={
            "symbol_id": "int64",
            "current": "float64",
            "a1_v": "float64",
            "a1_p": "float64",
            "b1_v": "float64",
            "b1_p": "float64",
            "position": "float64",
        },
    )
    values = execute_single_pro(snap_shots, tick_data)
    index_df = pd.DataFrame(
        values,
        columns=["symbol_id", "datetime"] + fields.split(",")[2:] + ["high", "low"],
    )
    index_df["datetime"] = index_df["datetime"].astype("datetime64[ms]")
    index_df["symbol_id"] = f"{product}8888.{exchange}"
    print(index_df.tail())
    save_db(index_df, product, date)
    insert_records(record)
    return f"{product}8888.{exchange}-{date}"


def process_single_product(product: str, exchange: str):
    """
    处理单个产品
    """
    date_lst = get_pro_dates(product, exchange)
    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for date in date_lst:
            futures.append(
                (date, executor.submit(save_1d_index, date, product, exchange))
            )
        for date, future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Thread Error: {product} {exchange} {date}: {e}")
    return results, product


@timeit
def cal_future_index():
    """
    计算期货指数主函数
    """
    prodct_dct = get_product_dict()
    results = {}
    with ProcessPoolExecutor() as executor:
        futures = [
            (product, executor.submit(process_single_product, product, exchange))
            for product, exchange in prodct_dct.items()
        ]
        for product, future in as_completed(futures):
            try:
                result, pro = future.result()
                results[pro] = result
            except Exception as e:
                print(f"Processe Error {product}: {e}")
    return results


if __name__ == "__main__":
    save_1d_index(product="pp", exchange="DCE", date="2024-07-19")
