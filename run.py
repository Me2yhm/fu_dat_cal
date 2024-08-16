import polars as pl
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from loguru import logger

from numba import njit, types

from utils import (
    get_hms,
    get_product_comma,
    in_trade_times,
    timeit,
)
from logger import Logger
from utils import DBHelper as db


class Processer:
    max_workers = 3

    @classmethod
    def process_major_contract(cls):
        """
        计算主力合约
        """
        prodct_dct = db.get_product_dict()
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for product, exchange in prodct_dct.items():
                futures[
                    executor.submit(
                        cls.process_single_mayjor_contract, product, exchange
                    )
                ] = product
            for future in as_completed(futures):
                product = futures[future]
                try:
                    major_contract = future.result()
                    if major_contract is not None:
                        results.append(major_contract)
                except Exception as e:
                    logger.error(
                        f"[{e.__class__.__name__}] Processe Error {product}: {e}"
                    )
        return results

    @staticmethod
    @timeit
    @njit()
    def execute_single_pro(
        snap_shots: np.ndarray,
        tick_data: np.ndarray,
        product_comma: str,
    ):
        """
        执行单个产品一天的指数数据的处理
        开盘价与同花顺不同，怀疑是集合竞价导致？
        """
        values = creat_value_array(snap_shots.shape[1] + 4)
        value_index = 0
        last_datetime = tick_data[0][1]  # datetime 精度为10ms
        last_datetime_intrade = True
        high = low = 0
        # 循环之前几乎占据一般的时间，需要优化
        for row in tick_data:
            timestamp = row[1] / 1e3  # 转化为秒
            hms = get_hms(timestamp)
            if not in_trade_times(product_comma, hms):
                last_datetime = row[1]
                last_datetime_intrade = False
                print(timestamp, "||||", hms, "not in trade times")
                continue
            symbol_idx = row[0]
            new_datetime = row[1]
            if new_datetime != last_datetime:
                if not last_datetime_intrade:
                    last_datetime = new_datetime
                    last_datetime_intrade = True
                    continue
                value, high, low = cal_value(snap_shots, last_datetime, high, low)
                values[value_index] = value
                value_index += 1
                last_datetime = new_datetime
            snap_shots[symbol_idx] = [
                row["current"],
                row["a1_v"],
                row["a1_p"],
                row["b1_v"],
                row["b1_p"],
                row["volume"],
                row["position"],
            ]  # symbol_idx会不会有缺失值？

        # 更新最后一个tick，但有一个问题，如果最后一个datetime恰好只有一个数据怎么办？
        value, high, low = cal_value(snap_shots, last_datetime, high, low)
        values[value_index] = value
        value_index += 1
        return values[:value_index]

    @classmethod
    @timeit
    def index_dat_preprocess(cls, tick_data: pl.DataFrame) -> pl.DataFrame:
        """
        指数数据预处理, 将symbol_id转化为分类索引, 并将datetime转化为timestamp
        """
        tick_data = tick_data.with_columns(
            pl.col("symbol_id")
            .cast(pl.Categorical)
            .to_physical()
            .cast(pl.Int64)
            .alias("symbol_id")
        )
        tick_data = tick_data.with_columns(
            ((pl.col("datetime").cast(pl.Int64) / 1e9).round(1) * 1e3).alias("datetime")
        )
        return tick_data

    @classmethod
    @timeit
    def dat_to_struct_arr(cls, tick_data: pl.DataFrame) -> np.ndarray:
        """
        将pl.DataFrame转换为numpy结构数组。根据列式存储加快转换速度
        """
        dtypes = [("symbol_id", "int64")] + [
            (col, "float64") for col in tick_data.columns[1:]
        ]
        struct_arr = np.zeros(len(tick_data), dtype=dtypes, order="C")
        for col in tick_data.columns:
            struct_arr[col] = tick_data[col].to_numpy()
        return struct_arr

    @classmethod
    @timeit
    def index_dat_postprocess(
        cls, values: np.ndarray, columns: list, product: str, exchange: str
    ) -> pd.DataFrame:
        """
        指数数据后处理, 将symbol_id转化为字符串, timestamp转化为datetime64[ms]
        """
        index_df = pd.DataFrame(
            values,
            columns=columns,
            copy=False,
        )
        index_df["datetime"] = index_df["datetime"].astype("datetime64[ms]")
        index_df["symbol_id"] = f"{product}8888.{exchange}"
        return index_df

    @classmethod
    @timeit
    def save_1d_index(
        cls, product: str = "ag", exchange: str = "SHFE", date: str = "2024-07-19"
    ):
        """
        计算一个产品一天的指数数据
        """
        symbol_id = f"{product}8888.{exchange}"
        record = {"symbol_id": symbol_id, "date": date}
        if db.is_processed(record):
            Logger.warning(f"{symbol_id} of {date} data has been processed")
            return
        # 新合约的处理：空行填充
        fields = "symbol_id,datetime,current,a1_v,a1_p,b1_v,b1_p,volume,position"
        symbol_ids = db.get_all_contracts(product, exchange, date)
        tick_data = db.get_symbols_tick(symbol_ids, date, fields).sort(
            by="datetime"
        )  # tick_df 的列顺序需要和snapshot保持一致
        symbol_ids = tick_data["symbol_id"].unique()
        snap_shots = db.get_last_snapshot(
            symbol_ids, db.get_last_trading_day(date), fields
        )
        tick_data = cls.index_dat_preprocess(tick_data)
        struct_arr = cls.dat_to_struct_arr(tick_data)
        product_comma = get_product_comma(product)
        values = Processer.execute_single_pro(snap_shots, struct_arr, product_comma)
        columns = fields.split(",") + ["high", "low"]
        index_df = cls.index_dat_postprocess(values, columns, product, exchange)
        print(index_df.head())
        # db.save_db(index_df, tick, date)
        # db.insert_records(record)
        # return f"{symbol_id}-{date}"

    @classmethod
    def process_single_index_product(cls, product: str, exchange: str):
        """
        处理单个产品的指数数据
        """
        date_lst = db.get_pro_dates(product, exchange)
        results = []
        with ThreadPoolExecutor(max_workers=cls.max_workers) as executor:
            futures = {}
            for date in date_lst[1:]:
                futures[executor.submit(cls.save_1d_index, product, exchange, date)] = (
                    date
                )

            for future in as_completed(futures):
                try:
                    result = future.result()
                    date = futures[future]
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(
                        f"[{e.__class__.__name__}] Thread Error: {product}_{exchange}_{date}: {e}"
                    )
        return results, product

    @classmethod
    @timeit
    def cal_future_index(cls):
        """
        计算期货指数主函数
        """
        prodct_dct = db.get_product_dict()
        results = {}
        with ThreadPoolExecutor(max_workers=cls.max_workers) as executor:
            futures = {
                executor.submit(
                    cls.process_single_index_product, product, exchange
                ): product
                for product, exchange in prodct_dct.items()
            }
            for future in as_completed(futures):
                try:
                    pro = futures[future]
                    result = future.result()
                    if result:
                        results[pro] = result
                        logger.success(f"{pro} have been processed successfully")
                except Exception as e:
                    logger.error(f"[{e.__class__.__name__}] Processe Error {pro}: {e}")
        return results

    @classmethod
    def save_1d_secondery(cls, product: str, exchange: str, date: str):
        """
        存一个期货品种一天的次主连数据
        """
        symbol_id = f"{product}7777.{exchange}"
        record = {"symbol_id": symbol_id, "date": date}
        if db.is_processed(record):
            Logger.warning(f"{symbol_id} of {date} data has been processed")
            return
        secondery = db.get_secondery_id(product, exchange, date)
        secondery_df = db.get_symbols_tick([secondery], date, "*").sort(by="datetime")
        secondery_df = cls.secondery_dat_postprocess(secondery_df, symbol_id)
        db.save_db(secondery_df, "tick", date)
        db.insert_records(record)
        return date

    @classmethod
    def process_single_secondery_product(cls, product: str, exchange: str):
        """
        存一个期货品种的次主连数据
        """
        date_lst = db.get_pro_dates(product, exchange)
        processed_dates = []
        with ThreadPoolExecutor(max_workers=cls.max_workers) as executor:
            futures = []
            for date in date_lst:
                futures.append(
                    executor.submit(cls.save_1d_secondery, product, exchange, date)
                )
            for future in as_completed(futures):
                try:
                    processed_date = future.result()
                    if processed_date:
                        processed_dates.append(processed_date)
                except Exception as e:
                    logger.error(
                        f"[{e.__class__.__name__}] Thread Error: {product} {exchange} {processed_date}: {e}"
                    )
        return processed_dates, product

    @staticmethod
    def process_secondery_contract():
        """
        计算期货次主连合约
        """
        prodct_dct = db.get_product_dict()
        results = {}
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    Processer.process_single_secondery_product, product, exchange
                ): product
                for product, exchange in prodct_dct.items()
            }
            for future in as_completed(futures):
                try:
                    processed_dates = future.result()
                    product = futures[future]
                    if processed_dates:
                        results[product] = processed_dates
                        logger.success(
                            f"{product} secondery contracts have been processed successfully"
                        )
                except Exception as e:
                    logger.error(
                        f"[{e.__class__.__name__}] Processe Error {product}: {e}"
                    )
            return results

    @classmethod
    def secondery_dat_postprocess(
        cls, secondery_dat: pl.DataFrame, symbol_id: str
    ) -> pd.DataFrame:
        """
        期货次主连数据后处理
        """
        secondery_dat = secondery_dat.with_columns(
            pl.lit(symbol_id).alias("symbol_id")
        ).sort("datetime")
        return secondery_dat

    @classmethod
    def process_single_mayjor_contract(cls, product: str, exchange: str):
        """
        计算单个主力合约
        """
        symbol_id = f"{product}9999.{exchange}"
        record = {"symbol_id": symbol_id, "date": "all"}
        if db.is_processed(record):
            Logger.warning(f"{symbol_id} of all data has been processed")
            return
        logger.info(f"Processing {symbol_id}")
        major_contract = db.get_mayjor_contract_dat(product, exchange)
        if major_contract:
            major_contract = cls.mayjor_dat_postprocess(major_contract, symbol_id)
            db.save_db(major_contract, "tick", "all")
            db.insert_records(record)
            logger.success(f"{symbol_id} have been processed successfully")
        return symbol_id

    @classmethod
    def mayjor_dat_postprocess(
        cls, mayjor_dat: pl.DataFrame, symbol_id: str
    ) -> pd.DataFrame:
        """
        主力合约后处理
        """
        mayjor_dat = mayjor_dat.with_columns(pl.lit(symbol_id).alias("symbol_id")).sort(
            by="datetime"
        )
        return mayjor_dat


@njit()
def creat_value_array(columns: int, dtype=np.float64) -> np.ndarray:
    """
    创建index值数组
    """
    # 注意value数组的columns和snapshot数组的columns不一致
    return np.zeros((100000, columns), dtype=dtype)


@njit()
def cal_value(snapshot: np.ndarray, datetime: float, high: float, low: float):
    """
    根据snapshot计算指数value
    """
    value = np.zeros(snapshot.shape[1] + 4)
    position = snapshot[:, -1]
    if (position == 0).all():
        return value, high, low
    total_position = position.sum()

    value[2:-4] = (position @ snapshot[:, :-2]) / total_position
    current = value[2]
    if current >= high or high == 0:
        high = current
    if current <= low or low == 0:
        low = current

    value[1] = datetime
    value[-4] = snapshot[:, -2].sum()
    value[-3] = total_position
    value[-2] = high
    value[-1] = low
    return value, high, low


if __name__ == "__main__":
    results = Processer.process_major_contract()
    breakpoint()
    # Processer.save_1d_index("ag", "SHFE", "2012-05-11")
