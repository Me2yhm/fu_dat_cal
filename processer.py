import sys
from typing import Literal, Union
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
    max_workers = 2
    FUNC_1D_DICT_MAP = {
        "mayjor": db.get_mayjor_dict,
        "index": db.get_index_dict,
        "secondery": db.get_secondery_dict,
    }

    @classmethod
    def process_single_mayjor_all(cls, product: str, exchange: str):
        """
        一次性处理单个主力合约的所有数据
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
            db.save_db(symbol_id, major_contract, "tick", "all")
            db.insert_records(record)
            logger.success(f"{symbol_id} have been processed successfully")
        return symbol_id

    @classmethod
    @timeit
    def process_future_major(cls):
        """
        计算主力合约
        """
        prodct_dct = db.get_product_dict()
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
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
                    processed_dates = future.result()
                    if processed_dates is not None:
                        results[product] = processed_dates
                except Exception as e:
                    print(processed_dates)
                    logger.error(
                        f"[{e.__class__.__name__}] Processe Error {product}: {e}"
                    )
        return results

    @classmethod
    @timeit
    def process_single_mayjor_contract(cls, product: str, exchange: str):
        """
        处理单个主力合约，跳过历史第一天（没有主力合约数据）
        """
        try:
            date_lst = db.get_pro_dates(product, exchange)
        except AssertionError:
            logger.warning(f"{product} {exchange} has no data")
            return
        results = []
        with ThreadPoolExecutor(max_workers=cls.max_workers) as executor:
            futures = {}
            # 每次处理30个日期
            for i in range(1, len(date_lst), 30):
                start_index = i
                end_index = i + 30 if i + 30 < len(date_lst) else len(date_lst) - 1
                futures[
                    executor.submit(
                        cls.save_1M_major,
                        product,
                        exchange,
                        date_lst[start_index:end_index],
                    )
                ] = (date_lst[start_index], date_lst[end_index])
            for future in as_completed(futures):
                try:
                    date_range = futures[future]
                    processed_date = future.result()
                    if processed_date:
                        results.append(processed_date)
                except Exception as e:
                    logger.error(
                        f"[{e.__class__.__name__}] Thread Error: {product}_{exchange}_{date_range}: {e}"
                    )
                    return None
        return results

    @classmethod
    @timeit
    def save_1M_major(cls, product: str, exchange: str, dates: list[str]):
        """
        计算一个产品一月的主力合约数据
        """
        symbol_id = f"{product}9999.{exchange}"
        if not dates:
            logger.warning(f"passed dates has no value")
            return None, None
        elif len(dates) == 1:
            mayjor_df, record = cls.save_1d_major(product, exchange, dates[0])
            if mayjor_df is not None:
                db.save_db(symbol_id, mayjor_df, "tick", dates[0])
                db.insert_records(record)
                logger.success(
                    f"{symbol_id} of {dates[0]} data have been processed successfully"
                )
            return dates[0], date[0]
        else:
            records = []
            mayjor_dfs = []
            for date in dates:
                mayjor_df, record = cls.save_1d_major(product, exchange, date)
                if mayjor_df is not None:
                    mayjor_dfs.append(mayjor_df)
                    records.append(record)
            if mayjor_dfs:
                mayjor_df = pl.concat(mayjor_dfs, how="vertical_relaxed")
                db.save_db(symbol_id, mayjor_df, "tick", (dates[0], dates[-1]))
                db.insert_records(records)
                logger.success(
                    f"{symbol_id} from {dates[0]} to {dates[-1]} have been processed successfully"
                )
            return dates[0], dates[-1]

    @classmethod
    def save_1d_major(cls, product: str, exchange: str, date: str):
        """
        存一个主力合约一天的数据
        """
        symbol_id = f"{product}9999.{exchange}"
        record = {"symbol_id": symbol_id, "date": date}
        field = "symbol_id,datetime,current,a1_p,b1_p,a1_v,b1_v,volume,position,high,low,money"
        if db.is_processed(record):
            Logger.warning(f"{symbol_id} of {date} data has been processed")
            return None, None
        try:
            major_contract = db.get_mayjor_contract_id(product, exchange, date)
            mayjor_df = db.get_symbols_tick([major_contract], date, fields=field)
        except AssertionError as e:
            logger.error(f"{product} {exchange} {date} error:{e}")
            return None, None
        mayjor_df = cls.mayjor_dat_postprocess(mayjor_df, symbol_id)
        return mayjor_df, record

    @classmethod
    @timeit
    def process_future_index(cls):
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
    def process_single_index_product(cls, product: str, exchange: str):
        """
        处理单个产品的指数数据
        """
        try:
            date_lst = db.get_pro_dates(product, exchange)
        except AssertionError:
            logger.warning(f"{product} {exchange} has no data")
            return
        results = []
        with ThreadPoolExecutor(max_workers=cls.max_workers) as executor:
            futures = {}
            # 将datelst分成30个一组，最后一个组可以不满30个
            for i in range(1, len(date_lst), 30):
                start_index = i
                end_index = i + 30 if i + 30 < len(date_lst) else len(date_lst) - 1
                futures[
                    executor.submit(
                        cls.save_1M_index,
                        product,
                        exchange,
                        date_lst[start_index:end_index],
                    )
                ] = (date_lst[start_index], date_lst[end_index])
            for future in as_completed(futures):
                try:
                    date_range = futures[future]
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(
                        f"[{e.__class__.__name__}] Thread Error: {product}_{exchange}_{date_range}: {e}"
                    )
        return results, product

    @classmethod
    @timeit
    def save_1M_index(cls, product: str, exchange: str, dates: list[str]):
        """
        计算一个产品一月的指数数据
        """
        symbol_id = f"{product}8888.{exchange}"
        if not dates:
            logger.warning(f"passed dates has no value")
            return None, None
        elif len(dates) == 1:
            index_df, record = cls.save_1d_index(product, exchange, dates[0])
            if index_df is not None:
                db.save_db(symbol_id, index_df, "tick", dates[0])
                db.insert_records(record)
                logger.success(
                    f"{symbol_id} of {dates[0]} data have been processed successfully"
                )
            return dates[0], date[0]
        else:
            records = []
            index_dfs = []
            for date in dates:
                index_df, record = cls.save_1d_index(product, exchange, date)
                if index_df is not None:
                    index_dfs.append(index_df)
                    records.append(record)
            if index_dfs:
                index_df = pl.concat(index_dfs, how="vertical_relaxed")
                db.save_db(symbol_id, index_df, "tick", (date[0], date[-1]))
                db.insert_records(records)
                logger.success(
                    f"{symbol_id} from {dates[0]} to {dates[-1]} have been processed successfully"
                )
            return dates[0], dates[-1]

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
            logger.warning(f"{symbol_id} of {date} data has been processed")
            return None, None
        # 新合约的处理：空行填充
        fields = "symbol_id,datetime,current,a1_p,b1_p,money,a1_v,b1_v,volume,position"
        try:
            symbol_ids = db.get_all_contracts(product, exchange, date)
            tick_data = db.get_symbols_tick(symbol_ids, date, fields).sort(
                by="datetime"
            )  # tick_df 的列顺序需要和snapshot保持一致
        except AssertionError:
            return None, None
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
        return index_df, record

    @staticmethod
    def process_future_secondery():
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
                    product = futures[future]
                    processed_dates = future.result()
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
    def process_single_secondery_product(cls, product: str, exchange: str):
        """
        存一个期货品种的次主连数据
        """
        try:
            date_lst = db.get_pro_dates(product, exchange)
        except AssertionError:
            logger.warning(f"{product} {exchange} has no data")
            return
        processed_dates = []
        with ThreadPoolExecutor(max_workers=cls.max_workers) as executor:
            futures = {}
            # 将datelst分成30个一组，最后一个组可以不满30个
            for i in range(1, len(date_lst), 30):
                start_index = i
                end_index = i + 30 if i + 30 < len(date_lst) else len(date_lst) - 1
                futures[
                    executor.submit(
                        cls.save_1M_secondery,
                        product,
                        exchange,
                        date_lst[start_index:end_index],
                    )
                ] = (date_lst[start_index], date_lst[end_index])
            for future in as_completed(futures):
                try:
                    date_range = futures[future]
                    processed_date = future.result()
                    if processed_date:
                        processed_dates.append(processed_date)
                except Exception as e:
                    logger.error(
                        f"[{e.__class__.__name__}] Thread Error: {product} {exchange} {date_range}: {e}"
                    )
        return processed_dates, product

    @classmethod
    def save_1M_secondery(cls, product: str, exchange: str, dates: list[str]):
        """
        计算一个主力合约一月的次主力合约数据
        """
        symbol_id = f"{product}7777.{exchange}"
        if not dates:
            logger.warning(f"passed dates has no value")
            return None, None
        elif len(dates) == 1:
            secondery_df, record = cls.save_1d_secondery(product, exchange, dates[0])
            if secondery_df is not None:
                db.save_db(symbol_id, secondery_df, "tick", dates[0])
                db.insert_records(record)
                logger.success(
                    f"{symbol_id} of {dates[0]} data have been processed successfully"
                )
            return dates[0], date[0]
        records = []
        major_dfs = []
        for date in dates:
            major_df, record = cls.save_1d_secondery(product, exchange, date)
            if major_df is not None:
                major_dfs.append(major_df)
                records.append(record)
        if major_dfs:
            major_df = pl.concat(major_dfs, how="vertical_relaxed")
            db.save_db(symbol_id, major_df, "tick", (date[0], date[-1]))
            db.insert_records(records)
            logger.success(
                f"{symbol_id} from {dates[0]} to {dates[-1]} have been processed successfully"
            )
        return dates[0], dates[-1]

    @classmethod
    def save_1d_secondery(cls, product: str, exchange: str, date: str):
        """
        存一个期货品种一天的次主连数据
        """
        symbol_id = f"{product}7777.{exchange}"
        record = {"symbol_id": symbol_id, "date": date}
        fields = "symbol_id,datetime,current,a1_p,b1_p,a1_v,b1_v,volume,position,high,low,money"
        if db.is_processed(record):
            Logger.warning(f"{symbol_id} of {date} data has been processed")
            return
        try:
            secondery = db.get_secondery_id(product, exchange, date)
            secondery_df = db.get_symbols_tick([secondery], date, fields).sort(
                by="datetime"
            )
        except AssertionError as e:
            logger.warning(f"{e}")
            return None, None
        secondery_df = cls.secondery_dat_postprocess(secondery_df, symbol_id)
        return secondery_df, record

    @classmethod
    def process_1m(cls, kind: Literal["mayjor", "index", "secondery"] = "index"):
        """
        处理期货指数分钟线数据
        """
        func = getattr(db, f"get_{kind}_dict")
        index_dct = db.func()
        results = {}
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    Processer.process_single_product_1m, product, exchange
                ): product
                for product, exchange in index_dct.items()
            }
            for future in as_completed(futures):
                try:
                    product = futures[future]
                    processed_dates = future.result()
                    if processed_dates:
                        results[product] = processed_dates
                        logger.success(
                            f"{product} index 1m data have been processed successfully"
                        )
                except Exception as e:
                    logger.error(
                        f"[{e.__class__.__name__}] Processe Error {product}: {e}"
                    )
            return results

    @classmethod
    def process_single_product_1m(cls, product: str, exchange: str):
        """
        处理单个产品的指数1分钟线数据
        """
        try:
            date_lst = db.get_pro_dates(product, exchange)
        except AssertionError:
            logger.warning(f"{product} {exchange} has no data")
            return
        processed_dates = []
        symbol_id = f"{product}.{exchange}"
        with ThreadPoolExecutor(max_workers=cls.max_workers) as executor:
            futures = {}
            # 将datelst分成30个一组，最后一个组可以不满30个
            for i in range(1, len(date_lst), 30):
                start_index = i
                end_index = i + 30 if i + 30 < len(date_lst) else len(date_lst) - 1
                futures[
                    executor.submit(
                        cls.save_month_1m,
                        symbol_id,
                        date_lst[start_index:end_index],
                    )
                ] = (date_lst[start_index], date_lst[end_index])
            for future in as_completed(futures):
                try:
                    date_range = futures[future]
                    processed_date = future.result()
                    if processed_date:
                        processed_dates.append(processed_date)
                except Exception as e:
                    logger.error(
                        f"[{e.__class__.__name__}] Thread Error: {product} {exchange} {date_range}: {e}"
                    )
        return processed_dates, product

    @classmethod
    def save_month_1m(cls, symbol_id: str, dates: list[str]):
        """
        计算一个指数的一个月的分钟线数据
        """
        if not dates:
            logger.warning(f"passed dates has no value")
            return None, None
        elif len(dates) == 1:
            index_df, record = cls.save_day_1m(symbol_id, dates[0])
            if index_df is not None:
                db.save_db(symbol_id, index_df, "1m", dates[0])
                db.insert_records(record)
                logger.success(
                    f"{symbol_id} of {dates[0]} 1m data have been processed successfully"
                )
            return dates[0], dates[0]
        records = []
        index_dfs = []
        for date in dates:
            index_df, record = cls.save_day_1m(symbol_id, date)
            if index_df is not None:
                index_dfs.append(index_df)
                records.append(record)
        if index_dfs:
            index_df = pl.concat(index_dfs, how="vertical_relaxed")
            db.save_db(symbol_id, index_df, "1m", (date[0], date[-1]))
            db.insert_records(records)
            logger.success(
                f"{symbol_id} from {dates[0]} to {dates[-1]} 1m have been processed successfully"
            )
        return dates[0], dates[-1]

    @classmethod
    def save_day_1m(cls, symbol_id: str, date: str) -> pl.DataFrame:
        """
        计算一个指数的分钟线数据
        """
        record = {"symbol_id": symbol_id, "date": date, "type": "1m"}
        if db.is_processed(record):
            logger.warning(f"{symbol_id} of {date} 1m data has been processed")
            return None, None
        sql = db.get_1m_sql(symbol_id, date, "default")
        rows = db.writer_conn.execute(sql, columnar=True, with_column_types=True)
        if not rows[0]:
            return None
        index_df = pl.DataFrame(rows[0], schema=[r[0] for r in rows[1]])
        return index_df, record

    @classmethod
    def process_1d(cls, kind: Literal["mayjor", "index", "secondery"] = "index"):
        """
        处理期货指数日线数据
        """
        func = getattr(db, f"get_{kind}_dict")
        index_dct = db.func()
        results = {}
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    Processer.process_single_product_1d, product, exchange
                ): product
                for product, exchange in index_dct.items()
            }
            for future in as_completed(futures):
                try:
                    product = futures[future]
                    processed_dates = future.result()
                    if processed_dates:
                        results[product] = processed_dates
                        logger.success(
                            f"{product} index 1d data have been processed successfully"
                        )
                except Exception as e:
                    logger.error(
                        f"[{e.__class__.__name__}] Processe Error {product}: {e}"
                    )
            return results

    @classmethod
    def process_single_product_1d(cls, product: str, exchange: str):
        """
        处理单个产品的指数日线数据
        """
        symbol_id = f"{product}.{exchange}"

        sql = db.get_1d_sql(symbol_id, "default")
        rows = db.writer_conn.execute(sql, columnar=True, with_column_types=True)
        if not rows[0]:
            logger.warning(f"{symbol_id} 1d data has no data")
            return
        index_df = pl.DataFrame(rows[0], schema=[r[0] for r in rows[1]])
        dates = index_df["datetime"]
        date_range = (dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d"))
        record = {"symbol_id": symbol_id, "type": "1d", "date": f"{date_range}"}
        if db.is_processed(record):
            logger.warning(f"{symbol_id} 1d data has been processed")
            return
        db.save_db(symbol_id, index_df, "1d", f"{date_range}")
        db.update_records(record)
        logger.success(f"{symbol_id} 1d data have been processed successfully")

    @staticmethod
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
                continue
            symbol_idx = row[0]
            new_datetime = row[1]
            if new_datetime != last_datetime:
                if last_datetime_intrade:
                    value, high, low = cal_value(snap_shots, last_datetime, high, low)
                    values[value_index] = value
                    value_index += 1
                    last_datetime = new_datetime
                else:
                    last_datetime = new_datetime
                    last_datetime_intrade = True
            snap_shots[symbol_idx] = [
                row["current"],
                row["a1_p"],
                row["b1_p"],
                row["money"],
                row["a1_v"],
                row["b1_v"],
                row["volume"],
                row["position"],
            ]  # symbol_idx会不会有缺失值？

        # 更新最后一个tick，但有一个问题，如果最后一个datetime恰好只有一个数据怎么办？
        value, high, low = cal_value(snap_shots, last_datetime, high, low)
        values[value_index] = value
        value_index += 1
        return values[:value_index]

    @classmethod
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
        index_df[["volume", "position", "a1_v", "b1_v"]] = index_df[
            ["volume", "position", "a1_v", "b1_v"]
        ].astype("int64")
        return pl.from_pandas(index_df, rechunk=True)

    @classmethod
    def secondery_dat_postprocess(
        cls, secondery_dat: pl.DataFrame, symbol_id: str
    ) -> pd.DataFrame:
        """
        期货次主连数据后处理
        """
        secondery_dat = secondery_dat.with_columns(
            pl.lit(symbol_id).alias("symbol_id"),
            pl.col("volume").cast(pl.Int64).alias("volume"),
            pl.col("position").cast(pl.Int64).alias("position"),
            pl.col("a1_v").cast(pl.Int64).alias("a1_v"),
            pl.col("b1_v").cast(pl.Int64).alias("b1_v"),
        ).sort("datetime")
        return secondery_dat

    @classmethod
    def mayjor_dat_postprocess(
        cls, mayjor_dat: pl.DataFrame, symbol_id: str
    ) -> pd.DataFrame:
        """
        主力合约后处理
        """
        mayjor_dat = mayjor_dat.with_columns(
            pl.lit(symbol_id).alias("symbol_id"),
            pl.col("volume").cast(pl.Int64).alias("volume"),
            pl.col("position").cast(pl.Int64).alias("position"),
            pl.col("a1_v").cast(pl.Int64).alias("a1_v"),
            pl.col("b1_v").cast(pl.Int64).alias("b1_v"),
        ).sort(by="datetime")
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

    value[2:5] = (position @ snapshot[:, :3]) / total_position
    for i in range(5, snapshot.shape[1] + 2):
        value[i] = snapshot[:, i - 2].sum()
    current = value[2]
    if current >= high or high == 0:
        high = current
    if current <= low or low == 0:
        low = current

    value[1] = datetime
    value[-2] = high
    value[-1] = low
    return value, high, low


if __name__ == "__main__":
    try:
        db.execute_1d_extract()
        logger.remove()
        logger.add(sys.stderr, level="SUCCESS")
        logger.add("logs/index.log", level="ERROR", rotation="10 MB")
        results = Processer.process_future_index()
        print(results)
        breakpoint()
    except Exception as e:
        logger.error(f"[{e.__class__.__name__}] Error:: {e}")
    finally:
        db.delete_1d_dbfile()
        pass
