import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from numba import njit

from logger import Logger
from utils import DBHelper as db
from utils import (
    get_hms,
    get_product_comma,
    in_trade_times,
    timeit,
)


class Processer:
    """集成管理数据处理部分的函数"""

    max_workers = 3
    max_workers_pro = 1
    FUNC_1D_DICT_MAP = {
        "major": db.get_major_dict,
        "index": db.get_index_dict,
        "secondery": db.get_secondery_dict,
    }

    @classmethod
    def thread_executor(cls, func, date_ranges, *args):
        """
        多线程执行函数
        """
        results = []
        if cls.max_workers_pro == 1:
            for date_range in date_ranges:
                try:
                    results.append(func(*args, date_range))
                except Exception as e:
                    logger.error(
                        f"[{e.__class__.__name__}] Thread Error: {args} from {date_range[0]} to {date_range[-1]}: {e}"
                    )
                    return results
        else:
            with ThreadPoolExecutor(max_workers=cls.max_workers_pro) as executor:
                futures = {}
                # 将datelst分成30个一组，最后一个组可以不满30个
                for date_range in date_ranges:
                    futures[
                        executor.submit(
                            func,
                            *args,
                            date_range,
                        )
                    ] = (date_range[0], date_range[-1])
                for future in as_completed(futures):
                    try:
                        date_range = futures[future]
                        processed_date = future.result()
                        if processed_date:
                            results.append(processed_date)
                    except Exception as e:
                        logger.error(
                            f"[{e.__class__.__name__}] Thread Error: {args} {date_range}: {e}"
                        )
        return results

    @classmethod
    def main_thread_executor(cls, func, pro_ranges, *args):
        results = {}
        with ThreadPoolExecutor(max_workers=cls.max_workers) as executor:
            futures = {}
            match pro_ranges:
                case dict():
                    for product, exchange in pro_ranges.items():
                        futures[executor.submit(func, product, exchange, *args)] = (
                            product,
                            exchange,
                        )
                case list():
                    for sym in pro_ranges:
                        futures[executor.submit(func, sym)] = sym
            for future in as_completed(futures):
                pro = futures[future]
                try:
                    processed_dates = future.result()
                    if processed_dates is not None:
                        results[pro] = processed_dates
                except Exception as e:
                    logger.error(f"[{e.__class__.__name__}] Processe Error {pro}: {e}")
        return results

    @classmethod
    def process_single_major_all(cls, product: str, exchange: str):
        """
        一次性处理单个主力合约的所有数据
        """
        symbol_id = f"{product}9999.{exchange}"
        record = {"symbol_id": symbol_id, "type": "tick", "date": "all"}
        if db.is_processed(record):
            Logger.warning(f"{symbol_id} of all data has been processed")
            return
        logger.info(f"Processing {symbol_id}")
        major_contract = db.get_major_contract_dat(product, exchange)
        if major_contract:
            major_contract = cls.major_dat_postprocess(major_contract, symbol_id)
            db.save_db(symbol_id, major_contract, "merged_tick", "all")
            db.insert_records(record)
            logger.success(f"{symbol_id} have been processed successfully")
        return symbol_id

    @classmethod
    @timeit
    def process_future_major(cls, lock=None):
        """
        计算主力合约
        """
        if lock is not None:
            print("add lock")
            db.lock = lock
        print("start")
        prodct_dct = db.get_product_dict_clickhouse()
        results = cls.main_thread_executor(
            cls.process_single_major_contract, prodct_dct
        )
        return results

    @classmethod
    @timeit
    def process_single_major_contract(cls, product: str, exchange: str):
        """
        处理单个主力合约，跳过历史第一天（没有主力合约数据）
        """
        try:
            date_lst = db.get_pro_dates(product, exchange, "9999", "tick")
        except AssertionError:
            logger.warning(f"{product} {exchange} has no data")
            return
        date_ranges = [date_lst[i : i + 30] for i in range(0, len(date_lst), 30)]
        results = cls.thread_executor(cls.save_1M_major, date_ranges, product, exchange)
        return results

    @classmethod
    @timeit
    def save_1M_major(cls, product: str, exchange: str, dates: list[str]):
        """
        计算一个产品一月的主力合约数据
        """
        symbol_id = f"{product}9999.{exchange}"
        assert dates, "dates is empty"
        if len(dates) == 1:
            major_df, record = cls.save_1d_major(product, exchange, dates[0])
            assert major_df is not None, f"{symbol_id}-{dates[0]} has no data"
            db.save_db(symbol_id, major_df, "merged_tick", dates[0])
            db.insert_records(record)
            logger.success(
                f"{symbol_id} of {dates[0]} data have been processed successfully"
            )
            return dates[0], dates[0]
        else:
            records = []
            major_dfs = []
            try:
                for date in dates:
                    major_df, record = cls.save_1d_major(product, exchange, date)
                    assert major_df is not None, f"{symbol_id}-{date} has no data"
                    major_dfs.append(major_df)
                    records.append(record)
            except Exception as e:
                logger.error(f"[{e.__class__.__name__}] {symbol_id}-{date} error:{e}")
                if major_dfs:
                    major_df = pl.concat(major_dfs, how="vertical_relaxed")
                    db.save_db(
                        symbol_id,
                        major_df,
                        "merged_tick",
                        (records[0][2], records[-1][2]),
                    )
                    db.insert_records(records)
                    logger.success(
                        f"{symbol_id} from {records[0][2]} to {records[-1][2]} have been processed successfully"
                    )
                raise AssertionError(f"{e}")
            if major_dfs:
                major_df = pl.concat(major_dfs, how="vertical_relaxed")
                db.save_db(symbol_id, major_df, "merged_tick", (dates[0], dates[-1]))
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
        logger.debug(f"Processing {product} {exchange} {date}")
        symbol_id = f"{product}9999.{exchange}"
        record = (symbol_id, "tick", date, "future", 0)
        field = "symbol_id,datetime,current,a1_p,b1_p,a1_v,b1_v,volume,position,high,low,money"
        if db.is_processed(record):
            Logger.warning(f"{symbol_id} of {date} data has been processed")
            return None, None
        try:
            major_contract = db.get_major_contract_id(product, exchange, date)
            major_df = db.get_symbols_tick([major_contract], date, fields=field)
        except AssertionError as e:
            logger.error(f"{product} {exchange} {date} error:{e}")
            return None, None
        major_df = cls.major_dat_postprocess(major_df, symbol_id)
        return major_df, record

    @classmethod
    @timeit
    def process_future_index(cls, lock=None):
        """
        计算期货指数主函数
        """
        if lock is not None:
            db.lock = lock
        prodct_dct = db.get_product_dict_clickhouse()
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
    @timeit
    def process_single_index_product(cls, product: str, exchange: str):
        """
        处理单个产品的指数数据
        """
        try:
            date_lst = db.get_pro_dates(product, exchange, "8888", "tick")
        except AssertionError:
            logger.warning(f"{product} {exchange} has no data")
            return
        date_ranges = [date_lst[i : i + 30] for i in range(0, len(date_lst), 30)]
        processed_dates = cls.thread_executor(
            cls.save_1M_index, date_ranges, product, exchange
        )
        return processed_dates

    @classmethod
    @timeit
    def save_1M_index(cls, product: str, exchange: str, dates: list[str]):
        """
        计算一个产品一月的指数数据
        """
        symbol_id = f"{product}8888.{exchange}"
        assert dates, "dates is empty"
        if len(dates) == 1:
            index_df, record = cls.save_1d_index(product, exchange, dates[0])
            assert index_df is not None, f"{symbol_id}-{dates[0]} has no data"
            db.save_db(symbol_id, index_df, "merged_tick", dates[0])
            db.insert_records(record)
            logger.success(
                f"{symbol_id} of {dates[0]} data have been processed successfully"
            )
            return dates[0], dates[0]
        else:
            records = []
            index_dfs = []
            try:
                for date in dates:
                    index_df, record = cls.save_1d_index(product, exchange, date)
                    assert index_df is not None, f"{symbol_id}-{date} has no data"
                    index_dfs.append(index_df)
                    records.append(record)
            except AssertionError as e:
                logger.warning(f"{e}")
                if index_dfs:
                    index_df = pl.concat(index_dfs, how="vertical_relaxed")
                    db.save_db(
                        symbol_id,
                        index_df,
                        "merged_tick",
                        (records[0][2], records[-1][2]),
                    )
                    db.insert_records(records)
                    logger.success(
                        f"{symbol_id} from {records[0][2]} to {records[-1][2]} have been processed successfully"
                    )
                raise AssertionError(f"{e}")
            if index_dfs:
                index_df = pl.concat(index_dfs, how="vertical_relaxed")
                db.save_db(symbol_id, index_df, "merged_tick", (dates[0], dates[-1]))
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
        logger.debug(f"Processing {product} {exchange} {date} index data")
        symbol_id = f"{product}8888.{exchange}"
        record = (symbol_id, "tick", date, "future", 0)
        if db.is_processed(record):
            logger.warning(f"{symbol_id} of {date} data has been processed")
            return None, None
        # 新合约的处理：空行填充
        fields = "symbol_id,datetime,current,a1_p,b1_p,money,a1_v,b1_v,volume,position"
        try:
            symbol_ids = db.get_all_contracts_clickhouse(product, exchange, date)
            tick_data = db.get_symbols_tick(symbol_ids, date, fields).sort(
                by="datetime"
            )  # tick_df 的列顺序需要和snapshot保持一致
        except AssertionError as e:
            logger.error(
                f"[{e.__class__.__name__}] {product} {exchange} {date} error:{e}"
            )
            return None, None
        symbol_ids = tick_data["symbol_id"].unique()
        snap_shots = db.get_last_snapshot(
            symbol_ids, db.get_last_trading_day(date), fields
        )
        tick_data = cls.index_dat_preprocess(tick_data)
        struct_arr = cls.dat_to_struct_arr(tick_data)
        try:
            cls.assert_tick_full(struct_arr["datetime"][-1] / 1e3)
        except AssertionError as e:
            logger.error(f"[{e.__class__.__name__}] {symbol_id} {date} error:{e}")
            return None, None
        except Exception as e:
            logger.error(f"[{e.__class__.__name__}] {symbol_id} {date} error:{e}")
            return None, None
        product_comma = get_product_comma(product)
        values = Processer.execute_single_pro(snap_shots, struct_arr, product_comma)
        columns = fields.split(",") + ["high", "low"]
        index_df = cls.index_dat_postprocess(values, columns, product, exchange)
        return index_df, record

    @classmethod
    def process_future_secondery(cls, lock=None):
        """
        计算期货次主连合约
        """
        if lock is not None:
            db.lock = lock
        prodct_dct = db.get_product_dict_clickhouse()
        results = {}
        with ThreadPoolExecutor(max_workers=cls.max_workers) as executor:
            futures = {
                executor.submit(
                    cls.process_single_secondery_product, product, exchange
                ): product
                for product, exchange in prodct_dct.items()
            }
            for future in as_completed(futures):
                try:
                    product = futures[future]
                    processed_dates = future.result()
                    if processed_dates is not None:
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
            date_lst = db.get_pro_dates(product, exchange, "7777", "tick")
        except AssertionError:
            logger.warning(f"{product} {exchange} has no data")
            return
        date_ranges = [date_lst[i : i + 30] for i in range(0, len(date_lst), 30)]
        processed_dates = cls.thread_executor(
            cls.save_1M_secondery, date_ranges, product, exchange
        )
        return processed_dates

    @classmethod
    def save_1M_secondery(cls, product: str, exchange: str, dates: list[str]):
        """
        计算一个主力合约一月的次主力合约数据
        """
        symbol_id = f"{product}7777.{exchange}"
        assert dates, "dates is empty"
        if len(dates) == 1:
            secondery_df, record = cls.save_1d_secondery(product, exchange, dates[0])
            assert secondery_df is not None, f"{symbol_id}-{dates[0]} has no data"
            db.save_db(symbol_id, secondery_df, "merged_tick", dates[0])
            db.insert_records(record)
            logger.success(
                f"{symbol_id} of {dates[0]} data have been processed successfully"
            )
            return dates[0], dates[0]
        records = []
        secondery_dfs = []
        try:
            for date in dates:
                secondery_df, record = cls.save_1d_secondery(product, exchange, date)
                assert secondery_df is not None, f"{symbol_id}-{date} has no data"
                secondery_dfs.append(secondery_df)
                records.append(record)
        except AssertionError as e:
            logger.warning(f"{e}")
            if secondery_dfs:
                secondery_df = pl.concat(secondery_dfs, how="vertical_relaxed")
                db.save_db(
                    symbol_id,
                    secondery_df,
                    "merged_tick",
                    (records[0][2], records[-1][2]),
                )
                db.insert_records(records)
                logger.success(
                    f"{symbol_id} from {records[0][2]} to {records[-1][2]} have been processed successfully"
                )
                raise AssertionError(f"{e}")
        if secondery_dfs:
            secondery_df = pl.concat(secondery_dfs, how="vertical_relaxed")
            db.save_db(symbol_id, secondery_df, "merged_tick", (dates[0], dates[-1]))
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
        logger.debug(f"Processing {product} {exchange}-{date} secondery contracts")
        symbol_id = f"{product}7777.{exchange}"
        record = (symbol_id, "tick", date, "future", 0)
        fields = "symbol_id,datetime,current,a1_p,b1_p,a1_v,b1_v,volume,position,high,low,money"
        if db.is_processed(record):
            Logger.warning(f"{symbol_id} of {date} data has been processed")
            return None, None
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
    def process_future_1m(
        cls, kind: Literal["9999", "8888", "7777"] = "8888", lock=None
    ):
        """
        处理期货指数分钟线数据
        """
        if lock:
            db.lock = lock
        product_dct = db.get_three_kinds_dic(kind)
        assert product_dct, f"kind {kind} has no data"
        results = cls.main_thread_executor(
            cls.process_single_future_1m, product_dct, kind
        )
        return results

    @classmethod
    def process_single_future_1m(
        cls, product: str, exchange: str, kind: Literal["9999", "8888", "7777"] = "8888"
    ):
        """
        处理单个产品的指数1分钟线数据
        """
        processed_dates = []
        symbol_id = f"{product}{kind}.{exchange}"
        try:
            date_lst = db.get_pro_dates(product, exchange, kind, "1m")
        except AssertionError:
            logger.warning(f"{symbol_id} has no data")
            return
        date_ranges = [date_lst[i : i + 30] for i in range(0, len(date_lst), 30)]
        processed_dates = cls.thread_executor(
            cls.save_month_future_1m, date_ranges, symbol_id
        )
        return processed_dates

    @classmethod
    def save_month_future_1m(
        cls,
        symbol_id: str,
        dates: list[str],
    ):
        """
        计算一个合约的一个月的分钟线数据
        """
        assert dates, "dates is empty"

        if len(dates) == 1:
            index_df, record = cls.save_day_future_1m(symbol_id, dates[0])
            assert index_df is not None, f"{symbol_id}-{dates[0]} has no 1m data"
            db.save_db(symbol_id, index_df, "merged_1m", dates[0])
            db.insert_records(record)
            logger.success(
                f"{symbol_id} of {dates[0]} 1m data have been processed successfully"
            )
            return dates[0], dates[0]
        records = []
        future_1m_dfs = []
        try:
            for date in dates:
                future_1m_df, record = cls.save_day_future_1m(symbol_id, date)
                assert future_1m_df is not None, f"{symbol_id}-{date} has no 1m data"
                future_1m_dfs.append(future_1m_df)
                records.append(record)
        except AssertionError as e:
            logger.warning(f"{e}")
            if future_1m_dfs:
                future_1m_df = pl.concat(future_1m_dfs, how="vertical_relaxed")
                db.save_db(
                    symbol_id,
                    future_1m_df,
                    "merged_1m",
                    (records[0][2], records[-1][2]),
                )
                db.insert_records(records)
                logger.success(
                    f"{symbol_id} from {records[0][2]} to {records[-1][2]} 1m have been processed successfully"
                )
            raise AssertionError(f"{e}")
        if future_1m_dfs:
            future_1m_df = pl.concat(future_1m_dfs, how="vertical_relaxed")
            db.save_db(symbol_id, future_1m_df, "merged_1m", (dates[0], dates[-1]))
            db.insert_records(records)
            logger.success(
                f"{symbol_id} from {dates[0]} to {dates[-1]} 1m have been processed successfully"
            )
        return dates[0], dates[-1]

    @classmethod
    def save_day_future_1m(cls, symbol_id: str, date: str) -> pl.DataFrame:
        """
        计算一个合约的分钟线数据
        """
        logger.debug(f"Processing {symbol_id} {date} 1m data")
        record = (symbol_id, "1m", date, "future", 0)
        if db.is_processed(record):
            logger.warning(f"{symbol_id} of {date} 1m data has been processed")
            return None, None
        rows = db.execute_1m_sql(symbol_id, date, "default")
        if not rows[0]:
            return None, None
        rows = cls.opt_dat_preprocess(rows)
        index_df = pl.DataFrame(rows[0], schema=[r[0] for r in rows[1]])
        return index_df, record

    @classmethod
    def process_option_1m(cls, lock=None):
        """
        处理期权1分钟线数据
        """
        if lock:
            db.lock = lock
        opt_lst = db.get_option_symbols("1m")
        assert opt_lst, "no option symbols"
        results = cls.main_thread_executor(cls.process_single_option_1m, opt_lst)
        return results

    @classmethod
    def process_single_option_1m(cls, symbol_id: str):
        """
        处理单个期权的1分钟线数据
        """
        try:
            opt_dates = db.get_sym_dates(symbol_id, "1m")
        except AssertionError:
            logger.warning(f"{symbol_id} has no data")
            return
        date_ranges = [opt_dates[i : i + 30] for i in range(0, len(opt_dates), 30)]
        expired_date = db.get_expired_date(symbol_id)
        processed_dates = cls.thread_executor(
            cls.save_month_opt_1m, date_ranges, symbol_id, expired_date
        )
        return processed_dates

    @classmethod
    def save_month_opt_1m(
        cls,
        symbol_id: str,
        expired_date: str,
        dates: list[str],
    ):
        """
        计算一个月的期权1分钟线数据
        """
        assert dates, "dates is empty"

        if len(dates) == 1:
            index_df, record = cls.save_day_opt_1m(symbol_id, dates[0], expired_date)
            assert index_df is not None, f"{symbol_id}-{dates[0]} has no 1m data"
            db.save_db(symbol_id, index_df, "merged_1m", dates[0])
            db.insert_records(record)
            logger.success(
                f"{symbol_id} of {dates[0]} 1m data have been processed successfully"
            )
            return dates[0], dates[0]
        records = []
        index_dfs = []
        try:
            for date in dates:
                index_df, record = cls.save_day_opt_1m(symbol_id, date, expired_date)
                assert index_df is not None, f"{symbol_id}-{date} has no 1m data"
                index_dfs.append(index_df)
                records.append(record)
        except AssertionError as e:
            logger.warning(f"{e}")
            if index_dfs:
                index_df = pl.concat(index_dfs, how="vertical_relaxed")
                db.save_db(
                    symbol_id, index_df, "merged_1m", (records[0][2], records[-1][2])
                )
                db.insert_records(records)
                logger.success(
                    f"{symbol_id} from {records[0][2]} to {records[-1][2]} 1m have been processed successfully"
                )
            raise AssertionError(f"{e}")
        if index_dfs:
            index_df = pl.concat(index_dfs, how="vertical_relaxed")
            db.save_db(symbol_id, index_df, "merged_1m", (dates[0], dates[-1]))
            db.insert_records(records)
            logger.success(
                f"{symbol_id} from {dates[0]} to {dates[-1]} 1m have been processed successfully"
            )
        return dates[0], dates[-1]

    @classmethod
    def save_day_opt_1m(
        cls, symbol_id: str, date: str, expired_date: str
    ) -> pl.DataFrame:
        """
        计算一天的期权1分钟线数据
        """
        logger.debug(f"Processing {symbol_id} {date} 1m data")
        if date == expired_date:
            record = (symbol_id, "1m", date, "option", 1)
        elif date < expired_date:
            record = (symbol_id, "1m", date, "option", 0)
        else:
            raise ValueError(
                f"symbol_id date: {date} should be before the expired_date: {expired_date}"
            )
        if db.is_processed(record):
            logger.warning(f"{symbol_id} of {date} 1m data has been processed")
            return None, None
        rows = db.execute_1m_sql(symbol_id, date, "jq")
        if not rows[0]:
            return None, None
        rows = cls.opt_dat_preprocess(rows)
        index_df = pl.DataFrame(rows[0], schema=[r[0] for r in rows[1]])
        return index_df, record

    @classmethod
    def process_future_1d(
        cls, kind: Literal["9999", "8888", "7777"] = "8888", lock=None
    ):
        """
        处理期货指数日线数据
        """
        if lock:
            db.lock = lock
        pro_dic = db.get_three_kinds_dic(kind)
        results = cls.main_thread_executor(cls.process_single_future_1d, pro_dic, kind)
        return results

    @classmethod
    def process_single_future_1d(
        cls, product: str, exchange: str, kind: Literal["9999", "8888", "7777"]
    ):
        """
        处理单个产品的日线数据. 注意日线数据处理的date_range是左开右闭
        """
        symbol_id = f"{product}{kind}.{exchange}"
        try:
            date_lst = db.get_pro_dates(product, exchange, kind, "1d")
        except AssertionError:
            logger.warning(f"{product} {exchange} has no data")
            return
        date_ranges = [date_lst[i : i + 30] for i in range(0, len(date_lst), 30)]
        processed_dates = cls.thread_executor(
            cls.save_month_future_1d, date_ranges, symbol_id
        )
        return processed_dates

    @classmethod
    def save_month_future_1d(cls, symbol_id: str, dates: list[str]):
        """
        计算一个月的日线数据
        """
        assert dates, "dates is empty"
        if len(dates) == 1:
            rows = db.execute_1m_sql(symbol_id, dates[0], "default")
            assert rows[0], f"{symbol_id} {dates[0]} has no 1d data"
            rows = cls.opt_dat_preprocess(rows)
            index_df = pl.DataFrame(rows[0], schema=[r[0] for r in rows[1]])
            db.save_db(symbol_id, index_df, "merged_1d", dates[0])
            db.insert_records((symbol_id, "1d", dates[0], "future", 0))
            logger.success(
                f"{symbol_id} of {dates[0]} 1d data have been processed successfully"
            )
            return dates[0], dates[0]
        rows = db.execute_month_1d_sql(symbol_id, dates[0], dates[-1], "default")
        assert rows[0], f"{symbol_id} from {dates[0]} to {dates[-1]} has no 1d data"
        rows = cls.opt_dat_preprocess(rows)
        index_df = pl.DataFrame(rows[0], schema=[r[0] for r in rows[1]])
        records = []
        for date in dates:
            records.append((symbol_id, "1d", date, "future", 0))
        db.save_db(symbol_id, index_df, "merged_1d", dates)
        db.insert_records(records)
        logger.success(f"{symbol_id} 1d data have been processed successfully")
        return dates

    @classmethod
    def process_option_1d(cls, lock=None):
        """
        处理期权日线数据
        """
        if lock:
            db.lock = lock
        opt_lst = db.get_option_symbols("1d")
        assert opt_lst, "no option symbols"
        results = cls.main_thread_executor(cls.process_single_option_1d, opt_lst)
        return results

    @classmethod
    def process_single_option_1d(cls, symbol_id: str):
        """
        处理单个期权的日线数据
        """
        try:
            opt_dates = db.get_sym_dates(symbol_id, "1d")
        except AssertionError:
            logger.warning(f"{symbol_id} has no data")
            return
        expired_date = db.get_expired_date(symbol_id)
        date_ranges = [opt_dates[i : i + 30] for i in range(0, len(opt_dates), 30)]
        processed_dates = cls.thread_executor(
            cls.save_month_option_1d, date_ranges, symbol_id, expired_date
        )
        return processed_dates

    @classmethod
    def save_month_option_1d(cls, symbol_id: str, expired_date: str, dates: list[str]):
        """
        计算一个月的期权日线数据
        """
        assert dates, "dates is empty"
        if len(dates) == 1:
            rows = db.execute_1d_sql(symbol_id, dates[0], "default")
            assert rows[0], f"{symbol_id} {dates[0]} has no 1d data"
            rows = cls.opt_dat_preprocess(rows)
            index_df = pl.DataFrame(rows[0], schema=[r[0] for r in rows[1]])
            record = cls.opt_record_genrate(symbol_id, dates[0], expired_date)
            db.save_db(symbol_id, index_df, "merged_1d", dates[0])
            db.insert_records(record)
            logger.success(
                f"{symbol_id} of {dates[0]} 1d data have been processed successfully"
            )
            return dates[0], dates[0]
        rows = db.execute_month_1d_sql(symbol_id, dates[0], dates[-1], "default")
        assert rows[0], f"{symbol_id} from {dates[0]} to {dates[-1]} has no 1d data"
        rows = cls.opt_dat_preprocess(rows)
        index_df = pl.DataFrame(rows[0], schema=[r[0] for r in rows[1]])
        records = []
        for date in dates:
            records.append(cls.opt_record_genrate(symbol_id, date, expired_date))
        db.save_db(
            symbol_id,
            index_df,
            "merged_1d",
            dates,
        )
        db.insert_records(records)
        logger.success(f"{symbol_id} 1d data have been processed successfully")
        return dates

    @classmethod
    def assert_tick_full(cls, timestamp: int):
        """校验tick数据是否到收盘"""
        hms = get_hms(timestamp)
        assert hms >= 145959, "tick数据不完整, 未到收盘"

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
        index_df = pl.DataFrame(values, schema=columns)
        index_df = index_df.with_columns(
            [
                pl.col("datetime")
                .cast(pl.Datetime(time_unit="ms"))
                .alias("datetime"),  # 将毫秒转换为秒
                pl.lit(f"{product}8888.{exchange}").alias("symbol_id"),
                pl.col("volume").cast(pl.Int64),
                pl.col("position").cast(pl.Int64),
                pl.col("a1_v").cast(pl.Int64),
                pl.col("b1_v").cast(pl.Int64),
            ]
        )

        return index_df

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
    def major_dat_postprocess(
        cls, major_dat: pl.DataFrame, symbol_id: str
    ) -> pd.DataFrame:
        """
        主力合约后处理
        """
        major_dat = major_dat.with_columns(
            pl.lit(symbol_id).alias("symbol_id"),
            pl.col("volume").cast(pl.Int64).alias("volume"),
            pl.col("position").cast(pl.Int64).alias("position"),
            pl.col("a1_v").cast(pl.Int64).alias("a1_v"),
            pl.col("b1_v").cast(pl.Int64).alias("b1_v"),
        ).sort(by="datetime")
        return major_dat

    @classmethod
    def opt_dat_preprocess(cls, rows: tuple) -> list:
        """
        期权数据预处理, 将symbol_id转化为分类索引, 并将datetime转化为timestamp
        """
        for i in range(len(rows[0])):
            if rows[0][i].dtype == np.dtype("datetime64[s]"):
                rows[0][1] = rows[0][1].astype("str")
            elif (
                rows[0][i].dtype == np.dtype("object")
                and rows[1][i][0] not in "volume,paused"
            ):
                rows[0][i] = rows[0][i].astype("float64")
            elif (
                rows[0][i].dtype == np.dtype("object")
                and rows[1][i][0] in "volume,paused,open_interest"
            ):
                rows[0][i] = rows[0][i].astype("int64")
        return rows

    @classmethod
    def opt_record_genrate(cls, symbol_id: str, date: str, expired_date: str):
        """生成期权处理记录, 如果date等于expired_date, 则表明该期权全部被处理."""
        if date < expired_date:
            record = (symbol_id, "1d", date, "option", 0)
        elif date == expired_date:
            record = (symbol_id, "1d", date, "option", 1)
        else:
            raise ValueError(
                f"symbol_id date: {date} should be before the expired_date: {expired_date}"
            )
        return record


@njit()
def creat_value_array(columns: int, dtype=np.float64) -> np.ndarray:
    """
    创建index值数组
    """
    # 注意value数组的columns和snapshot数组的columns不一致
    return np.zeros((1000000, columns), dtype=dtype)


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
        logger.remove()
        logger.add(sys.stderr, level="SUCCESS")
        logger.add("logs/index.log", level="ERROR", rotation="10 MB")
        results = Processer.process_future_index()
        breakpoint()
    except Exception as e:
        logger.error(f"[{e.__class__.__name__}] Error:: {e}")
    finally:
        pass
