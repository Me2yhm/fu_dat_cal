import functools
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Tuple, Union

import clickhouse_driver
import numba
import numpy as np
import pandas as pd
import polars as pl
import pymongo
import pymongo.collection
from jy_lib.clickhouse import ClickHouse
from loguru import logger
from numba.typed import List

from logger import Logger, setup_logger

log_root_dir = Path(__file__).parent / "log"
temp_dir = Path(__file__).parent / "temp"


def create_1d_dbfile(db_file: Path):
    temp_dir.mkdir(exist_ok=True)
    if not db_file.exists():
        db_file.touch()


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 调用被装饰的函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算运行时间
        logger.debug(
            f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds"
        )
        return result  # 返回函数的结果

    return wrapper


class DBHelper:
    """主要用来继承管理与数据库的交互, 调用的时候要注意reader和writer的clickhouse url是否正确, 并且放在环境变量中. 注意clickhouse url 的use_numpy参数需要为True"""

    lock = threading.Lock()
    # clickhouse
    reader_uri = os.environ.get("CLICKHOUSE_READER")
    conn_reader = clickhouse_driver.Client.from_url(url=reader_uri)
    # 本地数据库reader, 调试用
    reader_1d_uri = temp_dir / "future_1d.db"
    try:
        db_file = temp_dir / "future_1d.db"
        conn_1d = sqlite3.connect(reader_1d_uri, check_same_thread=False)
    except sqlite3.OperationalError:
        create_1d_dbfile(db_file)
        conn_1d = sqlite3.connect(reader_1d_uri, check_same_thread=False)
    cursor_1d = conn_1d.cursor()
    # 本地数据库writer, 调试用
    writer_uri = os.environ.get("CLICKHOUSE_WRITER")
    writer_conn = ClickHouse(writer_uri, 8123)

    logger = setup_logger("DBHelper", "db_helper.log")

    # mongodb_url = os.environ.get("MONGODB_URL")
    # record_conn = pymongo.MongoClient(mongodb_url)["Quote"]["FutureIndex"]
    columns_type_dict = {
        "symbol_id": str,
        "datetime": str,
        "current": float,
        "a1_v": float,
        "a1_p": float,
        "b1_v": float,
        "b1_p": float,
        "volume": float,
        "position": float,
    }

    @classmethod
    def init_writer_table(
        cls, table_name: Literal["merged_tick", "merged_1d", "merged_1m", "merged"]
    ):
        """
        创建clickhouse表, 用于写入数据
        """
        collection = "default"
        if table_name == "merged_tick":
            sql = f"""
            CREATE TABLE IF NOT EXISTS {collection}.{table_name}
            (

                `symbol_id` String CODEC(ZSTD(3)),

                `datetime` DateTime64(3) CODEC(DoubleDelta,
            ZSTD(3)),

                `current` Nullable(Float64) CODEC(ZSTD(3)),

                `high` Nullable(Float64) CODEC(ZSTD(3)),

                `low` Nullable(Float64) CODEC(ZSTD(3)),

                `volume` Nullable(Int64) CODEC(ZSTD(3)),

                `money` Nullable(Float64) CODEC(ZSTD(3)),

                `position` Nullable(Int64) CODEC(ZSTD(3)),

                `a5_v` Nullable(Int64) CODEC(ZSTD(3)),

                `a5_p` Nullable(Float64) CODEC(ZSTD(3)),

                `a4_v` Nullable(Int64) CODEC(ZSTD(3)),

                `a4_p` Nullable(Float64) CODEC(ZSTD(3)),

                `a3_v` Nullable(Int64) CODEC(ZSTD(3)),

                `a3_p` Nullable(Float64) CODEC(ZSTD(3)),

                `a2_v` Nullable(Int64) CODEC(ZSTD(3)),

                `a2_p` Nullable(Float64) CODEC(ZSTD(3)),

                `a1_v` Nullable(Int64) CODEC(ZSTD(3)),

                `a1_p` Nullable(Float64) CODEC(ZSTD(3)),

                `b1_v` Nullable(Int64) CODEC(ZSTD(3)),

                `b1_p` Nullable(Float64) CODEC(ZSTD(3)),

                `b2_v` Nullable(Int64) CODEC(ZSTD(3)),

                `b2_p` Nullable(Float64) CODEC(ZSTD(3)),

                `b3_v` Nullable(Int64) CODEC(ZSTD(3)),

                `b3_p` Nullable(Float64) CODEC(ZSTD(3)),

                `b4_v` Nullable(Int64) CODEC(ZSTD(3)),

                `b4_p` Nullable(Float64) CODEC(ZSTD(3)),

                `b5_v` Nullable(Int64) CODEC(ZSTD(3)),

                `b5_p` Nullable(Float64) CODEC(ZSTD(3))
            )
            ENGINE = ReplacingMergeTree
            PARTITION BY sipHash64(symbol_id) % 256
            PRIMARY KEY (symbol_id,
            datetime)
            ORDER BY (symbol_id,
            datetime)
            SETTINGS index_granularity = 8192;
            """
        elif table_name == "merged_1d":
            sql = f"""
                CREATE TABLE IF NOT EXISTS {collection}.`{table_name}`
                (

                    `symbol_id` String CODEC(ZSTD(3)),

                    `datetime` DateTime64(3) CODEC(DoubleDelta,
                ZSTD(3)),

                    `open` Nullable(Float64) CODEC(ZSTD(3)),

                    `high` Nullable(Float64) CODEC(ZSTD(3)),

                    `low` Nullable(Float64) CODEC(ZSTD(3)),

                    `close` Nullable(Float64) CODEC(ZSTD(3)),

                    `volume` Nullable(Int64) CODEC(ZSTD(3)),

                    `money` Nullable(Float64) CODEC(ZSTD(3)),

                    `factor` Nullable(Float64) CODEC(ZSTD(3)),

                    `high_limit` Nullable(Float64) CODEC(ZSTD(3)),

                    `low_limit` Nullable(Float64) CODEC(ZSTD(3)),

                    `avg` Nullable(Float64) CODEC(ZSTD(3)),

                    `pre_close` Nullable(Float64) CODEC(ZSTD(3)),

                    `paused` Nullable(UInt8) DEFAULT 0 CODEC(ZSTD(3)),

                    `open_interest` Nullable(Int64) CODEC(ZSTD(3))
                )
                ENGINE = ReplacingMergeTree
                PARTITION BY sipHash64(symbol_id) % 256
                PRIMARY KEY (symbol_id,
                datetime)
                ORDER BY (symbol_id,
                datetime)
                SETTINGS index_granularity = 8192;
            """
        elif table_name == "merged_1m":
            sql = f"""
                CREATE TABLE IF NOT EXISTS {collection}.`{table_name}`
                (

                    `symbol_id` String CODEC(ZSTD(3)),

                    `datetime` DateTime64(3) CODEC(DoubleDelta,
                ZSTD(3)),

                    `open` Nullable(Float64) CODEC(ZSTD(3)),

                    `high` Nullable(Float64) CODEC(ZSTD(3)),

                    `low` Nullable(Float64) CODEC(ZSTD(3)),

                    `close` Nullable(Float64) CODEC(ZSTD(3)),

                    `volume` Nullable(Int64) CODEC(ZSTD(3)),

                    `money` Nullable(Float64) CODEC(ZSTD(3)),

                    `factor` Nullable(Float64) CODEC(ZSTD(3)),

                    `high_limit` Nullable(Float64) CODEC(ZSTD(3)),

                    `low_limit` Nullable(Float64) CODEC(ZSTD(3)),

                    `avg` Nullable(Float64) CODEC(ZSTD(3)),

                    `pre_close` Nullable(Float64) CODEC(ZSTD(3)),

                    `paused` Nullable(UInt8) DEFAULT 0 CODEC(ZSTD(3)),

                    `open_interest` Nullable(Int64) CODEC(ZSTD(3))
                )
                ENGINE = ReplacingMergeTree
                PARTITION BY sipHash64(symbol_id) % 256
                PRIMARY KEY (symbol_id,
                datetime)
                ORDER BY (symbol_id,
                datetime)
                SETTINGS index_granularity = 8192;
            """
        elif table_name == "merged":
            sql = f"""
                CREATE TABLE IF NOT EXISTS {collection}.`{table_name}`
                (
                    `symbol_id` String CODEC(ZSTD(3)),
                    `type` String CODEC(ZSTD(3)),
                    `date` Date CODEC(DoubleDelta, ZSTD(3)),
                    `instrument_type` String CODEC(ZSTD(3)),
                    `status` Nullable(UInt8) DEFAULT 0 CODEC(ZSTD(3))
                    
                )
                ENGINE = ReplacingMergeTree
                PARTITION BY sipHash64(type) % 256
                PRIMARY KEY (symbol_id, date, type)
                ORDER BY (symbol_id, date, type)
                SETTINGS index_granularity = 8192;
            """
        cls.writer_conn.execute(sql)

    @classmethod
    def clear_writer_table(
        cls, table: Literal["merged_1d", "merged_1m", "merged_tick", "merged"]
    ):
        clear_sql = f" truncate table default.`{table}`"
        cls.writer_conn.execute(clear_sql)

    @classmethod
    @timeit
    def get_major_contract_dat(cls, product: str, exchange: str) -> pl.DataFrame:
        """
        获取主力合约数据

        """
        sql = f"""
                SELECT * FROM jq.tick t 
                WHERE symbol_id IN (
                SELECT symbol_id 
                FROM jq.`1d`
                WHERE money IN (
                    SELECT money 
                    FROM jq.`1d` d 
                    WHERE symbol_id = '{product}9999.{exchange}'
                )
                AND symbol_id != '{product}9999.{exchange}'
                );
        """
        with cls.lock:
            columns_info = cls.conn_reader.execute("DESCRIBE TABLE jq.`tick`")
            schemas = [column[0] for column in columns_info]
            cols = cls.conn_reader.execute(sql, columnar=True)
        df = pl.DataFrame(cols, schema=schemas)
        return df

    @classmethod
    def get_major_contract_id(
        cls, product: str, exchange: str, date: str = "2024-07-19"
    ) -> str:
        """
        获取主力合约代码
        """
        sql_xt = f"""
                SELECT symbol_id 
                FROM xt.`1d`
                WHERE (open_interest,settlement_price) in (
                    SELECT open_interest,settlement_price 
                    FROM xt.`1d` d 
                    WHERE symbol_id = '{product}9999.{exchange}'
                    AND `datetime` = '{date} 00:00:00'
                )
                AND `datetime` = '{date} 00:00:00'
                AND symbol_id LIKE '%.{exchange}%'
                AND symbol_id != '{product}9999.{exchange}'
                AND symbol_id != '{product}8888.{exchange}';
        """
        sq_jq = f"""
                SELECT symbol_id 
                FROM jq.`1d`
                WHERE (open_interest,avg) in (
                    SELECT open_interest,avg 
                    FROM jq.`1d` d 
                    WHERE symbol_id = '{product}9999.{exchange}'
                    AND `datetime` = '{date} 00:00:00'
                )
                AND `datetime` = '{date} 00:00:00'
                AND symbol_id LIKE '%.{exchange}%'
                AND symbol_id != '{product}9999.{exchange}'
                AND symbol_id != '{product}8888.{exchange}';
        """
        with cls.lock:
            rows = cls.conn_reader.execute(sql_xt, columnar=True)
        if not rows:
            with cls.lock:
                rows = cls.conn_reader.execute(sq_jq, columnar=True)
        assert rows, "没有主力合约数据"
        assert len(rows[0]) == 1, f"主力合约数量不唯一,得到{rows[0]}"
        major_contract = rows[0][0]
        if isinstance(major_contract, np.str_):
            return major_contract.__str__()
        return major_contract

    @classmethod
    def get_secondery_id(cls, product: str, exchange: str, date: str) -> str:
        """
        获得次主力合约代码, 规则是取日线数据持仓量第二大合约
        注意中金所主力合约的规则不一样
        同花顺规则也不一样（ag,2024-07-22）
        """
        symbol_id = (
            f"{product}___.{exchange}"
            if exchange == "CZCE"
            else f"{product}____.{exchange}"
        )
        sql = f"""
                SELECT symbol_id 
                FROM jq.`1d`
                WHERE `datetime` = '{date} 00:00:00'
                AND symbol_id LIKE '{symbol_id}'
                AND symbol_id != '{product}8888.{exchange}'
                AND symbol_id != '{product}9999.{exchange}'
                ORDER BY open_interest DESC limit 2;
        """
        with cls.lock:
            rows = cls.conn_reader.execute(sql, columnar=True)
        assert rows, "没有次主力合约数据"
        assert (
            len(rows[0]) == 2
        ), f"{product}-{exchange}-{date}次主力合约数量不确定, 仅包含{len(rows[0])}个"
        if isinstance(rows[0][1], np.str_):
            return rows[0][1].__str__()
        return rows[0][1]

    @classmethod
    def get_last_snapshot(
        cls,
        symbol_ids: np.ndarray,
        date: str = "2024-07-19",
        fields: str = "symbol_id,datetime,current,a1_p,b1_p,a1_v,b1_v,volume,position",
    ) -> np.ndarray:
        """
        获取上一个快照数据
        date 需要为交易日。
        """
        start_datetime = " ".join([date, "15:00:00"])
        end_datetime = " ".join([date, "15:02:00"])
        snapshot = creat_snapshot_array(len(symbol_ids), len(fields.split(",")) - 2)
        idx = 0
        with cls.lock:
            for symbol_id in symbol_ids:
                sql = f"""
                select {fields} from 
                jq.`tick` 
                where datetime between '{start_datetime}' and  '{end_datetime}' 
                and symbol_id = '{symbol_id}' order by datetime asc limit 1
                """
                row = cls.conn_reader.execute(sql)
                if not row:
                    idx += 1
                    continue
                snapshot[idx] = np.array(row[0][2:], dtype=np.float64)
                # volume要替换为0, 每日重新结算
                snapshot[idx, -2] = 0
                idx += 1
        return snapshot

    @classmethod
    def get_symbols_tick(
        cls,
        symbols: list,
        date: str = "2024-07-19",
        fields: str = "symbol_id,datetime,current,a1_p,b1_p,a1_v,b1_v,volume,position",
    ):
        """
        获取指定若干合约的Tick数据
        """
        start_datetime = " ".join([cls.get_last_trading_day(date), "20:55:00"])
        end_datetime = " ".join([date, "15:02:00"])
        sql = f"select {fields} from jq.`tick` where datetime between '{start_datetime}' and  '{end_datetime}' and symbol_id in {tuple(symbols)};"
        with cls.lock:
            rows = cls.conn_reader.execute(sql, columnar=True)
        if not rows:
            logger.error(f"No data found for {symbols} on {date}")
            raise AssertionError(f"No data found for {symbols} on {date}")
        if fields == "*":
            with cls.lock:
                columns_info = cls.conn_reader.execute("DESCRIBE TABLE jq.`tick`")
            schemas = [column[0] for column in columns_info]
        else:
            schemas = fields.split(",")
        for i in range(2, len(schemas)):
            rows[i] = rows[i].astype(np.float64)
        df = pl.DataFrame(rows, schema=schemas, orient="col", strict=False)
        return df

    @classmethod
    @timeit
    def get_1d_dat(cls, condition: str = ""):
        """
        获取日线数据
        """
        fields = "symbol_id,datetime"
        sql = f"select {fields} from jq.`1d` {condition}"
        with cls.lock:
            rows = cls.conn_reader.execute(sql, columnar=True)
        df = pl.DataFrame(rows, schema=fields.split(","))
        return df

    @classmethod
    def get_product_dict(cls) -> dict[str, str]:
        """
        获得所有产品和交易所组成的字典
        """
        with cls.lock:
            rows = cls.cursor_1d.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        product_dict = {}
        for row in rows:
            product_name, exchange_name = row[0].split("_")
            if product_name not in product_dict:
                product_dict[product_name] = exchange_name
        return product_dict

    @classmethod
    def get_product_dict_clickhouse(
        cls, kind: Literal["FUTURES", "OPTIONS"] = "FUTURES"
    ) -> dict[str, str]:
        """
        获得所有产品和交易所组成的字典
        """
        with cls.lock:
            sql = f"""
                select
                    distinct extract(symbol_id, '(^[a-zA-Z]+)\\d+.([A-Z]+)$') as lt,
                    extract(symbol_id, '\\.([A-Z]+)$') AS exchange_code
                from
                    jq.Symbols
                where
                    instrument_type = '{kind}'
        """
            result = DBHelper.conn_reader.execute(sql, columnar=True)
        return {str(result[0][i]): str(result[1][i]) for i in range(len(result[0]))}

    @classmethod
    def get_three_kinds_dic(cls, kind: Literal["9999", "8888", "7777"]):
        """
        获得clickhouse中有的主连|指数|次主力连所有产品和交易所组成的字典
        """
        sql = f"""
            select
                distinct extract(symbol_id, '(^[a-zA-Z]+)\\d+.([A-Z]+)$') as lt,
                extract(symbol_id, '\\.([A-Z]+)$') AS exchange_code
            from
                jq.Symbols
            where
                instrument_type = 'FUTURES'
                AND symbol_id like '%{kind}%'
                """
        with cls.lock:
            result = DBHelper.conn_reader.execute(sql, columnar=True)
        return {str(result[0][i]): str(result[1][i]) for i in range(len(result[0]))}

    @classmethod
    def get_option_symbols(cls, kind: Literal["1d", "1m"]) -> list[str]:
        """
        获得所有未处理的期权合约id
        """
        sql = """
            select
                distinct symbol_id,
            from
                jq.Symbols
            where
                instrument_type = 'OPTIONS'
        """
        processed_symbols = cls.get_processed_option_symbols(kind)
        with cls.lock:
            result = DBHelper.conn_reader.execute(sql, columnar=True)
        return [str(s) for s in result[0] if s not in processed_symbols]

    @classmethod
    def get_expired_date(cls, symbol_id: str) -> str:
        """
        获得期权到期日
        """
        sql = f"""
            select
                end_date
            from
                jq.Symbols
            where
                symbol_id = '{symbol_id}'
        """
        with cls.lock:
            result = DBHelper.conn_reader.execute(sql, columnar=True)
        return str(result[0][0])

    @classmethod
    def get_processed_option_symbols(cls, kind: Literal["1d", "1m"]) -> list[str]:
        """
        获得所有已经处理过的期权合约id
        """
        sql = f"""
            select
                distinct symbol_id
            from
                default.merged
            where
                instrument_type = 'option'
                AND status = 1
                AND type = '{kind}'
        """
        with cls.lock:
            result = DBHelper.writer_conn.execute(sql, columnar=True)
        return result[0]

    @classmethod
    def get_index_dict(cls) -> dict[str, str]:
        """
        获得所有指数产品和交易所组成的字典
        """
        return cls.get_three_kinds_dic("8888")

    @classmethod
    def get_major_dict(cls) -> dict[str, str]:
        """
        获得所有主连产品和交易所组成的字典
        """
        return cls.get_three_kinds_dic("9999")

    @classmethod
    def get_secondery_dict(cls) -> dict[str, str]:
        """
        获得所有次主力连产品和交易所组成的字典
        """
        return cls.get_three_kinds_dic("7777")

    @classmethod
    @timeit
    def get_pro_dates(
        cls,
        product: str,
        exchange: str,
        symbol_type: Literal["9999", "8888", "7777"],
        kind: Literal["1d", "1m", "tick"],
    ) -> list:
        """
        获取指定品种的交易日列表, 以日线交易日列表为基准
        """
        symbol_id = f"{product}{symbol_type}.{exchange}"
        last_date = cls.get_last_date(symbol_id, kind)
        symbol_pat = (
            f"{product}___.{exchange}"
            if exchange == "CZCE"
            else f"{product}____.{exchange}"
        )
        query_dates = f"""
                    SELECT
                        distinct Date(datetime) as date
                    FROM
                        jq.tick
                    WHERE
                        symbol_id like '{symbol_pat}'
                    order by
                        date asc
        """
        if last_date:
            query_dates = f"""
                        SELECT
                            distinct Date(datetime) as date
                        FROM
                            jq.tick
                        where
                            Date(datetime) > '{last_date}'
                            and symbol_id like '{symbol_pat}'
                        order by
                            date asc"""
        with cls.lock:
            try:
                day_date = cls.conn_reader.execute(query_dates, columnar=True)[0]
            except IndexError:
                logger.error(f"No data found for {product} of {exchange}")
                raise AssertionError
        dates = [str(d) for d in day_date]
        return dates

    @classmethod
    def get_sym_dates(
        cls,
        symbol_id: str,
        kind: Literal["1d", "1m", "tick"],
        table: Literal["1d", "1m", "tick"] = "tick",
    ) -> list:
        last_date = cls.get_last_date(symbol_id, kind)
        query_1d = f"""
                    SELECT
                        distinct Date(datetime) as date
                    FROM
                        jq.{table}
                    WHERE
                        symbol_id = '{symbol_id}'
                    order by
                        datetime asc
        """
        if last_date:
            query_1d = f"""
                        SELECT
                            distinct Date(datetime) as date
                        FROM
                            jq.{table}
                        where
                            Date(datetime) > '{last_date}'
                            and symbol_id = '{symbol_id}'
                        order by
                            datetime asc"""
        with cls.lock:
            try:
                day_date = cls.conn_reader.execute(query_1d, columnar=True)[0]
            except IndexError:
                logger.error(f"No data found for {symbol_id}")
                raise AssertionError
        dates = [str(d) for d in day_date]
        return dates

    @classmethod
    def get_last_trading_day(cls, date: str) -> str:
        """
        获取上一个交易日
        """
        sql = f"SELECT MAX(DATE(datetime)) as last_trading_day FROM jq.1d WHERE symbol_id = 'a9999.DCE' AND datetime < '{date} 00:00:00'"
        with cls.lock:
            try:
                last_trading_day = cls.conn_reader.execute(sql, columnar=True)[0][
                    0
                ].__str__()
            except IndexError:
                logger.warning(f"No trading date before {date}")
                return date
        return last_trading_day

    @classmethod
    def save_db(
        cls,
        symbol_id: str,
        df: pl.DataFrame | pd.DataFrame,
        table: Literal["merged_tick", "merged_1d", "merged_1m"],
        date: Union[str, tuple[str, str]] = "all",
    ):
        """
        保存数据到sqlite数据库
        """
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        with cls.lock:
            cls.writer_conn.import_df(table, df, ("symbol_id", "datetime"))
        if isinstance(date, str):
            logger.info(f"saved {symbol_id} of {date} data to {table} successfully")
        else:
            logger.info(
                f"saved {symbol_id} of {date[0]}-{date[1]} data to {table} successfully"
            )

    @classmethod
    def get_all_contracts(
        cls, product: str, exchange: str, date: str = "2024-07-19"
    ) -> np.ndarray:
        """
        获取所有合约
        """
        sql = f"select symbol_id from {product}_{exchange} where datetime = '{date} 00:00:00';"
        with cls.lock:
            rows = cls.cursor_1d.execute(sql).fetchall()
        symbol_ids = []
        for row in rows:
            symbol_ids.append(row[0])
        return symbol_ids

    @classmethod
    def get_all_contracts_clickhouse(
        cls, product: str, exchange: str, date: str = "2024-07-19"
    ) -> np.ndarray:
        """
        获取所有合约
        """
        symbol_pat = (
            f"{product}___.{exchange}"
            if exchange == "CZCE"
            else f"{product}____.{exchange}"
        )
        sql = f"""
                select
                    symbol_id
                from
                    jq.1d
                where
                    datetime = '{date} 00:00:00'
                    and symbol_id like '%{symbol_pat}'
                    and symbol_id != '{product}9999.{exchange}'
                    and symbol_id != '{product}7777.{exchange}'
                    and symbol_id != '{product}8888.{exchange}';
        """
        with cls.lock:
            result = DBHelper.conn_reader.execute(sql, columnar=True)
        assert result, "没有合约数据"
        return [str(r) for r in result[0]]

    @classmethod
    def insert_records(cls, record: Union[tuple, list]):
        """
        记录已经处理过的数据日期到clickhouse
        """
        with cls.lock:
            if isinstance(record, list):
                values = []
                for r in record:
                    sql = f"insert into merged (symbol_id, type, date, instrument_type,status) values {r};"
                    cls.writer_conn.execute(sql)
            else:
                sql = f"insert into merged (symbol_id, type, date,instrument_type,status) values {record};"
                cls.writer_conn.execute(sql)

    @classmethod
    def update_records(cls, record: Union[dict, list]):
        """
        更新已经处理过的数据日期到clickhouse
        """
        cls.insert_records(record)

    @classmethod
    def process_from_date(
        cls, date: str, symbol_id: str, kind: Literal["tick", "1m", "1d"]
    ):
        """
        从指定日期开始处理数据. 通过手动插入record, 来控制数据处理的开始日期
        """
        record = (symbol_id, kind, date)
        if cls.is_processed(record):
            logger.warning(f"{symbol_id} of {date} already processed")
            return
        with cls.lock:
            cls.insert_records(record)

    @classmethod
    def is_processed(cls, record) -> bool:
        """
        判断某个产品的某天数据是否已经处理过
        """
        with cls.lock:
            sql = f"select count(*) as count from default.merged where symbol_id = '{record[0]}' and type = '{record[1]}' and date = '{record[2]}';"
            count = cls.writer_conn.execute(sql, columnar=True)[0][0]
            is_processed = count > 0
        return is_processed

    @classmethod
    @timeit
    def group_by_product(cls, df: pl.DataFrame):
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

    @staticmethod
    def delete_1d_dbfile():
        global temp_dir
        db_file = temp_dir / "future_1d.db"
        if db_file.exists():
            DBHelper.conn_1d.close()
            db_file.unlink()

    @classmethod
    def execute_1d_extract(cls):
        """
        提取日线数据的symbl_id, datetime字段, 并按照产品分组, 保存到sqlite临时数据库
        """
        condition = "where match(symbol_id,'^[a-zA-Z]+\\d+.[a-zA-Z]+$') and symbol_id not like '%8888%' and symbol_id not like '%9999%' and paused = 0 and volume > 0;"
        df = cls.get_1d_dat(condition)
        product_gro = cls.group_by_product(df)
        logg = setup_logger("day", "1d_extract.log")
        for pro, dat in product_gro:
            record = {"product": pro[0], "date": "all", "type": "day"}
            if cls.is_processed(record):
                logg.warning(f"{pro[0]} of {dat['exchange'][0]} already processed")
                continue
            exchange = dat["exchange"][0]
            dat = dat.drop("pro_part")
            dat = dat.sort("datetime")
            cls.save_1d_dat(dat, pro[0], exchange)
            cls.insert_records(record)
            logg.info(f"Saved {pro[0]} of {exchange} data to future_1d.db successfully")

    @classmethod
    def extract_three_kinds_1d(cls, kind: Literal["9999", "8888", "7777"]):
        """
        提取单个类别日线数据的symbl_id, datetime字段, 并按照产品分组, 保存到sqlite临时数据库
        """
        condition = f"where symbol_id like '%{kind}%' and paused = 0 and volume > 0;"
        df = cls.get_1d_dat(condition)
        product_gro = cls.group_by_product(df)
        logg = setup_logger("day", "1d_extract.log")
        for pro, dat in product_gro:
            product = f"{pro[0]}{kind}"
            exchange = dat["exchange"][0]
            dat = dat.drop("pro_part")
            dat = dat.sort("datetime")
            cls.save_1d_dat(dat, product, exchange)
            logg.info(
                f"Saved {product} of {exchange} data to future_1d.db successfully"
            )

    @classmethod
    @timeit
    def extract_1d_1pro(
        cls,
        product: str,
        exchange: str,
        range_type: Literal["all", "last_date"],
        symbol_type: Literal["9999", "8888", "7777", None] = None,
        kind: Literal["tick", "1m", "1d", None] = None,
    ):
        """
        提取单个产品的日线数据的symbl_id, datetime字段, 保存到sqlite临时数据库
        @parameter
        product: 产品代码
        exchange: 交易所代码
        range_type: 提取类型, all: 提取所有数据, last_date: 从recorde表中获取最后日期开始提取
        symbol_type: 合约类型, 9999: 主连合约, 8888: 指数合约, 7777: 次主连合约, 只有last_date时才需要提供
        kind: 数据类型, tick: 盘口数据, 1m: 1分钟数据, 1d: 日线数据, 只有last_date时才需要提供
        """
        if range_type == "all":
            if exchange == "CZCE":
                condition = f"where symbol_id like '{product}___.{exchange}' and paused = 0 and volume > 0;"
            else:
                condition = f"where symbol_id like '{product}____.{exchange}' and paused = 0 and volume > 0;"

            df = cls.get_1d_dat(condition)
            if df.shape[0] == 0:
                logger.error(f"No data found for {product} of {exchange}")
                return
            df = df.sort("datetime")
            cls.save_1d_dat(df, product, exchange)
            logger.success(
                f"Saved {product} of {exchange} data to future_1d.db successfully"
            )
        elif range_type == "last_date":
            assert (
                type is not None and kind is not None
            ), "type and kind must be provided when range_type is last_date"
            symbol_id = f"{product}{symbol_type}.{exchange}"
            last_date = cls.get_last_date(symbol_id, kind)
            if not last_date:
                cls.extract_1d_1pro(product, exchange, "all")
            condition = f"where symbol_id like '{product}___.{exchange}' and datetime > '{last_date} 00:00:00' and paused = 0 and volume > 0;"
            df = cls.get_1d_dat(condition)
            if df.shape[0] == 0:
                logger.error(
                    f"No data found for {product} of {exchange} where datetime > '{last_date} 00:00:00'"
                )
                return
            df = df.sort("datetime")
            cls.save_1d_dat(df, product, exchange)
            logger.success(
                f"Saved {product} of {exchange} data to future_1d.db successfully"
            )

    @classmethod
    def extract_1d_all(cls):
        """
        从每个产品的最新记录开始, 提取所有产品的日线数据的symbl_id, datetime字段, 保存到sqlite临时数据库
        """
        if not cls.db_file.exists():
            create_1d_dbfile(cls.db_file)
        producnt_dict = cls.get_product_dict()
        for product, exchange in producnt_dict.items():
            cls.extract_1d_1pro(product, exchange, "last_date")

    @classmethod
    def get_pro_new_dates(
        cls,
        product: str,
        exchange: str,
        symbol_type: Literal["9999", "8888", "7777"],
        kind: Literal["tick", "1m", "1d"],
    ) -> list:
        """
        获取指定品种的最新交易日列表
        """
        symbol_id = f"{product}{symbol_type}.{exchange}"
        last_date = cls.get_last_date(symbol_id, kind=kind)
        if not last_date:
            return cls.get_pro_dates(product, exchange)
        else:
            pass

    @classmethod
    def get_last_date(
        cls,
        symbol_id: str,
        kind: Literal["tick", "1m", "1d"],
    ) -> str:
        """
        获取最后日期
        """
        with cls.lock:
            sql = f"select max(date) as last_date from merged where symbol_id = '{symbol_id}' and type = '{kind}'"
            last_date = cls.writer_conn.execute(sql, columnar=True)[0][0]
        return last_date

    @classmethod
    @timeit
    def save_1d_dat(cls, df: pl.DataFrame, product: str, exchange: str):
        """
        保存日线数据到sqlite数据库
        """
        pandas_df = df.to_pandas()
        pandas_df["datetime"] = pandas_df["datetime"].astype("datetime64[us]")
        with cls.lock:
            pandas_df.to_sql(
                f"{product}_{exchange}", cls.conn_1d, if_exists="append", index=False
            )

    @classmethod
    def clear_records(cls, mapping: dict = {}):
        """
        清空clickhouse记录
        """
        with cls.lock:
            if mapping:
                condition = []
                for key, value in mapping.items():
                    condition.append(f"{key} = '{value}'")
                condition = " and ".join(condition)
                sql = f"ALTER TABLE merged delete where {condition}"
                cls.writer_conn.execute(sql)
            else:
                sql = "truncate table merged"
                cls.writer_conn.execute(sql)

    @staticmethod
    def get_1m_sql(
        symbol_id: str,
        date: str,
        database: Literal["jq", "xt", "default"] = "jq",
    ) -> str:
        """
        获取1分钟数据sql, date为交易日
        拼index的时候, 需要先拼好上一个交易日的1d数据
        """
        last_date = DBHelper.get_last_trading_day(date)
        exchange = symbol_id.split(".")[1]
        if database == "default":
            table = "default.merged_1d"
            tick_table = "default.merged_tick"
        else:
            table = f"{database}.1d"
            tick_table = f"{database}.tick"
        if exchange == "CFFEX" or exchange == "SSE" or exchange == "SZSE":
            sql = f"""
                WITH
                    -- 获取前一日的收盘价
                    previous_close AS (
                        SELECT current AS last_close
                        FROM {tick_table}
                        WHERE symbol_id = '{symbol_id}' AND toDate(datetime) = '{last_date}'
                        ORDER by datetime DESC
                        LIMIT 1
                    ),

                    -- limits AS (
                    --     SELECT
                    --         (settlement_price * 1.1) AS high_limit,
                    --         (settlement_price * 0.9) AS low_limit
                    --     FROM xt.`1d`
                    --     WHERE symbol_id = '{symbol_id}' AND toDate(datetime) = '{last_date}'
                    --     LIMIT 1
                    -- ),

                    pre_today_data AS (
                        SELECT 
                            *, ROW_NUMBER() over (order by datetime) as rn
                        FROM 
                            {tick_table} t
                        WHERE symbol_id = '{symbol_id}'
                            AND datetime BETWEEN '{date} 09:25:00' AND '{date} 15:02:00'
                    ),

                    today_data as(
                        SELECT
                                t1.*,
                        CASE
                            WHEN t2.rn = 0 THEN (t1.volume)
                            ELSE t1.volume - t2.volume
                        END
                            AS volume_diff,
                        CASE
                            WHEN t2.rn = 0 THEN (t1.money)
                            ELSE t1.money - t2.money
                        END
                            AS money_diff
                        FROM
                            pre_today_data t1
                        LEFT JOIN
                            pre_today_data t2
                        ON
                            t1.rn = t2.rn + 1
                    ),


                    noon_close_data as(
                        SELECT
                            current,
                            money_diff,
                            volume_diff
                        from
                            today_data
                        WHERE
                            datetime BETWEEN '{date} 11:30:00' AND '{date} 11:40:00'
                        order by datetime DESC limit 1
                    ),

                    last_close_data as(
                        SELECT
                            current,
                            money_diff,
                            volume_diff
                        from
                            today_data
                        WHERE
                            datetime BETWEEN '{date} 15:00:00' AND '{date} 15:02:00'
                        order by datetime DESC limit 1
                    ),

                    -- 获取集合竞价期间的数据
                    -- 除了上期所外, 其他交易所只有一次集合竞价
                    auction_day AS (
                        SELECT
                            toDateTime('{date} 09:31:00') AS minute,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayFirst(x -> x>0, groupArray(current*(volume_diff>0)))) AS open,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayMax(groupArray(high*(volume_diff>0)))) AS high,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayMin(arrayFilter(x -> x > 0, groupArray(low * (volume_diff > 0))))) AS low,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayLast(x -> True, groupArray(current)),arrayLast(x -> x>0, groupArray(current*(volume_diff>0)))) AS close,
                            arrayLast(x -> true, groupArray(volume)) AS volume,
                            arrayLast(x -> true, groupArray(money)) AS money,
                            arrayLast(x -> true, groupArray(position)) AS open_interest
                        FROM today_data
                        WHERE datetime >= '{date} 09:29:00' AND datetime < '{date} 09:31:00'
                        HAVING COUNT(*) > 0
                    ),

                    -- 获取正常交易时段的数据
                    normal_data AS (
                        SELECT
                            toStartOfMinute(datetime) + INTERVAL 1 MINUTE AS minute,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayFirst(x -> x>0, groupArray(current*(volume_diff>0)))) AS open,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayMax(groupArray(current*(volume_diff>0)))) AS high,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayMin(arrayFilter(x -> x > 0, groupArray(current * (volume_diff > 0))))) AS low,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayLast(x -> True, groupArray(current)),arrayLast(x -> x>0, groupArray(current*(volume_diff>0)))) AS close,
                            arrayLast(x -> true, groupArray(volume)) AS volume,
                            arrayLast(x -> true, groupArray(money)) AS money,
                            arrayLast(x -> true, groupArray(position)) AS open_interest
                        FROM today_data
                        WHERE symbol_id = '{symbol_id}'
                            AND datetime BETWEEN '{date} 09:31:00' AND '{date} 15:00:00'
                        GROUP BY minute
                    ),

                    -- 处理集合竞价未成功的情况
                    combined_data AS (
                        SELECT
                            *,
                            row_number() OVER (ORDER BY minute) AS rn
                        FROM (
                            SELECT * FROM auction_day
                            UNION ALL
                            SELECT * FROM normal_data
                        ) ORDER BY minute
                    ),


                    diff AS (
                        SELECT
                            t1.*,
                            t1.volume - t2.volume AS volume_diff,
                            t1.money - t2.money AS money_diff,
                            CASE
                                WHEN t2.rn = 0 THEN (SELECT last_close FROM previous_close)
                                ELSE t2.close
                            END
                                AS pre_close
                        FROM
                            combined_data t1
                        LEFT JOIN
                            combined_data t2
                        ON
                            t1.rn = t2.rn + 1
                        )

                -- 最终查询
                SELECT
                    '{symbol_id}' AS symbol_id,
                    minute as datetime,
                    open,
                    case
                        when datetime='{date} 11:30:00' and (SELECT volume_diff from noon_close_data) > 0 THEN GREATEST(high,(SELECT current from noon_close_data))
                        when datetime='{date} 15:00:00' and (SELECT volume_diff from last_close_data) > 0 THEN GREATEST(high,(SELECT current from last_close_data))
                        ELSE high
                    END
                        as high,
                    case
                        when datetime='{date} 11:30:00' and (SELECT volume_diff from noon_close_data) > 0 THEN LEAST(low,(SELECT current from noon_close_data))
                        when datetime='{date} 15:00:00' and (SELECT volume_diff from last_close_data) > 0 THEN LEAST(low,(SELECT current from last_close_data))
                        ELSE low
                    END
                        as low,
                    case
                        when datetime='{date} 11:30:00' and (SELECT volume_diff from noon_close_data) > 0 THEN (SELECT current from noon_close_data)
                        when datetime='{date} 15:00:00' and (SELECT volume_diff from last_close_data) > 0 THEN (SELECT current from last_close_data)
                        ELSE close
                    END
                        as close,
                    case 
                        when datetime='{date} 11:30:00' THEN volume_diff + COALESCE((SELECT volume_diff FROM noon_close_data), 0)
                        when datetime='{date} 15:00:00' THEN volume_diff+COALESCE((SELECT volume_diff FROM last_close_data), 0)
                        ELSE volume_diff
                    END
                        as volume,
                    case 
                        when datetime='{date} 11:30:00' THEN money_diff+COALESCE((SELECT money_diff FROM noon_close_data), 0)
                        when datetime='{date} 15:00:00' THEN money_diff+COALESCE((SELECT money_diff FROM last_close_data), 0)
                        ELSE money_diff
                    END
                        as money,
                    pre_close,
                    -- (SELECT high_limit FROM limits) AS high_limit,
                    -- (SELECT low_limit FROM limits) AS low_limit,
                    open_interest
                FROM diff
                where
                    datetime not in ('{date} 11:31:00','{date} 15:01:00')
                ORDER BY minute;
                """
        else:
            sql = f"""
                WITH
                    -- 获取前一日的收盘价
                    previous_close AS (
                        SELECT current AS last_close
                        FROM {tick_table}
                        WHERE symbol_id = '{symbol_id}' AND toDate(datetime) = '{last_date}'
                        ORDER by datetime DESC
                        LIMIT 1
                    ),

                    -- limits AS (
                    --     SELECT
                    --        (settlement_price * 1.1) AS high_limit,
                    --        (settlement_price * 0.9) AS low_limit
                    --     FROM xt.`1d`
                    --     WHERE symbol_id = '{symbol_id}' AND toDate(datetime) = '{last_date}'
                    --     LIMIT 1
                    -- ),

                    pre_today_data AS (
                        SELECT 
                            *, ROW_NUMBER() over (order by datetime) as rn
                        FROM 
                            {tick_table} t
                        WHERE symbol_id = '{symbol_id}'
                            AND datetime BETWEEN '{last_date} 16:00:00' AND '{date} 15:02:00'
                    ),

                    today_data as(
                        SELECT
                                t1.*,
                        CASE
                            WHEN t2.rn = 0 THEN (t1.volume)
                            ELSE t1.volume - t2.volume
                        END
                            AS volume_diff,
                        CASE
                            WHEN t2.rn = 0 THEN (t1.money)
                            ELSE t1.money - t2.money
                        END
                            AS money_diff
                        FROM
                            pre_today_data t1
                        LEFT JOIN
                            pre_today_data t2
                        ON
                            t1.rn = t2.rn + 1
                    ),
                    
                    mid_close_data as(
                        SELECT
                            current,
                            money_diff,
                            volume_diff
                        from
                            today_data
                        WHERE 
                            datetime BETWEEN '{date} 10:15:00' AND '{date} 10:30:00'
                        order by datetime DESC limit 1
                    ),
                    
                    noon_close_data as(
                        SELECT
                            current,
                            money_diff,
                            volume_diff
                        from
                            today_data
                        WHERE 
                            datetime BETWEEN '{date} 11:30:00' AND '{date} 11:40:00'
                        order by datetime DESC limit 1
                    ),
                    
                    last_close_data as(
                        SELECT
                            current,
                            money_diff,
                            volume_diff
                        from
                            today_data
                        WHERE 
                            datetime BETWEEN '{date} 15:00:00' AND '{date} 15:02:00'
                        order by datetime DESC limit 1
                    ),

                    -- 获取集合竞价期间的数据
                    -- 节假日之前无夜盘, 因此要求COUNT>0
                    auction_night AS (
                        SELECT
                            toDateTime('{last_date} 21:01:00') AS minute,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayFirst(x -> x>0, groupArray(current*(volume_diff>0)))) AS open,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayMax(groupArray(current*(volume_diff>0)))) AS high,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayMin(arrayFilter(x -> x > 0, groupArray(current * (volume_diff > 0))))) AS low,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayLast(x -> True, groupArray(current)),arrayLast(x -> x>0, groupArray(current*(volume_diff>0)))) AS close,
                            arrayLast(x -> true, groupArray(volume)) AS volume,
                            arrayLast(x -> true, groupArray(money)) AS money,
                            arrayLast(x -> true, groupArray(position)) AS open_interest
                        FROM today_data
                        WHERE datetime >= '{last_date} 20:59:00' AND datetime < '{last_date} 21:01:00'
                        HAVING COUNT(*) > 0
                    ),

                    -- 除了上期所外, 其他交易所只有一次集合竞价
                    auction_day AS (
                        SELECT
                            toDateTime('{date} 09:01:00') AS minute,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayFirst(x -> x>0, groupArray(current*(volume_diff>0)))) AS open,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayMax(groupArray(current*(volume_diff>0)))) AS high,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayMin(arrayFilter(x -> x > 0, groupArray(current * (volume_diff > 0))))) AS low,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayLast(x -> True, groupArray(current)),arrayLast(x -> x>0, groupArray(current*(volume_diff>0)))) AS close,
                            arrayLast(x -> true, groupArray(volume)) AS volume,
                            arrayLast(x -> true, groupArray(money)) AS money,
                            arrayLast(x -> true, groupArray(position)) AS open_interest
                        FROM today_data
                        WHERE datetime >= '{date} 08:59:00' AND datetime < '{date} 09:01:00'
                        HAVING COUNT(*) > 0
                    ),

                    -- 获取正常交易时段的数据
                    normal_data AS (
                        SELECT
                            toStartOfMinute(datetime) + INTERVAL 1 MINUTE AS minute,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayFirst(x -> x>0, groupArray(current*(volume_diff>0)))) AS open,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayMax(groupArray(current*(volume_diff>0)))) AS high,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayFirst(x -> True, groupArray(current)),arrayMin(arrayFilter(x -> x > 0, groupArray(current * (volume_diff > 0))))) AS low,
                            if(arrayAll(x -> x = 0,groupArray(volume_diff>0)), arrayLast(x -> True, groupArray(current)),arrayLast(x -> x>0, groupArray(current*(volume_diff>0)))) AS close,
                            arrayLast(x -> true, groupArray(volume)) AS volume,
                            arrayLast(x -> true, groupArray(money)) AS money,
                            arrayLast(x -> true, groupArray(position)) AS open_interest
                        FROM today_data
                        WHERE symbol_id = '{symbol_id}'
                            AND datetime BETWEEN '{last_date} 21:01:00' AND '{date} 08:00:00'
                            or datetime BETWEEN '{date} 09:01:00' AND '{date} 15:00:00'
                        GROUP BY minute
                    ),

                    -- 处理集合竞价未成功的情况
                    combined_data AS (
                        SELECT
                            *,
                            row_number() OVER (ORDER BY minute) AS rn
                        FROM (
                            SELECT * FROM auction_day
                            UNION ALL
                            SELECT * FROM normal_data
                            UNION ALL
                            SELECT * FROM auction_night
                        ) ORDER BY minute
                    ),


                    diff AS (
                        SELECT
                            t1.*,
                            t1.volume - t2.volume AS volume_diff,
                            t1.money - t2.money AS money_diff,
                            CASE
                                WHEN t2.rn = 0 THEN (SELECT last_close FROM previous_close)
                                ELSE t2.close
                            END
                                AS pre_close
                        FROM
                            combined_data t1
                        LEFT JOIN
                            combined_data t2
                        ON
                            t1.rn = t2.rn + 1
                        )

                -- 最终查询
                SELECT
                    '{symbol_id}' AS symbol_id,
                    minute as datetime,
                    open,
                    case 
                        when datetime='{date} 10:15:00' and (SELECT volume_diff from mid_close_data) > 0 THEN GREATEST(high,(SELECT current from mid_close_data))
                        when datetime='{date} 11:30:00' and (SELECT volume_diff from noon_close_data) > 0 THEN GREATEST(high,(SELECT current from noon_close_data))
                        when datetime='{date} 15:00:00' and (SELECT volume_diff from last_close_data) > 0 THEN GREATEST(high,(SELECT current from last_close_data))
                        ELSE high
                    END
                        as high,
                    case 
                        when datetime='{date} 10:15:00' and (SELECT volume_diff from mid_close_data) > 0 THEN LEAST(low,(SELECT current from mid_close_data))
                        when datetime='{date} 11:30:00' and (SELECT volume_diff from noon_close_data) > 0 THEN LEAST(low,(SELECT current from noon_close_data))
                        when datetime='{date} 15:00:00' and (SELECT volume_diff from last_close_data) > 0 THEN LEAST(low,(SELECT current from last_close_data))
                        ELSE low
                    END
                        as low,
                    case 
                        when datetime='{date} 10:15:00' and (SELECT volume_diff from mid_close_data) > 0 THEN (SELECT current from mid_close_data)
                        when datetime='{date} 11:30:00' and (SELECT volume_diff from noon_close_data) > 0 THEN (SELECT current from noon_close_data)
                        when datetime='{date} 15:00:00' and (SELECT volume_diff from last_close_data) > 0 THEN (SELECT current from last_close_data)
                        ELSE close
                    END
                        as close,
                    case 
                        when datetime='{date} 10:15:00' THEN volume_diff+COALESCE((SELECT volume_diff FROM mid_close_data), 0)
                        when datetime='{date} 11:30:00' THEN volume_diff + COALESCE((SELECT volume_diff FROM noon_close_data), 0)
                        when datetime='{date} 15:00:00' THEN volume_diff+COALESCE((SELECT volume_diff FROM last_close_data), 0)
                        ELSE volume_diff
                    END
                        as volume,
                    case 
                        when datetime='{date} 10:15:00' THEN money_diff+COALESCE((SELECT money_diff FROM mid_close_data), 0)
                        when datetime='{date} 11:30:00' THEN money_diff+COALESCE((SELECT money_diff FROM noon_close_data), 0)
                        when datetime='{date} 15:00:00' THEN money_diff+COALESCE((SELECT money_diff FROM last_close_data), 0)
                        ELSE money_diff
                    END
                        as money,
                    pre_close,
                    -- (SELECT high_limit FROM limits) AS high_limit,
                    -- (SELECT low_limit FROM limits) AS low_limit,
                    open_interest
                FROM diff
                where 
                    datetime not in ('{date} 10:16:00', '{date} 11:31:00','{date} 15:01:00') 
                ORDER BY minute;

            """
        return sql

    @classmethod
    def get_basic_1d_sql(
        cls,
        symbol_id: str,
        condition: str = "",
        database: Literal["jq", "xt", "default"] = "jq",
    ) -> str:
        """
        获取基础日线数据sql语句
        """
        exchange = symbol_id.split(".")[1]
        if database == "default":
            table = "default.merged_1m"
        else:
            table = f"{database}.1m"
        if exchange == "CFFEX":
            sql = f"""
            WITH 
                data_table AS( 
                    SELECT
                        *,
                        toStartOfDay(datetime) AS datetime
                    FROM {table}
                    WHERE 
                        symbol_id = '{symbol_id}' {condition}
                    order by datetime asc
                ),

            --  Pre-aggregate minute data to handle open, close, high, and low values
                pre_aggregated_data AS (
                    SELECT
                        toStartOfDay(datetime) as datetime ,
                        arrayFirst(x -> x > 0, groupArray(open * (volume > 0))) AS open,
                        arrayLast(x -> x > 0, groupArray(close * (volume > 0))) AS close,
                        arrayFirst(x -> true, groupArray(pre_close)) AS pre_close,
                        maxIf(high, volume > 0) AS max_high,
                        minIf(low, volume > 0) AS min_low,
                        arrayLast(x -> true, groupArray(open_interest)) AS open_interest
                    FROM
                        data_table
                    GROUP BY
                        datetime
                )

            --  Aggregate final daily data
            SELECT
                '{symbol_id}' AS symbol_id,
                t1.datetime,
                t1.open,
                t1.max_high as high,
                t1.min_low as low,
                t1.close,
                t1.pre_close,
                sum(t2.volume) as volume,
                sum(t2.money) AS money,
                COALESCE(any(t2.factor),0) AS factor,
                any(t2.paused) AS paused,
                t1.open_interest
            FROM
                pre_aggregated_data t1
            JOIN
                data_table t2
            ON
                t1.datetime = t2.datetime
            GROUP BY
                t1.datetime, t1.open, t1.max_high, t1.min_low, t1.close,t1.open_interest,t1.pre_close
            ORDER BY
                t1.datetime;

            """
            return sql
        sql = f"""
            WITH 
                data_table AS( 
                    SELECT
                        *,
                        toStartOfDay(datetime - INTERVAL 3 hour) AS trading_date
                    FROM {table}
                    WHERE 
                        symbol_id = '{symbol_id}' {condition}
                    order by datetime asc
                ),
                
                -- Step 1: Identify distinct trading sessions
                trading_sessions AS (
                    SELECT
                        trading_date
                    FROM data_table
                    GROUP BY trading_date
                ),

                -- Step 2: Add row numbers to the trading sessions
                session_intervals AS (
                    SELECT 
                        trading_date, 
                        ROW_NUMBER() OVER (ORDER BY trading_date) AS rn
                    FROM trading_sessions
                ),

                -- Step 3: Create intervals by self-joining on row numbers
                --有个bug, 如果最后一天不是到期日, 则还有夜盘, 最后一天的交易日划分就会有问题
                date_intervals AS (
                    SELECT 
                        t1.trading_date AS start_datetime,
                        if(t2.rn >0,t2.trading_date,t1.trading_date) AS end_datetime
                    FROM 
                        session_intervals t1
                    LEFT JOIN 
                        session_intervals t2 ON t1.rn + 1 = t2.rn
                ),

                -- Step 4: Combine minute data with identified trading sessions
                minute_data_with_sessions AS (
                    SELECT 
                        t1.*,
                        CASE 
                            WHEN toHour(t1.datetime) >= 21 OR toHour(t1.datetime)<7 THEN t2.end_datetime
                            ELSE t2.start_datetime
                        END AS trading_day
                    FROM 
                        data_table t1
                    full JOIN 
                        date_intervals t2
                    ON 
                        t1.trading_date = t2.start_datetime 
                ),

            -- Step 5: Pre-aggregate minute data to handle open, close, high, and low values
                pre_aggregated_data AS (
                    SELECT
                        trading_day,
                        arrayFirst(x -> x > 0, groupArray(open * (volume > 0))) AS open,
                        arrayLast(x -> x > 0, groupArray(close * (volume > 0))) AS close,
                        arrayFirst(x -> true, groupArray(pre_close)) AS pre_close,
                        maxIf(high, volume > 0) AS max_high,
                        minIf(low, volume > 0) AS min_low,
                        arrayLast(x -> true, groupArray(open_interest)) AS open_interest
                    FROM 
                        minute_data_with_sessions
                    GROUP BY 
                        trading_day
                )

            -- Step 6: Aggregate final daily data
            SELECT
                '{symbol_id}' AS symbol_id,
                t1.trading_day AS datetime,
                t1.open,
                t1.max_high AS high,
                t1.min_low AS low,
                t1.close,
                t1.pre_close,
                sum(t2.volume) AS volume,
                sum(t2.money) AS money,
                COALESCE(any(t2.factor),0) AS factor,
                any(t2.paused) AS paused,
                t1.open_interest
            FROM 
                pre_aggregated_data t1
            JOIN 
                minute_data_with_sessions t2
            ON 
                t1.trading_day = t2.trading_day
            GROUP BY 
                t1.trading_day, t1.open, t1.max_high, t1.min_low, t1.close,t1.open_interest,t1.pre_close
            ORDER BY 
                t1.trading_day;

        """
        return sql

    @classmethod
    def get_1d_sql(
        cls,
        symbol_id: str,
        date: str,
        database: Literal["jq", "xt", "default"] = "jq",
    ):
        """获取一天的拼日线的sql语句"""
        exchange = symbol_id.split(".")[1]
        last_date = cls.get_last_trading_day(date)

        if exchange == "CFFEX":
            condition = f"AND datetime = '{date} 00:00:00'"
            return cls.get_basic_1d_sql(symbol_id, condition, database)
        condition = f"AND datetime BETWEEN '{last_date} 20:55:00' AND '{date} 15:00:02'"
        return cls.get_basic_1d_sql(symbol_id, condition, database)

    @classmethod
    def get_month_1d_sql(
        cls,
        symbol_id: str,
        date_start: str,
        date_end: str,
        database: Literal["jq", "xt", "default"] = "jq",
    ):
        """获取一个月的日线数据"""
        exchange = symbol_id.split(".")[1]

        if exchange == "CFFEX":
            condition = f"AND datetime between '{date_start} 00:00:00' and '{date_end} 23:59:59'"
            return cls.get_basic_1d_sql(symbol_id, condition, database)
        condition = (
            f"AND datetime BETWEEN '{date_start} 20:55:00' AND '{date_end} 15:00:02'"
        )
        return cls.get_basic_1d_sql(symbol_id, condition, database)

    @classmethod
    def get_settlement_price(cls, symbol_id: str, date: str) -> float:
        """
        获取结算价
        """
        sql = f"SELECT settlement_price FROM xt.1d where symbol_id='{symbol_id}' and toDate(datetime)='{date}'"
        with cls.lock:
            result = cls.conn_reader.execute(sql, columnar=True)
        assert result[0], "No settlement price found"
        return result[0][0]

    @classmethod
    def execute_1m_sql(
        cls, symbole_id: str, date: str, database: Literal["jq", "xt", "default"] = "jq"
    ) -> List[Tuple, Tuple]:
        """
        执行1分钟数据sql语句
        """
        sql = cls.get_1m_sql(symbole_id, date, database)
        conn = cls.writer_conn if database == "default" else cls.conn_reader
        with cls.lock:
            result = conn.execute(sql, columnar=True, with_column_types=True)
        return result

    @classmethod
    def execute_1d_sql(
        cls, symbole_id: str, date: str, database: Literal["jq", "xt", "default"] = "jq"
    ) -> List[Tuple, Tuple]:
        """
        执行日线数据sql语句
        """
        sql = cls.get_1d_sql(symbole_id, date, database)
        conn = cls.writer_conn if database == "default" else cls.conn_reader
        with cls.lock:
            result = conn.execute(sql, columnar=True, with_column_types=True)
        return result

    @classmethod
    def execute_month_1d_sql(
        cls,
        symbole_id: str,
        date_start: str,
        date_end: str,
        database: Literal["jq", "xt", "default"] = "jq",
    ) -> List[Tuple, Tuple]:
        """
        执行月线数据sql语句
        """
        sql = cls.get_month_1d_sql(symbole_id, date_start, date_end, database)
        conn = cls.writer_conn if database == "default" else cls.conn_reader
        with cls.lock:
            result = conn.execute(sql, columnar=True, with_column_types=True)
        return result


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


@numba.njit
def creat_snapshot_array(length: int, columns: int) -> np.ndarray:
    """
    创建快照数组
    """
    # 快照数组里面没有symbol_id和datetime列
    return np.zeros((length, columns), dtype=np.float64)


@numba.njit()
def get_product_comma(product: str) -> str:
    """
    获取大写品种代码并添加逗号
    """
    return f"{product.upper()},"


@numba.njit()
def get_hms(seconds):
    """将timestamp转换为hms"""
    seconds = divmod(seconds, 86_400)[1]
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return h * 10_000 + m * 100 + s


@numba.njit()
def in_trade_times(product_comma: str, hms: int) -> bool:
    """判断是否在交易时间"""
    if product_comma in "IM,IH,IC,IF,":
        return 92500 <= hms < 110001 or 130000 <= hms < 150001

    if product_comma in "T,TS,TL,TF,":
        return 92500 <= hms < 110001 or 130000 <= hms < 151501

    if 85500 <= hms < 101501 or 103000 <= hms < 113001 or 133000 <= hms < 150001:
        return True

    if product_comma in "AU,AG,SC,":
        return 0 <= hms < 23001 or 205500 <= hms < 240000

    if product_comma in "CU,PB,AL,ZN,SN,NI,SS,AO,BC,":
        return 0 <= hms < 10001 or 205500 <= hms < 240000

    if (
        product_comma
        in "FG,SA,SH,PX,MA,SR,TA,RM,OI,CF,CY,PF,ZC,I,J,JM,A,B,MA,P,Y,C,CS,PP,V,EB,EG,PG,RR,L,FU,RU,BR,BU,SP,RB,HC,LU,NR,"
    ):
        return 205500 <= hms < 230001

    return False


if __name__ == "__main__":
    # print(DBHelper.clear_writer_table("merged_1d"))
    # print(DBHelper.clear_writer_table("merged_1m"))
    # print(DBHelper.clear_records())
    print(DBHelper.get_major_contract_id("CY", "CZCE", "2017-08-18"))
