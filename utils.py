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
import threading

from numba.typed import List
import pymongo
import pymongo.collection

from logger import Logger, setup_logger

log_root_dir = Path(__file__).parent / "log"


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


class DBHelper:
    lock = threading.Lock()
    # clickhouse
    reader_uri = "clickhouse://reader:d9f6ed24@172.16.7.30:9900/jq?compression=lz4&use_numpy=true"
    conn_reader = clickhouse_driver.Client.from_url(url=reader_uri)
    # 本地数据库reader，调试用
    reader_1d_uri = "C:\\用户\\newf4\\database\\future_1d.db"
    conn_1d = sqlite3.connect(reader_1d_uri)
    cursor_1d = conn_1d.cursor()
    # 本地数据库writer，调试用
    writer_uri = "C:\\用户\\newf4\\database\\future_index.db"
    writer_conn = sqlite3.connect(writer_uri)
    writer_cursor = writer_conn.cursor()

    logger = setup_logger("DBHelper", "db_helper.log")

    mongodb_url = "mongodb://quote_rw:rx5cb0g3myoiw30g@172.16.7.31:27027/Quote"
    record_conn = pymongo.MongoClient(mongodb_url)["Quote"]["FutureIndex"]

    @classmethod
    @timeit
    def get_mayjor_contract_dat(cls, product: str, exchange: str) -> pl.DataFrame:
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
    def get_mayjor_contract_id(
        cls, product: str, exchange: str, date: str = "2024-07-19"
    ) -> str:
        """
        获取主力合约代码
        """
        sql = f"""
                SELECT symbol_id 
                FROM jq.`1d`
                WHERE money IN (
                    SELECT money 
                    FROM jq.`1d` d 
                    WHERE symbol_id = '{product}9999.{exchange}'
                )
                AND `datetime` = '{date} 00:00:00'
                AND symbol_id != '{product}9999.{exchange}';
        """
        with cls.lock:
            rows = cls.conn_reader.execute(sql, columnar=True)
        assert len(rows) == 1, "主力合约数量不唯一"
        major_contract = rows[0][0]
        return major_contract

    @classmethod
    def get_secondery_id(cls, product: str, exchange: str, date: str) -> str:
        """
        获得次主力合约代码，规则是取日线数据持仓量第二大合约
        注意中金所主力合约的规则不一样
        同花顺规则也不一样（ag,2024-07-22）
        """
        sql = f"""
                SELECT symbol_id 
                FROM jq.`1d`
                WHERE `datetime` = '{date} 00:00:00'
                AND symbol_id LIKE '{product}____.{exchange.upper()}'
                AND symbol_id != '{product}8888.{exchange}'
                AND symbol_id != '{product}9999.{exchange}'
                ORDER BY open_interest DESC limit 2;
        """
        with cls.lock:
            rows = cls.conn_reader.execute(sql, columnar=True)
            print(rows)
        return rows[0][1]

    @classmethod
    @timeit
    def get_last_snapshot(
        cls,
        symbol_ids: np.ndarray,
        date: str = "2024-07-19",
        fields: str = "symbol_id,datetime,current,a1_v,a1_p,b1_v,b1_p,position",
    ) -> np.ndarray:
        """
        获取上一个快照数据
        date 需要为交易日。
        """
        start_datetime = " ".join([date, "15:00:00"])
        end_datetime = " ".join([date, "16:00:00"])
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
                idx += 1
        return snapshot

    @classmethod
    @timeit
    def get_symbols_tick(
        cls,
        symbols: list,
        date: str = "2024-07-19",
        fields: str = "symbol_id,datetime,current,a1_v,a1_p,b1_v,b1_p,position",
    ):
        """
        获取指定若干合约的Tick数据
        """
        start_datetime = " ".join([cls.get_last_trading_day(date), "20:55:00"])
        end_datetime = " ".join([date, "16:00:00"])
        sql = f"select {fields} from jq.`tick` where datetime between '{start_datetime}' and  '{end_datetime}' and symbol_id in {tuple(symbols)};"
        with cls.lock:
            rows = cls.conn_reader.execute(sql, columnar=True)
        if fields == "*":
            columns_info = cls.conn_reader.execute("DESCRIBE TABLE jq.`tick`")
            schemas = [column[0] for column in columns_info]
        else:
            schemas = fields.split(",")
        for i in range(2, len(schemas)):
            rows[i] = rows[i].astype(np.float64)
        df = pl.DataFrame(rows, schema=schemas, orient="col", strict=False)
        return df

    @classmethod
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
    def get_pro_dates(cls, product: str, exchange: str) -> list:
        """
        获取指定品种的交易日列表
        """
        query = f"SELECT distinct Date(datetime) as date FROM {product}_{exchange} order by datetime asc"
        with cls.lock:
            df = pd.read_sql(query, cls.conn_1d, columns=["datetime"])
        return df["date"].tolist()

    @classmethod
    def get_last_trading_day(cls, date: str) -> str:
        """
        获取上一个交易日
        """
        sql = f"SELECT MAX(DATE(datetime)) as last_trading_day FROM a_DCE WHERE datetime < '{date} 00:00:00'"
        with cls.lock:
            last_trading_day = cls.conn_1d.execute(sql).fetchall()[0][0]
        return last_trading_day

    @classmethod
    @timeit
    def save_db(
        cls,
        df: pl.DataFrame | pd.DataFrame,
        product_id: str,
        date: str = "all",
        db_name: str = "future_index",
    ):
        """
        保存数据到sqlite数据库
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        with cls.lock:
            df.to_sql(product_id, cls.writer_conn, if_exists="append", index=False)
        Logger.info(f"saved {product_id} of {date} data to {db_name} successfully")

    @classmethod
    @timeit
    def get_all_contracts(
        cls, product: str, exchange: str, date: str = "2024-07-19"
    ) -> np.ndarray:
        """
        获取所有合约
        """
        sql = f"select symbol_id from {product}_{exchange} where datetime = '{date} 00:00:00';"
        rows = cls.cursor_1d.execute(sql).fetchall()
        symbol_ids = List.empty_list(numba.types.unicode_type)
        for row in rows:
            symbol_ids.append(row[0])
        return symbol_ids

    @classmethod
    def insert_records(cls, record):
        """
        记录已经处理过的数据日期到mongodb
        """
        with cls.lock:
            if cls.record_conn.find_one(record):
                print(f"Record {record} already exists in database.")
                return
            cls.record_conn.insert_one(record)

    @classmethod
    def is_processed(cls, record) -> bool:
        """
        判断某个产品的某天数据是否已经处理过
        """
        with cls.lock:
            is_processed = bool(cls.record_conn.find_one(record))
        return is_processed

    @classmethod
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

    @classmethod
    @timeit
    def execute_1d_extract(cls):
        condition = "where match(symbol_id,'^[a-zA-Z]+\\d+.[a-zA-Z]+$') and symbol_id not like '%8888%' and symbol_id not like '%9999%';"
        df = cls.get_1d_dat(condition)
        product_gro = cls.group_by_product(df)
        logg = setup_logger("day", "1d_extract.log")
        for pro, dat in product_gro:
            record = {"product": pro[0], "date": "all", "type": "day"}
            with cls.lock:
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
    def clear_records(cls):
        """
        清空mongodb记录
        """
        with cls.lock:
            result = cls.record_conn.delete_many({})
            cls.logger.info(f"Deleted {result.deleted_count} records from database.")
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


@numba.njit(nopython=True)
def get_product_comma(product: str) -> str:
    """
    获取大写品种代码并添加逗号
    """
    return f"{product.upper()},"


@numba.njit(nopython=True)
def get_hms(seconds):
    """将timestamp转换为hms"""
    seconds = divmod(seconds, 86_400)[1]
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return h * 10_000 + m * 100 + s


@numba.njit(nopython=True)
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
    symbol_ids = DBHelper.get_all_contracts("ag", "SHFE", "2012-05-10")
    dic = DBHelper.get_last_snapshot(
        symbol_ids,
        "2012-05-09",
    )
    print(dic)
