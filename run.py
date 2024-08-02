import polars as pl
import functools
import sqlite3
from datetime import datetime, timedelta
from multiprocessing import Pool

import clickhouse_driver

from utils import get_term


@functools.lru_cache
def get_conn():
    """
    获取ClickHouse连接
    """
    clickhouse_uri = "clickhouse://reader:d9f6ed24@172.16.7.30:9900/joinquant"
    conn = clickhouse_driver.Client.from_url(url=clickhouse_uri)
    return conn


def get_tick_dataframe(batch_size: int = 10000, offset: int = 0):
    """
    获取Tick数据
    """
    conn = get_conn()
    fields = "symbol_id,datetime,position,high,low,volume,current,a1_v,a1_p,b1_v,b1_p"
    sql = f"select {fields} from jq.`tick` where match (symbol_id,'^[a-z]+\\d+.[A-Z]+$') limit {batch_size} offset {offset}"
    rows = conn.execute(sql)
    df = pl.DataFrame(rows, schema=fields.split(","))
    print(f"Got {batch_size} rows from ClickHouse at offset {offset}")
    return df


def get_mayjor_contract(tick_df: pl.DataFrame) -> pl.DataFrame:
    """
    获取主力合约tick数据
    """
    major_tick_df = pl.DataFrame(schema=tick_df.schema)
    mayjor_contract = ""
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
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {product_id} (
            symbol_id TEXT,
            datetime TEXT,
            position INTEGER,
            high NUMERIC,
            low NUMERIC,
            volume INTEGER,
            current NUMERIC,
            a1_v INTEGER,
            a1_p NUMERIC,
            b1_v INTEGER,
            b1_p NUMERIC
        )
    """
    )
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
        save_db(dat, pro[0])
        print(f"Processed {pro[0]}")


def main():
    fields = "symbol_id,datetime,position,high,low,volume,current,a1_v,a1_p,b1_v,b1_p"
    conn = get_conn()
    batch_size = 100000  # 每批处理的数据量
    total_rows = conn.execute("SELECT count() FROM jq.`tick`")[0][0]  # 获取总行数
    print(f"Total rows: {total_rows}")

    # 使用多进程处理数据
    with Pool() as pool:
        offsets = range(0, total_rows, batch_size)
        results = pool.starmap(
            get_tick_dataframe, [(batch_size, offset) for offset in offsets]
        )

        # 处理每批数据
        processed_counts = pool.map(process_data, results)
        print(f"Total processed rows: {sum(processed_counts)}")


if __name__ == "__main__":
    main()
    # df = get_tick_dataframe(batch_size=100000, offset=0)
    # gropdf = group_by_product(df)
    # c = 0
    # for pro, dat in gropdf:
    #     print(dat)
    #     c += 1
    #     if c == 3:
    #         break
    # product_gro = df.group_by("symbol_id")
    # save_db(df, "future")
    # print(df)
    # for pro, dat in product_gro:
    #     save_db(dat, pro[0])
    #     break
