import time
import polars as pl
import functools
import sqlite3
from datetime import datetime, timedelta
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


from utils import get_last_trading_day, get_term, get_conn, timeit


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
    rows = conn.execute(sql)
    df = pl.DataFrame(rows, schema=fields.split(","))
    return df


@timeit
def get_tick_multithread(offsets: list, batch_size: int = 10000):
    """
    多线程处理Tick数据
    """
    producnts = get
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
    # main()
    offsets = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000]
    # results = get_tick_multithread(offsets, 100000)
    start_time = datetime.now()
    for offset in offsets:
        df = get_tick_dataframe(100000, offset)
    endtime = datetime.now()
    print(f"Total time: {endtime - start_time}")
