import functools
import re
import time

import polars as pl
import clickhouse_driver


@functools.lru_cache
def get_conn():
    """
    获取ClickHouse连接
    """
    clickhouse_uri = "clickhouse://reader:d9f6ed24@172.16.7.30:9900/joinquant"
    conn = clickhouse_driver.Client.from_url(url=clickhouse_uri)
    return conn


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


def get_last_close_dat(product: str, date: str) -> pl.DataFrame:
    """
    get the last close dat of the product
    """
    pass


def get_last_mayjor(product: str, date: str) -> str:
    """
    get the last mayjor of the product
    """
    pass


def get_last_secondery(product: str, date: str) -> str:
    """
    get the last secondery of the product
    """
    pass


def get_last_contract(product: str, date: str) -> list:
    """
    get the last contract of the product
    """
    pass


if __name__ == "__main__":
    code = "2403.shfe"
    print(get_term(code))
