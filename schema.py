from datetime import datetime
import sqlite3
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Union

import polars as pl
import clickhouse_driver
from numba.typed import List, Dict

from fu_dat_cal.utils import get_nearest_hour

fields = (
    "symbol_id, datetime, open, high, low, close, volume, money, a1_v, a1_p, a2_v, a2_p"
)


class dataIter:
    def __init__(self, database: str, table: str, fields: str, batch_size: int = 1):
        self.conn = sqlite3.connect(database)
        self.cur = self.conn.cursor()
        self.table = table
        self.fields = fields
        self.batche_size = batch_size
        self.offset = 0
        self.sql = (
            f"SELECT {fields} FROM {table} LIMIT {batch_size} OFFSET {self.offset}"
        )
        self.count = self.__len__()
        self.last_dat = []
        self.datetime = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset >= self.count:
            raise StopIteration
        self.cur.execute(self.sql)
        data = self.cur.fetchall()
        self.last_dat = data
        self.datetime = data[1]
        self.offset += self.batche_size
        self.sql = f"SELECT {self.fields} FROM {self.table} LIMIT {self.batche_size} OFFSET {self.offset}"
        return data

    def __len__(self):
        self.cur.execute(f"SELECT COUNT(*) FROM {self.table}")
        return self.cur.fetchone()[0]

    def close(self):
        self.conn.close()

    def set_batch_size(self, batch_size: int):
        self.batche_size = batch_size
        self.sql = f"SELECT {self.fields} FROM {self.table} LIMIT {self.batche_size} OFFSET {self.offset}"

    def set_offset(self, offset: int):
        self.offset = offset
        self.sql = f"SELECT {self.fields} FROM {self.table} LIMIT {self.batche_size} OFFSET {self.offset}"


class container:
    def __init__(self, symbol_id: str):
        self.symbol_id = symbol_id
        self.high = 0
        self.low = 0
        self.current = 0
        self.position = 0
        self.volume = 0
        self.money = 0
        self.a1_v = 0
        self.a1_p = 0
        self.a2_v = 0
        self.a2_p = 0
        self.data = pl.DataFrame(
            schema=[
                ("symbol_id", pl.Utf8),
                ("high", pl.Float64),
                ("low", pl.Float64),
                ("current", pl.Float64),
                ("position", pl.Int64),
                ("volume", pl.Float64),
                ("money", pl.Float64),
                ("a1_v", pl.Float64),
                ("a1_p", pl.Float64),
                ("a2_v", pl.Float64),
                ("a2_p", pl.Float64),
            ]
        )

    def add_data(self):
        self.data = self.data.append(
            {
                "symbol_id": self.symbol_id / self.position,
                "high": self.high / self.position,
                "low": self.low,
                "current": self.current / self.position,
                "position": 1,
                "volume": self.volume / self.position,
                "money": self.money / self.position,
                "a1_v": self.a1_v / self.position,
                "a1_p": self.a1_p / self.position,
                "a2_v": self.a2_v / self.position,
                "a2_p": self.a2_p / self.position,
            },
            ignore_index=True,
        )


def get_index_tick(symbol_ids: list, db: str):
    iter_list = [dataIter(db, symbol_id, fields) for symbol_id in symbol_ids]


class plIter:
    def __init__(self, values: pl.DataFrame) -> None:
        self.cols = values.columns
        self.values = values.iter_rows()

    def __iter__(self) -> Iterable:
        return self

    def __next__(self) -> Dict:
        return Dict(zip(self.cols, next(self.values)))


class contractContainer(ABC):
    """
    abstract class for contract container
    """

    tick_iter: plIter
    new_tick: Dict
    last_data: Union[pl.DataFrame, Dict]
    value: pl.DataFrame

    def __init__(self, product: str, exchange: str):
        self.exchange = exchange
        self.product = product
        self.date_time = self._get_start_datetime(product, exchange)
        self.end_datetime = self.date_time.replace(
            hour=15, minute=00, second=00, microsecond=500
        )
        self.value = pl.DataFrame(
            schema=[
                ("symbol_id", pl.Utf8),
                ("high", pl.Float64),
                ("low", pl.Float64),
                ("current", pl.Float64),
                ("position", pl.Int64),
                ("volume", pl.Float64),
                ("money", pl.Float64),
                ("a1_v", pl.Float64),
                ("a1_p", pl.Float64),
                ("a2_v", pl.Float64),
                ("a2_p", pl.Float64),
            ]
        )
        self.high = 0.0
        self.low = 0.0

    def _get_start_datetime(self, product: str, exchange: str) -> datetime:
        """
        get start datetime of contract
        """
        first_tick = next(self.tick_iter)
        self.new_tick = first_tick
        first_datetime: datetime = first_tick["datetime"]
        start_datetime = get_nearest_hour(first_datetime)
        return start_datetime

    @abstractmethod
    def add_tick(self):
        """
        add new tick to container
        """


class mayjorContainer(contractContainer):
    """
    container for mayjor contract
    """

    def __init__(
        self, product: str, exchange: str, last_mayjor: Dict, tick_iter: plIter
    ) -> None:
        self.last_mayjor = last_mayjor
        self.tick_iter = tick_iter
        super().__init__(product, exchange)

    def add_tick(self):
        """
        add new tick to container
        """
        if self.new_tick["datetime"] > self.end_datetime:
            newtick = next(self.tick_iter)
            if self.check_new_mayjor(newtick):
                newtick["symbol_id"] = self.symbol_id
                self.new_tick = newtick
            else:
                self.new_tick["datetime"] = newtick["datetime"]
            self.value = self.value.vstack(pl.DataFrame([newtick]))
        else:
            if self.new_tick["high"] > self.high:
                self.high = self.new_tick["high"]
            if self.new_tick["low"] < self.low:
                self.low = self.new_tick["low"]
            self.value = self.value.append(
                {
                    "symbol_id": self.product,
                    "high": self.high,
                    "low": self.low,
                    "current": self.new_tick["close"],
                    "position": 1,
                    "volume": self.new_tick["volume"],
                    "money": self.new_tick["money"],
                    "a1_v": self.new_tick["a1_v"],
                    "a1_p": self.new_tick["a1_p"],
                    "a2_v": self.new_tick["a2_v"],
                    "a2_p": self.new_tick["a2_p"],
                },
                ignore_index=True,
            )


if __name__ == "__main__":
    pass
