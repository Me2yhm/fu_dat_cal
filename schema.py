from datetime import datetime
import sqlite3
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Union

import polars as pl
import polars.selectors as cs
from numba.typed import List, Dict

from utils import get_nearest_hour, get_term, next_term

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
        return dict(zip(self.cols, next(self.values)))


class contractContainer(ABC):
    """
    abstract class for contract container
    """

    last_tick: Dict
    symbol_id: str
    last_data: Union[pl.DataFrame, Dict]
    value: pl.DataFrame

    def __init__(self, product: str, exchange: str, start_datetime: datetime):
        self.exchange = exchange
        self.product = product
        self.date_time = start_datetime
        if start_datetime > start_datetime.replace(
            hour=15, minute=00, second=00, microsecond=500
        ):
            self.end_datetime = self.date_time.replace(
                hour=15,
                minute=00,
                second=00,
                microsecond=500,
                day=self.date_time.day + 1,
            )
        else:
            self.end_datetime = self.date_time.replace(
                hour=15, minute=00, second=00, microsecond=500
            )
        self.value = pl.DataFrame(
            schema=[
                ("symbol_id", pl.Utf8),
                ("datetime", pl.Datetime),
                # ("high", pl.Float64),
                # ("low", pl.Float64),
                ("current", pl.Float64),
                ("position", pl.Int64),
                # ("volume", pl.Float64),
                # ("money", pl.Float64),
                # ("a1_v", pl.Float64),
                # ("a1_p", pl.Float64),
                # ("a2_v", pl.Float64),
                # ("a2_p", pl.Float64),
            ]
        )
        self.high = 0.0
        self.low = 0.0

    @abstractmethod
    def add_tick(self):
        """
        add new tick to container
        """

    def _assert_new_tick(self, newtick: Dict) -> None:
        """
        check if new tick is valid
        """
        for value in newtick.values():
            try:
                assert (
                    value is not None
                    and value != 0
                    and value != "0"
                    and value != "0.0"
                    and value != "None"
                    and value != "null"
                    and value != ""
                    and value != "nan"
                    and value != "NaN"
                    and value != "NAN"
                    and value != "NaT"
                    and value != "nat"
                    and value != "N/A"
                    and value != "n/a"
                    and value != "N/A N/A"
                    and value != "n/a n/a"
                    and value != "N/A N/A N/A"
                    and value != "n/a n/a n/a"
                    and value != "N/A N/A N/A N/A"
                    and value != "n/a n/a n/a n/a"
                )
            except AssertionError:
                print(f"Invalid value: {newtick}")
                raise AssertionError


class mayjorContainer(contractContainer):
    """
    container for mayjor contract
    """

    def __init__(
        self,
        product: str,
        exchange: str,
        last_mayjor: Dict,
        start_datetime,
        last_tick: dict,
    ) -> None:
        self.last_mayjor = last_mayjor
        self.symbol_id = product + "9999." + exchange.upper()
        self.last_tick = last_tick
        super().__init__(product, exchange, start_datetime)

    def add_tick(self, newtick: Dict):
        """
        add new tick to container
        """
        self._assert_new_tick(newtick)
        if newtick["datetime"] < self.end_datetime:
            if self.check_new_mayjor(newtick):
                newtick["symbol_id"] = self.symbol_id
                self.last_tick = newtick
            else:
                self.last_tick["datetime"] = newtick["datetime"]
            self.value = self.value.vstack(
                pl.DataFrame(self.last_tick, schema=self.value.schema)
            )
        else:
            pass

    def check_new_mayjor(self, newtick: Dict) -> bool:
        """
        check if new tick is a new mayjor
        """
        if self.exchange != "INE":
            condition1 = newtick["position"] > self.last_mayjor["position"]
            condition2 = get_term(newtick["symbol_id"]) >= get_term(
                self.last_mayjor["symbol_id"]
            )
            return condition1 and condition2
        else:
            return self._is_latest_month_contract(
                newtick["symbol_id"], newtick["datetime"]
            )

    def _is_latest_month_contract(self, symbol_id: str, datetime: datetime) -> bool:
        """
        check if new tick is the latest month contract
        """
        year, month, day = datetime.year, datetime.month, datetime.day
        expire_date = self._get_expire_date(self.last_mayjor["symbol_id"])
        if (year, month, day) > expire_date:
            return symbol_id == (
                self.product
                + next_term(get_term(self.last_mayjor["symbol_id"]))
                + "."
                + self.exchange.upper()
            )


class indexContainer(contractContainer):
    """
    container for future index contract
    """

    def __init__(
        self, product: str, exchange: str, last_snapshot: pl.DataFrame
    ) -> None:
        self.last_snapshot = last_snapshot
        self.symbol_id = product + "8888." + exchange.upper()
        self.high = 0.0
        self.low = 0.0
        super().__init__(product, exchange)

    def add_tick(self, newtick: Dict):
        """
        add new tick to container
        """
        self._assert_new_tick(newtick)
        if newtick["datetime"] < self.end_datetime:
            self.date_time = newtick["datetime"]
            self.update_snapshot(newtick)
            value = self.update_value()
            self.value = self.value.vstack(value)
        else:
            pass

    def update_snapshot(self, newtick: Dict) -> None:
        """
        upadate snapshot of all contracts
        """
        if newtick["symbol_id"] not in self.last_snapshot["symbol_id"]:
            self.last_snapshot = self.last_snapshot.vstack(pl.DataFrame(newtick))
        condition1 = self.last_snapshot["symbol_id"] == newtick["symbol_id"]
        condition2 = self.last_snapshot["position"] == 0
        self.last_snapshot = self.last_snapshot.select(
            [
                pl.when(condition1).then(newtick[col]).otherwise(pl.col(col).alias(col))
                for col in newtick.keys()
            ]
        ).filter(~condition2)

    def update_value(self) -> pl.DataFrame:
        """
        update value of index contract
        """
        condition = cs.numeric()
        numeric_snap = self.last_snapshot.select(condition)
        new_value = (
            numeric_snap.select(
                [
                    pl.col(col) * pl.col("position")
                    for col in numeric_snap.columns
                    if col != "position"
                ]
            ).sum()
            / numeric_snap["position"].sum()
        )
        self.update_extream(new_value["current"].item())
        lit_info = pl.DataFrame(
            {
                "symbol_id": self.symbol_id,
                "datetime": self.date_time,
                "high": self.high,
                "low": self.low,
            }
        )
        new_value.hstack(lit_info, in_place=True)
        return new_value

    def update_extream(self, value: float) -> None:
        """
        update extream value of index contract
        """
        if value > self.high:
            self.high = value
        if value < self.low or self.low == 0.0:
            self.low = value


if __name__ == "__main__":
    pass
