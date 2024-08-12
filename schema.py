from datetime import datetime
import sqlite3
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Union

import polars as pl
import polars.selectors as cs
from numba.typed import List, Dict

from utils import get_term, next_term, timeit

fields = (
    "symbol_id, datetime, open, high, low, close, volume, money, a1_v, a1_p, a2_v, a2_p"
)


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
        last_mayjor: dict,
        start_datetime: datetime,
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
                self.last_mayjor["symbol_id"] = newtick["symbol_id"]
                self.last_mayjor["position"] = newtick["position"]
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
            condition1 = newtick["position"] >= self.last_mayjor["position"]
            condition2 = get_term(newtick["symbol_id"]) >= get_term(
                self.last_mayjor["symbol_id"]
            )
            condition3 = newtick["symbol_id"] == self.last_mayjor["symbol_id"]
            return (condition1 and condition2) or condition3
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
        self,
        product: str,
        exchange: str,
        last_snapshot: pl.DataFrame,
        start_datetime: datetime,
    ) -> None:
        self.last_snapshot = last_snapshot
        self.symbol_id = product + "8888." + exchange.upper()
        self.high = 0.0
        self.low = 0.0
        super().__init__(product, exchange, start_datetime)
        self.value = self.value.drop("position")
        self.weight = self.last_snapshot.select("position")

    def add_tick(self, newtick: Dict):
        """
        add new tick to container
        """
        self._assert_new_tick(newtick)
        if newtick["datetime"] < self.end_datetime:
            self.update_snapshot(newtick)
            if self.date_time != newtick["datetime"]:
                value = self.update_value()
                # self.value = value.select(self.value.columns).vstack(self.value)
                self.value = self.value.vstack(value.select(self.value.columns))
            self.date_time = newtick["datetime"]
        else:
            pass

    def update_snapshot(self, newtick: Dict) -> None:
        """
        upadate snapshot of all contracts
        """
        if newtick["symbol_id"] not in self.last_snapshot["symbol_id"]:
            self.last_snapshot = self.last_snapshot.vstack(
                pl.DataFrame(newtick, schema=self.last_snapshot.schema)
            )
        condition1 = self.last_snapshot["symbol_id"] == newtick["symbol_id"]
        condition2 = self.last_snapshot["position"] == 0
        del newtick["symbol_id"]
        self.last_snapshot = self.last_snapshot.select(
            ["symbol_id"]
            + [
                pl.when(condition1).then(newtick[col]).otherwise(pl.col(col)).alias(col)
                for col in newtick.keys()
            ]
        ).filter(~condition2)

    def update_value(self) -> pl.DataFrame:
        """
        update value of index contract
        """
        condition = cs.numeric()
        numeric_snap = self.last_snapshot.select(condition)
        # numeric_snap = numeric_snap.drop("position").hstack(self.weight)
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


def assert_new_tick(newtick: Dict) -> None:
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


def check_new_mayjor(newtick: Dict, exchange: str, last_mayjor: dict) -> bool:
    """
    判断新进tick是否是新的主力合约tick
    """
    if exchange != "INE":
        # 新进tick的合约代码就是主力合约代码，直接返回True
        condition1 = newtick["symbol_id"] == last_mayjor["symbol_id"]
        if condition1:
            return True
        # 新tick的持仓量大于主力合约的持仓量，且新tick的合约代码大于主力合约的合约代码，则返回True
        condition2 = newtick["position"] >= last_mayjor["position"]
        if condition2:
            condition3 = get_term(newtick["symbol_id"]) >= get_term(
                last_mayjor["symbol_id"]
            )
            return condition3
        return False
    else:
        # 中金所的近月合约为主力合约
        return _is_latest_month_contract(newtick["symbol_id"], newtick["datetime"])


def _is_latest_month_contract(
    symbol_id: str,
    datetime: datetime,
    last_mayjor: dict,
    product: str,
    exchange: str,
) -> bool:
    """
    check if new tick is the latest month contract
    """
    year, month, day = datetime.year, datetime.month, datetime.day
    expire_date = _get_expire_date(last_mayjor["symbol_id"])
    if (year, month, day) > expire_date:
        return symbol_id == (
            product
            + next_term(get_term(last_mayjor["symbol_id"]))
            + "."
            + exchange.upper()
        )


def _get_expire_date(symbol_id: str) -> Tuple[int, int, int]:
    pass


def upadate_major_tick(newtick: Dict, last_mayjor: dict, symbol_id: str):
    """
    add new tick to container
    """
    assert_new_tick(newtick)
    if check_new_mayjor(newtick):
        last_mayjor["symbol_id"] = newtick["symbol_id"]
        last_mayjor["position"] = newtick["position"]
        newtick["symbol_id"] = symbol_id
        last_mayjor = newtick

    else:
        last_mayjor["datetime"] = newtick["datetime"]
    return last_mayjor


def upadate_index_tick(
    newtick: Dict, last_snapshot: pl.DataFrame, symbol_id: str, high: float, low: float
):
    """
    add new tick to container
    """
    assert_new_tick(newtick)
    if newtick["datetime"] < last_snapshot["datetime"].max():
        last_snapshot = last_snapshot.vstack(
            pl.DataFrame(newtick, schema=last_snapshot.schema)
        )
    else:
        last_snapshot = last_snapshot.vstack(
            pl.DataFrame(
                {
                    "symbol_id": symbol_id,
                    "datetime": newtick["datetime"],
                    "open": newtick["open"],
                    "high": newtick["high"],
                    "low": newtick["low"],
                    "close": newtick["close"],
                    "volume": newtick["volume"],
                    "money": newtick["money"],
                    "position": 0,
                },
                schema=last_snapshot.schema,
            )
        )


def update_extream(value: float, high: float, low: float) -> Tuple[float, float]:
    """
    update extream value of index contract
    """
    if value > high:
        high = value
    if value < low or low == 0.0:
        low = value
    return high, low


def update_index_value(
    last_snapshot: pl.DataFrame, symbol_id: str, high: float, low: float
) -> pl.DataFrame:
    """
    update value of index contract
    """
    condition = cs.numeric()
    numeric_snap = last_snapshot.select(condition)
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
    high, low = update_extream(new_value["current"].item(), high, low)
    lit_info = pl.DataFrame(
        {
            "symbol_id": symbol_id,
            "datetime": new_value["datetime"],
            "high": high,
            "low": low,
        }
    )
    new_value.hstack(lit_info, in_place=True)
    return new_value


def update_index_snapshot(
    newtick: Dict, last_snapshot: pl.DataFrame, symbol_id: str
) -> pl.DataFrame:
    """
    upadate snapshot of all contracts
    """
    if newtick["symbol_id"] not in last_snapshot["symbol_id"]:
        last_snapshot = last_snapshot.vstack(
            pl.DataFrame(newtick, schema=last_snapshot.schema)
        )
    condition1 = last_snapshot["symbol_id"] == newtick["symbol_id"]
    condition2 = last_snapshot["position"] == 0
    del newtick["symbol_id"]
    last_snapshot = last_snapshot.select(
        ["symbol_id"]
        + [
            pl.when(condition1).then(newtick[col]).otherwise(pl.col(col)).alias(col)
            for col in newtick.keys()
        ]
    ).filter(~condition2)
    return last_snapshot


if __name__ == "__main__":
    pass
