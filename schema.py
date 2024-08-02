import sqlite3

import polars as pl
import clickhouse_driver

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


if __name__ == "__main__":
    db = r"C:\用户\newf4\database\future.db"
    fields = "symbol_id"
    dat = dataIter(db, "ag", fields, 1)
