import sys

from processer import Processer
from utils import DBHelper as db
from loguru import logger

FUNC_MAP = {
    "index_tick": Processer.process_future_index,
    "mayjor_tick": Processer.process_future_major,
    "secondery_tick": Processer.process_future_secondery,
}


def run_tick(func_name):
    try:
        func = FUNC_MAP[func_name]
        db.execute_1d_extract()
        logger.remove()
        logger.add(sys.stderr, level="SUCCESS")
        logger.add("logs/index.log", level="WARNING", rotation="10 MB")
        results = func()
        logger.success(f"[{func_name}] Success:: {results}")
    except Exception as e:
        logger.error(f"[{e.__class__.__name__}] Error:: {e}")
    finally:
        db.delete_1d_dbfile()


def run_1m():
    pass
