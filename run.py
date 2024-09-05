import multiprocessing
import sys
from typing import Literal

from processer import Processer
from utils import DBHelper as db
from loguru import logger

FUNC_MAP = {
    "index_tick": Processer.process_future_index,
    "mayjor_tick": Processer.process_future_mayjor,
    "secondery_tick": Processer.process_future_secondery,
    "future_1m": Processer.process_future_1m,
    "future_1d": Processer.process_future_1d,
    "option_1m": Processer.process_option_1m,
    "option_1d": Processer.process_option_1d,
}


def run_pre(
    func_name: Literal["index_tick", "mayjor_tick", "secondery_tick", "option_1m"]
):
    try:
        func = FUNC_MAP[func_name]
        logger.add(f"logs/{func_name}.log", level="WARNING", rotation="10 MB")
        results = func()
        logger.success(f"[{func_name}] Success:: {results}")
    except Exception as e:
        logger.error(f"[{e.__class__.__name__}] Error:: {e}")
    finally:
        pass


def run_after(
    func_name: Literal["future_1m", "future_1d", "option_1d"],
    kind: Literal["index", "mayjor", "secondery"],
):
    try:
        func = FUNC_MAP[func_name]
        logger.add(f"logs/{func_name}/{kind}.log", level="WARNING", rotation="10 MB")
        match func_name.split("_")[0]:
            case "future":
                results = func(kind)
            case "option":
                results = func()
            case _:
                raise ValueError(
                    "Invalid function name. It should be 'future_1m' or 'future_1d' or 'option_1d'."
                )
        logger.success(f"[{kind}] Success:: {results}")
    except Exception as e:
        logger.error(f"[{e.__class__.__name__}] Error:: {e}")
    finally:
        pass


def run_all_pre():
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.apply_async(run_pre, args=("index_tick",))
        pool.apply_async(run_pre, args=("mayjor_tick",))
        pool.apply_async(run_pre, args=("secondery_tick",))
        pool.apply_async(run_pre, args=("option_1m",))
        pool.close()
        pool.join()


def run_all_1m():
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.apply_async(run_after, args=("future_1m", "index"))
        pool.apply_async(run_after, args=("future_1m", "mayjor"))
        pool.apply_async(run_after, args=("future_1m", "secondery"))
        pool.close()
        pool.join()


def run_all_1d():
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.apply_async(run_after, args=("futrue_1d", "index"))
        pool.apply_async(run_after, args=("future_1d", "mayjor"))
        pool.apply_async(run_after, args=("future_1d", "secondery"))
        pool.apply_async(run_after, args=("option_1d", "option"))
        pool.close()
        pool.join()


def run_data_supplement():
    logger.remove()
    logger.add(sys.stderr, level="SUCCESS")
    run_all_pre()
    run_all_1m()
    run_all_1d()


if __name__ == "__main__":
    run_data_supplement()
