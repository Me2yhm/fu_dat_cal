import multiprocessing
import sys
from typing import Literal

from loguru import logger

from processer import Processer
from utils import DBHelper as db

FUNC_MAP = {
    "index_tick": Processer.process_future_index,
    "major_tick": Processer.process_future_major,
    "secondery_tick": Processer.process_future_secondery,
    "future_1m": Processer.process_future_1m,
    "future_1d": Processer.process_future_1d,
    "option_1m": Processer.process_option_1m,
    "option_1d": Processer.process_option_1d,
}


class DataRunner:

    @classmethod
    def run_pre(
        cls,
        func_name: Literal["index_tick", "major_tick", "secondery_tick", "option_1m"],
        lock=None,
    ):
        """
        处理主连、次主力连、指数合约的tick数据或者期权的1分钟数据。因为这些数据可以直接处理, 不需要依赖其他数据.

        :param func_name: `index_tick`代表处理指数tick, `major_tick`代表主力合约tick,\
                          `secondery_tick`代表处理次主力合约tick, `option_1m`代表处理期权的1分钟数据
        :param lock: 进程锁
        """
        try:
            func = FUNC_MAP[func_name]
            logger.add(f"logs/{func_name}.log", level="WARNING", rotation="10 MB")
            results = func(lock)
            logger.success(f"[{func_name}] Success:: {results}")
        except Exception as e:
            logger.error(f"[{e.__class__.__name__}] Error:: {e}")
        finally:
            pass

    @classmethod
    def run_after(
        cls,
        func_name: Literal["future_1m", "future_1d", "option_1d"],
        kind: Literal["index", "major", "secondery", "option"],
        lock,
    ):
        """
        处理1m数据需要现有tick数据, 处理1d数据需要先有1m数据. 因此运行该函数之前, 需确保有相应的数据.
        
        :param func_name: `future_1m`代表处理期货的1分钟线数据, 其他类推.
        :param kind: 期货数据的类别, `index`代表指数, `major`代表主力合约, `secondery`代表次主力合约.\
                    如果第一个参数是`option_1d`, 则此参数无实际意义, 仅代表日志名, 建议为`option`
        :param lock: 进程锁
        """
        try:
            func = FUNC_MAP[func_name]
            logger.add(
                f"logs/{func_name}/{kind}.log", level="WARNING", rotation="10 MB"
            )
            match func_name.split("_")[0]:
                case "future":
                    results = func(kind, lock)
                case "option":
                    results = func(lock)
                case _:
                    raise ValueError(
                        "Invalid function name. It should be 'future_1m' or 'future_1d' or 'option_1d'."
                    )
            logger.success(f"[{kind}] Success:: {results}")
        except Exception as e:
            logger.error(f"[{e.__class__.__name__}] Error:: {e}")
        finally:
            pass

    @classmethod
    def run_all_pre(cls):
        """
        处理主连、次主力连、指数合约的tick数据以及期权的1分钟数据。因为这些数据可以直接处理, 不需要依赖其他数据.
        """
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Manager() as manager:
            lock = manager.Lock()
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.apply_async(cls.run_pre, args=("index_tick", lock))
                pool.apply_async(cls.run_pre, args=("major_tick", lock))
                pool.apply_async(cls.run_pre, args=("secondery_tick", lock))
                pool.apply_async(cls.run_pre, args=("option_1m", lock))
                pool.close()
                pool.join()
                breakpoint()

    @classmethod
    def run_all_1m(cls):
        """
        处理主连、次主力连、指数合约的1m数据. 需要确保有相应的tick数据.
        """
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Manager() as manager:
            lock = manager.Lock()
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.apply_async(cls.run_after, args=("future_1m", "index", lock))
                pool.apply_async(cls.run_after, args=("future_1m", "major", lock))
                pool.apply_async(cls.run_after, args=("future_1m", "secondery", lock))
                pool.close()
                pool.join()

    @classmethod
    def run_all_1d(cls):
        """
        处理主连、次主力连、指数合约以及期权的1d数据. 需要确保有相应的1m数据.
        """
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Manager() as manager:
            lock = manager.Lock()
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.apply_async(cls.run_after, args=("future_1d", "index", lock))
                pool.apply_async(cls.run_after, args=("future_1d", "major", lock))
                pool.apply_async(cls.run_after, args=("future_1d", "secondery", lock))
                pool.apply_async(cls.run_after, args=("option_1d", "option", lock))
                pool.close()
                pool.join()

    @classmethod
    def run_data_supplement(cls):
        """
        补充主连、次主力连、指数合约以及期权的tick、1m、1d数据. 主函数接口
        """
        logger.remove()
        logger.add(sys.stderr, level="SUCCESS")
        cls.run_all_pre()
        cls.run_all_1m()
        cls.run_all_1d()


if __name__ == "__main__":
    DataRunner.run_data_supplement()
