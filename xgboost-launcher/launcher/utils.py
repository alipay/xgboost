import logging
import multiprocessing
import os
import time
from threading import Thread

import numpy as np
import psutil
from xgboost import rabit

logger = logging.getLogger(__name__)


def max_thread_num():
    return multiprocessing.cpu_count()


def memory_percent():
    process = psutil.Process(os.getpid())
    return process.memory_percent()


def cpu_percent():
    return psutil.cpu_percent(percpu=True)


def logging_machine_status():
    # TODO: add CPU usage, which should take cluster mode (cpu sharing) into consideration.
    logger.info('Memory usage: %.2f%%' % memory_percent())


def backend_logging(interval=10):
    if backend_logging._start:
        return

    def interval_logger():
        while True:
            time.sleep(interval)
            logging_machine_status()

    th = Thread(target=interval_logger)
    th.setDaemon(True)
    th.start()
    backend_logging._start = True


backend_logging._start = False


def rabit_sync_exec(fn, sync_code):
    """
    Execute a function(fn) on each node, and not return the result until all functions has been done.
    Here, we use Rabit.allreduce to sync the status. So, Rabit must be initialized before.

    :param fn: function to execute on each node
    :param sync_code: a unique code to identify the sync_exec
    :return: return of fn
    """
    assert callable(fn), 'function not callable!'
    try:
        world_size = rabit.get_world_size()
        rank = rabit.get_rank()
    except Exception as e:
        raise EnvironmentError('Rabit uninitialized!')

    ret = fn()

    sync_array = np.zeros(world_size)
    sync_array[rank] = sync_code
    sync_array = rabit.allreduce(sync_array, op=2)
    # check if all elements of return array are sync_code
    if np.sum(sync_array - sync_code) != 0:
        raise RuntimeError('Sync codes mismatch, %s!' % sync_array.tolist())

    logger.info('rabit sync job has finished successfully, whose sync code is %d.' % sync_code)

    return ret
