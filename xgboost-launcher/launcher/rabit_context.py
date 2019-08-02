import logging
import subprocess
import time

import xgboost as xgb

from .env_parser import DistributedEnv
from .tracker import RabitTracker

logger = logging.getLogger(__name__)


class RabitContext:
    def __init__(self, dist_env):
        assert isinstance(dist_env, DistributedEnv)
        addr, port, world_size, rank = dist_env.unbox()
        self.addr = addr
        if port is not None:
            assert 0 < port <= 65536
        self.port = port
        assert 0 <= rank < world_size
        self.world_size = world_size
        self.rank = rank
        self._tracker = None
        self._rabit_init = False

    def __enter__(self):
        if self.world_size == 1:
            logger.info("Local environment detected.")
            return self

        logger.info("Distributed environment detected, world size is %d." % self.world_size)
        if self.rank > 0:
            logger.info("Current node running as a worker, whose rank is %d." % self.rank)
        else:
            # rabit tracker init
            logger.info("Current node running as a master, whose hosts the RabitTracker.")
            # make tracker listening all requests
            self._tracker = RabitTracker("0.0.0.0", self.world_size, self.port, self.port + 1)
            self._tracker.start(self.world_size)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._rabit_init:
            xgb.rabit.finalize()
        if self._tracker:
            self._tracker.join()

    def client_init(self, timeout=300):
        if self.world_size > 1:
            rabit_init(self.addr, self.port, self.rank, max_wait_time=timeout)
            self._rabit_init = True


def rabit_init(addr, port, rank, reconnect_interval=30, max_wait_time=300):
    wait_time = 0
    while True:
        proc = subprocess.run(['ping', '-c', '1', addr])
        if proc.returncode == 0:
            break
        elif proc.returncode == 2:
            if wait_time >= max_wait_time:
                raise RuntimeError('Failed to connect RabitTracker in %d secs, given up!' % max_wait_time)
            logger.warning(
                "RabitTracker is still not ready. Wait %d secs to re-connect!" % reconnect_interval)
            time.sleep(reconnect_interval)
            wait_time += reconnect_interval
        else:
            raise RuntimeError("Illegal return code(%d) of %s!" % (proc.returncode, proc.args))

    rabit_env = [
        str.encode('DMLC_TRACKER_URI=%s' % addr),
        str.encode('DMLC_TRACKER_PORT=%d' % port),
        str.encode('DMLC_TASK_ID=%d' % rank)]
    xgb.rabit.init(rabit_env)
    logger.info('Rabit rank of current worker is %d.' % xgb.rabit.get_rank())
