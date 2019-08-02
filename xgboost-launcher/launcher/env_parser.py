import json
import os
from typing import NamedTuple

import logging

logger = logging.getLogger(__name__)


class DistributedEnv(NamedTuple):
    master_addr: str
    master_port: int
    world_size: int
    rank: int

    def unbox(self):
        logger.info("MASTER_ADDR: %s", self.master_addr)
        logger.info("MASTER_PORT: %s", self.master_port)
        logger.info("RANK: %d" % self.rank)
        logger.info("WORLD_SIZE: %d" % self.world_size)
        return self.master_addr, self.master_port, self.world_size, self.rank


__DistEnv = [None]


def extract_dist_env():
    if __DistEnv[0]:
        return __DistEnv[0]
    envs = os.environ
    if 'TF_CONFIG' in envs:
        logger.info('TF distributed environments detected!')
        __DistEnv[0] = extract_tf_env()
    elif 'MASTER_ADDR' in envs and 'MASTER_PORT' in envs and \
            'RANK' in envs and 'WORLD_SIZE' in envs:
        logger.info('PyTorch distributed environments detected!')
        __DistEnv[0] = extract_pytorch_env()
    else:
        logger.info('No distributed environments detected! Assert single machine env!')
        __DistEnv[0] = DistributedEnv(None, None, 1, 0)
    return __DistEnv[0]


def extract_tf_env():
    tf_spec = json.loads(os.environ['TF_CONFIG'])
    logging.info("TF_CONFIG: %s" % json.dumps(tf_spec, indent=2))
    world_size = 0
    master_addr = ''
    master_port = 0
    n_chief = 0
    for k, spec in tf_spec['cluster'].items():
        if k == 'master':
            master_addr = spec[0][:spec[0].rfind(':')]
            master_port = int(spec[0][spec[0].rfind(':') + 1:])
            n_chief = len(spec)
            world_size += len(spec)
        elif k == 'worker':
            world_size += len(spec)
        else:
            raise RuntimeError("Illegal tfClusterSpec!")

    rank = int(tf_spec['task']['index'])
    if tf_spec['task']['type'].lower() == 'worker':
        rank += n_chief

    if rank == 0:
        master_addr = '127.0.0.1'

    return DistributedEnv(master_addr, master_port, world_size, rank)


def extract_pytorch_env():
    return DistributedEnv(
        master_addr=os.environ['MASTER_ADDR'],
        master_port=int(os.environ['MASTER_PORT']),
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE']))
