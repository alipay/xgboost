from __future__ import absolute_import
from __future__ import print_function

import logging
import random
from concurrent.futures import ThreadPoolExecutor as Executor
from typing import List

import numpy as np
import odps
import requests
import urllib3

from .odps_utils import create_download_session, RecordReader
from ...utils import logger, max_thread_num
from odps import ODPS


class ODPSReader:
    def __init__(self, project: str,
                 access_id: str, access_key: str, endpoint: str,
                 table: str, partition: str = None,
                 num_processes: int = None):
        """
        :param project: ODPS project
        :param access_id: ODPS user access ID
        :param access_key: ODPS user access key
        :param endpoint: ODPS cluster endpoint
        :param table: ODPS table name
        :param partition: ODPS table's partition, `None` if no-partitioned table
        :param num_processes: Number of multi-process. If `None`, use core number instead
        """

        assert project is not None
        assert access_id is not None
        assert access_key is not None
        assert endpoint is not None
        assert table is not None

        if table.find('.') > 0:
            project, table = table.split('.')
        self._project = project
        self._access_id = access_id
        self._access_key = access_key
        self._endpoint = endpoint
        self._table = table
        self._partition = partition
        self._num_processes = max([num_processes or 1, max_thread_num()])
        self._session = None

        odps.options.retry_times = 5
        odps.options.read_timeout = 200
        odps.options.connect_timeout = 200
        odps.options.tunnel.endpoint = None

    def to_iterator(self, num_worker, index_worker, batch_size,
                    shuffle=False,
                    columns=None):
        """
        Load slices of table data with Python Generator.
        :param num_worker: Number of worker in distributed cluster
        :param index_worker: Current index of worker of workers in in distributed cluster
        :param batch_size: Size of a slice
        :param shuffle: Shuffle order or not
        :param columns: Chosen columns. Will use all schema names of ODPS table if `None`
        :return Generator of List[row]
        """

        if not index_worker < num_worker:
            raise ValueError('index of worker should be less than number of worker')
        if not batch_size > 0:
            raise ValueError('batch_size should be positive')
        odps_table = ODPS(self._access_id, self._access_key, self._project, self._endpoint).get_table(self._table)
        table_size = self._count_table_size(odps_table)
        if columns is None:
            columns = odps_table.schema.names

        overall_items = range(0, table_size, batch_size)
        worker_items = list(np.array_split(np.asarray(overall_items), num_worker)[index_worker])
        if shuffle:
            random.shuffle(worker_items)

        self._num_processes = min(self._num_processes, len(worker_items))

        # Create download session, which is shared by reading threads of this machine.
        self._session = create_download_session(
            self._access_id, self._access_key, self._project, self._endpoint,
            self._table, self._partition)
        logger.info("download id: %s" % self._session.id)

        with Executor(max_workers=self._num_processes) as executor:
            futures = []
            running_items = []
            # producer: submit reading tasks
            for i, range_start in enumerate(worker_items):
                range_end = min(range_start + batch_size, table_size)
                logging.debug('read range: %d - %d' % (range_start, range_end))
                future = executor.submit(self._block_reading, range_start, range_end, columns)
                futures.append(future)
                running_items.append(i)

            # consumer: fetch available task results asynchronously
            while len(running_items) > 0:
                active_index = -1
                for index in running_items:
                    if futures[index].done():
                        running_items.remove(index)
                        active_index = index
                        break

                if active_index < 0:
                    continue

                yield futures[active_index].result()

    def _count_table_size(self, odps_table):
        with odps_table.open_reader(partition=self._partition) as reader:
            return reader.count

    def _block_reading(self,
                       start: int, end: int,
                       columns: List[str],
                       max_retry_times: int = 3):
        """
        Read ODPS table in chosen row range [`start`, `end`) with columns `columns`
        :param start: row range start
        :param end: row range end
        :param columns: chosen columns
        :param max_retry_times : max_retry_times
        :return: Two-dimension python list with shape: (end - start, len(columns))
        """

        retry_time = 0
        batch_record = []
        while len(batch_record) == 0:
            try:
                with RecordReader(self._session) as reader:
                    for record in reader.read(start=start, count=end - start, columns=columns):
                        batch_record.append(record.values)
            except (requests.exceptions.ReadTimeout, odps.errors.ConnectTimeout, urllib3.exceptions.ReadTimeoutError):
                import time
                if retry_time > max_retry_times:
                    raise RuntimeError('Reading table failure more than %d times!' % max_retry_times)

                logger.info('connect timeout. retrying {} time'.format(retry_time))
                time.sleep(5)
                retry_time += 1

        return batch_record
