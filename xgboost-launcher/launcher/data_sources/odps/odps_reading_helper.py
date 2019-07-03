#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import
from __future__ import print_function

import logging
import random
from concurrent.futures import ProcessPoolExecutor as Executor
from queue import Queue

import numpy as np
import odps
import requests
import urllib3
from odps import ODPS

from ... import utils

logger = logging.getLogger(__name__)


def read_odps_one_shot(project, access_id, access_key, endpoint, table, partition, start, end, columns,
                       max_retry_times=3):
    """
    Read ODPS table in chosen row range [ `start`, `end` ) with columns `columns`

    Args:
        project: ODPS project
        access_id: ODPS user access ID
        access_key: ODPS user access key
        endpoint: ODPS cluster endpoint
        table: ODPS table name
        partition: ODPS table's partition, `None` if no-partitioned table
        start: row range start
        end: row range end
        columns: chosen columns
        max_retry_times : max_retry_times
    Returns: Two-dimension python list with shape: (end - start, len(columns))
    """
    odps_table = ODPS(access_id, access_key, project, endpoint).get_table(table)

    retry_time = 0

    while retry_time < max_retry_times:
        try:
            batch_record = []
            with odps_table.open_reader(partition=partition, reopen=True) as reader:
                for record in reader.read(start=start, count=end - start, columns=columns):
                    batch_record.append([record[column] for column in columns])

            return batch_record

        except (requests.exceptions.ReadTimeout, odps.errors.ConnectTimeout,
                urllib3.exceptions.ReadTimeoutError):
            import time
            # if exceeds max timeout, raise with stack
            if retry_time >= max_retry_times:
                raise

            logging.info('connect timeout. retrying {} time'.format(retry_time))
            time.sleep(5)
            retry_time += 1

    # should never get here
    assert False, 'A bug occurs. Please contact admin.'


class ODPSReader:

    def __init__(self, project, access_id, access_key, endpoint, table, partition=None,
                 num_processes=None, options=None):
        """
        Constructs a `ODPSReader` instance.

        Args:
            project: ODPS project
            access_id: ODPS user access ID
            access_key: ODPS user access key
            endpoint: ODPS cluster endpoint
            table: ODPS table name
            partition: ODPS table's partition, `None` if no-partitioned table
            num_processes: Number of multi-process. If `None`, use core number instead
            options: Other defined options: https://yuque.antfin-inc.com/modelbuilding/api/uh2k0f
        """
        assert project is not None
        assert access_id is not None
        assert access_key is not None
        assert endpoint is not None
        assert table is not None

        if table.find('.') > 0:
            project, table = table.split('.')
        if options is None:
            options = {}
        self._project = project
        self._access_id = access_id
        self._access_key = access_key
        self._endpoint = endpoint
        self._table = table
        self._partition = partition
        self._num_processes = max([num_processes or 1, utils.max_thread_num()])
        self._options = options

        odps.options.retry_times = options.get('odps.options.retry_times', 5)
        odps.options.read_timeout = options.get('odps.options.read_timeout', 200)
        odps.options.connect_timeout = options.get('odps.options.connect_timeout', 200)
        odps.options.tunnel.endpoint = options.get('odps.options.tunnel.endpoint', None)
        # TODO: Need to generalize the logic in order to work for both China and overseas projects
        if odps.options.tunnel.endpoint is None and 'service.odps.aliyun-inc.com/api' in self._endpoint:
            odps.options.tunnel.endpoint = 'http://dt.odps.aliyun-inc.com'

    def to_iterator(self, num_worker, index_worker, batch_size,
                    shuffle=False,
                    columns=None):
        """
        Load slices of ODPS table (partition of table if `partition` was specified) data with Python Generator.

        Args:
            num_worker: Number of worker in distributed cluster
            index_worker: Current index of worker of workers in in distributed cluster
            batch_size: Size of a slice
            shuffle: Shuffle order or not
            columns: Chosen columns. Will use all schema names of ODPS table if `None`
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

        with Executor(max_workers=self._num_processes) as executor:

            futures = Queue()
            # initialize concurrently running processes according to num_processes
            for i in range(self._num_processes):
                range_start = worker_items[i]
                range_end = min(range_start + batch_size, table_size)
                logging.info('read range: %d - %d' % (range_start, range_end))
                future = executor.submit(read_odps_one_shot, self._project, self._access_id, self._access_key,
                                         self._endpoint, self._table, self._partition, range_start, range_end, columns)
                futures.put(future)

            worker_items_index = self._num_processes

            while not futures.empty():
                if worker_items_index < len(worker_items):
                    range_start = worker_items[worker_items_index]
                    range_end = min(range_start + batch_size, table_size)
                    logging.info('read range: %d - %d' % (range_start, range_end))
                    future = executor.submit(read_odps_one_shot, self._project, self._access_id, self._access_key,
                                             self._endpoint, self._table, self._partition, range_start, range_end,
                                             columns)
                    futures.put(future)
                    worker_items_index = worker_items_index + 1

                head_future = futures.get()
                records = head_future.result()
                for i in range(0, len(records), batch_size):
                    yield records[i:i + batch_size]

    def _count_table_size(self, odps_table):
        with odps_table.open_reader(partition=self._partition) as reader:
            return reader.count
