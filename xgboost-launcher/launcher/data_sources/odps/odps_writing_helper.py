from __future__ import absolute_import
from __future__ import print_function

import logging
from threading import Thread

from odps import ODPS
from odps.types import OdpsSchema

from ...utils import rabit_sync_exec

logger = logging.getLogger(__name__)

RENEW_TABLE_MAGIC_CODE = 10103488


class ODPSWritingHelper:

    def __init__(self, project, access_id, access_key, endpoint, table, partition=None,
                 lifecycle=None, drop_if_exists=True, comment=None):

        """
        Construct a `ODPSWriter` instance

        Args:
            project: String, ODPS project
            access_id: String, ODPS user access ID
            access_key: String, ODPS user access key
            endpoint: String, ODPS cluster endpoint
            table: String, ODPS table name
            partition: String or PartitionSpec, ODPS table's partition, `None` if no-partitioned table
            lifecycle: Int, table lifecycle. If absent, options.lifecycle will be used.
            drop_if_exists: Bool, drop table or partition if exists, False as default
            comment: String, comment when create table
        """

        assert project is not None
        assert access_id is not None
        assert access_key is not None
        assert endpoint is not None
        assert table is not None

        table_project = project
        if table.find('.') > 0:
            table_project, table = table.split('.')

        self._project = project
        self._access_id = access_id
        self._access_key = access_key
        self._endpoint = endpoint
        self._table = table
        self._table_project = table_project
        self._partition = partition
        self._lifecycle = lifecycle
        self._drop_if_exists = drop_if_exists
        self._comment = comment
        self._o = ODPS(self._access_id, self._access_key, self._project, self._endpoint)

    def get_or_create_table(self, schema):
        self._drop_table_or_partition()

        if not self._o.exist_table(self._table, self._table_project):
            table = self._o.create_table(
                self._table, project=self._table_project, schema=schema,
                lifecycle=self._lifecycle, comment=self._comment)
            if self._partition is not None:
                table.create_partition(self._partition)
        else:
            table = self.get_table()

        return table

    def get_table(self):
        return self._o.get_table(self._table, self._table_project)

    def _drop_table_or_partition(self):

        if self._drop_if_exists:
            if self._o.exist_table(self._table, self._table_project):
                table = self._o.get_table(self._table, self._table_project)
                if self._partition is not None:
                    if table.exist_partition(self._partition):
                        logging.debug('Partition %s exists in table %s, drop partition first' % self._partition,
                                      self._table)
                        table.delete_partition(self._partition, if_exists=True)
                else:
                    logging.debug('Table %s exists, drop table first' % self._table)
                    self._o.delete_table(self._table, self._project)

    def _check_exists_table_or_partition(self):
        if not self._o.exist_table(self._table, self._table_project):
            return False

        table = self._o.get_table(self._table, self._table_project)
        if self._partition is not None:
            return table.exist_partition(self._partition)

        return True

    def create_block_writer(self, schema, rank=0, world_size=1, local_blocks=1):
        """
        Create a writer which can write data into different ODPS blocks in parallel.
        This writer can work both in local and distributed runtime.

        :param schema: OdpsSchema, schema of table
        :param rank: int, rank of current machine, default is 0
        :param world_size: int, size of distributed env, default is 1
        :param local_blocks: int, number of blocks each local machine holds
        :return: the block writer
        """
        assert isinstance(schema, OdpsSchema), "Illegal table schema!"

        if world_size > 1:
            def renew_table_fn():
                if rank == 0:
                    self.get_or_create_table(schema)
            # In distributed env, all nodes wait node(rank=0) to renew the table before get its ref.
            rabit_sync_exec(renew_table_fn, sync_code=RENEW_TABLE_MAGIC_CODE)
            table = self.get_table()
        else:
            table = self.get_or_create_table(schema)

        part = self._partition

        class BlockWriter:
            def __init__(self):
                self._blocks = [x + rank for x in range(0, world_size * local_blocks, world_size)]
                self._blocks = self._blocks[::-1]
                self._writer = table.open_writer(blocks=self._blocks.copy(), partition=part)
                self._th_pool = []

            def async_block_write(self, data):
                if len(self._blocks) == 0:
                    self._writer.close()
                    raise RuntimeError('No more block left in BlockWriter!')
                block_id = self._blocks.pop()

                def write_impl():
                    self._writer.write(data, block_id=block_id)
                    local_block_id = (block_id - rank) / world_size
                    print('ODPS block writing finished, local block id=%d!\n' % local_block_id)

                th = Thread(target=write_impl)
                th.start()
                self._th_pool.append(th)

            def wait_to_close(self):
                for th in self._th_pool:
                    th.join()
                self._writer.close()

        return BlockWriter()
