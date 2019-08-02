from __future__ import absolute_import
from __future__ import print_function

from odps import readers, ODPS


class RecordReader(readers.AbstractRecordReader):
    """
    hack from odps python sdk
    """
    def __init__(self, download_session):
        self._it = iter(self)
        self._sess = download_session

    @property
    def download_id(self):
        return self._sess.id

    @property
    def count(self):
        return self._sess.count

    @property
    def status(self):
        return self._sess.status

    def __iter__(self):
        for record in self.read():
            yield record

    def __next__(self):
        return next(self._it)

    next = __next__

    def _iter(self, start=None, end=None, step=None):
        count = self._calc_count(start, end, step)
        return self.read(start=start, count=count, step=step)

    def read(self, start=None, count=None, step=None,
             compress=False, columns=None):
        start = start or 0
        step = step or 1
        count = count * step if count is not None else self.count - start

        if count == 0:
            return

        with self._sess.open_record_reader(
                start, count, compress=compress, columns=columns) as reader:
            for record in reader[::step]:
                yield record

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def create_download_session(access_id: str,
                            access_key: str,
                            project: str,
                            endpoint: str,
                            table: str,
                            partition: str = None):
    """
    create download session of odps.tunnel, which can be shared by multiple reading processes.
    :param access_id:
    :param access_key:
    :param project:
    :param endpoint:
    :param table:
    :param partition:
    :return:
    """
    odps = ODPS(access_id, access_key, project, endpoint)
    table = odps.get_table(table)
    tunnel = table._create_table_tunnel()
    return tunnel.create_download_session(table, partition_spec=partition)
