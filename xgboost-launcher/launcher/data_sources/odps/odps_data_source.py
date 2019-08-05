from typing import NamedTuple, Iterator, List

from odps.types import Column, OdpsSchema

from ... import config_fields
from ...data_source import DataSource, WriterUtils
from .odps_reading_helper import ODPSReader
from .odps_writing_helper import ODPSWritingHelper
from ...utils import logger
from ...data_units import RecordBuilder, XGBoostRecord, XGBoostResult


class ODPSFields(NamedTuple):
    access_id: str
    access_key: str
    project: str
    endpoint: str
    # configs for data reading
    input_table: str
    input_partition: str = None
    read_batch_size: int = 1024
    read_parallelism: int = 8
    # configs for data writing
    output_table: str = None
    output_partition: str = None
    lifecycle: int = 7
    drop_if_exists: bool = True
    write_batch_size: int = 2048


class ODPSDataSource(DataSource):
    def __init__(self, rank: int, num_worker: int,
                 column_conf: config_fields.ColumnFields,
                 source_conf):
        super().__init__(rank, num_worker, column_conf, source_conf)
        assert isinstance(source_conf, ODPSFields), "Illegal source conf for ODPSDataSource!"
        self._odps_conf = source_conf
        self._read_bs = self._odps_conf.read_batch_size
        self._read_par = self._odps_conf.read_parallelism
        self._write_bs = self._odps_conf.write_batch_size

        # TODO: support it
        if self._odps_conf.output_partition:
            raise AssertionError('Output partitions are not supported for now!')

        self._w_counter = WriterUtils.Counter()

    def read(self) -> Iterator[XGBoostRecord]:
        batch_iterator, offsets = self._read_impl()
        rcd_builder = RecordBuilder(self.col_conf.features)
        for rows in batch_iterator:
            for row in rows:
                yield rcd_builder.build(
                    feat=row[:len(self.col_conf.features.columns)],
                    label=row[offsets[0]] if offsets[0] > 0 else None,
                    group=row[offsets[1]] if offsets[1] > 0 else None,
                    weight=row[offsets[2]] if offsets[2] > 0 else None,
                    append_info=row[offsets[3]:] if offsets[3] > 0 else None)

    def _read_impl(self):
        partition = self._odps_conf.input_partition if self._odps_conf.input_partition else None
        reader = ODPSReader(
            project=self._odps_conf.project,
            access_id=self._odps_conf.access_id,
            access_key=self._odps_conf.access_key,
            endpoint=self._odps_conf.endpoint,
            table=self._odps_conf.input_table,
            partition=partition)

        # FIXME: if append_cols is duplicated with other columns, it will crash
        fetch_columns = self.col_conf.features.columns
        fetch_columns = [col for col in fetch_columns if col is not None]
        label_offset = len(fetch_columns) if self.col_conf.label else -1
        if label_offset > 0:
            fetch_columns.append(self.col_conf.label)
        group_offset = len(fetch_columns) if self.col_conf.group else -1
        if group_offset > 0:
            fetch_columns.append(self.col_conf.group)
        weight_offset = len(fetch_columns) if self.col_conf.weight else -1
        if weight_offset > 0:
            fetch_columns.append(self.col_conf.weight)
        append_offset = len(fetch_columns) if self.col_conf.append_columns else -1
        if append_offset > 0:
            fetch_columns.extend(self.col_conf.append_columns)

        reader._num_processes = self._read_par
        batch_iterator = reader.to_iterator(
            self.num_worker, self.rank,
            batch_size=self._read_bs, columns=fetch_columns,
            shuffle=True)

        return batch_iterator, [label_offset, group_offset, weight_offset, append_offset]

    def write(self, result_iter: Iterator[XGBoostResult]):
        assert self._odps_conf.output_table, 'Missing output table name!'
        helper = ODPSWritingHelper(
            project=self._odps_conf.project,
            access_id=self._odps_conf.access_id,
            access_key=self._odps_conf.access_key,
            endpoint=self._odps_conf.endpoint,
            table=self._odps_conf.output_table,
            partition=self._odps_conf.output_partition,
            lifecycle=self._odps_conf.lifecycle,
            drop_if_exists=self._odps_conf.drop_if_exists)

        # infer odps schema according to first result record
        first = result_iter.__next__()
        schema, transformers = self._infer_schema_and_transformer(first)

        # a buffer to hold records
        buffer = [[]]
        for i, (_, v) in enumerate(first._asdict().items()):
            transformers[i](buffer[0], v)
        self._w_counter.inc()

        # block_writer writing different batches into different blocks
        block_writer = helper.create_block_writer(
            schema=schema,
            rank=self.rank,
            world_size=self.num_worker,
            local_blocks=1000)
        remain_blocks = 1000
        for ret in result_iter:
            record = []
            for i, (_, value) in enumerate(ret._asdict().items()):
                transformers[i](record, value)
            buffer.append(record)
            self._w_counter.inc()

            if buffer and len(buffer) % self._write_bs == 0 and remain_blocks > 1:
                remain_blocks -= 1
                block_writer.async_block_write(buffer)
                buffer = []
                logger.info('ODPSWriter has submitted %d records!' % self._w_counter.count)
        # write the remaining part
        if buffer:
            block_writer.async_block_write(buffer)
            logger.info('ODPSWriter has submitted %d records!' % self._w_counter.count)

        block_writer.wait_to_close()

    def _infer_schema_and_transformer(self, ret: XGBoostResult) -> (OdpsSchema, List):
        ret_cols = self.col_conf.result_columns
        columns = [Column(name=ret_cols.result_column, typo='double')]
        transformers = [WriterUtils.identity_transformer]
        if ret_cols.probability_column:
            columns.append(Column(name=ret_cols.probability_column, typo='double'))
            transformers.append(WriterUtils.identity_transformer)
        else:
            transformers.append(WriterUtils.empty_transformer)
        if ret_cols.detail_column:
            columns.append(Column(name=ret_cols.detail_column, typo='string'))
            transformers.append(WriterUtils.detail_transformer)
        else:
            transformers.append(WriterUtils.empty_transformer)
        if ret_cols.leaf_column:
            columns.append(Column(name=ret_cols.leaf_column, typo='string'))
            transformers.append(WriterUtils.leaf_transformer)
        else:
            transformers.append(WriterUtils.empty_transformer)
        if self.col_conf.append_columns:
            for name, value in zip(*(self.col_conf.append_columns, ret.append_info)):
                if isinstance(value, float):
                    columns.append(Column(name=name, typo='double'))
                elif isinstance(value, int):
                    columns.append(Column(name=name, typo='bigint'))
                elif isinstance(value, bool):
                    columns.append(Column(name=name, typo='boolean'))
                elif isinstance(value, str):
                    columns.append(Column(name=name, typo='string'))
                else:
                    raise TypeError('Illegal data type of append info: %s!' % value)

            transformers.append(WriterUtils.batch_transformer)
        else:
            transformers.append(WriterUtils.empty_transformer)

        return OdpsSchema(columns=columns), transformers
