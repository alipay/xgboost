from typing import Iterator
from typing import NamedTuple, Iterator, List,Dict

from ...data_units import XGBoostResult, XGBoostRecord,  XGBoostResult
from ...data_source import DataSource, WriterUtils
from ...utils import logger
from ... import config_fields

import pandas as pd
import numpy as np

class CsvFields(NamedTuple):
    input_table: str
    output_table: str


class CsvDataSource(DataSource):
    def __init__(self, rank: int, 
                 num_worker: int,
                 column_conf: config_fields.ColumnFields,
                 source_conf):
        super().__init__(rank, num_worker, column_conf, source_conf)
        assert isinstance(source_conf, CsvFields), "Illegal source conf for CsvDataSource!"
        self._csv_conf = source_conf
        self._w_counter = WriterUtils.Counter()

    def _read_impl(self):
        input_table = self._csv_conf.input_table
        input_df = pd.read_csv(input_table, header=None)
        # FIXME: if append_cols is duplicated with other columns, it will crash
        self.col_conf.features.columns
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

        return input_df, [label_offset, group_offset, weight_offset, append_offset]

    def read(self) -> Iterator[XGBoostRecord]:
        input_df, offsets = self._read_impl()
        rcd_builder = RecordBuilder(self.col_conf.features)
       	for row in input_df.values:
            yield rcd_builder.build(
                feat=row[:len(self.col_conf.features.columns)],
                label=row[offsets[0]] if offsets[0] > 0 else None,
                group=row[offsets[1]] if offsets[1] > 0 else None,
                weight=row[offsets[2]] if offsets[2] > 0 else None,
                append_info=row[offsets[3]:] if offsets[3] > 0 else None)


    def write(self, result_iter: Iterator[XGBoostResult]):
        assert self._csv_conf.output_table, 'Missing output table name!'
        outputTable=self._csv_conf.output_table
        first = result_iter.__next__()
        columns, columns_type, transformers = self._infer_schema_and_transformer(first)
        # a buffer to hold records
        buffer = [[]]
        for i, (_, v) in enumerate(first._asdict().items()):
            transformers[i](buffer[0], v)
        self._w_counter.inc()
        for ret in result_iter:
            record = []
            for i, (_, value) in enumerate(ret._asdict().items()):
                transformers[i](record, value)
            buffer.append(record)
            self._w_counter.inc()
        if buffer:
        	logger.info('CsvWriter has submitted %d records!' % self._w_counter.count)
        buffer = np.array(buffer)
        outputDF = pd.DataFrame(buffer,columns=columns)
        # transfer columns's type 
        for c in outputDF.columns:
        	outputDF[c] = outputDF[c].astype(columns_type[c])
        # save predict results to csv file
        outputDF.to_csv(outputTable, index=False)
        print(outputDF)


    def _infer_schema_and_transformer(self, ret: XGBoostResult) -> (List, Dict, List):
        ret_cols = self.col_conf.result_columns
        columns = []
        columns_type = {}
        columns.append(ret_cols.result_column)
        columns_type[ret_cols.result_column] = 'double'
        transformers = [WriterUtils.identity_transformer]
        if ret_cols.probability_column:
            columns.append(ret_cols.probability_column)
            columns_type[ret_cols.probability_column] = 'double'
            transformers.append(WriterUtils.identity_transformer)
        else:
            transformers.append(WriterUtils.empty_transformer)
        if ret_cols.detail_column:
            columns.append(ret_cols.detail_column)
            columns_type[ret_cols.detail_column] = 'str'
            transformers.append(WriterUtils.detail_transformer)
        else:
            transformers.append(WriterUtils.empty_transformer)
        if ret_cols.leaf_column:
            columns.append(ret_cols.leaf_column)
            columns_type[ret_cols.leaf_column] = 'str'
            transformers.append(WriterUtils.leaf_transformer)
        else:
            transformers.append(WriterUtils.empty_transformer)
        if self.col_conf.append_columns:
            for name, value in zip(*(self.col_conf.append_columns, ret.append_info)):
                columns.append(name)
                if isinstance(value, float):
                	columns_type[name] = 'float'
                elif isinstance(value, int):
                	columns_type[name] = 'int'
                elif isinstance(value, bool):
                	columns_type[name] = 'bool'
                elif isinstance(value, str):
                	columns_type[name] = 'str'
                else:
                    raise TypeError('Illegal data type of append info: %s!' % value)
            transformers.append(WriterUtils.batch_transformer)
        else:
            transformers.append(WriterUtils.empty_transformer)
        return columns, columns_type , transformers
