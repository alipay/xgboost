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
        df = pd.read_csv(self._csv_conf.input_table)

        feature_cols = df[self.col_conf.features.columns] if self.col_conf.features.columns else None
        label_cols = df[self.col_conf.label] if self.col_conf.label else None 
        group_cols = df[self.col_conf.group] if self.col_conf.group else None
        weight_cols = df[self.col_conf.weight] if self.col_conf.weight else None
        append_cols = df[self.col_conf.append_columns] if self.col_conf.append_columns else None
        return feature_cols, label_cols, group_cols, weight_cols, append_cols

    def read(self) -> Iterator[XGBoostRecord]:
        feature_cols, label_cols, group_cols, weight_cols, append_cols = self._read_impl()
        rcd_builder = RecordBuilder(self.col_conf.features)
        for row in range(len(feature_cols)):
            yield rcd_builder.build(
                feat = feature_cols.loc[row,:] if feature_cols is not None else None,
                label = label_cols[row] if label_cols is not None  else None,
                group = group_cols[row] if group_cols  is not None else None,
                weight = weight_cols[row] if weight_cols  is not None else None,
                append_info = list(append_cols.loc[row,:]) if append_cols is not None else None
                )

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
