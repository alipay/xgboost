from __future__ import absolute_import
from __future__ import print_function

import logging
import sys
from abc import abstractmethod
from typing import Iterator, NamedTuple, Any, List

import xgboost as xgb
from scipy.sparse import csr_matrix

from . import config_fields
from .data_source import DataSource

logger = logging.getLogger(__name__)


class XGBoostData(NamedTuple):
    d_matrix: xgb.DMatrix
    append_info: List[List[Any]] = None


class XGBoostDataBuilder:
    """
    Base class of XGBoost data builder, producing XGBoostData from XGBoostRecord (output of data_source.reader)
    in form of Generator.
    """

    def __init__(self, data_reader: DataSource,
                 batch_size: int = -1,
                 logging_interval: int = 2000):
        """
        :param data_reader: specific DataSource of input data
        :param batch_size: batch size of XGBoostData, default (-1) means full batch
        :param logging_interval: record interval of progress logging (unit: seconds)
        """

        self.num_features = data_reader.num_features
        self._iterator_fn = lambda: data_reader.read()
        self.batch_size = batch_size
        self.logging_interval = logging_interval

    @abstractmethod
    def build(self) -> Iterator[XGBoostData]:
        pass

    @classmethod
    def create(cls, config: config_fields.DataBuilderFields,
               data_reader: DataSource):
        """
        factory of XGBoostDataBuilder
        :param config: configs.DataBuilderFields
        :param data_reader: specific DataSource of input data
        :return: a XGBoostDataBuilder instance
        """

        if config.name == "CSRMatrixBuilder":
            return CSRMatBuilder(data_reader, config.batch_size)
        else:
            raise NameError("Unknown DMatrixBuilder %s!" % config.name)


class CSRMatBuilder(XGBoostDataBuilder):
    """
    Convert XGBoostRecords into scipy.csr_matrix, and then build xgb.DMatrix from csr_matrix.
    """

    def build(self) -> Iterator[XGBoostData]:
        for mat, label, group, weight, base_margin, append_info in self._build_csr():
            d_matrix = xgb.DMatrix(mat)
            if label:
                d_matrix.set_label(label)
            if group:
                d_matrix.set_group(group)
            if weight:
                d_matrix.set_weight(weight)
            if base_margin:
                d_matrix.set_base_margin(base_margin)
            yield XGBoostData(d_matrix, append_info)

    def _build_csr(self) -> Iterator:
        row_size = self.batch_size if self.batch_size > 0 else sys.maxsize
        col_size = self.num_features
        data = []
        ind = []
        indptr = [0]
        label_buf = []
        group_buf = []
        weight_buf = []
        base_margin_buf = []
        append_info_buf = []
        count = 0
        for rcd in self._iterator_fn():
            data_length = len(rcd.indices)
            ind.extend(rcd.indices)
            data.extend(rcd.values)
            last_ind = indptr[-1]
            indptr.append(last_ind + data_length)
            if rcd.label is not None:
                label_buf.append(rcd.label)
            if rcd.weight is not None:
                weight_buf.append(rcd.weight)
            if rcd.group is not None:
                group_buf.append(rcd.group)
            if rcd.base_margin is not None:
                base_margin_buf.append(rcd.base_margin)
            if rcd.append_info is not None and len(rcd.append_info) > 0:
                append_info_buf.append(rcd.append_info)
            count += 1
            if count % min([self.logging_interval, row_size]) == 0:
                logging.info('CSRMatrixBuilder has fetched %d records.' % count)
            if count % row_size == 0:
                # if col_size == 0, let csr_matrix do shape inference.
                if col_size > 0:
                    mat = csr_matrix((data, ind, indptr), [row_size, col_size])
                else:
                    mat = csr_matrix((data, ind, indptr))
                data.clear()
                ind.clear()
                indptr.clear()
                indptr.append(0)
                label = label_buf[:]
                label_buf.clear()
                group = group_buf[:]
                group_buf.clear()
                weight = weight_buf[:]
                weight_buf.clear()
                base_margin = base_margin_buf[:]
                base_margin_buf.clear()
                append_info = append_info_buf[:]
                append_info_buf.clear()
                yield mat, label, group, weight, base_margin, append_info

        if data:
            if col_size > 0:
                mat = csr_matrix((data, ind, indptr), [count % row_size, col_size])
            else:
                mat = csr_matrix((data, ind, indptr))
            yield mat, label_buf, group_buf, weight_buf, base_margin_buf, append_info_buf
