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

from typing import Iterator

import numpy as np
import pytest

from .. import config_helper
from .. import config_fields
from ..data_builder import CSRMatBuilder
from ..data_source import DataSource
from ..data_units import RecordBuilder, XGBoostResult, XGBoostRecord


def test_record_builder():
    dense_builder = RecordBuilder(config_helper.load_config(config_fields.FeatureFields, **{
        'columns': 'feat1,feat2,feat3,feat4,feat5',
    }))
    rcd = dense_builder.build([1.0, 2.0, 0.0, 4.0, 5.0])
    assert rcd.indices == [0, 1, 2, 3, 4]
    assert rcd.values == [1.0, 2.0, 0.0, 4.0, 5.0]
    with pytest.raises(RuntimeError, match=r"Unexpected feature length.*"):
        dense_builder.build([1.0, 2.0])
    rcd = dense_builder.build(
        feat=[1, 2, 3, 4, 5],
        label=1, group=3, weight=3.5, base_margin=0.1,
        append_info=['1', 2, 3.3])
    assert rcd.label == 1
    assert rcd.group == 3
    assert rcd.weight == 3.5
    assert rcd.base_margin == 0.1
    assert rcd.append_info == ['1', 2, 3.3]

    sp_builder = RecordBuilder(config_helper.load_config(config_fields.FeatureFields, **{
        'columns': 'kv',
        'feature_num': 10,
        'is_sparse': True,
        'item_delimiter': ' '
    }))
    rcd = sp_builder.build([''])
    assert rcd.indices == []
    assert rcd.values == []
    rcd = sp_builder.build(['0:1 2:100.5 8:-1000.3'])
    assert rcd.indices == [0, 2, 8]
    assert rcd.values == [1, 100.5, -1000.3]
    rcd = sp_builder.build(['0:1 2:0 3:0'])
    assert rcd.indices == [0, 2, 3]
    assert rcd.values == [1, 0, 0]
    rcd = sp_builder.build(['1:0 8:-1000.3 0:1 2:100.5'])
    assert rcd.indices == [0, 1, 2, 8]
    assert rcd.values == [1, 0, 100.5, -1000.3]
    with pytest.raises(IndexError, match=r".*which should be smaller"):
        sp_builder.build(['2:1 11:100'])
    with pytest.raises(IndexError, match=r".*it occurs more"):
        sp_builder.build(['3:1 2:2 3:1 4:5'])


class TestDataSource(DataSource):
    def __init__(self, n_rows, n_cols,
                 has_label=False, has_group=False, has_weight=False, has_margin=False,
                 n_append_cols=0):
        num_features = n_cols - n_append_cols
        self._append = num_features
        self._label = self._group = self._weight = self._margin = -1
        if has_margin:
            num_features -= 1
            self._margin = num_features
        if has_weight:
            num_features -= 1
            self._weight = num_features
        if has_group:
            num_features -= 1
            self._group = num_features
        if has_label:
            num_features -= 1
            self._label = num_features
        assert num_features > 0
        self._num_features = num_features
        features = config_fields.FeatureFields(columns=['a'] * num_features)
        column_conf = config_fields.ColumnFields(features=features)
        super().__init__(0, 1, column_conf, None)
        self._data = np.random.random((n_rows, n_cols))

    def read(self) -> Iterator[XGBoostRecord]:
        rcd_builder = RecordBuilder(self.col_conf.features)
        for row in self._data:
            yield rcd_builder.build(
                feat=row[:self._num_features],
                label=row[self._label] if self._label > 0 else None,
                group=row[self._group].astype(np.int) if self._group > 0 else None,
                weight=row[self._weight] if self._weight > 0 else 1.0,
                base_margin=row[self._margin] if self._margin > 0 else None,
                append_info=row[self._append:] if self._append > 0 else None)

    def write(self, result_iter: Iterator[XGBoostResult]):
        pass

    def get_feat_mat(self):
        return self._data[:, :self.num_features]

    def get_label(self):
        if self._label < 0:
            return None
        return self._data[:, self._label]

    def get_group(self):
        if self._group < 0:
            return None
        return self._data[:, self._group]

    def get_weight(self):
        if self._weight < 0:
            return None
        return self._data[:, self._weight]

    def get_margin(self):
        if self._margin < 0:
            return None
        return self._data[:, self._margin]

    def get_append(self):
        if not self._append:
            return None
        return self._data[:, self._append:]


def test_d_matrix_built_by_csr_mat_builder():
    ds = TestDataSource(100, 6)
    # test full batch transform
    builder = CSRMatBuilder(ds)
    # we can't fetch data from a DMatrix, so just check some meta
    data_iter = builder.build()
    data = data_iter.__next__()
    with pytest.raises(StopIteration):
        data_iter.__next__()
    assert not data.append_info
    assert data.d_matrix.num_row() == 100
    assert data.d_matrix.num_col() == 6
    # check data equality between csr_matrix and source data
    csr_mat = builder._build_csr().__next__()[0]
    assert np.all(csr_mat.todense() == ds.get_feat_mat())

    # test mini batch transform
    builder = CSRMatBuilder(ds, batch_size=10)
    data_iter = builder.build()
    for i in range(10):
        data = data_iter.__next__()
        assert data.d_matrix.num_row() == 10
        assert data.d_matrix.num_col() == 6
    with pytest.raises(StopIteration):
        data_iter.__next__()

    batches = [np.array(batch[0].todense()) for batch in builder._build_csr()]
    data_from_csr = np.concatenate(batches, axis=0)
    assert np.all(data_from_csr == ds.get_feat_mat())

    builder = CSRMatBuilder(ds, batch_size=13)
    data_iter = builder.build()
    for i in range(7):
        data = data_iter.__next__()
        assert data.d_matrix.num_row() == 13
        assert data.d_matrix.num_col() == 6
    assert data_iter.__next__().d_matrix.num_row() == 9
    with pytest.raises(StopIteration):
        data_iter.__next__()

    batches = [np.array(batch[0].todense()) for batch in builder._build_csr()]
    data_from_csr = np.concatenate(batches, axis=0)
    assert np.all(data_from_csr == ds.get_feat_mat())


def test_other_cols_built_by_csr_mat_builder():
    ds = TestDataSource(
        n_rows=100, n_cols=10,
        has_label=True, has_group=True, has_weight=True, has_margin=True,
        n_append_cols=3)
    builder = CSRMatBuilder(ds, batch_size=10)
    data_iter = builder.build()
    for i in range(10):
        data = data_iter.__next__()
        assert data.d_matrix.num_col() == 3
        assert np.all(np.isclose(data.d_matrix.get_label(), ds.get_label()[i*10:i*10+10]))
        assert np.all(np.isclose(data.d_matrix.get_weight(), ds.get_weight()[i*10:i*10+10]))
        assert np.all(np.isclose(data.d_matrix.get_base_margin(), ds.get_margin()[i*10:i*10+10]))
        assert np.all(data.append_info == ds.get_append()[i*10:i*10+10, :])

    batches = [np.array(batch[0].todense()) for batch in builder._build_csr()]
    data_from_csr = np.concatenate(batches, axis=0)
    assert np.all(data_from_csr == ds.get_feat_mat())
