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

from typing import Iterator, NamedTuple, List

from .. import config_helper
from .. import config_fields
from ..data_units import XGBoostResult, XGBoostRecord
from ..data_source import DataSource, register_data_source, create_data_source_init_fn
from ..model_source import register_model_source, ModelSource, create_model_source


class MockDataConfigA(NamedTuple):
    field_x: str
    field_y: int
    field_z: float


class MockDataSourceA(DataSource):
    def read(self) -> Iterator[XGBoostRecord]:
        pass

    def write(self, result_iter: Iterator[XGBoostResult]):
        pass


class MockDataConfigB(NamedTuple):
    field_a: str
    field_b: int
    field_c: float


class MockDataSourceB(DataSource):
    def read(self) -> Iterator[XGBoostRecord]:
        pass

    def write(self, result_iter: Iterator[XGBoostResult]):
        pass


__mock_col_format = config_helper.load_config(config_fields.ColumnFields, **{
    'features': {
        'columns': 'a,b,c,d,e'
    },
    'label': 'mock_label'
})


def test_data_source():
    register_data_source('mock_a', MockDataConfigA, MockDataSourceA)
    register_data_source('mock_b', MockDataConfigB, MockDataSourceB)
    register_data_source('mock_b', MockDataConfigB, MockDataSourceB)

    ds_partial_a = create_data_source_init_fn(config_fields.DataSourceFields(**{
        'name': 'mock_a',
        'config': {
            'field_x': 'A',
            'field_y': 11,
            'field_z': 12.345
        }
    }))
    ds_a = ds_partial_a(0, 1, __mock_col_format)
    assert isinstance(ds_a, MockDataSourceA)
    src_conf = ds_a.source_conf
    assert src_conf.field_x == 'A'
    assert src_conf.field_y == 11
    assert src_conf.field_z == 12.345
    assert ds_a.col_conf == __mock_col_format

    ds_partial_b = create_data_source_init_fn(config_fields.DataSourceFields(**{
        'name': 'mock_b',
        'config': {
            'field_a': 'A',
            'field_b': 11,
            'field_c': 12.345
        }
    }))
    ds_b = ds_partial_b(0, 1, __mock_col_format)
    assert isinstance(ds_b, MockDataSourceB)
    src_conf = ds_b.source_conf
    assert src_conf.field_a == 'A'
    assert src_conf.field_b == 11
    assert src_conf.field_c == 12.345
    assert ds_b.col_conf == __mock_col_format


class MockModelConfig(NamedTuple):
    field_a: str
    field_b: int
    field_c: float


class MockModelSource(ModelSource):
    def read_buffer(self, model_path: str) -> bytes:
        pass

    def write_buffer(self, buf: bytes, model_path):
        pass

    def read_lines(self, model_path: str) -> List[str]:
        pass

    def write_lines(self, lines: List[str], model_path):
        pass


def test_model_source():
    register_model_source('mock_a', MockModelConfig, MockModelSource)
    register_model_source('mock_b', MockModelConfig, MockModelSource)

    mk_src_a = create_model_source(config_fields.ModelSourceFields(**{
        'name': 'mock_a',
        'config': {
            'field_a': 'A',
            'field_b': 11,
            'field_c': 12.345
        }
    }))
    mk_src_b = create_model_source(config_fields.ModelSourceFields(**{
        'name': 'mock_b',
        'config': {
            'field_a': 'A',
            'field_b': 11,
            'field_c': 12.345
        }
    }))
    assert mk_src_a.source_conf == mk_src_b.source_conf
