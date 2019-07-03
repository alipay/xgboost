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

import os

import pytest

from ..config_fields import ModelFields, XGBoostTrainFields
from ..model_helper import load_launcher_model, save_launcher_model
from ..model_source import register_model_source, LocalModelSource
from tempfile import mktemp
import numpy as np

from ..predictor import binary_logistic_converter, multi_softprob_converter, common_converter, XGBoostResultBatch

register_model_source('local', None, LocalModelSource)


def test_launcher_model():
    path = os.path.join(os.getcwd(), 'test_resources/test_booster')
    model = load_launcher_model(ModelFields(model_path=path))
    assert isinstance(model.meta, XGBoostTrainFields)
    params = model.meta.params
    assert params.objective == 'multi:softprob'
    assert params.num_class == 3
    assert params.max_depth == 5
    assert params.tree_method == 'hist'
    assert params.nthread == 2
    assert not model.meta.auto_train
    assert model.meta.num_boost_round == 100

    meta_cpy = model.meta._asdict()
    if not os.path.exists('test_resources/tmp/'):
        os.mkdir('test_resources/tmp/')
    tmp_path = mktemp(prefix='test_resources/tmp/')[5:]
    save_launcher_model(
        model.booster,
        configs.TrainFields(model.meta, None, ModelFields(model_path=tmp_path)))
    model = load_launcher_model(ModelFields(model_path=tmp_path))
    assert model.meta._asdict() == meta_cpy


def test_converters():
    reg_results = np.random.rand(100) * 1000
    i = 0
    for xgb_ret in common_converter(reg_results).to_iterator():
        assert reg_results[i] == xgb_ret.result
        i += 1

    binary_results = np.random.rand(100)
    ret_batch = binary_logistic_converter(binary_results)
    i = 0
    for xgb_ret in ret_batch.to_iterator():
        origin_ret = binary_results[i]
        assert (origin_ret >= 0.5) == xgb_ret.result
        if xgb_ret.result == 1:
            assert origin_ret == xgb_ret.classification_prob
        else:
            assert origin_ret == 1 - xgb_ret.classification_prob
        assert [1 - origin_ret, origin_ret] == xgb_ret.classification_detail
        i += 1

    num_class = 4
    multi_results = np.random.rand(100, num_class)
    ret_batch = multi_softprob_converter(multi_results)
    i = 0
    for xgb_ret in ret_batch.to_iterator():
        origin_ret = multi_results[i]
        assert origin_ret.argmax() == xgb_ret.result
        assert origin_ret.max() == xgb_ret.classification_prob
        assert origin_ret.tolist() == xgb_ret.classification_detail
        i += 1


def test_xgb_ret_batch():
    with pytest.raises(AssertionError, match='result should be 1-d array!'):
        XGBoostResultBatch(result=np.zeros([100, 2]))
    with pytest.raises(AssertionError, match='prob should has 100 rows!'):
        XGBoostResultBatch(result=np.zeros(100), prob=np.zeros(50))
    with pytest.raises(AssertionError, match='detail should has 100 rows!'):
        XGBoostResultBatch(result=np.zeros(100), prob=np.zeros(100), detail=np.zeros([50, 3]))

    ret_batch = XGBoostResultBatch(result=np.random.rand(100))
    assert ret_batch.num_rows == 100
    leaf_indices = np.random.randint(0, 10000000, (100, 10))
    ret_batch.set_leaf(leaf_indices)
    append_info = np.random.randint(-10000, -100, (100, 3))
    ret_batch.set_append_info(append_info.tolist())
    i = 0
    for ret in ret_batch.to_iterator():
        assert ret.append_info == append_info[i, :].tolist()
        assert ret.leaf_indices == leaf_indices[i, :].tolist()
        i += 1


# TODO
def test_predictor():
    pass
