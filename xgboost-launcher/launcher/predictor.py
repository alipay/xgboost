from typing import Iterator, List, Any

import numpy as np

from . import config_fields
from . import model_helper
from .data_builder import XGBoostData
from .env_parser import extract_dist_env
from .rabit_context import RabitContext
from .data_units import XGBoostResult


class XGBoostPredictor:
    def __init__(self, conf: config_fields.PredictFields):
        col_conf = conf.data_conf.column_format
        self._ret_cols = col_conf.result_columns
        assert self._ret_cols, 'info of ResultColumns is not set!'
        assert self._ret_cols.result_column, 'name of result_column should not be None!'
        assert self._ret_cols.result_column not in col_conf.append_columns, \
            'Name of result_column is duplicated with one of the append columns!'

        self._predict_leaf = self._ret_cols.leaf_column is not None

        model = model_helper.load_launcher_model(conf.model_conf)
        self._booster = model.booster

        if model.meta.params.objective == 'binary:logistic':
            self._converter = binary_logistic_converter
        elif model.meta.params.objective == 'multi:softprob':
            self._converter = multi_softprob_converter
        else:
            self._converter = common_converter

        self._rabit_ctx = RabitContext(extract_dist_env())

    def __enter__(self):
        self._rabit_ctx.__enter__()
        self._rabit_ctx.client_init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._rabit_ctx.__exit__(exc_type, exc_val, exc_tb)

    def predict(self, data: XGBoostData) -> Iterator[XGBoostResult]:
        batch_ret = self._booster.predict(data.d_matrix)
        batch_ret = self._converter(batch_ret)
        if data.append_info:
            batch_ret.set_append_info(data.append_info)
        if self._predict_leaf:
            batch_ret.set_leaf(self._booster.predict(data.d_matrix, pred_leaf=True))
        return batch_ret.to_iterator()


class XGBoostResultBatch:
    def __init__(self, result=None, prob=None, detail=None):
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 1, 'result should be 1-d array!'
        self.num_rows = result.shape[0]
        self.result = result
        if prob is not None:
            assert isinstance(prob, np.ndarray)
            assert len(prob.shape) == 1, 'prob should be 1-d array!'
            assert prob.shape[0] == self.num_rows, 'prob should has %d rows!' % self.num_rows
        self.prob = prob
        if detail is not None:
            assert isinstance(detail, np.ndarray)
            assert len(detail.shape) == 2, 'detail should be 2-d array'
            assert detail.shape[0] == self.num_rows, 'detail should has %d rows!' % self.num_rows
        self.detail = detail
        self.leaf = None
        self.append_info = None

    def set_leaf(self, leaf_result: np.ndarray):
        assert isinstance(leaf_result, np.ndarray)
        assert len(leaf_result.shape) == 2
        assert leaf_result.shape[0] == self.num_rows
        self.leaf = leaf_result

    def set_append_info(self, append_info: List[List[Any]]):
        assert isinstance(append_info[0], list)
        assert len(append_info) == self.num_rows
        self.append_info = append_info

    def to_iterator(self) -> Iterator[XGBoostResult]:
        for i in range(self.num_rows):
            yield XGBoostResult(
                result=self.result[i],
                classification_prob=self.prob[i] if self.prob is not None else None,
                classification_detail=self.detail[i].tolist() if self.detail is not None else None,
                leaf_indices=self.leaf[i].tolist() if self.leaf is not None else None,
                append_info=self.append_info[i] if self.append_info is not None else None)


def common_converter(raw_result: np.ndarray):
    return XGBoostResultBatch(raw_result, None, None)


def binary_logistic_converter(raw_result: np.ndarray):
    detail = np.column_stack([1.0 - raw_result, raw_result])
    result = detail.argmax(axis=1)
    prob = detail.max(axis=1)
    return XGBoostResultBatch(result, prob, detail)


def multi_softprob_converter(raw_result: np.ndarray):
    detail = raw_result
    result = detail.argmax(axis=1)
    prob = raw_result.max(axis=1)
    return XGBoostResultBatch(result, prob, detail)
