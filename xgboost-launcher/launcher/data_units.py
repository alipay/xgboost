from typing import NamedTuple, List, Any

import numpy as np

from . import config_fields


class XGBoostRecord(NamedTuple):
    indices: List[int]
    values: List[float]
    label: float = None
    group: int = None
    weight: float = 1.0
    base_margin: float = None
    append_info: List[Any] = None


class RecordBuilder:
    def __init__(self, config: config_fields.FeatureFields):
        self.conf = config
        # TODO: add param check
        self._feature_fn = \
            self._build_sparse_feature if self.conf.is_sparse else self._build_dense_feature

    def build(self,
              feat: List[Any],
              label: float = None,
              group: int = None,
              weight: float = 1.0,
              base_margin: float = None,
              append_info: List[Any] = None):
        indices, values = self._feature_fn(feat)
        return XGBoostRecord(indices, values, label, group, weight, base_margin, append_info)

    def _build_dense_feature(self, feat: List[Any]):
        if len(feat) != len(self.conf.columns):
            raise RuntimeError('Unexpected feature length({}), it should be {}!'
                               .format(len(feat), len(self.conf.columns)))
        indices = []
        values = []
        for i in range(len(feat)):
            val = feat[i]
            if val is not None and val != np.nan:
                try:
                    val = float(val)
                except ValueError as e:
                    raise RuntimeError('Non numerical value {}'.format(val))
            else:
                val = np.nan
            if val != np.nan:
                indices.append(i)
                values.append(val)
        return indices, values

    def _build_sparse_feature(self, feat: List[Any]):
        feat = feat[0]
        if not feat:
            return [], []
        if self.conf.kv_delimiter not in feat:
            raise RuntimeError('Found no kv_delimiter({}) in sparse feature: {}.'
                               .format(self.conf.kv_delimiter, feat))

        buffer = {}
        for kv in feat.split(self.conf.item_delimiter):
            items = kv.split(self.conf.kv_delimiter)
            if len(items) != 2:
                raise RuntimeError('Not a key-value pair: {}'.format(kv))
            else:
                key = int(items[0])
                val = float(items[1])
                if key >= self.conf.feature_num:
                    raise IndexError(
                        'Invalid sparse index %d, which should be smaller than feature_num %d!'
                        % (key, self.conf.feature_num))
                if key in buffer:
                    raise IndexError('Index sparse index %d, it occurs more than once!' % key)
                if val != np.nan:
                    buffer[key] = val

        indices = []
        values = []
        for k, v in sorted(buffer.items()):
            indices.append(k)
            values.append(v)

        return indices, values


class XGBoostResult(NamedTuple):
    # TODO: support pred_contribs
    result: float
    classification_prob: float = None
    classification_detail: List[float] = None
    leaf_indices: List[int] = None
    append_info: List[Any] = None
