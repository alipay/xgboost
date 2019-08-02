import json
import logging
from abc import abstractmethod
from typing import Iterator

from . import config_helper
from . import config_fields
from .data_units import XGBoostRecord, XGBoostResult

logger = logging.getLogger(__name__)


class DataSource:
    def __init__(self,
                 rank: int,
                 num_worker: int,
                 column_conf: config_fields.ColumnFields,
                 source_conf):
        self.rank = rank
        self.num_worker = num_worker
        assert config_helper.field_keys_equal(column_conf, config_fields.ColumnFields)
        self.col_conf = column_conf
        self.source_conf = source_conf
        if column_conf.features.is_sparse:
            # if num_features <= 0, compute num_col internally.
            self.num_features = column_conf.features.feature_num
        else:
            self.num_features = len(column_conf.features.columns)

    @abstractmethod
    def read(self) -> Iterator[XGBoostRecord]:
        pass

    @abstractmethod
    def write(self, result_iter: Iterator[XGBoostResult]):
        pass


class DataSourceCollection:
    def __init__(self):
        self._sources = {}
        self._source_impls = {}

    def register(self, name, config_def, source_impl):
        meta_dict = source_impl.__dict__
        assert 'read' in meta_dict and 'write' in meta_dict
        assert hasattr(config_def, '_field_types')
        if name in self._sources:
            raise KeyError("There exists data source named %s!" % name)
        self._sources[name] = config_def
        self._source_impls[name] = source_impl
        logger.info("Data source %s registered successfully!" % name)

    def create_init_fn(self, name, config):
        assert name in self._sources, "Unregistered data source %s!" % name
        assert isinstance(config, dict)
        # convert Dict to target data source config (NamedTuple)
        data_source_conf = self._sources[name](**config)
        return lambda rank, world_size, col_format: \
            self._source_impls[name](rank, world_size, col_format, data_source_conf)


__DataSources = DataSourceCollection()


def register_data_source(name, config_def, source_impl):
    """
    register a new data source into local system
    :param name: unique id of data source
    :param config_def: config definition of data source, which is subclass of NamedTuple.
    :param source_impl: meta class of data source, which is subclass of xgboost_launcher.DataSource.
    :return:
    """
    if name not in __DataSources._sources:
        __DataSources.register(name, config_def, source_impl)


def create_data_source_init_fn(conf: config_fields.DataSourceFields):
    """
    build partial function for creating a specific data source
    :param conf: config_fields.DataSourceFields
    :return: the partial function
    """

    return __DataSources.create_init_fn(conf.name, conf.config)


class WriterUtils:
    @staticmethod
    def empty_transformer(record, item):
        return record

    @staticmethod
    def identity_transformer(record, item):
        return record.append(item)

    @staticmethod
    def leaf_transformer(record, x):
        assert isinstance(x, list)
        return record.append(','.join(map(str, x)))

    @staticmethod
    def detail_transformer(record, x):
        assert isinstance(x, list)
        x = json.dumps({str(i): e for i, e in enumerate(x)})
        return record.append(x)

    @staticmethod
    def batch_transformer(record, items):
        return record.extend(items)

    @staticmethod
    class Counter:
        """
        This Counter is typically used for counting records when writing to ds.
        Since (upstreaming) data feed lazily, we need a global counter.
        """

        def __init__(self, init_value=0):
            self._value = init_value

        def inc(self):
            self._value += 1

        def reset(self):
            self._value = 0

        @property
        def count(self):
            return self._value
