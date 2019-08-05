import json
import logging
from abc import abstractmethod
from typing import List

import xgboost as xgb

from . import config_fields

logger = logging.getLogger(__name__)


class ModelSource:
    """
        Base class of ModelSource, who is responsible for r/w booster and related files.
    """
    def __init__(self, source_conf):
        self.source_conf = source_conf

    @abstractmethod
    def read_buffer(self, model_path: str) -> bytes:
        pass

    @abstractmethod
    def write_buffer(self, buf: bytes, model_path: str):
        pass

    @abstractmethod
    def read_lines(self, model_path: str) -> List[str]:
        pass

    @abstractmethod
    def write_lines(self, lines: List[str], model_path: str):
        pass

    def load_booster(self, model_path: str, booster_params=None) -> xgb.Booster:
        buf = self.read_buffer(model_path)
        return xgb.Booster(model_file=bytearray(buf), params=booster_params)

    def save_booster(self, bst: xgb.Booster, model_path: str):
        buf = bytes(bst.save_raw())
        self.write_buffer(buf, model_path)
        logger.info('Booster binary has been saved at path(%s) successfully!' % model_path)

    def dump_booster_info(self, bst: xgb.Booster, conf: config_fields.DumpInfoFields):
        if conf.path is None:
            return
        # dump model info
        path = conf.path
        if conf.dump_format == 'json' and not path.endswith('json'):
            path = path + '.json'
        if conf.dump_format == 'text' and not path.endswith('txt'):
            path = path + '.txt'
        info_list = bst.get_dump(
            fmap=conf.fmap,
            with_stats=conf.with_stats,
            dump_format=conf.dump_format)
        self.write_lines(info_list, path)
        logger.info('Booster info has been saved at path(%s) successfully!' % path)
        # dump feature scores
        if not conf.is_dump_fscore:
            return
        if path.endswith('.txt'):
            path = path[:-4] + '_fscore.json'
        elif path.endswith('.json'):
            path = path[:-5] + '_fscore.json'
        f_score = json.dumps(bst.get_score(conf.fmap, conf.importance_type), indent=2)
        self.write_lines([f_score], path)
        logger.info('F_score description has been saved at path(%s) successfully!' % path)


class ModelSourceCollection:
    def __init__(self):
        self._sources = {}
        self._source_impls = {}

    def register(self, name, config_def, source_impl):
        meta_dict = source_impl.__dict__
        assert 'read_buffer' in meta_dict and \
               'write_buffer' in meta_dict and \
               'read_lines' in meta_dict and \
               'write_lines' in meta_dict
        assert config_def is None or hasattr(config_def, '_field_types')
        if name in self._sources:
            raise KeyError("There exists data source named %s!" % name)
        self._sources[name] = config_def
        self._source_impls[name] = source_impl
        logger.info("Model source %s registered successfully!" % name)

    def create(self, name, config):
        assert name in self._sources, "Unregistered model source %s!" % name
        assert isinstance(config, dict)
        # convert Dict to target model source config(NamedTuple)
        # if config contains nothing, then data_source_conf mean nothing.
        data_source_conf = self._sources[name](**config) if config else None
        return self._source_impls[name](data_source_conf)


__ModelSources = ModelSourceCollection()


def register_model_source(name, config_def, source_impl):
    """
    register a new model source into local system
    :param name:
    :param config_def:
    :param source_impl:
    :return:
    """
    if name not in __ModelSources._sources:
        __ModelSources.register(name, config_def, source_impl)


def create_model_source(conf: config_fields.ModelSourceFields) -> ModelSource:
    """
    create a specific model source
    :param conf: configs.ModelSourceFields
    :return: the partial function
    """

    return __ModelSources.create(conf.name, conf.config)
