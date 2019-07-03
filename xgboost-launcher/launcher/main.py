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

import json
import logging
import os
import sys
from typing import Iterator

import xgboost as xgb
import yaml

from . import XGBOOST_LAUNCHER_ENV_TAG
from . import config_fields
from . import config_helper
from . import model_helper
from .data_builder import XGBoostDataBuilder, XGBoostData
from .data_source import create_data_source_init_fn, register_data_source, DataSource
from .env_parser import extract_dist_env
from .model_source import register_model_source, LocalModelSource
from .predictor import XGBoostPredictor
from .rabit_context import RabitContext
from .utils import backend_logging

logger = logging.getLogger(__name__)


def create_data_source(conf: config_fields.DataFields):
    dist_env = extract_dist_env()
    # create data source from config
    ds_partial = create_data_source_init_fn(conf.data_source)
    ds = ds_partial(dist_env.rank, dist_env.world_size, conf.column_format)

    vds = None
    # create valid data source from config
    if conf.valid_data_source is not None:
        vds_partial = create_data_source_init_fn(conf.valid_data_source)
        vds = vds_partial(dist_env.rank, dist_env.world_size, conf.column_format)

    return ds, vds


def data_prepare(ds: DataSource, conf: config_fields.DataBuilderFields) -> Iterator[XGBoostData]:
    dm_builder = XGBoostDataBuilder.create(conf, ds)
    # build Iterator of DMatrix via DMatrixBuilder
    return dm_builder.build()


def train_pipeline(conf: config_fields.TrainFields):
    def _iteration_progress_callback(env):
        current = env.iteration
        logger.info('iteration: %d / %d' % (current + 1, env.end_iteration))

    def build_train_d_matrix(data_source: DataSource):
        d_matrix = None
        for xgboost_data in data_prepare(data_source, conf.data_conf.builder):
            if d_matrix is None:
                d_matrix = xgboost_data.d_matrix
                logger.info(
                    "DMatrix has been built successfully, whose shape is [%d, %d]!"
                    % (d_matrix.num_row(), d_matrix.num_col()))
            else:
                raise RuntimeError('In training pipeline, data generator should contain only one element!')
        return d_matrix

    ds, vds = create_data_source(conf.data_conf)
    with RabitContext(extract_dist_env()) as ctx:
        logger.info("start build training matrix...")
        train_mat = build_train_d_matrix(ds)
        logger.info("start build validating matrix...")
        valid_mat = build_train_d_matrix(vds) if vds else None

        ctx.client_init()

        xgboost_conf = config_helper.dump_config(conf.xgboost_conf)
        logger.info('xgboost train conf: %s' % json.dumps(xgboost_conf, indent=2))

        eval_mats = [(train_mat, 'train')]
        if valid_mat is not None:
            eval_mats.append((valid_mat, 'eval'))
        callbacks = [_iteration_progress_callback]
        reserved_args = {'dtrain': train_mat, 'evals': eval_mats, 'callbacks': callbacks}
        xgboost_conf.update(reserved_args)
        xgboost_conf['params']['verbosity'] = 3
        if xgboost_conf.pop('auto_train'):
            bst = xgb.auto_train(**xgboost_conf)
        else:
            bst = xgb.train(**xgboost_conf)

        model_helper.save_launcher_model(bst, conf)


def predict_pipeline(conf: config_fields.PredictFields):
    with XGBoostPredictor(conf) as predictor:
        ds, _ = create_data_source(conf.data_conf)

        def predict_fn():
            for xgboost_data in data_prepare(ds, conf.data_conf.builder):
                for ret in predictor.predict(xgboost_data):
                    yield ret

        ds.write(predict_fn())


def launch(conf: config_fields.JobFields):
    if conf.job == config_fields.JobType.TRAIN:
        assert conf.train_conf, "Running training job without any configuration!"
        train_pipeline(conf.train_conf)
    elif conf.job == config_fields.JobType.PREDICT:
        assert conf.predict_conf, "Running prediction job without any configuration!"
        predict_pipeline(conf.predict_conf)


# TODO: register sources dynamically
def register_sources():
    from .data_sources.odps.odps_data_source import ODPSFields, ODPSDataSource
    register_data_source('odps', ODPSFields, ODPSDataSource)
    # unimplemented
    from .data_sources.csv_data_source import CsvFields, CsvDataSource
    register_data_source('csv', CsvFields, CsvDataSource)

    from .model_sources.oss_model_source import OSSFields, OssModelSource
    register_model_source('oss', OSSFields, OssModelSource)
    register_model_source('local', None, LocalModelSource)


# TODO: add examples
def run(raw_conf: str):
    """
    Main method of XGBoostLauncher, which parses the configuration (config_fields.JobFields) from raw string,
    and then runs train/predict pipeline according to the configuration.

    The raw string can be a json-like string, or a yaml-like (multi-line) string.
    In addition to passing an argument directly, raw string can also be fetched as an env-var keyed by `xgb_launcher_env`,
    when `raw_conf` is set to `xgb_launcher_env`.

    :param raw_conf: (handler of) raw string configuration describing the XGBLauncher job
    :return:
    """

    try:
        if raw_conf.strip() == XGBOOST_LAUNCHER_ENV_TAG:
            env = os.environ
            if XGBOOST_LAUNCHER_ENV_TAG not in env:
                raise KeyError("Couldn't find XGBoost specific key(%s) in env vars!" % XGBOOST_LAUNCHER_ENV_TAG)
            raw_conf = env[XGBOOST_LAUNCHER_ENV_TAG]
        if raw_conf.lstrip().startswith('{'):
            conf = json.loads(raw_conf)
        else:
            from yaml import CLoader as Loader
            conf = yaml.load(raw_conf, Loader)
        job_conf = config_helper.load_config(config_fields.JobFields, **conf)
    except Exception as e:
        raise RuntimeError('XGBLauncher config loading failed!')
    try:
        # logging cpu & memory usage every 30 seconds
        backend_logging(30)
        launch(job_conf)
    except Exception as e:
        raise RuntimeError('XGBLauncher running failed!')


if __name__ == '__main__':
    register_sources()
    run(sys.argv[1])
