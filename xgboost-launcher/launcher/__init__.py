from __future__ import absolute_import

# order of import clauses matters
from .config_fields import LearningFields, DataFields, ModelFields, JobFields, TrainFields, PredictFields
from .rabit_context import RabitContext
from .data_units import XGBoostResult, XGBoostRecord
from .data_source import DataSource, register_data_source
from .model_source import ModelSource, register_model_source
from .data_sources.odps.odps_data_source import ODPSDataSource, ODPSFields
from .model_sources.local_model_source import LocalModelSource
from .main import launch, run as launch_c, train_pipeline as train, predict_pipeline as predict
from .version import get_launcher_version

__version__ = get_launcher_version()

__all__ = ['launch', 'launch_c', 'train', 'predict',
           'XGBoostResult', 'XGBoostRecord',
           'DataSource', 'register_data_source',
           'ModelSource', 'register_model_source',
           'ODPSDataSource', 'ODPSFields',
           'LocalModelSource',
           'LearningFields', 'DataFields', 'ModelFields', 'JobFields', 'TrainFields', 'PredictFields',
           'RabitContext']
