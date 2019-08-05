import os

from .main import launch, run as launch_c, train_pipeline as train, predict_pipeline as predict
from .data_units import XGBoostResult, XGBoostRecord
from .data_source import DataSource, register_data_source
from .model_source import ModelSource, register_model_source
from .data_sources.odps.odps_data_source import ODPSDataSource, ODPSFields
from .model_sources.local_model_source import LocalModelSource
from .config_fields import LearningFields, DataFields, ModelFields, JobFields, TrainFields, PredictFields
from .rabit_context import RabitContext

VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
with open(VERSION_FILE) as f:
    __version__ = f.read().strip()

__all__ = ['launch', 'launch_c', 'train', 'predict',
           'XGBoostResult', 'XGBoostRecord',
           'DataSource', 'register_data_source',
           'ModelSource', 'register_model_source',
           'ODPSDataSource', 'ODPSFields',
           'LocalModelSource',
           'LearningFields', 'DataFields', 'ModelFields', 'JobFields', 'TrainFields', 'PredictFields',
           'RabitContext']
