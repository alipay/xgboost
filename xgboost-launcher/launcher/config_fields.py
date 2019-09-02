from enum import Enum
from typing import NamedTuple, List, Dict

from .utils import max_thread_num


class BoosterFields(NamedTuple):
    objective: str = 'binary:logistic'
    eval_metric: str = None
    booster: str = 'gbtree'
    seed: int = 0
    num_class: int = None
    eta: float = None
    gamma: float = None
    max_depth: int = None
    min_child_weight: int = None
    subsample: float = None
    colsample_bytree: float = None
    colsample_bylevel: float = None
    colsample_bynode: float = None
    reg_lambda: float = None
    reg_alpha: float = None
    tree_method: str = 'auto'
    sketch_eps: float = None
    scale_pos_weight: float = None
    grow_policy: str = None
    max_leaves: int = None
    max_bin: int = None
    num_parallel_tree: int = None
    nthread: int = max_thread_num()
    gpu_id: int = None
    convergence_criteria: str = None
    verbosity: int = None


class LearningFields(NamedTuple):
    params: BoosterFields
    num_boost_round: int = 10
    # if true, using experimental feature: xgboost.auto_train
    auto_train: bool = False


class FeatureFields(NamedTuple):
    columns: List[str]
    feature_num: int = 0
    is_sparse: bool = False
    item_delimiter: str = ','
    kv_delimiter: str = ':'

    @classmethod
    def convert_columns(cls, value):
        if isinstance(value, str):
            return value.split(',')
        elif isinstance(value, List):
            return value
        else:
            raise ValueError('Invalid features: %s' % value)


class ResultColumns(NamedTuple):
    result_column: str = 'prediction_result'
    probability_column: str = 'prediction_probability'
    detail_column: str = 'prediction_detail'
    leaf_column: str = None


class ColumnFields(NamedTuple):
    features: FeatureFields
    label: str = None
    group: str = None
    weight: str = None
    append_columns: List[str] = None
    result_columns: ResultColumns = None

    @classmethod
    def convert_append_columns(cls, value):
        if isinstance(value, str):
            return value.split(',')
        elif isinstance(value, List):
            return value
        else:
            raise ValueError('Invalid features: %s' % value)


class DataSourceFields(NamedTuple):
    name: str
    config: Dict


class DataBuilderFields(NamedTuple):
    name: str = 'CSRMatrixBuilder'
    batch_size: int = -1


class DataFields(NamedTuple):
    data_source: DataSourceFields
    column_format: ColumnFields
    builder: DataBuilderFields = DataBuilderFields()
    # source of validation data in train mode, if None, skip validation
    valid_data_source: DataSourceFields = None


class ModelSourceFields(NamedTuple):
    name: str
    config: Dict = {}


LOCAL_MODEL_SOURCE = ModelSourceFields(name='local')


class DumpInfoFields(NamedTuple):
    """
    config for dumping info of booster
    """
    path: str = None
    fmap: str = ''
    with_stats: bool = False
    dump_format: str = 'text'
    is_dump_fscore: bool = False
    importance_type: str = 'weight'


class ModelFields(NamedTuple):
    model_path: str
    model_source: ModelSourceFields = LOCAL_MODEL_SOURCE
    dump_conf: DumpInfoFields = DumpInfoFields()


class TrainFields(NamedTuple):
    xgboost_conf: LearningFields
    data_conf: DataFields
    model_conf: ModelFields


class PredictFields(NamedTuple):
    data_conf: DataFields
    model_conf: ModelFields


class JobType(Enum):
    TRAIN = 1
    PREDICT = 2


class JobFields(NamedTuple):
    job: JobType
    train_conf: TrainFields = None
    predict_conf: PredictFields = None

    @classmethod
    def convert_job(cls, value):
        if value == 1 or value == 'train':
            return JobType.TRAIN
        elif value == 2 or value == 'predict':
            return JobType.PREDICT
        else:
            raise ValueError("Illegal JobType: %s" % value)
