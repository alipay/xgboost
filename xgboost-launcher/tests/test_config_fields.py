import yaml
from yaml import CLoader as Loader

from launcher.config_helper import load_config, dump_config
from launcher import config_fields

__data_fields_demo = """
data_source:
    name: hive
    config:
        table_name: table_xxxxx
        cluster_ip: xxxxxx
column_format:
    features:
        columns: 'col_a,col_b,col_c'
        feature_num: 3
    label: label_Y
    group: group_G
    append_columns: 'app_a,app_b,app_c'
    result_columns:
        detail_column: 'detail'
        leaf_column: 'leafs'
builder:
    batch_size: 1024
"""

__model_fields_demo = """
model_source:
    name: hdfs
    config:
        cluster_ip: xxxxxx
model_path: 'hdfs://xxxxxxxxxx'
dump_conf:
    path: 'hdfs://yyyyyyyyyy'
    with_stats: true
    dump_format: json
    is_dump_fscore: true
"""

__xgb_train_fields_demo = """
auto_train: true
num_boost_round: 500
params:
    objective: "multi:softprob"
    num_class: 3
    max_depth: 8
    tree_method: hist
    eval_metric: merror
    subsample: 0.8
    convergence_criteria: true
    colsample_bytree: 0.5
    colsample_bylevel: 0.5
"""


def test_data_fields():
    conf = yaml.load(__data_fields_demo, Loader)
    fields = load_config(config_fields.DataFields, **conf)
    assert isinstance(fields, config_fields.DataFields)
    assert fields.data_source.name == 'hive'
    assert fields.data_source.config == {
        'table_name': 'table_xxxxx',
        'cluster_ip': 'xxxxxx'
    }
    assert fields.column_format.features.columns == ['col_a', 'col_b', 'col_c']
    assert fields.column_format.features.feature_num == 3
    assert fields.column_format.features.is_sparse == False
    assert fields.column_format.features.kv_delimiter == ':'
    assert fields.column_format.features.item_delimiter == ','
    assert fields.column_format.label == 'label_Y'
    assert fields.column_format.group == 'group_G'
    assert fields.column_format.weight is None
    assert fields.column_format.append_columns == ['app_a', 'app_b', 'app_c']
    ret_cols = fields.column_format.result_columns
    assert ret_cols.result_column == 'prediction_result'
    assert ret_cols.probability_column == 'prediction_probability'
    assert ret_cols.detail_column == 'detail'
    assert ret_cols.leaf_column == 'leafs'
    assert fields.builder.name == 'CSRMatrixBuilder'
    assert fields.builder.batch_size == 1024
    assert fields.valid_data_source is None

    valid_data_source = {'name': 'A', 'config': {'x': 1, 'y': 2}}
    conf.update(**{'valid_data_source': valid_data_source})
    fields = load_config(config_fields.DataFields, **conf)
    assert fields.valid_data_source.name == 'A'
    assert fields.valid_data_source.config == {'x': 1, 'y': 2}


def test_model_fields():
    conf = yaml.load(__model_fields_demo, Loader)
    fields = load_config(config_fields.ModelFields, **conf)
    assert isinstance(fields, config_fields.ModelFields)
    assert fields.model_source.name == 'hdfs'
    assert fields.model_source.config == {'cluster_ip': 'xxxxxx'}
    assert fields.model_path == 'hdfs://xxxxxxxxxx'
    assert fields.dump_conf.path == 'hdfs://yyyyyyyyyy'
    assert fields.dump_conf.fmap == ''
    assert fields.dump_conf.with_stats
    assert fields.dump_conf.is_dump_fscore
    assert fields.dump_conf.dump_format == 'json'
    assert fields.dump_conf.importance_type == 'weight'


def test_xgb_train_fields():
    conf = yaml.load(__xgb_train_fields_demo, Loader)
    fields = load_config(config_fields.LearningFields, **conf)
    assert isinstance(fields, config_fields.LearningFields)
    assert fields.auto_train
    assert fields.num_boost_round == 500
    expected = {k: v for k, v in config_fields.BoosterFields()._asdict().items() if v is not None}
    expected.update({
        'objective': "multi:softprob",
        'num_class': 3,
        'max_depth': 8,
        'tree_method': 'hist',
        'eval_metric': 'merror',
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'colsample_bylevel': 0.5,
        'convergence_criteria': True
    })
    assert dump_config(fields.params) == expected


def test_job_fields():
    data_conf = yaml.load(__data_fields_demo, Loader)
    model_conf = yaml.load(__model_fields_demo, Loader)
    xgb_conf = yaml.load(__xgb_train_fields_demo, Loader)
    fields = load_config(config_fields.JobFields, **{
        'job': 'train',
        'train_conf': {
            'xgboost_conf': xgb_conf,
            'data_conf': data_conf,
            'model_conf': model_conf
        }
    })
    assert isinstance(fields, config_fields.JobFields)
    assert fields.job == config_fields.JobType.TRAIN
    assert fields.train_conf.data_conf == load_config(config_fields.DataFields, **data_conf)
    assert fields.train_conf.model_conf == load_config(config_fields.ModelFields, **model_conf)
    assert fields.train_conf.xgboost_conf == load_config(config_fields.LearningFields, **xgb_conf)
    fields = load_config(config_fields.JobFields, **{
        'job': 'predict',
        'predict_conf': {
            'data_conf': data_conf,
            'model_conf': model_conf
        }
    })
    assert isinstance(fields, config_fields.JobFields)
    assert fields.job == config_fields.JobType.PREDICT
    assert fields.predict_conf.data_conf == load_config(config_fields.DataFields, **data_conf)
    assert fields.predict_conf.model_conf == load_config(config_fields.ModelFields, **model_conf)
