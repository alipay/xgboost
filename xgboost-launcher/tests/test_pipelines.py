import json
import os
from typing import Iterator, NamedTuple

from launcher import train, predict, config_fields as cf
from launcher import DataSource, XGBoostResult, XGBoostRecord, register_data_source

file_path = os.path.dirname(os.path.abspath(__file__))


class TestDSConfig(NamedTuple):
    is_train: bool


class TestDataSource(DataSource):
    def __init__(self, rank: int, num_worker: int,
                 column_conf: cf.ColumnFields,
                 source_conf):
        super().__init__(rank, num_worker, column_conf, source_conf)
        assert isinstance(source_conf, TestDSConfig)
        self._is_train = source_conf.is_train

    def read(self) -> Iterator[XGBoostRecord]:
        suffix = 'train' if self._is_train else 'test'
        path = os.path.join(file_path, '../../demo/data/agaricus.txt.%s' % suffix)
        with open(path, 'r') as f:
            for line in f.readlines():
                label, *features = line.split(' ')
                label = float(label)
                indices = []
                values = []
                for pair in features:
                    sp = pair.index(':')
                    indices.append(int(pair[:sp]))
                    values.append(float(pair[sp + 1:]))
                if self._is_train:
                    yield XGBoostRecord(indices, values, label)
                else:
                    yield XGBoostRecord(indices, values, append_info=[label])

    def write(self, result_iter: Iterator[XGBoostResult]):
        # check results here
        for ret in result_iter:
            label = ret.append_info[0]
            assert ret.result == label
            assert ret.classification_prob > 0.9


register_data_source('test', TestDSConfig, TestDataSource)


def test_train_and_test_pipeline():
    ds_fields = cf.DataSourceFields('test', {'is_train': True})
    # ColumnFields is unnecessary for TestDataSource, so we just fill it with blank.
    data_fields = cf.DataFields(
        data_source=ds_fields,
        column_format=cf.ColumnFields(cf.FeatureFields([])))
    learning_fields = cf.LearningFields(
        num_boost_round=30,
        params=cf.BoosterFields(verbosity=3, eval_metric='auc'))
    model_path = os.path.join(file_path, 'TestModel')
    model_fields = cf.ModelFields(
        model_path=model_path,
        dump_conf=cf.DumpInfoFields(
            path=os.path.join(file_path, 'TestModelInfo'),
            with_stats=True,
            is_dump_fscore=True))
    train_fields = cf.TrainFields(learning_fields, data_fields, model_fields)
    train(train_fields)

    assert os.path.exists(os.path.join(file_path, 'TestModelInfo.json'))
    f_score = open(os.path.join(file_path, 'TestModelInfo_fscore.json')).read()
    assert isinstance(json.loads(f_score), dict)

    data_fields = cf.DataFields(
        data_source=cf.DataSourceFields('test', {'is_train': False}),
        column_format=cf.ColumnFields(
            features=cf.FeatureFields([]),
            append_columns=['label'],
            result_columns=cf.ResultColumns()))
    pred_fields = cf.PredictFields(data_fields, model_fields)
    predict(pred_fields)

    os.remove(model_path)
    os.remove(os.path.join(file_path, 'TestModelInfo.json'))
    os.remove(os.path.join(file_path, 'TestModelInfo_fscore.json'))
