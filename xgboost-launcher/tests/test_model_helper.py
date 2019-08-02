import os
from tempfile import mktemp

from launcher import config_fields
from launcher.config_fields import ModelFields, XGBoostTrainFields
from launcher.model_helper import load_launcher_model, save_launcher_model
from launcher.model_source import register_model_source, LocalModelSource

register_model_source('local', None, LocalModelSource)


def test_launcher_model():
    file_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(file_path, 'test_resources/test_booster')
    model = load_launcher_model(ModelFields(model_path=path))
    assert isinstance(model.meta, XGBoostTrainFields)
    params = model.meta.params
    assert params.objective == 'multi:softprob'
    assert params.num_class == 3
    assert params.max_depth == 5
    assert params.tree_method == 'hist'
    assert params.nthread == 2
    assert not model.meta.auto_train
    assert model.meta.num_boost_round == 100

    meta_cpy = model.meta._asdict()
    tmp_path = os.path.join(file_path, 'test_resources/tmp/')
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    tmp_path = mktemp(prefix=tmp_path)[5:]
    save_launcher_model(
        model.booster,
        config_fields.TrainFields(model.meta, None, ModelFields(model_path=tmp_path)))
    model = load_launcher_model(ModelFields(model_path=tmp_path))
    assert model.meta._asdict() == meta_cpy
