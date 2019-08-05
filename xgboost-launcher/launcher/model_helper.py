import json

import xgboost as xgb

from . import config_helper
from . import config_fields
from .env_parser import extract_dist_env
from .model_source import create_model_source


class LauncherModel:
    def __init__(self, bst: xgb.Booster, meta: config_fields.LearningFields = None):
        self.booster = bst
        if meta is not None:
            self.meta = meta
            self.booster.set_attr(**{
                'kind': 'xgboost_launcher_model',
                'config': json.dumps(config_helper.dump_config(meta))})
        else:
            attrs = self.booster.attributes()
            assert attrs.get('kind', '') == 'xgboost_launcher_model', \
                'Failed to find attr[kind]="xgboost_launcher_model" in booster, not a LauncherModel!'
            meta = json.loads(attrs.get('config'))
            self.meta = config_helper.load_config(config_fields.LearningFields, **meta)


def save_launcher_model(bst: xgb.Booster, conf: config_fields.TrainFields):
    dist_env = extract_dist_env()
    # do model saving in master node
    if dist_env.rank > 0:
        return
    model = LauncherModel(bst, conf.xgboost_conf)
    model_source = create_model_source(conf.model_conf.model_source)
    model_source.save_booster(model.booster, conf.model_conf.model_path)
    model_source.dump_booster_info(model.booster, conf.model_conf.dump_conf)


def load_launcher_model(conf: config_fields.ModelFields,
                        booster_conf: config_fields.BoosterFields = None):
    model_source = create_model_source(conf.model_source)
    if booster_conf is not None:
        booster_conf = booster_conf._asdict()
    bst = model_source.load_booster(conf.model_path, booster_params=booster_conf)
    return LauncherModel(bst)
