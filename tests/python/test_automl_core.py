"""Unit tests for AutoML Library functions."""
import unittest
import os
from xgboost.automl_core import ParamError, check_xgb_parameter
import xgboost as xgb

class TestAutomlCore(unittest.TestCase):
    """
    A class providing tests for automl_core main functions
    """

    def test_default_values(self):
        """
        A test checking the setting up of default values for hyper-parameters
        """
        params = {'objective': 'binary:logistic'}
        expected_params = {'objective': 'binary:logistic', 'max_depth': 6, \
                           'eta': 0.3, 'num_round': 100, 'eval_metric': 'error', \
                           'maximize_eval_metric': 'False'}
        with self.assertWarns(Warning):
            params = check_xgb_parameter(params, 0, 2)
        self.assertEqual(params, expected_params)

    def test_objective(self):
        """
        A test for the objective
        """
        params = {}
        with self.assertRaises(ParamError):
            check_xgb_parameter(params, 100, 2)

    def test_num_class(self):
        """
        A test for number of classes
        """
        params = {'objective': 'binary:logistic'}
        with self.assertRaises(ParamError):
            check_xgb_parameter(params, 100, 4)

    def test_metric(self):
        """
        A test for evaluation metric
        """
        params = {'objective': 'binary:logistic', 'max_depth': 6, \
                  'learning_rate': 0.3, 'num_round': 100, \
                  'eval_metric': 'ndcg@ab'}
        with self.assertRaises(ParamError):
            check_xgb_parameter(params, 100, 2)

    def test_metric_optimization_direction(self):
        """
        A test for metric optimization direction
        """
        params = {'objective': 'binary:logistic', 'max_depth': 6, \
                  'learning_rate': 0.3, 'num_round': 100, \
                  'eval_metric': 'auc'}
        check_xgb_parameter(params, 100)
        maximize_eval_metric = params['maximize_eval_metric'].lower() == 'true'
        self.assertTrue(maximize_eval_metric)

    def test_xgboost_auto_train1(self):
        """
        A test for auto xgboost on depth and learning rate
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dtrain = xgb.DMatrix(dir_path + '/../../demo/data/agaricus.txt.train')
        dtest = xgb.DMatrix(dir_path + '/../../demo/data/agaricus.txt.test')
        param = {'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
        # specify validations set to watch performance
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        best_model = xgb.auto_train(param, dtrain, 10, watchlist)
        self.assertTrue(float(best_model.attr('best_score')) == 1.0)

    def test_xgboost_auto_train2(self):
        """
        A test for auto xgboost when learning rate is fixed
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dtrain = xgb.DMatrix(dir_path + '/../../demo/data/agaricus.txt.train')
        dtest = xgb.DMatrix(dir_path + '/../../demo/data/agaricus.txt.test')
        param = {'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'eta': 0.01}
        # specify validations set to watch performance
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        best_model = xgb.auto_train(param, dtrain, 10, watchlist)
        self.assertTrue(float(best_model.attr('best_score')) == 1.0)

if __name__ == '__main__':
    unittest.main()
