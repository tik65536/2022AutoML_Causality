import logging
import time

from flaml import tune
from flaml.data import (CLASSIFICATION, NLG_TASKS, SEQCLASSIFICATION,
                        SEQREGRESSION, SUMMARIZATION, TOKENCLASSIFICATION,
                        TS_FORECASTREGRESSION, TS_TIMESTAMP_COL, TS_VALUE_COL,
                        add_time_idx_col, group_counts)
from flaml.model import SKLearnEstimator

# Logistic Regression (aka logit, MaxEnt) classifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("flaml.automl")


class LogisticRegressionL1Estimator(SKLearnEstimator):
    def __init__(self, task='classification', **config):
        super().__init__(task, **config)
        self.estimator_class = LogisticRegression

    @classmethod
    def search_space(cls, data_size, task):
        space = {
            'C': {
                "domain": tune.loguniform(lower=0.03125, upper=32768.0), 
                'init_value': 1.0},
            'solver': {
                'domain': tune.choice(["liblinear", "saga"]), 
                'init_value': "saga"},
            'random_state': {
                'domain': tune.choice([0, 10, 20, 30]), 
                'init_value': 0},
        }
        return space
    
    def config2params(self, config: dict) -> dict:
        params = super().config2params(config)
        params["tol"] = params.get("tol", 0.0001)
        params["penalty"] = params.get("penalty", "l1")
        params["multi_class"] = params.get("multi_class", "ovr")
        return params

    def _fit(self, X_train, y_train, **kwargs):
        current_time = time.time()
        if "groups" in kwargs:
            kwargs = kwargs.copy()
            groups = kwargs.pop("groups")
            if self._task == "rank":
                kwargs["group"] = group_counts(groups)
                # groups_val = kwargs.get('groups_val')
                # if groups_val is not None:
                #     kwargs['eval_group'] = [group_counts(groups_val)]
                #     kwargs['eval_set'] = [
                #         (kwargs['X_val'], kwargs['y_val'])]
                #     kwargs['verbose'] = False
                #     del kwargs['groups_val'], kwargs['X_val'], kwargs['y_val']
        X_train = self._preprocess(X_train)
        params = self.params
        del params['n_jobs']
        model = self.estimator_class(**params)
        if logger.level == logging.DEBUG:
            # xgboost 1.6 doesn't display all the params in the model str
            logger.debug(
                f"flaml.model - {model} fit started with params {self.params}")
        model.fit(X_train, y_train, **kwargs)
        if logger.level == logging.DEBUG:
            logger.debug(f"flaml.model - {model} fit finished")
        train_time = time.time() - current_time
        self._model = model
        return train_time


class LogisticRegressionL2Estimator(SKLearnEstimator):
    def __init__(self, task='classification', **config):
        super().__init__(task, **config)
        self.estimator_class = LogisticRegression

    @classmethod
    def search_space(cls, data_size, task):
        space = {
            'C': {
                "domain": tune.loguniform(lower=0.03125, upper=32768.0), 
                'init_value': 1.0},
            'solver': {
                'domain': tune.choice(["newton-cg", "lbfgs", "liblinear", "sag", "saga"]), 
                'init_value': "lbfgs"},
            'random_state': {
                'domain': tune.choice([0, 10, 20, 30]), 
                'init_value': 0},
        }
        return space
    
    def config2params(self, config: dict) -> dict:
        params = super().config2params(config)
        params["tol"] = params.get("tol", 0.0001)
        params["penalty"] = params.get("penalty", "l2")
        params["multi_class"] = params.get("multi_class", "ovr")
        return params

    def _fit(self, X_train, y_train, **kwargs):
        current_time = time.time()
        if "groups" in kwargs:
            kwargs = kwargs.copy()
            groups = kwargs.pop("groups")
            if self._task == "rank":
                kwargs["group"] = group_counts(groups)
                # groups_val = kwargs.get('groups_val')
                # if groups_val is not None:
                #     kwargs['eval_group'] = [group_counts(groups_val)]
                #     kwargs['eval_set'] = [
                #         (kwargs['X_val'], kwargs['y_val'])]
                #     kwargs['verbose'] = False
                #     del kwargs['groups_val'], kwargs['X_val'], kwargs['y_val']
        X_train = self._preprocess(X_train)
        params = self.params
        del params['n_jobs']
        model = self.estimator_class(**params)
        if logger.level == logging.DEBUG:
            # xgboost 1.6 doesn't display all the params in the model str
            logger.debug(
                f"flaml.model - {model} fit started with params {self.params}")
        model.fit(X_train, y_train, **kwargs)
        if logger.level == logging.DEBUG:
            logger.debug(f"flaml.model - {model} fit finished")
        train_time = time.time() - current_time
        self._model = model
        return train_time
    
    
    
class LogisticRegressionElasticNetEstimator(SKLearnEstimator):
    def __init__(self, task='classification', **config):
        super().__init__(task, **config)
        self.estimator_class = LogisticRegression

    @classmethod
    def search_space(cls, data_size, task):
        space = {
            'C': {
                "domain": tune.loguniform(lower=0.03125, upper=32768.0), 
                'init_value': 1.0},
            'l1_ratio': {
                "domain": tune.loguniform(lower=0.1, upper=1.0), 
                'init_value': 0.5},
            'random_state': {
                'domain': tune.choice([0, 10, 20, 30]), 
                'init_value': 0},
        }
        return space
    
    def config2params(self, config: dict) -> dict:
        params = super().config2params(config)
        params["tol"] = params.get("tol", 0.0001)
        params["penalty"] = params.get("penalty", "elasticnet")
        params["solver"] = params.get("solver", "saga")
        params["multi_class"] = params.get("multi_class", "ovr")
        return params

    def _fit(self, X_train, y_train, **kwargs):
        current_time = time.time()
        if "groups" in kwargs:
            kwargs = kwargs.copy()
            groups = kwargs.pop("groups")
            if self._task == "rank":
                kwargs["group"] = group_counts(groups)
                # groups_val = kwargs.get('groups_val')
                # if groups_val is not None:
                #     kwargs['eval_group'] = [group_counts(groups_val)]
                #     kwargs['eval_set'] = [
                #         (kwargs['X_val'], kwargs['y_val'])]
                #     kwargs['verbose'] = False
                #     del kwargs['groups_val'], kwargs['X_val'], kwargs['y_val']
        X_train = self._preprocess(X_train)
        params = self.params
        del params['n_jobs']
        model = self.estimator_class(**params)
        if logger.level == logging.DEBUG:
            # xgboost 1.6 doesn't display all the params in the model str
            logger.debug(
                f"flaml.model - {model} fit started with params {self.params}")
        model.fit(X_train, y_train, **kwargs)
        if logger.level == logging.DEBUG:
            logger.debug(f"flaml.model - {model} fit finished")
        train_time = time.time() - current_time
        self._model = model
        return train_time
