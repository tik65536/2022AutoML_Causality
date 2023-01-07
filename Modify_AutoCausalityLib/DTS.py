import logging
import time

from flaml import tune
from flaml.data import (CLASSIFICATION, NLG_TASKS, SEQCLASSIFICATION,
                        SEQREGRESSION, SUMMARIZATION, TOKENCLASSIFICATION,
                        TS_FORECASTREGRESSION, TS_TIMESTAMP_COL, TS_VALUE_COL,
                        add_time_idx_col, group_counts)
from flaml.model import SKLearnEstimator

# decision tree classifier
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger("flaml.automl")


class DTS(SKLearnEstimator):
    def __init__(self, task='classification', **config):
        super().__init__(task, **config)
        self.estimator_class = DecisionTreeClassifier

    @classmethod
    def search_space(cls, data_size, task):
        space = {
            'criterion': {'domain': tune.choice(['gini', 'entropy', 'log_loss']), 'init_value': 'entropy'},
            'splitter': {'domain': tune.choice(['best', 'random']), 'init_value': 'best'},
            'max_depth': {'domain': tune.choice([3, 5, 10, 15, 20]), 'init_value': 3},
            'random_state': {'domain': tune.choice([0, 10, 20, 30, 40]), 'init_value': 0},
        }
        return space

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
