from flaml.model import SKLearnEstimator
from flaml import tune
from sklearn.neural_network import MLPClassifier
import logging
from flaml.data import (
    group_counts,
)
import time
logger = logging.getLogger("flaml.automl")

class MLP(SKLearnEstimator):
    def __init__(self,task='classification',**config):
        super().__init__(task,**config)
        self.estimator_class=MLPClassifier

    @classmethod
    def search_space(cls,data_size,task):
        space={
            'hidden_layer_sizes':{'domain':tune.randint(10,200),'init_value':100},
            'activation':{'domain':tune.choice(['identity', 'logistic', 'tanh', 'relu']),'init_value':'relu'},
            'alpha':{'domain':tune.uniform(0,1),'init_value':0.0001},
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
            params=self.params
            del params['n_jobs']
            model = self.estimator_class(**params)
            if logger.level == logging.DEBUG:
                # xgboost 1.6 doesn't display all the params in the model str
                logger.debug(f"flaml.model - {model} fit started with params {self.params}")
            model.fit(X_train, y_train, **kwargs)
            if logger.level == logging.DEBUG:
                logger.debug(f"flaml.model - {model} fit finished")
            train_time = time.time() - current_time
            self._model = model
            return train_time

