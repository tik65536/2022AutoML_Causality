from flaml.model import SKLearnEstimator
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from flaml import tune
import logging
from flaml.data import (
    group_counts,
)
import time
logger = logging.getLogger("flaml.automl")

class gpClassifier(SKLearnEstimator):
    def __init__(self,task='classification',**config):
        super().__init__(task,**config)
        self.estimator_class=GaussianProcessClassifier

    @classmethod
    def search_space(cls,data_size,task):
        space={
            'kernel':{'domain':tune.choice(['rbf','dot','matern','rq','wh']),'init_value':'rbf' },
            'length_scale':{'domain':tune.uniform(0.1,2),'init_value':1},
            'sigma_0':{'domain':tune.uniform(0.1,2),'init_value':1},
            'alpha':{'domain':tune.uniform(0.1,2),'init_value':1},
            'noise_level':{'domain':tune.uniform(0.1,2),'init_value':1}
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
            if(self.params['kernel']=='rbf'):
                params={'kernel':1*RBF(self.params['length_scale']),'n_jobs':self.params['n_jobs']}
                model = self.estimator_class(**params)
            elif(self.params['kernel']=='dot'):
                params={'kernel':1*DotProduct(self.params['sigma_0']),'n_jobs':self.params['n_jobs']}
                model = self.estimator_class(**params)
            elif(self.params['kernel']=='matern'):
                params={'kernel':1*Matern(self.params['length_scale']),'n_jobs':self.params['n_jobs']}
                model = self.estimator_class(**params)
            elif(self.params['kernel']=='wh'):
                params={'kernel':1*WhiteKernel(self.params['noise_level']),'n_jobs':self.params['n_jobs']}
                model = self.estimator_class(**params)
            elif(self.params['kernel']=='rq'):
                params={'kernel':1*RationalQuadratic(length_scale=self.params['length_scale'],alpha=self.params['alpha']),'n_jobs':self.params['n_jobs']}
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

