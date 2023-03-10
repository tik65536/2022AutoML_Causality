import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Union
import pymp

import warnings
warnings.filterwarnings('ignore') # suppress sklearn deprecation warnings for now..

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from auto_causality import AutoCausality
from auto_causality.data_utils import preprocess_dataset
from auto_causality.datasets import generate_synthetic_data
metrics = ["norm_erupt", "qini","energy_distance"]
n_samples = 10000
test_size = 0.33 # equal train,val,test
components_time_budget = 300
estimator_list = "all"
n_runs = 1
out_dir = "./GaussianNB/"
filename_out = "synthetic_observational_cate"

dataset = generate_synthetic_data(n_samples=n_samples, confounding=True, linear_confounder=True, noisy_outcomes=True)
#dataset.preprocess_dataset()
#features_X=dataset.effect_modifiers
#features_W=dataset.common_causes
#data_df=dataset.data
data_df, features_X, features_W = preprocess_dataset(
    dataset.data, treatment=dataset.treatment, targets=dataset.outcomes
)
# drop true effect:
features_X = [f for f in features_X if f != "true_effect"]
print(f"features_X: {features_X}")
print(f"features_W: {features_W}")

with pymp.Parallel(2) as p:
    for i_run in p.range(1,n_runs+1):

        train_df, test_df = train_test_split(data_df, test_size=test_size)
        test_df = test_df.reset_index(drop=True)
        for metric in metrics:
            clf_pf = GaussianNB()
            ac = AutoCausality(
                metric=metric,
                verbose=1,
                components_verbose=1,
                components_time_budget=components_time_budget,
                estimator_list=estimator_list,
                store_all_estimators=True,
                propensity_model=clf_pf,
            )

            ac.fit(
                train_df,
                treatment="treatment",
                outcome=["outcome"],
                common_causes=features_W,
                effect_modifiers=features_X,
            )
            # compute relevant scores (skip newdummy)
            datasets = {"train": ac.train_df, "validation": ac.test_df, "test": test_df}
            # get scores on train,val,test for each trial,
            # sort trials by validation set performance
            # assign trials to estimators
            estimator_scores = {est: [] for est in ac.scores.keys() if "NewDummy" not in est}
            for trial in ac.results.trials:
                # estimator name:
                estimator_name = trial.last_result["estimator_name"]
                if  trial.last_result["estimator"]:
                    estimator = trial.last_result["estimator"]
                    scores = {}
                    for ds_name, df in datasets.items():
                        scores[ds_name] = {}
                        # make scores
                        est_scores = ac.scorer.make_scores(
                            estimator,
                            df,
                            #problem=ac.problem,
                            metrics_to_report=ac.metrics_to_report,
                        )

                        # add cate:
                        scores[ds_name]["CATE_estimate"] = estimator.estimator.effect(df)
                        # add ground truth for convenience
                        scores[ds_name]["CATE_groundtruth"] = df["true_effect"]
                        scores[ds_name][metric] = est_scores[metric]
                    estimator_scores[estimator_name].append(scores)


            # sort trials by validation performance
            for k in estimator_scores.keys():
                estimator_scores[k] = sorted(
                    estimator_scores[k],
                    key=lambda x: x["validation"][metric],
                    reverse=False if metric == "energy_distance" else True,
                )
            results = {
                "best_estimator": ac.best_estimator,
                "best_config": ac.best_config,
                "best_score": ac.best_score,
                "optimised_metric": metric,
                "scores_per_estimator": estimator_scores,
            }


            with open(f"{out_dir}{filename_out}_{metric}_run_{i_run}.pkl", "wb") as f:
                pickle.dump(results, f)
