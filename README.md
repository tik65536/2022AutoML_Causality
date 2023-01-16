# 2022AutoML_Causality

This github repo is for 2022 AutoML project on the topic of `"Out-of-sample scoring and automatic selection of causal estimators"`

Here is the original paper for reference : [Out-of-sample scoring and automatic selection of causal estimators](https://arxiv.org/pdf/2212.10076.pdf)

## Repo Structure

1. `DataSet` : It contains 5 synthetic data set generated from `auto_causality` lib for experiment
2. `HPC_Script` : It contains slurm script for running the experiment on HPC
3. `Modify_AutoCausalityLib` : It contains custom Model(inherit from FLAML) for additional propensity model and necessary change on auto_causality lib
4. `NewData_plot` : It contains plots for the experiments
5. `main_py` : It contains all experiment's main python file
6. `reult` : It contains those experiment's reuslt dataset export from running those python in main_py.
7. `AnalysisReport.ipynb` : The analysis notebook for the Project

## Modify_AutoCausalityLib

1. Code Change in `auto_causality` lib :
   a. `optimiser.py`
      - The change is mainly in the `init_propensity` function to include those custom Model :

        ex :

            elif propensity_model == "super":
            self.propensity_model = AutoML(
                **{**self._settings["component_models"], "task": "classification","estimator_list":['Gaussian','SVM','MLP','DTC','ExTC','LR']}
            )
            self.propensity_model.add_learner(learner_name='Gaussian', learner_class=Gaussian)
            self.propensity_model.add_learner(learner_name='SVM', learner_class=SVM)
            self.propensity_model.add_learner(learner_name='MLP', learner_class=MLP)
            self.propensity_model.add_learner(learner_name='DTC', learner_class=DTC)
            self.propensity_model.add_learner(learner_name='ExTC', learner_class=ExTC)
            self.propensity_model.add_learner(learner_name='LR', learner_class=LR)

   b. `scoring.py`
      - The Change here is main on capturing more result data for analysis.

        ex :

        To capture the propensity model's HPO  

            out['#_Propensity_model']=type(_model._trained_estimator).__name__
            out['#_Propensity_model_param']=_propensity_param

2. Custom Model Include :
    - Bayesian.py
    - SVM.py
    - MLP.py
    - DtsEstimator.py
    - LogisticRegressionEstimator.py

2.1 Research Space
Here is the list of search space for each custom model in FLAML.

- **Bayesian**

    ```python
    space = {
            'var_smoothing':{
                'domain':tune.uniform(lower=0,upper=0.5),
                'init_value':1e-9, }
            }
    ```

- **SVM**

    ```python
    space = {
            'nu':{
                'domain':tune.uniform(lower=0,upper=0.9),
                'init_value': 0.5 },
            'kernel':{
                'domain':tune.choice(['linear','poly','rbf']),
                'init_value':'rbf'},
            'degree':{
                'domain':tune.choice([1,2,3,4,5]),
                'init_value':3},
            'gamma':{
                'domain':tune.choice(['auto','scale']),
                'init_value':'scale'},
            }
    ```

- **MLP**

    ```python
    space = {
            'hidden_layer_sizes':{
                'domain':tune.randint(10,200),
                'init_value':100},
            'activation':{
                'domain':tune.choice(['identity', 'logistic', 'tanh', 'relu']),
                'init_value':'relu'},
            'alpha':{
                'domain':tune.uniform(0,1),
                'init_value':0.0001},
            }
    ```

- **DTs**

    ```python
    space = {
            # The log_loss option for the parameter criterion was added only in the latest scikit-learn version 1.1.2
            # 'criterion': {'domain': tune.choice(['gini', 'entropy', 'log_loss']), 'init_value': 'entropy'},
            'criterion': {
                'domain': tune.choice(['gini', 'entropy']), 
                'init_value': 'gini'},
            'splitter': {
                'domain': tune.choice(['best', 'random']), 
                'init_value': 'best'},
            'max_depth': {
                'domain': tune.choice([3, 5, 7, 10]), 
                'init_value': 3},
            'random_state': {
                'domain': tune.choice([0, 10, 20, 30]), 
                'init_value': 0},
        }
    ```

- **Logistic Regression**

    ```python
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
    
    # 'penalty' elasticnet incoulds both l1 and l2 depens on 'l1_ratio'
    params["penalty"] = params.get("penalty", "elasticnet")
    params["solver"] = params.get("solver", "saga")
    params["multi_class"] = params.get("multi_class", "ovr")
    ```

## Result DataSet

As the result dataset is large in size (~410MB) , so it is put on google drive for access, it can be download from the link :

[1] [ResultData](https://drive.google.com/file/d/1BlBfDPVjoxWoiaWy0ajFu2u19aerikLi/view?usp=sharing)

After download, decompress it into result folder and modify the variable out_dir in the notebook.

## Initial setup

1. Execute `pip install --user git+https://github.com/transferwise/auto-causality.git`
2. go to `~/.local/lib/python3.8/site-packages/auto_causality/`
3. `mkdir thridparty/causalml`
4. copy `metrics.py` from [this](https://github.com/transferwise/auto-causality/tree/main/auto_causality/thirdparty/causalml) github repos
5. `module load python/3.8.6`
6. Replace the `scoring.py` from the attachment to `~/.local/lib/python3.8/site-packages/auto_causality/`
