# 2022AutoML_Causality

This github repo is for 2022 AutoML project on the topic of "Out-of-sample scoring and automatic selection of causal estimators"

Here is the original paper for reference : https://arxiv.org/pdf/2212.10076.pdf

# Repo Structure:

1. DataSet : It contains 5 synthetic data set generated from auto_causality lib for experiment
2. HPC_Script : It contains slurm script for running the experiment on HPC
3. Modify_AutoCausalityLib : It contains custom Model(inherit from FLAML) for additional propensity model and necessary change on auto_causality lib
4. NewData_plot : It contains plots for the experiments
5. main_py : It contains all experiment's main python file
6. reult : It contains those experiment's reuslt dataset export from running those python in main_py.
7. AnalysisReport.ipynb : The analysis notebook for the Project


## Modify_AutoCausalityLib
1. Code Change in auto_causality lib :
   a. optimiser.py
      - The change is mainly in the init_propensity function to include those custom Model :
      
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
            
   b. scoring.py
      - The Change here is main on capturing more result data for analysis 
      
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
    
## Result DataSet 
As the result dataset is large in size (~410MB) , so it is put on google drive for access, it can be download from the link :

ResultData: https://drive.google.com/file/d/1BlBfDPVjoxWoiaWy0ajFu2u19aerikLi/view?usp=sharing

After download, decompress it into result folder and modify the variable out_dir in the notebook.

## Initial setup

1. Execute `pip install --user git+https://github.com/transferwise/auto-causality.git`
2. go to `~/.local/lib/python3.8/site-packages/auto_causality/`
3. `mkdir thridparty/causalml`
4. copy `metrics.py` from [this](https://github.com/transferwise/auto-causality/tree/main/auto_causality/thirdparty/causalml) github repos
5. `module load python/3.8.6`
6. Replace the `scoring.py` from the attachment to `~/.local/lib/python3.8/site-packages/auto_causality/`
