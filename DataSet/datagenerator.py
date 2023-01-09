import pickle
from sklearn.model_selection import train_test_split
from auto_causality.datasets import generate_synthetic_data
n_samples = 10000
test_size = 0.33 # equal train,val,test
datacount=5

for i in range(datacount+1):
    dataset = generate_synthetic_data(n_samples=n_samples, confounding=True, linear_confounder=True, noisy_outcomes=True)
    dataset.preprocess_dataset()
    data_df=dataset.data
    train_df, test_df = train_test_split(data_df, test_size=test_size)
    test_df = test_df.reset_index(drop=True)
    data={
        'train_df':train_df,
        'test_df':test_df,
        'features_X':dataset.effect_modifiers,
        'features_W':dataset.common_causes
    }
    with open(f"./dataset_run_{i+1}.data", "wb") as f:
        pickle.dump(data, f)
