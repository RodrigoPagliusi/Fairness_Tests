from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def load_ori_population(folder, path, sensitive_name, label):
    file = pd.read_csv(folder + path, header=0)
    labels = file[label]
    group_col = file.loc[:, sensitive_name]
    feature_name = list([i for i in file.columns if i not in [sensitive_name, label]])
    population = file[feature_name]
    return population, labels, group_col, feature_name


def simulate_dataset(folder, sensitive_name, label_name, random=None):
    population, labels = make_classification(
                                            n_samples=1000,
                                            n_features=16,
                                            n_informative=10,
                                            flip_y=0.2,
                                            random_state=random
                                            )
    population[:, 6:11] = np.abs(np.floor(population[:, 6:11]))
    population[:, :6] = np.round(1/(1 + np.exp(-population[:, :6])))
    population = pd.DataFrame(population)
    population[label_name] = labels
    population = population.rename(columns={0: sensitive_name})
    population.to_csv(folder + "simulated.csv", index=False)
    print(f"The reference dataset saved to {folder}simulated.csv.")
    return population

