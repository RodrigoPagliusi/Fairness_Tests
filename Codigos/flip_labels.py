from adult_dataset import save_adult

import os
import sys
import argparse
import numpy as np
import pandas as pd

# import warnings
# # Ignore all warnings
# warnings.simplefilter("ignore")
# warnings.filterwarnings("ignore")

def flip_labels(folder,datafile,sensitive_name,label_name,threshold):
    dataframe = pd.read_csv(os.path.join(folder, datafile))
    condition = ((dataframe[sensitive_name] == 0) & (dataframe[label_name] == 1)) | ((dataframe[sensitive_name] == 1) & (dataframe[label_name] == 0))
    random_numbers = np.random.rand(len(dataframe.loc[condition, label_name]))
    dataframe.loc[condition, label_name] = [
        1 - label if rn < threshold else label
        for label, rn in zip(dataframe.loc[condition, label_name], random_numbers)
    ]
    print(dataframe[['sex','income']].value_counts())
    return dataframe

if __name__ == '__main__':

    label_name = 'income'
    sensitive_name = 'sex'
    protected_value = 'Female'
    threshold = 0.05
    num_rows = None
    num_datasets = 20
    sequential = False
    unfair_method = "flip"
    datafile = "encoded_adult.csv"

    # TEMPORARY
    # num_rows = 100

    available_datasets = ['encoded_adult.csv']
    if not datafile in available_datasets: sys.exit("Dataset inválido")

    folder = "./datasets/"
    if not os.path.exists(folder): os.makedirs(folder)

    save_adult(folder, datafile, sensitive_name, protected_value, num_rows)

    if num_rows: datafile = str(num_rows) + "_" + datafile

    for number in range(num_datasets):
        threshold = round(threshold,2)
        print(f"\n\nNEW ITERARION - NUMBER_{number}")
        print(
            datafile,"|",
            "Sequential =",
            str(sequential),"|"
            )

        population = flip_labels(folder,datafile,sensitive_name,label_name,threshold)

        datafile_name = datafile.split('.')[0]
        extension = '.' + datafile.split('.')[1]

        # Default dataset name:
        # encoded_adult_unfair_flip_seq_True_set_0.csv
        if number == 0:
            filename = \
                datafile_name + \
                '_unfair_' + unfair_method + \
                "_seq_"  + str(sequential) + \
                '_set_' + str(number) + \
                extension
        else:
            filename = \
                filename.rsplit('_', 1)[0] + "_" +\
                str(number) + \
                extension

        population.to_csv(folder + filename, index=False)

        threshold = threshold + 0.05
        if sequential: datafile = filename

    # Chamar a função para analisar os datasets
