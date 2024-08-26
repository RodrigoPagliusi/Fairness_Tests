import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

def prepare_adult(protected_groups, partial):

    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    # 1 quer dizer que pertence ao grupo protegido
    # 0 quer dizer que pertence ao grupo privilegiado
    for k,v in protected_groups.items():
        if pd.api.types.is_string_dtype(X[k]):
            X[k] = X[k].apply(lambda x: 1 if x == v else 0)
        elif pd.api.types.is_numeric_dtype(X[k]):
            X[k] = X[k].apply(lambda x: 1 if x > v else 0)

    label_enc = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        if col not in list(protected_groups.keys()):
            X[col] = label_enc.fit_transform(X[col])

    y['income'] = y['income'].replace(r'\.','',regex=True)
    y['income'] = label_enc.fit_transform(y['income'])

    X = X.astype('int64')
    y = y['income'].astype('int64')
    X['income'] = y

    if partial: X = X.iloc[0:partial]

    return X

def save_adult(folder, dataset, sensitive_name, protected_value, partial):

    if partial: partial_str = str(partial) + '_'
    else: partial_str = ""

    if os.path.isfile(folder + partial_str + dataset): return

    protected_groups = {
        sensitive_name:protected_value
    }

    (
    encoded_adult
    ) = prepare_adult(
            protected_groups,
            partial = partial
            )

    filename = partial_str + dataset
    encoded_adult.to_csv(folder + filename, index = False)
    print('Dataset adult saved to: ' + folder + filename)


if __name__ == "__main__":

    import os

    dataset_name = "encoded_adult.csv"
    folder = "./datasets/"
    if not os.path.exists(folder): os.makedirs(folder)

    protected_groups = {
        "sex":"Female"
    }

    # Use any False option to get the full dataset
    number_rows = 0

    encoded_adult = prepare_adult(
                        protected_groups,
                        partial = number_rows
                        )

    if number_rows: partial_str = str(number_rows) + '_'
    else: partial_str = ""
    filename = partial_str + dataset_name
    encoded_adult.to_csv(folder + filename, index = False)
    print('Dataset adult saved to: ' + folder + filename)