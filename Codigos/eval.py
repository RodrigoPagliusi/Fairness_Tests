import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
from itertools import cycle

# TENTAR AUMENTAR O COEFICIENTE, PARA VER SE AUMENTA A INJUSTIÇA!


# OBS: calcular métricas de injustiça nos próprios datasets

# Positivo e negativos dos privilegiados en ao privilegaidos
# nas metrifcas dos datasets

# Veja o código do Ygor, veja como ele fez o processamento paralelo (se fez)

# Fazer a integracao entre a geracao de datasets e a avaliacao dele.
# Roda o Logistic Regression sim e também o Statistical Parity

# Ver o que acontece sem os warnings no seu data.py
# Ver o que acontece sem os warnings no seu main_paralelo.py
# Ver o que acontece sem os warnings no seu unfairness_metrics.py
# Ver o que acontece sem os warnings no seu genetic_algorithm.py
# Ver o que acontece sem os warnings no seu genetic_algorithm_base.py

# Ver o que acontece no seu main_paralelo.py com e sem random_state=98958

def eval_datasets(dataset, label, sensitive, priv, unpriv, unfair_metric, seq):

    datasets_path = "././datasets/"

    # Criar as pastas de plots e tables se não existirem
    plots = "./plots/"
    if not os.path.exists(plots): os.makedirs(plots)

    tables = "./tables/"
    if not os.path.exists(tables): os.makedirs(tables)

    file_pattern = dataset + "_unfair_" + unfair_metric + "_seq_" + str(seq) + "_set_*.csv"
    matching_files = glob.glob(os.path.join(datasets_path, file_pattern))
    matching_files.insert(0, datasets_path + dataset + ".csv")

    # Create models
    models = {
        # 'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        # 'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(),
        # 'KNN': KNeighborsClassifier()
    }

    # Initialize results list
    data_metrics = []
    results = []

    # Iterate through each dataset
    for file_path in matching_files:
        file_name = os.path.basename(file_path)

        # Load the dataset
        df = pd.read_csv(file_path)

        # Prepare data
        X = df.drop(columns=[label]).copy()
        y = df[[label]].copy()
        sensitive_feature = df[[sensitive]].copy()

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        df_X_test_sensitive = X_test[sensitive].copy().to_numpy().astype('int64')
        X_train = X_train.to_numpy().astype('int64')
        y_train = y_train.to_numpy().astype('int64').ravel()
        X_test = X_test.to_numpy().astype('int64')
        y_test = y_test.to_numpy().astype('int64').ravel()

        # Max Pearson's correlation
        # max_corr = max(abs(pearsonr(sensitive_feature, X[col])[0]) for col in X.columns)

        # Value counts
        label_counts = df[label].value_counts().to_dict()
        sensitive_counts = df[sensitive].value_counts().to_dict()
        label_counts = {int(k): v for k, v in label_counts.items()}
        sensitive_counts = {int(k): v for k, v in sensitive_counts.items()}

        condition = ((df[sensitive] == 0) & (df[label] == 1))
        positive_priviledged = df.loc[condition,label].copy().shape[0]
        condition = ((df[sensitive] == 0) & (df[label] == 0))
        negative_priviledged = df.loc[condition,label].copy().shape[0]
        condition = ((df[sensitive] == 1) & (df[label] == 1))
        positive_unpriviledged = df.loc[condition,label].copy().shape[0]
        condition = ((df[sensitive] == 1) & (df[label] == 0))
        negative_unpriviledged = df.loc[condition,label].copy().shape[0]

        unprivileged = df[df[sensitive] == 1].copy()
        p_positive_unprivileged = unprivileged[label].mean()
        privileged = df[df[sensitive] == 0].copy()
        p_positive_privileged = privileged[label].mean()
        statistical_parity = abs(p_positive_unprivileged - p_positive_privileged)

        data_metrics.append({
            "Dataset": file_name,
            "Number of Features": X.shape[1],
            "Number of Instances": X.shape[0],
            "Sensitive Attribute": sensitive,
            "Label Column": label,
            "Label Counts Positive": label_counts[1],
            "Label Counts Negative": label_counts[0],
            "Sensitive Attribute Counts Unpriviledged": sensitive_counts[1],
            "Sensitive Attribute Counts Priviledged": sensitive_counts[0],
            "Positive Priviledged":positive_priviledged,
            "Positive Unpriviledged":negative_priviledged,
            "Negative Priviledged":positive_unpriviledged,
            "Negative Unpriviledged":negative_unpriviledged,
            # "Max Pearson's Correlation": max_corr,
            "Statistical Parity": statistical_parity
        })

        for model_name, clf in models.items():

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Compute traditional metrics
            # ACHO QUE EU DEVIA FAZER 0 NÃO PRIVILEGIADO E 1 PRIVILEGIADO
            # COLOQUEI MESMO TIPO DE DADOS PARA y_test e y_pred
            # CALCULE AS MÉTRICAS DE FAIRNESS VOCÊ MESMO
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            p_positive_unprivileged = np.mean(y_pred[df_X_test_sensitive == 1])
            p_positive_privileged = np.mean(y_pred[df_X_test_sensitive == 0])
            stat_parity = abs(p_positive_unprivileged - p_positive_privileged)

            true_positive_unprivileged = y_pred[(df_X_test_sensitive == 1) & (y_test == 1)]
            true_positive_privileged = y_pred[(df_X_test_sensitive == 0) & (y_test == 1)]
            tpr_unprivileged = np.mean(true_positive_unprivileged) if len(true_positive_unprivileged) > 0 else 0
            tpr_privileged = np.mean(true_positive_privileged) if len(true_positive_privileged) > 0 else 0
            eq_opportunity = abs(tpr_unprivileged - tpr_privileged)

            false_positive_unprivileged = y_pred[(df_X_test_sensitive == 1) & (y_test == 0)]
            false_positive_privileged = y_pred[(df_X_test_sensitive == 0) & (y_test == 0)]
            fpr_unprivileged = np.mean(false_positive_unprivileged) if len(false_positive_unprivileged) > 0 else 0
            fpr_privileged = np.mean(false_positive_privileged) if len(false_positive_privileged) > 0 else 0
            fpr_diff = abs(fpr_unprivileged - fpr_privileged)

            eq_odds = np.mean([eq_opportunity,fpr_diff])

            results.append({
                "Dataset": file_name,
                "Model": model_name,
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Matthews Correlation Coefficient": mcc,
                "Statistical Parity": stat_parity,
                "Equal Opportunity": eq_opportunity,
                "Equalized Odds": eq_odds,
            })

    data_metrics = pd.DataFrame(data_metrics)
    data_metrics = data_metrics.set_index("Dataset").T
    data_metrics.columns.name = "Dataset"

    results = pd.DataFrame(results)
    results = results.pivot(index="Dataset", columns="Model")
    results.columns = [f'{col[1]}_{col[0]}' for col in results.columns]
    results = results.T
    results = results.reset_index()
    results[['Model', 'Metric']] = results['index'].str.rsplit('_', expand=True)
    results = results.drop(columns=['index'])
    results = results.set_index(['Metric', 'Model'])

    new_column_names = [f'set_{i}' for i in range(results.shape[1])]
    new_column_names[0] = "Original"
    data_metrics.columns = new_column_names
    results.columns = new_column_names

    data_metrics.to_excel(tables + "datasets_" + dataset + "_unfair_" + unfair_metric + "_seq_" + str(seq) + ".xlsx")
    results.to_excel(tables + "models_" + dataset + "_unfair_" + unfair_metric + "_seq_" + str(seq) + ".xlsx")

    results_performance = results.iloc[:3]
    results_fariness = results.iloc[3:]

    plt.figure(figsize=(10, 6))
    colors = cycle(['b', 'g', 'r'])
    for (metric, model), color in zip(results_performance.index, colors):
        plt.plot(results_performance.columns, results_performance.loc[(metric, model)], marker='o', label=f'{metric}', color=color)
        plt.title(f'Performance Metrics for ' + dataset + "_unfair_" + unfair_metric + "_seq_" + str(seq))
    plt.xlabel('Dataset')
    plt.ylabel('Metric Value')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plots + 'performance_metrics_' + dataset + "_unfair_" + unfair_metric + "_seq_" + str(seq) + '.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for (metric, model), color in zip(results_fariness.index, colors):
        plt.plot(results_fariness.columns, results_fariness.loc[(metric, model)], marker='o', label=f'{metric}', color=color)
        plt.title(f'Fairness Metrics for ' + dataset + "_unfair_" + unfair_metric + "_seq_" + str(seq))
    plt.xlabel('Dataset')
    plt.ylabel('Metric Value')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plots + 'fariness_metrics_' + dataset + "_unfair_" + unfair_metric + "_seq_" + str(seq) + '.png', bbox_inches='tight')
    plt.close()

    print("All metrics plotted and saved successfully.")




if __name__ == "__main__":

    eval_datasets("encoded_adult", "income", "sex", 1, 0, 'flip', False)
    # def eval_datasets(dataset, label, sensitive, priv, unpriv, unfair_metric, seq):