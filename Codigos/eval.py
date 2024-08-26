import os
import glob
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
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.metrics import BinaryLabelDatasetMetric
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

def calculate_statistical_parity(df, group_column, outcome_column):
    """
    Calculate statistical parity between privileged and unprivileged groups based on outcome results.

    Parameters:
    df (pd.DataFrame): The dataframe containing the group and outcome columns.
    group_column (str): The name of the column representing the privileged (0) and unprivileged (1) groups.
    outcome_column (str): The name of the column representing the negative (0) and positive (1) outcomes.

    Returns:
    float: The statistical parity difference between the unprivileged and privileged groups.
    """

    # Filter privileged and unprivileged groups
    privileged_group = df[df[group_column] == 0]
    unprivileged_group = df[df[group_column] == 1]

    # Calculate the rate of positive outcomes for each group
    privileged_positive_rate = privileged_group[outcome_column].mean()
    unprivileged_positive_rate = unprivileged_group[outcome_column].mean()

    # Calculate statistical parity difference
    statistical_parity = unprivileged_positive_rate - privileged_positive_rate

    return statistical_parity





def eval_datasets(dataset, label, sensitive, priv, unpriv, unfair_metric, seq):

    datasets_path = "././datasets/"

    # Criar as pastas de plots e tables se não existirem
    plots = "./plots/"
    if not os.path.exists(plots): os.makedirs(plots)

    tables = "./tables/"
    if not os.path.exists(tables): os.makedirs(tables)

    unfair_metric -= 1
    unfair_metric = str(unfair_metric)

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
        X = df.drop(columns=[label])
        y = df[[label]]
        sensitive_feature = df[[sensitive]]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Create test dataset for AIF360
        test_dataset = BinaryLabelDataset(df=pd.concat([X_test, y_test], axis=1),
                                        label_names=[label],
                                        protected_attribute_names=[sensitive])

        full_dataset = BinaryLabelDataset(
                        favorable_label=priv,
                        unfavorable_label=unpriv,
                        df=df,
                        label_names=[label],
                        protected_attribute_names=[sensitive]
                    )

        # Max Pearson's correlation
        # max_corr = max(abs(pearsonr(sensitive_feature, X[col])[0]) for col in X.columns)

        # Value counts
        label_counts = df[label].value_counts().to_dict()
        sensitive_counts = df[sensitive].value_counts().to_dict()
        label_counts = {int(k): v for k, v in label_counts.items()}
        sensitive_counts = {int(k): v for k, v in sensitive_counts.items()}

        direct_metric = BinaryLabelDatasetMetric(
            full_dataset,
            privileged_groups=[{sensitive: priv}],
            unprivileged_groups=[{sensitive: unpriv}]
        )


        # Calcular as métricas você mesmo e comparar com o AIF360

        statistical_parity_difference = direct_metric.statistical_parity_difference()

        # Tenho que calcular positivos e negativos de privilegiados e não-privilegiados
        data_metrics.append({ # Incluir Statistical Parity Aqui
            "Dataset": file_name,
            "Number of Features": X.shape[1],
            "Number of Instances": X.shape[0],
            "Sensitive Attribute": sensitive,
            "Label Column": label,
            "Label Counts Positive": label_counts[1],
            "Label Counts Negative": label_counts[0],
            "Sensitive Attribute Counts Unpriviledged": sensitive_counts[1],
            "Sensitive Attribute Counts Priviledged": sensitive_counts[0],
            "Positive Priviledged":,
            "Positive Unpriviledged":,
            "Negative Priviledged":,
            "Negative Unpriviledged":,
            #"Max Pearson's Correlation": max_corr,
            "Statistical Parity": statistical_parity_difference
        })

        for model_name, clf in models.items():

            # Train the model
            clf.fit(X_train, y_train.values.ravel())

            # Make predictions
            y_pred = clf.predict(X_test)

            # Create predicted dataset for AIF360
            predicted_dataset = test_dataset.copy(deepcopy=True)
            predicted_dataset.labels = y_pred

            # Compute traditional metrics
            accuracy = accuracy_score(y_test, y_pred)
            # ACHO QUE EU DEVIA FAZER 0 NÃO PRIVILEGIADO E 1 PRIVILEGIADO
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            # Compute fairness metrics using AIF360
            metric = ClassificationMetric(
                test_dataset,
                predicted_dataset,
                unprivileged_groups=[{test_dataset.protected_attribute_names[0]: unpriv}],
                privileged_groups=[{test_dataset.protected_attribute_names[0]: priv}]
            )
            stat_parity = abs(metric.statistical_parity_difference())
            eq_opportunity = abs(metric.equal_opportunity_difference())
            eq_odds = abs(metric.average_odds_difference())

            # Append results
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

    # Convert results to a DataFrame and save to excel
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
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

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

    eval_datasets("encoded_adult", "income", "sex", 0, 1, 9, True)