from genetic_algorithm import ContinuousGenAlgSolver
from adult_dataset import save_adult
from data import load_ori_population, simulate_dataset
from unfairness_metrics import UNFAIRNESS_METRICS, UnfairnessMetric

from sklearn import metrics, model_selection, pipeline, preprocessing, linear_model, ensemble

import os
import sys
import argparse
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import warnings
# Ignore all warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


class calculation():
    def __init__(self, dataset, labels, groups, index, groupkfold=None, random = None):
        self.data = dataset
        self.labels = labels
        self.groups = groups
        self.groupkfold = groupkfold
        self.unfair_index = index
        self.last_unfair = 0
        self.iter = 1
        self.random = random
        self.inital_lr = self.initial_calculate(dataset, linear_model.LogisticRegression(max_iter=200, random_state=self.random))
        self.inital_rf = self.initial_calculate(dataset, ensemble.RandomForestClassifier(random_state=self.random))
        self.inital_et = self.initial_calculate(dataset, ensemble.ExtraTreesClassifier(random_state=self.random))

    def initial_calculate(self, dataset, clf):
        if self.groupkfold is None:
            xval = model_selection.KFold(4, shuffle=True, random_state=self.random)
        else:
            xval = model_selection.GroupKFold(4)

        scoring = {}
        for m in UNFAIRNESS_METRICS:
            if m == "calibration":
                metric = UnfairnessMetric(pd.Series(self.groups), m)
                scoring[m] = metrics.make_scorer(metric, needs_proba=True)
            else:
                metric = UnfairnessMetric(pd.Series(self.groups), m)
                scoring[m] = metrics.make_scorer(metric)
        scoring['auc'] = metrics.make_scorer(metrics.roc_auc_score)
        scoring['acc'] = metrics.make_scorer(metrics.accuracy_score)
        pipe = pipeline.Pipeline([
            ('standardize', preprocessing.StandardScaler()),
            ('model', clf)
        ])
        if self.groupkfold is None:
            result = model_selection.cross_validate(pipe, dataset, pd.Series(self.labels), verbose=0, cv=xval,
                                                    scoring=scoring, return_estimator=True)
        else:
            result = model_selection.cross_validate(pipe, dataset, pd.Series(self.labels), verbose=0,
                                                    cv=xval, groups=self.groupkfold,
                                                    scoring=scoring, return_estimator=True)
        unfair_score = []
        unfair_score.append(result['test_' + UNFAIRNESS_METRICS[self.unfair_index]].mean())
        unfair_score.append(result['test_auc'].mean())
        unfair_score.append(result['test_acc'].mean())
        print('UNFAIRNESS SCORE: ', unfair_score)
        return unfair_score[0]

    def calculate_corr(self, dataset):
        corr = np.corrcoef(dataset.T)
        pos = np.where(np.abs(corr) > 0.4)
        return pos, corr[pos]

    def fit_single_score(self, individual):
        if self.groupkfold is not None:
            xval = model_selection.GroupKFold(4)
        else:
            xval = model_selection.KFold(4, shuffle=True, random_state=self.random)

        clf = linear_model.LogisticRegression(max_iter=200, random_state=self.random)

        sml = np.count_nonzero(np.equal(self.data, individual) == 1) / (self.data.shape[0] * self.data.shape[1])

        scoring = {}
        for m in UNFAIRNESS_METRICS:
            if m == "calibration":
                metric = UnfairnessMetric(pd.Series(self.groups), m)
                scoring[m] = metrics.make_scorer(metric, needs_proba=True)
            else:
                metric = UnfairnessMetric(pd.Series(self.groups), m)
                scoring[m] = metrics.make_scorer(metric)
        scoring['auc'] = metrics.make_scorer(metrics.roc_auc_score)
        scoring['acc'] = metrics.make_scorer(metrics.accuracy_score)

        pipe = pipeline.Pipeline([
            ('standardize', preprocessing.StandardScaler()),
            ('model', clf),
        ])

        if self.groupkfold is not None:
            result = model_selection.cross_validate(pipe, individual, pd.Series(self.labels), verbose=0,
                                                    cv=xval, groups=self.groupkfold,
                                                    scoring=scoring, return_estimator=True)
        else:
            result = model_selection.cross_validate(pipe, individual, pd.Series(self.labels), verbose=0,
                                                    cv=xval, scoring=scoring, return_estimator=True)
        unfair_score = []

        for id_unfair in range(len(UNFAIRNESS_METRICS)):
            if id_unfair == self.unfair_index:
                unfair_score.append(result['test_' + UNFAIRNESS_METRICS[id_unfair]].mean())
        unfair_score.append(result['test_auc'].mean())
        unfair_score.append(result['test_acc'].mean())

        return sml, unfair_score

    def fit_scores(self, population, labels, idx_groups, gen_percent, expect_score, pnt=False):

        self.coeff = 1
        results = Parallel(n_jobs=-1)(delayed(self.fit_single_score)(population[i]) for i in range(population.shape[0]))

        score, unfair_scores = [], []
        for sml, unfair_score in results:
            if pnt:
                print(unfair_score)
            unfair_scores.append(unfair_score[0])
            score.append([sml, unfair_score[0]])

        n_top_10 = int(len(unfair_scores) * 0.1)

        # adjust weight
        if sum(unfair_scores[:n_top_10]) / n_top_10 < expect_score:
            if gen_percent > 1 - (expect_score - sum(unfair_scores[:n_top_10]) / n_top_10) / (expect_score - self.inital_lr):
                current_lr = sum(unfair_scores[:n_top_10]) / n_top_10
                current_percent = (current_lr - self.inital_lr) / (expect_score - self.inital_lr)
                self.coeff = gen_percent - current_percent + 1

        for i_scores in range(len(score)):
            score[i_scores].insert(0, self.coeff * 4 * score[i_scores][1] + score[i_scores][0])

        return score, self.coeff

    def post_evaluate(self, population, labels, group, clf):
        if self.groupkfold is not None:
            xval = model_selection.GroupKFold(4)
        else:
            xval = model_selection.KFold(4, shuffle=True, random_state=self.random)

        groups_syn = group
        scoring = {}
        for m in UNFAIRNESS_METRICS:
            if m == "calibration":
                metric = UnfairnessMetric(pd.Series(groups_syn), m)
                scoring[m] = metrics.make_scorer(metric, needs_proba=True)
            else:
                metric = UnfairnessMetric(pd.Series(groups_syn), m)
                scoring[m] = metrics.make_scorer(metric)
        scoring['auc'] = metrics.make_scorer(metrics.roc_auc_score)
        scoring['acc'] = metrics.make_scorer(metrics.accuracy_score)

        pipe = pipeline.Pipeline([
            ('standardize', preprocessing.StandardScaler()),
            ('model', clf),
        ])
        if self.groupkfold is not None:
            result = model_selection.cross_validate(pipe, population, pd.Series(labels), verbose=0,
                                                    cv=xval, groups=self.groupkfold,
                                                    scoring=scoring, return_estimator=True)
        else:
            result = model_selection.cross_validate(pipe, population, pd.Series(labels), verbose=0,
                                                    cv=xval, scoring=scoring, return_estimator=True)

        unfair_score = []
        unfair_score.append(result['test_' + UNFAIRNESS_METRICS[self.unfair_index]].mean())
        unfair_score.append(result['test_auc'].mean())
        unfair_score.append(result['test_acc'].mean())
        print('UNFAIRNESS SCORE: ', unfair_score)
        print('PERCENTAGE SAME: ', np.count_nonzero(np.equal(self.data, population) == 1) / (self.data.shape[0] * self.data.shape[1]))
        self.last_unfair = unfair_score[0]

        c = 0
        scores = []
        if self.groupkfold is not None:
            for train_index, test_index in xval.split(self.data, groups=self.groupkfold):
                X_test, y_test = self.data.iloc[test_index], self.labels.iloc[test_index]
                pred = result['estimator'][c].predict(X_test)
                scores.append(metrics.roc_auc_score(y_test, pred))
        else:
            for train_index, test_index in xval.split(self.data):
                X_test, y_test = self.data.loc[test_index], self.labels.loc[test_index]
                pred = result['estimator'][c].predict(X_test)
                scores.append(metrics.roc_auc_score(y_test, pred))
                c += 1
        self.score = sum(scores) / len(scores)
        print('AUC ON ORIGINAL DATASET: ', sum(scores) / len(scores))
        return unfair_score[0]


def run(folder, datafile, metric, sensitive_name, label, random=None):
    dataset, labels, groups, _ = load_ori_population(folder, datafile, sensitive_name, label)
    fitness_function = calculation(dataset, labels, groups, metric)

    solver = ContinuousGenAlgSolver(
        fitness_function=fitness_function.fit_scores,
        expect_score=0.5,
        dataset=dataset, labels=labels, group_col=groups, feature_name=_,
        pop_size=100,  # population size (number of individuals)
        max_gen=50,  # maximum number of generations
        gene_mutation_rate=0.002,
        mutation_rate=0.002,  # mutation rate to apply to the population
        selection_rate=0.6,  # percentage of the population to select for mating
        selection_strategy="roulette_wheel",  # strategy to use for selection. see below for more details
        plot_results=False,
        random_state=random
    )

    population, labels, group = solver.solve()
    unfair_lr = fitness_function.post_evaluate(population, labels, group, linear_model.LogisticRegression(max_iter=200, random_state=random))
    unfair_rf = fitness_function.post_evaluate(population, labels, group, ensemble.RandomForestClassifier(random_state=random))
    unfair_et = fitness_function.post_evaluate(population, labels, group, ensemble.ExtraTreesClassifier(random_state=random))

    print("DIFFERENCE LOGISTICREGRESSION:", unfair_lr - fitness_function.inital_lr)
    print("DIFFERENCE RANDOMFOREST:", unfair_rf - fitness_function.inital_rf)
    print("DIFFERENCE EXTATREE:", unfair_et - fitness_function.inital_et)
    return population, labels, group

def flip_labels(folder,datafile,sensitive_name,label_name,threshold):
    dataframe = pd.read_csv(folder + datafile)
    condition = ((dataframe[sensitive_name] == 1) & (dataframe[label_name] == 1)) | ((dataframe[sensitive_name] == 0) & (dataframe[label_name] == 0))
    random_numbers = np.random.rand(len(dataframe.loc[condition,label_name]))
    mask = random_numbers < threshold
    dataframe.loc[condition,label_name] = dataframe.loc[condition,label_name].where(~mask, 1 - dataframe.loc[condition,label_name])
    return dataframe

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="adult dataset")

    # If this is called, creates a new reference dataset and nothing more
    parser.add_argument('--generate_reference', action="store_true")

    parser.add_argument('--dataset',
                        help="""
                        Specify the dataset to use. Options are:
                        1. "encoded_adult.csv"
                        """,
                        default="encoded_adult.csv",
                        type=str
                        )
    parser.add_argument('--num_rows',
                        help="""
                        Number of rows of the dataset to use.
                        Not specifying will use entire dataset.
                        """,
                        default = None,
                        type=int
                        )
    parser.add_argument('--sequential',
                        action="store_false")
    parser.add_argument('--num_datasets',
                        default = 20,
                        type=int)
    parser.add_argument('--unfair_metric',
        help="""
            Specify the unfair metric to use. Options are:
            1. overall_accuracy_equality
            2. statistical_parity
            3. conditional_procedure
            4. conditional_use_accuracy_equality
            5. treatment_equality
            6. all_equality
            7. calibration
            8. equalized_odds
            9. Simple
        """,
        type=int,
        default=9
    )
    parser.add_argument('--label_name',
                        help="name of prediction/label variable",
                        default='income')
    parser.add_argument('--sensitive_name',
                        help="name of sensitive feature",
                        default='sex')
    parser.add_argument('--protected_value',
                        help="protected value of the sensitive feature",
                        default='Female')
    parser.add_argument('--threshold',
                        help="Probability to flip the label",
                        default = 0.05,
                        type=float)

    args = parser.parse_args()
    print(args)

    label_name = args.label_name
    sensitive_name = args.sensitive_name
    protected_value = args.protected_value
    threshold = args.threshold
    num_rows = args.num_rows
    num_datasets = args.num_datasets
    sequential = args.sequential
    generate_reference = args.generate_reference
    unfair_metric = args.unfair_metric
    datafile = args.dataset

    # TEMPORARY
    # num_rows = 100

    available_datasets = ['encoded_adult.csv']
    if not datafile in available_datasets: sys.exit("Dataset inválido")

    folder = "./datasets/"
    if not os.path.exists(folder): os.makedirs(folder)

    save_adult(folder, datafile, sensitive_name, protected_value, num_rows)

    if num_rows: datafile = str(num_rows) + "_" + datafile

    if generate_reference:
        print("Generating a reference dataset.")
        simulate_dataset(folder, sensitive_name, label_name)
    else:
        u_metric_index = unfair_metric - 1

        for number in range(num_datasets):
            print(f"\n\nNEW ITERARION - NUMBER_{number}")
            print(
                datafile,"|",
                "Sequential =",
                str(sequential),"|",
                UNFAIRNESS_METRICS[u_metric_index]
                )

            if u_metric_index == 8:
                population = flip_labels(folder,datafile,sensitive_name,label_name,threshold)
            else:
                population, group, labels = run(
                                            folder,
                                            datafile,
                                            u_metric_index,
                                            sensitive_name,
                                            label_name
                                            )
                population[sensitive_name] = group
                population[label_name] = labels

            datafile_name = datafile.split('.')[0]
            extension = '.' + datafile.split('.')[1]

            # Default dataset name:
            # encoded_adult_unfair_7_seq_True_set_0.csv
            if number == 0:
                filename = \
                    datafile_name + \
                    '_unfair_' + str(u_metric_index) + \
                    "_seq_"  + str(sequential) + \
                    '_set_' + str(number) + \
                    extension
            else:
                filename = \
                    filename.rsplit('_', 1)[0] + "_" +\
                    str(number) + \
                    extension

            population.to_csv(folder + filename, index=False)

            if sequential:
                if u_metric_index == 8: threshold += 0.05
                else: datafile = filename
        # Chamar a função para analisar os datasets
