import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind
from tabulate import tabulate


def missing_values(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    percentage_final = (round(percentage, 2) * 100)
    total_percent = pd.concat(objs=[total, percentage_final], axis=1, keys=['Total', '%'])
    return total_percent


# DATA PREPROCESSING

df = pd.read_csv("Data/ILPD.csv")
df['gender'].replace(['Female', 'Male'], [0, 1], inplace=True)

# print(missing_values(df))

df['alkphos'].fillna(df['alkphos'].mean(), inplace=True)

# print(missing_values(df))

dataset = df.to_numpy()

X = dataset[:, :-1]
y = dataset[:, -1].astype(int)


# FEATURE SELECTION

fs = SelectKBest(score_func=f_classif, k='all')
fs.fit(X, y)

#print(fs.scores_)

random_state = 42

classifiers = {
    '2N': MLPClassifier(solver='sgd', hidden_layer_sizes=(2,), max_iter=1500, momentum=0, random_state=random_state),
    '5N': MLPClassifier(solver='sgd', hidden_layer_sizes=(5,), max_iter=1500, momentum=0, random_state=random_state),
    '10N': MLPClassifier(solver='sgd', hidden_layer_sizes=(10,), max_iter=1500, momentum=0, random_state=random_state),
    '2N momentum': MLPClassifier(solver='sgd', hidden_layer_sizes=(2,), max_iter=1500, momentum=0.8, random_state=random_state),
    '5N momentum': MLPClassifier(solver='sgd', hidden_layer_sizes=(5,), max_iter=1500, momentum=0.8, random_state=random_state),
    '10N momentum': MLPClassifier(solver='sgd', hidden_layer_sizes=(10,), max_iter=1500, momentum=0.8, random_state=random_state)
}

n_repeats = 5
n_splits = 2
n_total_features = len(X[0])

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
scores = np.zeros((len(classifiers),n_total_features, n_splits * n_repeats))

for n_features in range(1, n_total_features+1):
    X_selected = SelectKBest(score_func=f_classif, k=n_features).fit_transform(X, y)
    for fold_id, (train, test) in enumerate(rskf.split(X_selected, y)):
        for clf_id, clf_name in enumerate(classifiers):
            clf = clone(classifiers[clf_name])
            clf.fit(X_selected[train], y[train])
            y_pred = clf.predict(X_selected[test])
            scores[clf_id, n_features-1,  fold_id] = accuracy_score(y[test], y_pred)

mean_scores = np.mean(scores, axis=2)

headers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
columns = np.array([["2N"], ["5N"], ["10N"], ["2N momentum"], ["5N momentum"], ["10N momentum"]])
score_table = np.concatenate((columns, mean_scores), axis=1)
score_table = tabulate(score_table, headers, floatfmt=".4f")
print("\nMean scores:\n", score_table)

best_score_indexes = mean_scores.argmax(axis=1)
print(best_score_indexes)

alpha = 0.05
t_statistic = np.zeros((len(classifiers), len(classifiers)))
p_value = np.zeros((len(classifiers), len(classifiers)))

for i in range(len(classifiers)):
    for j in range(len(classifiers)):
        t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i, best_score_indexes[i]], scores[j, best_score_indexes[j]])

headers = ["2N", "5N", "10N", "2N momentum", "5N momentum", "10N momentum"]
columns = np.array([["2N"], ["5N"], ["10N"], ["2N momentum"], ["5N momentum"], ["10N momentum"]])
t_statistic_table = np.concatenate((columns, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((columns, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nt-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(classifiers), len(classifiers)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (columns, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((len(classifiers), len(classifiers)))
significance[p_value <= alpha] = 1
significance_table = tabulate(np.concatenate(
    (columns, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (columns, stat_better), axis=1), headers)
print("\nStatistically significantly better:\n", stat_better_table)



