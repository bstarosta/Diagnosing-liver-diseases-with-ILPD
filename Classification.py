import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score


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
X_selected = SelectKBest(score_func=f_classif, k=1).fit_transform(X, y)
print(X_selected.shape)
print(fs.scores_)


classifiers = {
    '2N': MLPClassifier(solver='sgd', hidden_layer_sizes=(2,), max_iter=1500, momentum=0, random_state=42),
    '5N': MLPClassifier(solver='sgd', hidden_layer_sizes=(5,), max_iter=1500, momentum=0, random_state=42),
    '10N': MLPClassifier(solver='sgd', hidden_layer_sizes=(10,), max_iter=1500, momentum=0, random_state=42),
    '2N_momentum': MLPClassifier(solver='sgd', hidden_layer_sizes=(2,), max_iter=1500, momentum=0.8, random_state=42),
    '5N_momentum': MLPClassifier(solver='sgd', hidden_layer_sizes=(5,), max_iter=1500, momentum=0.8, random_state=42),
    '10N_momentum': MLPClassifier(solver='sgd', hidden_layer_sizes=(10,), max_iter=1500, momentum=0.8, random_state=42)
}

n_repeats = 5
n_splits = 2

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
scores = np.zeros((len(classifiers), n_splits * n_repeats))

for fold_id, (train, test) in enumerate(rskf.split(X_selected, y)):
    for clf_id, clf_name in enumerate(classifiers):
        clf = clone(classifiers[clf_name])
        clf.fit(X_selected[train], y[train])
        y_pred = clf.predict(X_selected[test])
        scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

print(scores)
mean = np.mean(scores, axis=1)
std = np.std(scores, axis=1)

for clf_id, clf_name in enumerate(classifiers):
    print("%s: %.3f (std: %.2f)" % (clf_name, mean[clf_id], std[clf_id]))
