from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_rel
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

class RandomStumps(BaseEstimator, ClassifierMixin): # Definiowanie klasy RandomStumps
    def __init__(self, n_stumps): 
        self.n_stumps = n_stumps
        self.stumps = [DecisionTreeClassifier(max_depth=1) for _ in range(n_stumps)] # max_depth jest liczbą rozwinięć węzłów drzewa

    def fit(self, X, y): 
        for stump in self.stumps: # Dla każdego pnia dobieramy losowo połowę danych
            subset_indices = np.random.choice(range(X.shape[0]), size=int(X.shape[0] * 0.5), replace=False)
            X_subset = X[subset_indices]
            y_subset = y[subset_indices]
            stump.fit(X_subset, y_subset)
        return self

    def predict(self, X): # Przewidywanie etykiet dla zioru X na podstawie większości głosów
        predictions = np.array([stump.predict(X) for stump in self.stumps]) # Tworzenie tablicy z przwidywaniami
        majority_vote = stats.mode(predictions, axis=0)[0]
        return majority_vote

dir_list = listdir('datasets')

# print(dir_list)

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
clfs = [
    GaussianNB(),
    KNeighborsClassifier(),
    MLPClassifier(),
    RandomStumps(n_stumps=10)
]

results = np.zeros((len(dir_list), 10, len(clfs)))

# for d_id, d_name in enumerate(dir_list):
#     data = np.loadtxt('datasets/%s' % d_name, delimiter=',')
#     print(data.shape)
#     X, y = data[:, :-1], data[:, -1]

#     for fold, (train, test) in enumerate(rskf.split(X, y)):
#         for clf_id, clf in enumerate(clfs):

#             clf_clone = clone(clf)
#             clf_clone.fit(X[train], y[train])
#             pred = clf_clone.predict(X[test])

#             acc = accuracy_score(y[test], pred)
#             results[d_id, fold, clf_id] = acc

# np.save('results.npy', results)

# 2 zad

results = np.load('results.npy')
print(results.shape)

res = results[9]
print(res.shape)

t_stat_matrix = np.zeros((4,4))
p_val_matrix = np.zeros((4,4))
better_matrix = np.zeros((4,4))

for i in range(4):
    for j in range(4):

        res_i = res[:, i]
        res_j = res[:, j]

        t_stat, p_val = ttest_rel(res_i, res_j)
        t_stat_matrix[i, j] = t_stat
        p_val_matrix[i, j] = p_val

        better_matrix[i, j] = np.mean(res_i) > np.mean(res_j)

print(t_stat_matrix)
print(p_val_matrix)
print(better_matrix)

alpha = 0.05
stat_significant = p_val_matrix < alpha
print(stat_significant)

stat_better = stat_significant*better_matrix
print(stat_better)

clfs = ['GNB', 'KNN', 'MLP', 'RST']

for i in range(4):
    for j in range(4):
        if stat_better[i, j]:
            print('%s (acc=%0.3f) Jest lepszy statystycznie of %s (acc=%0.3f)' %
            (
                clfs[i], np.mean(res[:, i]), clfs[j], np.mean(res[:, j])
            ))