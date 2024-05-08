# Importowanie potrzebnych bibliotek do implementacji RandomStumps i obliczenia Accuracy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes # Importowanie database'a iris do eksperymentu
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
import matplotlib.pyplot as plt

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
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_vote

# iris = load_iris() # Wczytywanie danych z iris
# X, y = iris.data, iris.target # Przypisanie danych do X i y

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # Podział danych na zbiór treningowy i testowy

# random_stumps = RandomStumps(n_stumps=10) # Tworzenie RandomStumps z 10 pniami
# random_stumps.fit(X_train, y_train) # Dopasowanie modelu na podstawie danych treningowych

# y_pred = random_stumps.predict(X_test) # Przewidywanie erykiet za pomocą danych testowych
# accuracy = accuracy_score(y_test, y_pred, normalize=True) # Obliczanie dokładności modelu
# recall = recall_score(y_test, y_pred, average='weighted') # Obliczanie recall
# precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) # Obliczanie precision

# print(f'Accuracy: {accuracy}')
# print(f'Recall: {recall}')
# print(f'Precision: {precision}')
# # print(f'y_test: {y_test}')
# # print(f'y_pred: {y_pred}')

# cv_score = cross_val_score(random_stumps, X, y, cv=10) # Obliczanie dokładności modelu za pomocą walidacji krzyżowej
# # print(f'Cross validation score: {cv_score}')
# print(f'Cross validation score mean: {np.mean(cv_score)}')

# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.xlabel('True Labels')
# plt.ylabel('Predicted Labels')
# plt.title('True vs Predicted Labels')
# plt.show()

breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # Podział danych na zbiór treningowy i testowy

random_stumps = RandomStumps(n_stumps=10) # Tworzenie RandomStumps z 10 pniami
random_stumps.fit(X_train, y_train) # Dopasowanie modelu na podstawie danych treningowych

y_pred = random_stumps.predict(X_test) # Przewidywanie erykiet za pomocą danych testowych
accuracy = accuracy_score(y_test, y_pred, normalize=True)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
# print(f'y_test: {y_test}')
# print(f'y_pred: {y_pred}')

cv_score = cross_val_score(random_stumps, X, y, cv=10) # Obliczanie dokładności modelu za pomocą walidacji krzyżowej
print(f'Cross validation score: {cv_score}')
print(f'Cross validation score mean: {np.mean(cv_score)}')

# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.xlabel('True Labels')
# plt.ylabel('Predicted Labels')
# plt.title('True vs Predicted Labels')
# plt.show()