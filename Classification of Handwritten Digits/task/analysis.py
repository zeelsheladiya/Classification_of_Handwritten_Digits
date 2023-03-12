# write your code here

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
from math import sqrt
import random

# Stage 1/5
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

# x_train_flattened = x_train.reshape(len(x_train), 28*28)
#
# print(f"Classes: {np.unique(y_train)}")
# print(f"Features' shape: {x_train_flattened.shape}")
# print(f"Target's shape: {y_train.shape}")
# print(f"min: {x_train.min()}, max: {x_train.max()}")

# Stage 2/5
num_of_row = 6000
data = np.concatenate((x_train, x_test))
target = np.concatenate((y_train, y_test))

x_train, x_test, y_train, y_test = train_test_split(data[:num_of_row], target[:num_of_row], test_size=0.3, random_state=40)
x_train_flatten = x_train.reshape(len(x_train), 28*28)
x_test_flatten = x_test.reshape(len(x_test), 28*28)

#
# print(f"x_train shape: {x_train_flatten.shape}")
# print(f"x_test shape: {x_test_flatten.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"y_test shape: {y_test.shape}")
# print("Proportion of samples per class in train set:")
# print(pd.Series(y_train).value_counts(normalize=True))

# Stage 3/5
scores = {}

# def fit_predict_eval(model, features_train, features_test, target_train, target_test):
#     # here you fit the model
#     model.fit(features_train, target_train)
#     # make a prediction
#     y_pred = model.predict(features_test)
#     # calculate accuracy and save it to score
#     score = accuracy_score(y_test, y_pred)
#     scores[model] = score
#     print(f'Model: {model}\nAccuracy: {score}\n')

# fit_predict_eval(KNeighborsClassifier(), x_train_flatten, x_test_flatten, y_train, y_test)
# fit_predict_eval(DecisionTreeClassifier(random_state=40), x_train_flatten, x_test_flatten, y_train, y_test)
# fit_predict_eval(LogisticRegression(random_state=40), x_train_flatten, x_test_flatten, y_train, y_test)
# fit_predict_eval(RandomForestClassifier(random_state=40), x_train_flatten, x_test_flatten, y_train, y_test)

# print(f"The answer to the question: {str(list(scores.keys())[list(scores.values()).index(max(scores.values()))]).split('(')[0]} - {round(max(scores.values()), 3)}")

# Stage 4/5
x_train_norm = Normalizer().transform(x_train_flatten)
x_test_norm = Normalizer().transform(x_test_flatten)

# fit_predict_eval(KNeighborsClassifier(), x_train_norm, x_test_norm, y_train, y_test)
# fit_predict_eval(DecisionTreeClassifier(random_state=40), x_train_norm, x_test_norm, y_train, y_test)
# fit_predict_eval(LogisticRegression(random_state=40), x_train_norm, x_test_norm, y_train, y_test)
# fit_predict_eval(RandomForestClassifier(random_state=40), x_train_norm, x_test_norm, y_train, y_test)
# print(f"The answer to the 1st question: yes\n")
# print(f"The answer to the 2nd question: {str(list(scores.keys())[list(scores.values()).index(max(scores.values()))]).split('(')[0]} - {round(max(scores.values()), 3)}, {str(list(scores.keys())[list(scores.values()).index(sorted(scores.values())[-2])]).split('(')[0]} - {round(sorted(scores.values())[-2], 3)}")

# Stage 5/5
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # here you fit the model
    model.fit(features_train, target_train)
    # make a prediction
    y_pred = model.predict(features_test)
    # calculate accuracy and save it to score
    score = accuracy_score(y_test, y_pred)
    return score


knn = KNeighborsClassifier()
rfc = RandomForestClassifier(random_state=40)

knn_parameters = dict(n_neighbors=[3, 4], weights=['uniform', 'distance'], algorithm=['auto', 'brute'])
rfc_parameters = dict(n_estimators=[300, 500], max_features=['auto', 'log2'], class_weight=['balanced', 'balanced_subsample'])

knn_clf = GridSearchCV(knn, knn_parameters, scoring='accuracy', n_jobs=-1)
knn_clf.fit(x_train_norm, y_train)

rfc_clf = GridSearchCV(rfc, rfc_parameters, scoring='accuracy', n_jobs=-1)
rfc_clf.fit(x_train_norm, y_train)

print("K-nearest neighbours algorithm")
print(f"best estimator: {knn_clf.best_estimator_}")
# y_pred = knn_clf.predict(x_test)
print(f"accuracy: {fit_predict_eval(knn_clf, x_train_norm, x_test_norm, y_train, y_test)}\n")

print("Random forest algorithm")
print(f"best estimator: {rfc_clf.best_estimator_}")
print(f"accuracy: {sqrt(fit_predict_eval(rfc_clf, x_train_norm, x_test_norm, y_train, y_test))}\n")

