## Stacking ensemble: Binary classification

# Ref: Brownlee, Chapter 28

###################

# Set random seed for reproducibility.

seed = 543
from numpy import random
random.seed(seed)
from tensorflow.random import set_seed
set_seed(seed)

###################

# Load the dataset.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

dataset= np.loadtxt("C:/Users/PC1110223B/Desktop/112-2HW/Fundamental_Enselmble_Learning/project/2330.txt", delimiter = " ")

X = dataset[:, 0:13]
y = dataset[:, 13]

X.shape
y.shape

# Standardize the input data.

#np.mean(X, axis = 0)
#np.std(X, ddof = 1, axis = 0)

standardize = lambda x: (x - np.mean(x, axis = 0)) / np.std(x, axis = 0, ddof = 1)

X_scaled = standardize(X)

# Split data in training and testing datasets.

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size = 0.7, stratify = y, random_state = 47)

print("number of traning samples = ", len(X_train)) 
print("number of testing samples = ", len(X_test))  

###################

# Base learners

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

n = 30  # number of estimators

bagging = BaggingClassifier(n_estimators = n)
random_forest = RandomForestClassifier(n_estimators = n, min_samples_split = 10, min_samples_leaf = 10)
adaboost = AdaBoostClassifier(n_estimators = 20)
gradient_boost = GradientBoostingClassifier(n_estimators = 35, min_samples_split = 10, min_samples_leaf = 10)

# Get a stacking ensemble.

def get_stacking():
	# Define base learners.
	base_learners = list()
	base_learners.append(('bagging', bagging))
	base_learners.append(('random_forest', random_forest))
	base_learners.append(('adaboost', adaboost))
	base_learners.append(('gradient_boost', gradient_boost))
	# Define the meta learner.
	meta_learner = LogisticRegression()
	# Define the stacking ensemble.
	stacking_model = StackingClassifier(estimators = base_learners, final_estimator = meta_learner, cv = 5)
	return stacking_model

# Get a list of models.

def get_models():
	models = dict()
	models['bagging'] = bagging
	models['random_forest'] = random_forest
	models['adaboost'] = adaboost
	models['gradient_boost'] = gradient_boost
	models['stacking'] = get_stacking()
	return models

# Evaluate a given model using cross validation.

def evaluate_model(model, X, y):
	cv_scheme = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 47)
	cv_scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv_scheme)
	return cv_scores

# Get models.

models = get_models()

# Evaluate the models and store results.

names, cv_scores = list(), list()

for name, model in models.items():
	scores = evaluate_model(model, X_scaled, y)
	names.append(name)
	cv_scores.append(scores)
	print("Classifier = ", name, "  mean accuracy = ", np.mean(scores).round(4), "  standard deviation = ", np.std(scores, ddof = 1).round(4))

import pandas as pd

mean_scores = np.mean(cv_scores, axis = 1)
stds = np.std(cv_scores, ddof = 1, axis = 1)

df = pd.DataFrame({'Mean score': mean_scores, 'standard deviation': stds}, index = names)

print(df)

mean_scores.round(4)
stds.round(4)

# Find the best estimator.

best_index = np.argmax(mean_scores)
names[best_index]

# Plot model performance for comparison.

np.shape(cv_scores)

plt.boxplot(cv_scores, notch = True, labels = names, showmeans = True)
plt.xlabel("classifier")
plt.ylabel("accuracy")
plt.title("Stacking ensemble")
plt.show()

# Train the models using training set and evaluate on testing set.

names = []
test_scores = []

for name, model in models.items():
	model.fit(X_train, y_train)
	y_predicted = model.predict(X_test)
	score = accuracy_score(y_test, y_predicted)
	names.append(name)
	test_scores.append(score)
	print("Classifier = ", name, "  testing accuracy = ", np.round(score, 4))

import pandas as pd

df = pd.DataFrame({'Testing accuracy': test_scores}, index = names)

print(df)

np.round(test_scores, 4)

###################
