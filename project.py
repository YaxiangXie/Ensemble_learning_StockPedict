import numpy as np
from numpy import random
random.seed(47)
import os
path = f"C:/Users/PC1110223B/Desktop/112-2HW/Fundamental_Enselmble_Learning/project"
os.chdir(path)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing # 標準化
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


class Models:
    def __init__(self):
        self.bagging = BaggingClassifier(n_estimators = 30)
        self.adaboost = AdaBoostClassifier(n_estimators = 20)
        self.gradient_boost = GradientBoostingClassifier(n_estimators = 35, min_samples_split = 10, min_samples_leaf = 10)
        self.random_forest = RandomForestClassifier(n_estimators = 30, min_samples_split = 10, min_samples_leaf = 10)
        self.stacking = None

    def Modelset(self):
        LearnerList = {'Bagging':self.bagging,
                       'Adaboost':self.adaboost,
                       'Gradient_boost':self.gradient_boost,
                       'Random_forest':self.random_forest,
                       'StackingLearner': Models.getStacking()
                       }
        return LearnerList
        
        
    def getStacking(self):
        base = [['Bagging',self.bagging],
                ['Adaboost',self.adaboost],
                ['Gradient_boost',self.gradient_boost],
                ['Random_forest',self.random_forest]
                ]
        metaLearner = LogisticRegression()
        StackingLearner = StackingClassifier(estimators = base, final_estimator = metaLearner, cv = 5)
        return StackingLearner
    
    

   
def evaluate_model(model, X, y):
	cv_scheme = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 47)
	cv_scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv_scheme)
	return cv_scores 
    
        
    
    




if __name__ == '__main__':
    #Load Dataset and standardize
    dataset = np.loadtxt("2330.txt", delimiter = " ")
    dataset_X = dataset[:, 0:13]
    target = dataset[:, 13]

    print(dataset_X.shape)    #(994, 13)
    print(target.shape)    #(994)

    dataset_X = preprocessing.scale(dataset_X)

    ############################################
    X_train, X_test, y_train, y_test = train_test_split(dataset_X, target, train_size = 0.7, stratify = target, random_state = 47)
    print(f"訓練資料樣本數: {len(X_train)}")  #695
    print(f"測試資料樣本數: {len(X_test)}")  #299
    