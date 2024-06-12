import numpy as np
from numpy import random
random.seed(47)
import os
path = f"C:/Users/謝亞翔/Desktop/112-2HW/Fundamental_Enselmble_Learning/Ensemble_learning_StockPedict"
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
        self.knn = KNeighborsClassifier(n_neighbors = 5, algorithm = 'auto')
        self.decision_tree = DecisionTreeClassifier(criterion = 'entropy', splitter = 'random', max_depth = 15)
        self.GNB = GaussianNB()
        self.stacking = None
        self.LearnerList = {}

    def Modelset(self):
        self.LearnerList = {'Bagging':self.bagging,
                       'Adaboost':self.adaboost,
                       'Gradient_boost':self.gradient_boost,
                       'Random_forest':self.random_forest,
                       'KnnClassifier':self.knn,
                       'DecisionTree':self.decision_tree,
                       'GaussianNB':self.GNB,
                       'StackingLearner': Models.getStacking()
                       }
        
        return self.LearnerList
    
    
    def getStacking(self):
        base = [['Bagging',self.bagging],
                ['Adaboost',self.adaboost],
                ['Gradient_boost',self.gradient_boost],
                ['Random_forest',self.random_forest],
                ['KnnClassifier',self.knn],
                ['DecisionTree',self.decision_tree],
                ['GaussianNB',self.GNB]
                ]
        metaLearner = LogisticRegression()
        StackingLearner = StackingClassifier(estimators = base, 
                                             final_estimator = metaLearner, 
                                             cv = 5,
                                             stack_method = 'auto'
                                             )
        return StackingLearner
    
    

   
def evaluate_model(model, X, y):
	cv_scheme = RepeatedStratifiedKFold(n_splits = 20, n_repeats = 3, random_state = 47)
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

    X_train, X_test, y_train, y_test = train_test_split(dataset_X, target, train_size = 0.7, stratify = target, random_state = 47)
    print(f"訓練資料樣本數: {len(X_train)}")  #695
    print(f"測試資料樣本數: {len(X_test)}")  #299

    ############################################
    Models = Models()

    Modelset = Models.Modelset()

    ModelNameList = list(Modelset.keys())
    
    CrossVal_score = []
    for model in Modelset.values():
        scores = evaluate_model(model, dataset_X, target)
        CrossVal_score.append(np.mean(scores).round(4))

    ModelScore = []
    for model in Modelset.values():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        ModelScore.append(np.round(score, 4))

    
    df = pd.DataFrame({'Testing accuracy': CrossVal_score}, index = ModelNameList)
    df2 = pd.DataFrame({'Testing accuracy': ModelScore}, index = ModelNameList)

    print("\033[33m k-fold cross validation score\033[0m")
    print(df)
    print('-' * 30)
    print("\033[33m Accuracy score\033[0m")
    print(df2)
    