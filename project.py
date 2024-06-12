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
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

class Models:
    # 初始建構各基本學習機
    def __init__(self):
        self.bagging = BaggingClassifier(n_estimators = 30)
        self.adaboost = AdaBoostClassifier(n_estimators = 20)
        self.gradient_boost = GradientBoostingClassifier(n_estimators = 35, min_samples_split = 10, min_samples_leaf = 10)
        self.random_forest = RandomForestClassifier(n_estimators = 30, min_samples_split = 10, min_samples_leaf = 10)
        self.knn = KNeighborsClassifier(n_neighbors = 5, algorithm = 'auto')
        self.decision_tree = DecisionTreeClassifier(criterion = 'entropy', splitter = 'random', max_depth = 15)
        self.stacking = None
        self.LearnerList = {}

    def Modelset(self):
        self.LearnerList = {'Bagging':self.bagging,
                            'Adaboost':self.adaboost,
                            'Gradient_boost':self.gradient_boost,
                            'Random_forest':self.random_forest,
                            'KnnClassifier':self.knn,
                            'DecisionTree':self.decision_tree,
                            'StackingLearner': Models.getStacking()
                            }
        return self.LearnerList
    
    
    def getStacking(self):
        #第一層學習機集合
        base = [['Bagging',self.bagging],
                ['Adaboost',self.adaboost],
                ['Gradient_boost',self.gradient_boost],
                ['Random_forest',self.random_forest],
                ['KnnClassifier',self.knn],
                ['DecisionTree',self.decision_tree],
                ]
        #second level learner
        #StackingClassifier: final_estimator 預設使用 logistiregressor
        StackingLearner = StackingClassifier(estimators = base, 
                                             cv = 5,
                                             stack_method = 'auto'
                                             )
        return StackingLearner
    
#訓練基本學習機及stacking學習機，並做K-fold 交叉驗證
def evaluate_model_K_Fold(model, X, y):
	cv_scheme = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 47)
	cv_scores = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv_scheme)
	return cv_scores 


#訓練基本學習機及stacking學習機，並求 Accuracy score
def evaluate_model_Accuracy_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
    
if __name__ == '__main__':
    #Load Dataset and standardize
    dataset = np.loadtxt("2330.txt", delimiter = " ")
    dataset_X = dataset[:, 0:13]
    target = dataset[:, 13]
    
    # 測試使用: 減少部分特徵向量，觀察是否能提高學習機準確率，最終僅發現刪除布林通道中軌指標(20均線)，會導致整體學習機準確率下降
    #dataset_X = np.delete(dataset_X, [0,1], axis= 1)  

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
    
    #計算基本學習機、stacking學習機的準確率
    CrossVal_score = []
    for model in Modelset.values():
        scores = evaluate_model_K_Fold(model, dataset_X, target)
        CrossVal_score.append(np.mean(scores).round(4))

    #計算基本學習機、stacking學習機的準確率
    ModelScore = []
    for model in Modelset.values():
        score = evaluate_model_Accuracy_score(model, X_train, X_test, y_train, y_test)
        ModelScore.append(np.round(score, 4))

    
    df = pd.DataFrame({'Testing accuracy': CrossVal_score}, index = ModelNameList)
    df2 = pd.DataFrame({'Testing accuracy': ModelScore}, index = ModelNameList)

    print("\033[33m k-fold cross validation score\033[0m")
    print(df)
    print('-' * 30)
    print("\033[33m Accuracy score\033[0m")
    print(df2)
    