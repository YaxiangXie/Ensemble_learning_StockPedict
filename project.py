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
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

class SuperLearner:
    def __init__(self):
        
    
    




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




    