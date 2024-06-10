import numpy as np
from numpy import random
random.seed(47)
import os
path = f"C:/Users/PC1110223B/Desktop/112-2HW/Fundamental_Enselmble_Learning/project"
os.chdir(path)
import pandas as pd

from sklearn.model_selection import train_test_split









if __name__ == '__main__':
    #讀取資料集
    dataset = np.loadtxt("2330.txt", delimiter = " ")
    