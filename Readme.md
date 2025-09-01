#  Ensemble Learning for Stock Prediction

本專案使用 **台灣股票市場資料** (以所羅門 [2359] 為例，可自行設定)，透過 **技術指標 (TA-Lib)** 建立特徵，
並應用多種 **Ensemble Learning 演算法** (Bagging、Boosting、Stacking) 進行股價漲跌分類預測。  

---

## 研究背景
股價具備高波動性與隨機性，單一學習模型容易受到噪音影響。  
因此，本專案採用 **Ensemble Learning**，利用多個弱分類器組合成更強的學習模型，以提升預測穩健性與泛化能力。  

## 專案架構
project/
│── stock_feature_extract.py # 抓取股價並生成技術指標
│── ensemble_models.py # Ensemble 模型訓練與評估
│── 2359.txt # 預處理後的股票資料 (特徵 + 標籤)
│── README.md # 專案說明

---

## 特徵工程

### 使用技術指標：
1. **基本特徵**
   - 當日最高價 (high)  
   - 當日最低價 (low)  
   - 收盤價 (close)  
   - 成交量 (volume)  

2. **價格相關指標**
   - 布林通道 (BBANDS) → 上軌 / 中軌 / 下軌  
   - 相對強弱指標 (RSI)  
   - 平均趨向指標 (ADX)  
   - 隨機震盪指標 (KD) → %K, %D  

3. **成交量相關指標**
   - 累積/派發線 (ADL)  
   - 能量潮指標 (OBV)  

4. **標籤 (Class)**
   - 若 **明日收盤價 ≥ 今日收盤價 → 1 (上漲)**  
   - 否則 → 0 (下跌)  

---
## 使用模型 (Learners)

- **BaggingClassifier**  
- **AdaBoostClassifier**  
- **GradientBoostingClassifier**  
- **RandomForestClassifier**  
- **KNeighborsClassifier (KNN)**  
- **DecisionTreeClassifier**  
- **StackingClassifier** (整合以上模型，Logistic Regression 作為最終分類器)

---
## 執行流程
1. 安裝環境
```shell
pip install numpy pandas scikit-learn talib twstock
```
2. 資料前處理 (產生股票資料)

```shell
## 修改想要測試的股票代碼
python stock_feature_extract.py
```
3. 訓練與測試模型
```shell

python ensemble_models.py

# 輸出結果
k-fold cross validation score
                 CV mean accuracy
Bagging                  0.6824
Adaboost                 0.6941
Gradient_boost           0.7012
Random_forest            0.7136
KnnClassifier            0.6650
DecisionTree             0.6725
StackingLearner          0.7241
------------------------------
Accuracy score
                 Holdout accuracy
Bagging                  0.6780
Adaboost                 0.6906
Gradient_boost           0.7045
Random_forest            0.7184
KnnClassifier            0.6592
DecisionTree             0.6742
StackingLearner          0.7300
```

