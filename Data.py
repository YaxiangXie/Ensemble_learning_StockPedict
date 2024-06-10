import numpy as np
import twstock
from twstock import Stock
# https://twstock.readthedocs.io/zh-tw/latest/reference/stock.html
# https://hackmd.io/@s02260441/HJcMcnds8
import pandas as pd
import talib
from talib import abstract


#布林通道
def getBollingerBand(close):
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0,matype=0)
    return upper, middle, lower

#RSI
def getRSI(close):
    df_rsi = abstract.RSI(close, 6)
    return df_rsi

#ADX
def getADX(high, low, close):
    df_ADX = talib.ADX(high, low, close, timeperiod = 14)
    return df_ADX

def getKD(high, low, close):
    slowk, slowd = talib.STOCH(high, low, close,slowk_period=5, slowd_period=5, fastk_period=9)
    return slowk, slowd

def getADL(high, low, close, volume):
    df_ADL = talib.AD(high, low, close, volume)
    return df_ADL

def getOBV(close, volume):
    df_OBV = talib.OBV(close, volume)
    return df_OBV
    


if __name__ == '__main__':
    #stock_id = input("請輸入股票代碼: ")
    stock_id = '2330'
    stock = Stock(stock_id)
    stock_name = twstock.codes[stock_id].name
    print(f"股票: {stock_name} ({stock_id}) ")
    
    target_list = [] # 使用 target_list 包 feature
    data = stock.fetch_from(2024,2) # 擷取2024/4 至今股票資料
    
    #將股票最高價、最低價、收盤價、成交量等資訊紀錄於獨立的list，方便使用於股票指標計算
    high = [] #當日最高價list
    low = [] #當日最低價list
    close = [] #收盤價list
    volume = []
    for item in data:
        high.append(item.high)
        low.append(item.low)
        close.append(item.close)
        volume.append(item.Transcation)
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    volume = np.array((volume))
    '''
    使用以下股票指標作為特徵:
    股價相關指標: 
    
    1、布林通道:
        產生上軌、中軌、下軌等三種軌道線，模擬常態分佈假設。
        當日收盤價-上軌值 / 收盤價，做特徵值upperLoss
        當日收盤價-中軌值 / 收盤價，做特徵值middleLoss
        當日收盤價-下軌值 / 收盤價，做特徵值lowerLoss
    2、相對強弱指標(RSI)
    3、平均趨向指標(ADX)
    4、隨機震盪指標(KD)
        K line : 快線
        D line : 慢線
    5、A/D line
    
    交易量相關指標:
    
    6、能量潮指標(OBV)
    
    7、成交量
    
    -----------------------------------------------------
    
    
    
    '''
    # (一) 布林通道
    upper, middle, lower = getBollingerBand(close)
    upperLoss, middleLoss, lowerLoss = [], [], []
    
    #for upperData, middleData, lowerData in zip(upper, middle, lower, ):
    
    # (二) RSI
    df_RSI = getRSI(close)
    
    # (三) ADX
    df_ADX = getADX(high, low, close)
    
    # (四) KD
    slowk, slowd = getKD(high, low, close)
    
    # (五) A/D line
    df_ADL = getADL(high, low, close, volume)
    
    # (六) OBV
    df_OBV = getOBV(close, volume)
    
    # (七) 成交量
    
    
    
    
    
    
    
    
    # # 建立資料表並合併資料
    # name_attribute = ['Date', 'Capacity', 'Turnover', 'Open', 'High', 'Low', 'Close', 'Change', 'Transcation','Class']
    # df = pd.DataFrame(columns = name_attribute, data = target_list)

    # #存取資料
    # filename = f'C:/Users/PC1110223B/Desktop/112-2HW/Fundamental_Enselmble_Learning/project/{stock_id}.csv'
    # df.to_csv(filename)