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
    stock_id = '2359' # 本實驗以所羅門(2359)做研究對象
    stock = Stock(stock_id)
    stock_name = twstock.codes[stock_id].name
    print(f"股票: {stock_name} ({stock_id}) 資料開始匯出 ")
    
    data = stock.fetch_from(2020,6) # 擷取2020/6 至今該股票資料
    
    #將股票最高價、最低價、收盤價、成交量等資訊紀錄於獨立的list，方便使用於股票指標計算
    high = [] #當日最高價
    low = [] #當日最低價
    close = [] #收盤價
    volume = [] #成交量
    change = [] #漲跌幅
    
    for item in data:
        high.append(item.high)
        low.append(item.low)
        close.append(item.close)
        volume.append(float(item.capacity))
        change.append(item.change)
    close_pred = close  # 比較隔日股價用 
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    volume = np.array(volume)
    change = np.array(change)
    '''
    使用以下股票指標作為特徵:
    基本指標: 
    1、最高價
    2、最低價
    3、收盤價
    4、成交量
    
    股價相關指標: 
    5 6 7、布林通道:
        產生上軌、中軌、下軌等三種軌道線，模擬常態分佈假設，
        將三種指標分別加入特徵列表中
    8、相對強弱指標(RSI)
    9、平均趨向指標(ADX)
    10 11、隨機震盪指標(KD)
        K line : 快線
        D line : 慢線
    
    交易量相關指標:
    
    12、A/D line
    13、能量潮指標(OBV)
    
    最後整理特徵列表時，發現平均趨向指標(ADX)，前27筆資料作為計算值，沒有準確的資料值
    因此，將所有指標捨棄前27筆資料。
    -----------------------------------------------------
    14、Class:
    依據當日是否收紅做分類，若當日收盤價大於等於前日收盤價，則回傳 1 
    若小於前日收盤價則回傳 0
    
    '''
    # (五 六 七) 布林通道
    upper, middle, lower = getBollingerBand(close)
    
    # (八) RSI 
    df_RSI = getRSI(close) 
    
    # (九) ADX
    df_ADX = getADX(high, low, close)
   
    # (十 十一) KD
    slowk, slowd = getKD(high, low, close)
    
    # (十二) A/D line
    df_ADL = getADL(high, low, close, volume)
    
    # (十三) OBV
    df_OBV = getOBV(close, volume)
    
    # (十四) Class
    classList = []
    close_pred.pop(0)#移項
    close_pred.append(0)
    #今日收盤價和明日收盤價比較，若上漲，則回傳 1
    for i in range(len(close)):
        if close[i] > close_pred[i]:
            classList.append(0)
        else:
            classList.append(1)
    classList = np.array(classList)
    #整理特徵列表
    ##########################################
    data_num = len(data)
    
    # 建立資料表並合併資料
    df = pd.DataFrame({"high":high[27:data_num],
                       'low':low[27:data_num],
                       'close':close[27:data_num],
                       'volume':volume[27:data_num],
                       'upper':upper[27:data_num],
                       'middle':middle[27:data_num],
                       'lower':lower[27:data_num],
                       'RSI':df_RSI[27:data_num],
                       'ADX':df_ADX[27:data_num],
                       'slowk':slowk[27:data_num],
                       'slowd':slowd[27:data_num],
                       'ADL':df_ADL[27:data_num],
                       'OBV':df_OBV[27:data_num],
                       'Class':classList[27:data_num],
                       })
    
    #存取資料
    filename = f'C:/Users/PC1110223B/Desktop/stock_pred/Ensemble_learning_StockPedict/{stock_id}.txt'
    df.to_csv(filename, sep=' ', index=False, header = False)