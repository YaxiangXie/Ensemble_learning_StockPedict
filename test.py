import numpy as np

import twstock
from twstock import Stock
from twstock import BestFourPoint
# https://pypi.org/project/twstock/ twstock 教學

stock_id = input("請輸入股票代碼: ")

stock = Stock(stock_id)

ma_p = stock.moving_average(stock.price, 5)  #五日均價
print(ma_p)
ma_c = stock.moving_average(stock.capacity, 5) #五日均量

ma_p_cont = stock.continuous(ma_p) #五日均價持續天數

ma_br = stock.ma_bias_ratio(5, 10) #五日、十日乖離值

stock.fetch_from(2024, 5) #2024/3 至今之資料

stock.price

stock.capacity

stock.data[0]

#stock info
twstock.codes[stock_id] #stock all info
twstock.codes[stock_id].name
twstock.codes[stock_id].start

#四大買賣點分析
bfp = BestFourPoint(stock)

bfp.best_four_point_to_buy()    # 判斷是否為四大買點
bfp.best_four_point_to_sell()   # 判斷是否為四大賣點
bfp.best_four_point()           # 綜合判斷

#即時股票資訊查詢
twstock.realtime.get('2330')    # 擷取當前台積電股票資訊
twstock.realtime.get(['2330', '2337', '2409'])  # 擷取當前三檔資訊