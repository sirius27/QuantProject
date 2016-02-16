# -*- coding: utf-8 -*-
# import packages
import pandas as pd
import numpy as np
from pandas import DataFrame
import statsmodels.api as sm
from sklearn.decomposition import RandomizedPCA

'''
记录：1.np.std的公式是sqrt(sum(x-mean(x))**2/n)
       pd.rolling_std的公式是sqrt(sum(x-mean(x))**2/(n-1))
     2.
'''


def getClose(df, row_start, column_start, column_end):
    str_close = df.values[(row_start - 2):, (column_start - 1):column_end]
    close = str_close.astype(np.double)
    isnotzero = close != 0
    not_zero_num = np.sum(isnotzero, axis=1)
    threshold = 0.9 * close.shape[1]
    less_zero_than_threshold = not_zero_num > threshold
    first_not_zero = np.min(np.where(less_zero_than_threshold == True))
    close = close[first_not_zero:, :]
    # return the line number of the first lien with >90% non_zero term
    close = pd.DataFrame(close)
    return [close, first_not_zero + 1]


def getData(df, row_start, first_not_zero, column_start, column_end):
    data = df.values[(row_start - 2):, (column_start - 1):column_end]
    data = data[(first_not_zero - 1):, :]
    data = pd.DataFrame(data)
    return data


def getReturn(close):
    [row, col] = close.shape
    ret = np.zeros([row, col])
    up = close.iloc[1:, :].reindex()
    up.index = up.index - 1
    down = close.iloc[:-1, :]
    daily_return = up / down
    ret = np.log(daily_return)
    ret[ret.isnull()] = 0
    ret[np.isinf(ret)] = 0
    return ret


def getVol(ret, interval):  # interval is the period where standard error is measured
    interval=26
    standard_error = pd.rolling_std(ret, interval)
    standard_error = pd.DataFrame(standard_error.dropna(axis=0).values)
    return standard_error


def getKDJ(close, high, low, kdj_interval, N):
    # calculate RSV
    [row, col] = close.shape
    close = pd.DataFrame(close.iloc[(kdj_interval - 1):, :].values)
    # shape of RSV matrix after cutting kdj_interval in row number
    high_max_in_interval = pd.rolling_max(high, kdj_interval)
    low_min_in_interval = pd.rolling_min(low, kdj_interval)
    high_max_in_interval = pd.DataFrame(high_max_in_interval.fillna(0).values)
    low_min_in_interval = pd.DataFrame(low_min_in_interval.fillna(0).values)
    # calculate RSV
    RSV = 100 * (close - low_min_in_interval) / (high_max_in_interval - low_min_in_interval)
    RSV[RSV.isnull()] = 0
    RSV[np.isinf(RSV)] = 0
    # update matrix shape
    [row, col] = RSV.shape
    # calculate K
    # assuming N equals n in the formula
    # initialize both N and K with 50
    K = pd.DataFrame(np.zeros([row, col]))
    D = pd.DataFrame(np.zeros([row, col]))
    K.iloc[0, :] = 50 * np.ones([1, col])
    D.iloc[0, :] = 50 * np.ones([1, col])
    for i in range(1, row):
        K.iloc[i, :] = (RSV.iloc[i, :] + K.iloc[(i - 1), :]) / N
        D.iloc[i, :] = (K.iloc[i, :] - D.iloc[(i - 1), :]) / N
    KDJ = 3 * K - 2 * D
    return [RSV, K, D, KDJ]


def getEMA(close, n1=12, n2=26, n3=9):
    MA12 = pd.rolling_mean(close, n1)
    MA12 = pd.DataFrame(MA12.fillna(0).values)
    MA26 = pd.rolling_mean(close, n2)
    MA26 = pd.DataFrame(MA26.fillna(0).values)
    [row, col] = MA26.shape
    DIF = pd.DataFrame(MA12.iloc[(-row):, :].values) - MA26
    tmp = pd.rolling_mean(DIF, n3)
    tmp = pd.DataFrame(tmp.fillna(0).values)
    [row, col] = tmp.shape
    DIF = pd.DataFrame(DIF.iloc[(-row):, :].values)
    EMA = DIF - tmp
    return EMA



def getTrade(df, row_start, first_not_zero, trade_start, trade_end):
    trade = getData(df, row_start, first_not_zero, trade_start, trade_end)
    trade=trade.astype(np.float)
    return trade


def getBuySignal(EMA, trade):
    [row, col] = EMA.shape
    trade_copy = trade.iloc[(-(row + 1)):, :]
    trade_increment = pd.DataFrame(trade_copy.diff().fillna(0).values)
    signal_EMA = EMA > 0
    signal_trade = trade_increment > 0
    signal = signal_EMA * signal_trade
    return signal.astype(np.bool)


def getSaleSignal(EMA, trade):
    [row, col] = EMA.shape
    trade_copy = trade.iloc[(-(row + 1)):, :]
    trade_increment = pd.DataFrame(trade_copy.diff().fillna(0).values)
    signal_EMA = EMA < 0
    signal_sale = trade_increment < 0
    signal = signal_EMA * signal_sale
    return signal.astype(np.bool)


def getRSI(close, n=3):
    close_increment = pd.DataFrame(close.diff().fillna(0).values)
    close_pos = close_increment.copy()
    close_neg = close_increment.copy()
    close_pos[close_pos < 0] = 0
    close_abs = np.abs(close_increment)
    sum_pos = pd.DataFrame(pd.rolling_sum(close_pos, n).fillna(0).values)
    sum_abs = pd.DataFrame(pd.rolling_sum(close_abs, n).fillna(0).values)
    RSI = sum_pos / sum_abs
    RSI=pd.DataFrame(RSI.fillna(0).values)
    return RSI


def getMTM(close):
    MTM = pd.DataFrame(close.diff(3).dropna().values)
    return MTM

def fb_reg_over_time(ret,factor_list):
    X=np.zeros([len(factor_list),ret.shape[1]])
    #X[i,j]是期货j的收益率在因子i上的暴露
    tvalue_threshold = 1.96
    significant_futures_list = np.zeros(len(factor_list))
    # for i in range(len(factor_list)):
    for i in range(len(factor_list)):
        data = factor_list[i]
        [row, col] = data.shape
        significant_futures = 0
        for j in range(col):
            model = sm.OLS(ret[j].values, data[j].values).fit()
            X[i,j]=model.params[0]
            if abs(model.tvalues) > tvalue_threshold:
                significant_futures = significant_futures+1
        significant_futures_list[i] = significant_futures
    X=pd.DataFrame(X)
    return [significant_futures_list,X]
def fb_reg_over_future(overloading,ret):
    factor_return=np.zeros([ret.shape[0],overloading.shape[0]])
    significant_days_list=np.zeros(overloading.shape[0])
    tvalue_threshold = 1.96
    #normalization
    # ret=ret.transpose()
    # ret=(ret-np.mean(ret,axis=0))/np.std(ret,axis=0)
    # ret=ret.transpose()
    # overloading=overloading.transpose()
    # overloading=(overloading-np.mean(overloading,axis=0))/np.std(overloading,axis=0)
    # overloading=overloading.transpose()

    for i in range(overloading.shape[0]):
        significant_days=0
        for j in range(ret.shape[0]):
            model=sm.OLS(ret.iloc[j,:].values,overloading.iloc[i,:].values).fit()
            factor_return[j,i]=model.params[0]
            if abs(model.tvalues) > tvalue_threshold:
                significant_days=significant_days+1
        significant_days_list[i]=significant_days
    factor_return=pd.DataFrame(factor_return)
    return [significant_days_list,factor_return]
# def reg_over_future(ret,factor_list):
#     factor_return=np.zeros([ret.shape[0],len(factor_list)])
#     significant_days_list = np.zeros(len(factor_list))
#     tvalue_threshold = 1.96
#     # factor_return.shape is 280*8,for example
#     for i in range(len(factor_list)):
#         data=factor_list[i]
#         # normalization
#         data=data.transpose()
#         data=(data-np.mean(data,axis=0))/np.std(data,axis=0)
#         data=data.transpose()
#         significant_days = 0
#         for j in range(ret.shape[0]):
#             model=sm.OLS(ret.iloc[j].values,data.iloc[j].values).fit()
#             factor_return[j,i]=model.params[0]
#             if model.tvalues > tvalue_threshold:
#                 significant_days = significant_days+1
#         significant_days_list[i] = significant_days
#     return [significant_days_list,factor_return]
if __name__ == '__main__':
    # start row number of valid data
    row_start = 5
    # start & end columns for valid close data
    close_start = 198
    close_end = 246
    # start & end columns for valid high data
    high_start = 100
    high_end = 148
    # start & end columns for valid low data
    low_start = 149
    low_end = 197
    # start & end columns for valid trade data
    trade_start = 296
    trade_end = 344
    # interval of standard error of close data
    close_interval = 26
    # interval and N of KDJ data
    kdj_interval = 9
    kdj_N = 3
    df = pd.read_csv('New_Data.csv')
    df.ix[0] = 0
    df.ix[0] = 0
    # first_not_zero is the first LINE NUMBER of >90% non zero line
    [close, first_not_zero] = getClose(df, row_start, close_start, close_end)
    ret = getReturn(close)
    vol = getVol(ret, close_interval)
    # x is dummy receipt of the first non_zero line number
    high = getData(df, row_start, first_not_zero, high_start, high_end)
    low = getData(df, row_start, first_not_zero, low_start, low_end)
    [RSV, K, D, KDJ] = getKDJ(close, high, low, kdj_interval, kdj_N)
    EMA = getEMA(close)
    trade = getTrade(df, row_start, first_not_zero, trade_start, trade_end)
    buy_signal = getBuySignal(EMA, trade)
    sale_signal = getSaleSignal(EMA, trade)
    RSI = getRSI(close)
    MTM = getMTM(close)
    minimum_size = min(close.shape, ret.shape, vol.shape, vol.shape, EMA.shape, RSI.shape
                       , KDJ.shape, trade.shape, buy_signal.shape, sale_signal.shape, MTM.shape)
    close = close.iloc[(-minimum_size[0]):, :]
    ret = ret.iloc[(-minimum_size[0]):, :]
    vol = vol.iloc[(-minimum_size[0]):, :]
    EMA = EMA.iloc[(-minimum_size[0]):, :]
    RSI = RSI.iloc[(-minimum_size[0]):, :]
    KDJ = KDJ.iloc[(-minimum_size[0]):, :]
    trade = trade.iloc[(-minimum_size[0]):, :]
    buy_signal = buy_signal.iloc[(-minimum_size[0]):, :]
    sale_signal = sale_signal.iloc[(-minimum_size[0]):, :]
    MTM = MTM.iloc[(-minimum_size[0]):, :]
    # time series regression
    index_list = ['KDJ', 'EMA', 'vol', 'MTM', 'buy_signal', 'sale_signal', 'trade', 'RSI']
    factor_list = [KDJ, EMA, vol, MTM, buy_signal, sale_signal, trade, RSI]
    [significant_futures_of_factor,future_overload_on_factor]=fb_reg_over_time(ret,factor_list)
    # [significant_days_of_factor,factor_return]=reg_over_future(ret,factor_list)
    [significant_days_list,factor_return]=fb_reg_over_future(future_overload_on_factor,ret)