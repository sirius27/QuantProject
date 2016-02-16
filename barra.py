# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def load_file(file_name, sheet_name_list):
    print 'loading file'
    file = pd.ExcelFile(file_name)
    raw_data = {}
    for i in range(len(sheet_name_list)):
        print 'parsing sheet', sheet_name_list[i]
        temp = file.parse(sheet_name_list[i]).values[2:, :]
        raw_data[sheet_name_list[i]] = pd.DataFrame(temp)
    return raw_data

def pre_processing(raw_data,sheet_name_list):
    template=raw_data['close']
    num_of_zero=sum(template,axis=1)
    for i in range(len(sheet_name_list)):
        num_of_zero=num_of_zero+1
    return 0

def getReturn(close):
    up = close.iloc[1:, :].reindex()
    up.index = up.index - 1
    down = close.iloc[:-1, :]
    daily_return = up / down
    ret = np.log(daily_return)
    ret[ret.isnull()] = 0
    ret[np.isinf(ret)] = 0
    return ret


def getVol(ret, interval):  # interval is the period where standard error is measured
    interval = 26
    standard_error = pd.rolling_std(ret, interval)
    standard_error = pd.DataFrame(standard_error.dropna(axis=0).values)
    return standard_error


def getKDJ(close, high, low):
    kdj_interval = 9
    N = 3
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


if __name__ == '__main__':
    file_name = 'technical_analysis.xlsx'
    tecnical_index_list = ['close', 'high', 'low', 'trade', 'growth', 'volume', 'PE', 'rm']
    raw_data = load_file(file_name, tecnical_index_list)
    print raw_data[tecnical_index_list[3]]
    ret = getReturn(raw_data['close'])
    [RSV, K, D, KDJ] = getKDJ(close, raw_data[tecnical_index_list['high']], raw_data[tecnical_index_list[low]])
