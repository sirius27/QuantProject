# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

'''
question:
1.sale_signal 就用bool值去回归？
'''
'''
module description: this module provides methods to load data from file,calculate factor value
from them,and format data so that they have proper shape.
'''


def load_file(file_name, sheet_name_list):
    '''
    load xlsx file into a dictionary indexed by sheet names
    :param string file_name:name of file
    :param [string] sheet_name_list: name of selected sheets in the xlsx file
    :return: {string:DataFrame} raw_data: {name of sheet:pure data retrieved from xlsx
    with column and index 0,1,2,...}
    '''
    print 'loading file...'
    cut_head = 2
    file = pd.ExcelFile(file_name)
    raw_data = {}
    # iterate over every sheet and retrieve useful data into raw_data
    for i in range(len(sheet_name_list)):
        print 'parsing sheet', sheet_name_list[i]
        # parse a sheet from the whole file into a DataFrame with headers cut off
        temp = file.parse(sheet_name_list[i]).iloc[cut_head:, :]
        # now temp.dtype = object,because the data read in contains string.Here convert it to float
        temp = temp.astype(np.float)
        # reset index and column with 0,1,2,...,
        temp.columns = range(temp.shape[1])
        temp.index = range(temp.shape[0])
        raw_data[sheet_name_list[i]] = temp
    return raw_data


def pre_processing(raw_data, sheet_name_list):
    '''
    find the first row where more than (threshold_fraction) fraction of elements are non-zero
    and cut off all rows before this one
    :param {string:DataFrame} raw_data: {name of sheet:pure data retrieved from xlsx with column
    and index 0,1,2,...}
    :param [string] sheet_name_list: name of selected sheets in the xlsx file
    :return:{string:DataFrame} data: {name of sheet: data with acceptable zeroes}
    '''
    # find the target row index in close
    template = raw_data['close']
    threshold_fraction = 0.9
    threshold = threshold_fraction * template.shape[1]
    # number of non-zeroe elements in each row
    num_of_non_zero = np.sum(template != 0, axis=1)
    # index of target row
    target_index = np.min(np.where(num_of_non_zero >= threshold)[0])
    # use this row index to prune all sheets
    data = dict()
    for i in range(len(sheet_name_list)):
        # prune
        temp = raw_data[sheet_name_list[i]].iloc[target_index:, :]
        # initialize index with 0,1,2...,otherwise it will be threshold,threshold+1,...
        temp.index = range(temp.shape[0])
        data[sheet_name_list[i]] = temp
    return data


def getReturn(close):
    '''
    calculate log return ratio with close price
    :param DataFrame close:close price
    :return: DataFrame ret:log return ratio
    '''
    # get numerator
    up = close.iloc[1:, :]
    up.index = up.index - 1
    # get denominator
    down = close.iloc[:-1, :]
    daily_return = up / down
    ret = np.log(daily_return)
    # replace null,inf values with 0
    ret.replace([np.inf, np.nan], 0, inplace=True)
    return ret


def getVol(ret):
    '''
    calculate volatility value of log return ratio
    :param DataFrame ret: return value
    :param int interval: interval over which volatility is calculated
    :return: DataFrame standard_error: volatility value
    '''
    print '''*************************************************************************************
    a kind WARNING from the programmer(not the evil interpreter) function getVol:
    we have different values for interval in test code and real code,because the sample file
    may not have sufficient rows for real interval,leading to empty matrix.So be careful of
    the value you choose
    **************************************************************************************
          '''
    # real value
    # interval = 26
    # test value
    interval = 4
    standard_error = pd.rolling_std(ret, interval)
    standard_error.dropna(inplace=True)
    standard_error.index=range(standard_error.shape[0])
    return standard_error


def getKDJ(close, high, low):
    '''
    calculate KDJ value
    :param DataFrame close:close price
    :param DataFrame high:highest price of a day
    :param DataFrame low: lowest price of a day
    :return: [DataFrame,DataFrame,DataFrame,DataFrame] [RSV, K, D, KDJ]:KDJ value and some subproducts
    '''
    # interval over which KDJ is calculated
    kdj_interval = 9
    N = 3
    # calculate RSV
    # get the close value to be used
    close = pd.DataFrame(close.iloc[(kdj_interval - 1):, :].values)
    # calculate maximum in (kdj_interval) days in high value
    high_max_in_interval = pd.rolling_max(high, kdj_interval)
    # rolling_sum function will set the first (kdj_interval-1) days as np.nan,drop them
    high_max_in_interval.dropna(inplace=True)
    # set index with 0,1,2...,otherwise it will be kdj_interval,kdj_interval+1,...(may not be explicit but fuck the index)
    high_max_in_interval.index = range(high_max_in_interval.shape[0])
    low_min_in_interval = pd.rolling_min(low, kdj_interval)
    low_min_in_interval.dropna(inplace=True)
    low_min_in_interval.index = range(low_min_in_interval.shape[0])
    # calculate RSV
    RSV = 100 * (close - low_min_in_interval) / (high_max_in_interval - low_min_in_interval)
    # replace np.nan and np.inf in RSV because there might be 0 in the denominator of the last formula
    RSV.replace([np.nan, np.inf], 0, inplace=True)
    # get matrix shape
    [row, col] = RSV.shape
    # calculate K
    # assuming N equals n in the formula
    # initialize both N and K with 50
    K = pd.DataFrame(np.zeros([row, col]))
    D = pd.DataFrame(np.zeros([row, col]))
    K.iloc[0, :] = 50 * np.ones([1, col])
    D.iloc[0, :] = 50 * np.ones([1, col])
    # calculate K and D iteratively
    for i in range(1, row):
        K.iloc[i, :] = (RSV.iloc[i, :] + K.iloc[(i - 1), :]) / N
        D.iloc[i, :] = (K.iloc[i, :] - D.iloc[(i - 1), :]) / N
    KDJ = 3 * K - 2 * D
    return [RSV, K, D, KDJ]


def getEMA(close):
    '''
    calculate EMA value
    :param DataFrame close: close price
    :return: DataFrame EMA: EMA value
    '''
    print '''*************************************************************************************
    a kind WARNING from the programmer(not the evil interpreter) function getEMA:
    we have different values for n1,n2,n3 in test code and real code,because the sample file
    may not have sufficient rows for real n1,n2,n3,leading to empty matrix.So be careful of
    the value you choose
    **************************************************************************************
          '''
    # real n1,n2,n3
    # n1=12
    # n2=26
    # n3=9
    # n1,n2,n3 for test
    n1 = 3
    n2 = 6
    n3 = 5
    # calculate MA12
    MA12 = pd.rolling_mean(close, n1)
    # drop np.nan in the first (n1-1) rows
    MA12.dropna(inplace=True)
    # set index with 0,1,2...
    MA12.index = range(MA12.shape[0])
    MA26 = pd.rolling_mean(close, n2)
    MA26.dropna(inplace=True)
    MA26.index = range(MA26.shape[0])
    [row, col] = MA26.shape
    DIF = pd.DataFrame(MA12.iloc[(-row):, :].values) - MA26
    tmp = pd.rolling_mean(DIF, n3)
    tmp.dropna(inplace=True)
    tmp.index = range(tmp.shape[0])
    [row, col] = tmp.shape
    DIF = pd.DataFrame(DIF.iloc[(-row):, :].values)
    EMA = DIF - tmp
    return EMA


def getBuySignal(EMA, trade):
    '''
    calculate buy signal
    :param DataFrame EMA: EMA value
    :param DataFrame trade:trade value
    :return: DataFrame(bool) signal:buy or not
    '''
    [row, col] = EMA.shape
    # here trade_copy has one more row than EMA,so when the .diff() function is applied
    # and the first row full of null is dropped,they have the same shape
    trade_copy = trade.iloc[(-(row + 1)):, :]
    trade_increment = trade_copy.diff()
    trade_increment.dropna(inplace=True)
    trade_increment.index = range(trade_increment.shape[0])
    signal_EMA = EMA > 0
    signal_trade = trade_increment > 0
    signal = signal_EMA * signal_trade
    return signal.astype(np.bool)


def getSellSignal(EMA, trade):
    '''
    calculate buy signal
    :param DataFrame EMA: EMA value
    :param DataFrame trade:trade value
    :return: DataFrame(bool) signal:buy or not
    '''
    [row, col] = EMA.shape
    # here trade_copy has one more row than EMA,so when the .diff() function is applied
    # and the first row full of null is dropped,they have the same shape
    trade_copy = trade.iloc[(-(row + 1)):, :]
    trade_increment = trade_copy.diff()
    trade_increment.dropna(inplace=True)
    trade_increment.index = range(trade_increment.shape[0])
    signal_EMA = EMA < 0
    signal_trade = trade_increment < 0
    signal = signal_EMA * signal_trade
    return signal.astype(np.bool)


def getRSI(close):
    '''
    calculate RSI value
    :param DataFrame close: close price
    :return: DataFrame RSI: RSI value
    '''
    n = 3
    # calculate increment of close price of two succeeding days
    close_increment = close.diff()
    close_increment.dropna(inplace=True)
    close_increment.index = range(close_increment.shape[0])
    close_pos = close_increment.copy()
    close_pos[close_pos < 0] = 0
    close_abs = np.abs(close_increment)
    sum_pos = pd.rolling_sum(close_pos, n)
    sum_pos.dropna(inplace=True)
    sum_pos.index = range(sum_pos.shape[0])
    sum_abs = pd.rolling_sum(close_abs, n)
    sum_abs.dropna(inplace=True)
    sum_abs.index = range(sum_abs.shape[0])
    RSI = sum_pos / sum_abs
    RSI.replace([np.nan, np.inf], 0, inplace=True)
    return RSI


def getMTM(close):
    '''
    calculate MTM value
    :param DataFrame close: close price
    :return: DataFrame MTM: MTM value
    '''
    print '''*************************************************************************************
    a kind WARNING from the programmer(not the evil interpreter) function getEMA:
    we have different values for interval in test code and real code,because the sample file
    may not have sufficient rows for real interval leading to empty matrix.So be careful of
    the value you choose
    **************************************************************************************
    '''
    # real value
    # interval=9
    #test value
    interval=3
    MTM = close.diff(interval)
    MTM.dropna(inplace=True)
    MTM.index = range(MTM.shape[0])
    return MTM


def clean_data(file_name,index_list):
    '''
    从文件读取数据并清理
    :param string file_name: xlsx文件路径
    :param [string] index_list: 原始指标名列表
    :return: [{string:DataFrame},DataFrame] [factor_data,ret]: 所用的每个指标的数据，各自放在一个DataFrame中，
    每个DataFrame的[i,j]元素是在第(i+1)天第(j+1)只股票在这个指标上的值.并且用相同的方法对ret进行裁剪，以便回归
    '''
    # raw_data:all pure data from file
    raw_data = load_file(file_name, index_list)
    # data:data with acceptable amount of zeroes
    data = pre_processing(raw_data, index_list)
    # close:close value as factor
    close = data['close']
    # trade:trade value as factor
    trade = data['trade']
    # ret:return value as factor
    ret = getReturn(data['close'])
    # vol:return volatility as factor
    vol = getVol(ret)
    # KDJ:KDJ value as factor
    [RSV, K, D, KDJ] = getKDJ(close, data['high'], data['low'])
    # ema:EMA value as factor
    EMA = getEMA(close)
    # buy_signal:buy or not?It's a signal,as factor
    buy_signal = getBuySignal(EMA, trade)
    # sell_signal:another signal,as factor
    sell_signal = getSellSignal(EMA, trade)
    # rsi:RSI value as factor
    RSI = getRSI(close)
    # mtm:mtm value as factor
    MTM = getMTM(close)
    # 将计算出来的指标存入字典，并找出其最小行数
    unpruned_factor_data = {'KDJ': KDJ, 'EMA': EMA, 'vol': vol, 'MTM': MTM, 'buy_signal': buy_signal,
                   'sale_signal': sell_signal, 'trade': trade, 'RSI': RSI}
    min_row_number = unpruned_factor_data['KDJ'].shape[0]
    for i in unpruned_factor_data.items():
        if i[1].shape[0] < min_row_number:
            min_row_number = i[1].shape[0]
    factor_data=dict()
    # 从后往前取min_row_number行，对每个指标的数据进行裁剪
    for i in unpruned_factor_data.items():
        temp=i[1].iloc[(-min_row_number):,:]
        temp.index=range(temp.shape[0])
        factor_data[i[0]]=temp
    ret=ret.iloc[(-min_row_number):,:]
    ret.index=range(ret.shape[0])
    ret.to_csv('return.csv')
    factor_data['KDJ'].to_csv('kdj.csv')
    return [factor_data,ret]