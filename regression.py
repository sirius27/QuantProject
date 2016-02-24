# -*- coding: utf-8 -*-
from preprocessing import *
import statsmodels.api as sm
from macro_process import *


def fb_reg_over_time(ret, data):
    '''
    用每只股票在一段时间的收益率与这只股票某个因子在这段时间的值做回归，将回归系数每只股票收益率在每个因子上的暴露。
    并统计每个因子在多少只股票上显著
    :param DataFrame ret: 收益率
    :param {string:DataFrame} data: 每个因子相关的数据
    :return: [{string:int},DataFrame] [significant_futures_list, X]: 每个因子在几个股票上显著？;因子暴露矩阵
    '''
    # 标准化
    # for i in data.items():
    #     data[i[0]]=(i[1]-i[1].mean())/i[1].std()
    # ret=(ret-ret.mean())/ret.std()
    # X用于记录因子暴露（以回归斜率来刻画），X[i,j]是股票(i+1)的收益率在因子(j+1)上的暴露(row个股票，col个因子)
    X = np.zeros([ret.shape[1], len(data)])
    # 判断鲜猪肚所用的t检验阈值
    tvalue_threshold = 1.96
    significant_futures_list = dict()
    # num_of_factor是当前正在研究的factor的序号，每个大循环结束之后加1
    num_of_factor = 0
    # name of factors,prepared for converting X to DataFrame,with columns=factor_name
    factor_name = []
    # 对每个因子进行研究,i是一个tuple,i[0]是指标名，i[1]是一个DataFrame，存有[某一天,某个股票]这个因子的值
    for i in data.items():
        factor_name = factor_name + [i[0]]
        # 将这个因子显著的股票数目初始化为0
        significant_futures = 0
        for j in range(i[1].shape[1]):
            # 取第j个股票在所有时间的收益率与它的因子值进行回归
            model = sm.OLS(ret[j].values, i[1][j].values).fit()
            # 用回归的斜率来表征因子暴露
            if model.params[0] == np.nan:
                print i[0], j
            X[j, num_of_factor] = model.params[0]
            # 如果在这个股票上显著，就加1
            if abs(model.tvalues) > tvalue_threshold:
                significant_futures += 1
        # 将这个因子显著的股票数目放入列表相应的位置中
        significant_futures_list[i[0]] = significant_futures
        num_of_factor += 1
    # 把X转为DataFrame方便处理
    X = pd.DataFrame(X)
    X.fillna(0, inplace=True)
    X.columns = factor_name
    return [significant_futures_list, X]


def fb_reg_over_stock(loading, ret):
    '''
    固定的loading(股票数*因子数)，对每一天的股票收益率进行回归，得到每一天的因子收益率，
    :param DataFrame loading: 因子暴露矩阵，overload[i,j]是第(i+1)个股票在第(j+1)个因子上的暴露
    :param DataFrame ret: 收益率，ret[i,j]是第(j+1)个股票在第(i+1)天的收益率
    :return:
    '''
    print loading
    factor_return = np.zeros([ret.shape[0], loading.shape[1]])
    significant_days_list = dict()
    tvalue_threshold = 1.96
    # 取第j+1天的收益率进行回归
    for j in range(ret.shape[0]):
        model = sm.OLS(ret.iloc[j, :].values, loading.values).fit()
        # 回归系数是每一个
        factor_return[j] = model.params
        significant_days_list[np.where(abs(model.params) > tvalue_threshold)] += 1
    factor_return = pd.DataFrame(factor_return)
    return [significant_days_list, factor_return]


if __name__ == '__main__':
    fname = 'hushen_tech.xlsx'
    tecnical_index_list = ['close', 'high', 'low', 'trade', 'growth', 'ev']
    # 指标值和收盘价
    [close, data, ret] = clean_data(fname, tecnical_index_list)
    # factor used for regression:['buy_signal', 'EMA', 'RSI', 'KDJ', 'vol', 'trade', 'sell_signal', 'MTM','ev']
    [significant_futures_list, stock_load_on_factor] = fb_reg_over_time(ret, data)
    # 选择被判断为显著的因子进行第二步回归
    print significant_futures_list
    X = stock_load_on_factor['buy_signal', 'RSI', 'KDJ', 'vol', 'trade', 'sell_signal', 'MTM']
    [significant_days_list, factor_return] = fb_reg_over_stock(X, ret)
    print significant_days_list
    # calculating return covariance
    V = np.cov(ret.transpose().values)
    F = np.cov(factor_return.transpose().values)
    explained_cov = stock_load_on_factor.values.dot(np.dot(F, stock_load_on_factor.transpose().values))
    delta = pd.DataFrame(V - explained_cov)
    delta.to_csv('delta.csv')
