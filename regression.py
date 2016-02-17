# -*- coding: utf-8 -*-
from preprocessing import *
import statsmodels.api as sm

def fb_reg_over_time(ret, data):
    '''
    判断每个因子对收益率的鲜猪肚，取
    :param DataFrame ret: 收益率
    :param {string:DataFrame} data: 每个因子相关的数据
    :return: [int,DataFrame] [significant_futures_list, X]: 每个因子在几个股票上显著？;因子暴露矩阵
    '''
    # X用于记录因子暴露（以回归斜率来刻画），X[i,j]是股票(j+1)的收益率在因子(i+1)上的暴露(row个因子，col个股票)
    X = np.zeros([len(data), ret.shape[1]])
    # 判断鲜猪肚所用的t检验阈值
    tvalue_threshold = 1.96
    significant_futures_list = np.zeros(len(data))
    # num_of_factor是当前正在研究的factor的序号，每个大循环结束之后加1
    num_of_factor=0
    # 对每个因子进行研究,i是一个tuple,i[0]是指标名，i[1]是一个DataFrame，存有[某一天,某个股票]这个因子的值
    for i in data.items():
        # 将这个因子显著的股票数目初始化为0
        significant_futures = 0
        for j in range(i[1].shape[1]):
            model = sm.OLS(ret[j].values, i[1][j].values).fit()
            # 用回归的斜率来表征因子暴露
            X[num_of_factor, j] = model.params[0]
            # 如果在这个股票上显著，就加1
            if abs(model.tvalues) > tvalue_threshold:
                significant_futures+=1
        # 将这个因子显著的股票数目放入列表相应的位置中
        significant_futures_list[num_of_factor] = significant_futures
        num_of_factor+=1
    # 把X转为DataFrame方便处理
    X = pd.DataFrame(X)
    return [significant_futures_list, X]
def fb_reg_over_stock(overloading,ret):
    '''
    对这个的作用并不是很清楚
    :param DataFrame overloading: 因子暴露矩阵，overload[i,j]是第(j+1)个股票在第(i+1)个因子上的暴露
    :param DataFrame ret: 收益率，ret[i,j]是第(j+1)个股票在第(i+1)天的收益率
    :return:
    '''
    factor_return=np.zeros([ret.shape[0],overloading.shape[0]])
    significant_days_list=np.zeros(overloading.shape[0])
    tvalue_threshold = 1.96
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
if __name__ == '__main__':
    fname = 'technical_analysis.xlsx'
    tecnical_index_list = ['close', 'high', 'low', 'trade', 'growth', 'volume', 'PE']
    # 指标值和收盘价
    [data,ret]=clean_data(fname,tecnical_index_list)
    # 收盘价
    # factor used for regression:['buy_signal', 'EMA', 'RSI', 'KDJ', 'vol', 'trade', 'sale_signal', 'MTM']
    [significant_futures_list, stock_overload_on_factor]=fb_reg_over_time(ret,data)
    [significant_days_list,factor_return]=fb_reg_over_stock(stock_overload_on_factor,ret)
    print significant_futures_list
    print significant_days_list
    print stock_overload_on_factor