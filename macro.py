# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm


def load_macro_index(fname, index_list):
    '''
    从xlsx文档中读取原始数据并存于一个DataFrame
    :param string fname: 文件路径
    :param [string]factor_list: 宏观因子的名字
    :return: DataFrame macro_index
    '''
    macro_index = pd.read_excel('macro.xlsx')
    return macro_index


def macro_regression(ret, macro_index,macro_index_list):
    '''
    抽取每一只股票的收益率时间序列，与每个宏观因子在这段时间的收益率做单元回归，回归系数就是这只股票在
    这个宏观因子上的暴露。将所得结果排成一个大小为股票数*宏观因子数的矩阵，横向拼入股票-因子暴露矩阵
    :param DataFrame ret: 每只股票每天的收益率
    :param DataFrame macro_index: 每天每个宏观因子的值
    :return: [DataFrame,dict] [loading,significant_list]: 每只股票在每个宏观因子上的暴露;每个因子显著的股票数
    '''
    loading = pd.DataFrame(np.zeros([ret.shape[1], macro_index.shape[1]]), columns=macro_index.columns)
    # 分别记录每个因子在多少只股票上面显著
    significant_list = dict()
    # 初始化显著数
    for i in range(len(macro_index_list)):
        significant_list[macro_index_list[i]] = 0
    # p值小于这个的被认为显著
    pvalue_threshold = 0.1
    # 选取一只股票用于回归
    for i in range(ret.shape[1]):
        y = ret.values[:, i]
        # 用每一个因子去回归股票收益率
        for j in range(len(macro_index_list)):
            x = macro_index.values[:, j]
            model = sm.OLS(y, x).fit()
            loading.iloc[i, j] = model.params[0]
            if model.pvalues < pvalue_threshold:
                significant_list[macro_index_list[j]] += 1
    return [loading, significant_list]