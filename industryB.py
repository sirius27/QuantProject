# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import tech
def load_industry_close(fname,ind_code):
    '''
    从已有的文件中读取行业收盘价矩阵，计算收益率，取其中的最后192行
    :param string fname: 文件路径
    :param [string] ind_code: 因子代码的列表
    :return: DataFrame ind_return: 因子收盘价，月数*行业数
    '''
    ind_return=pd.read_excel(fname)
    ind_return.columns=ind_code
    return ind_return