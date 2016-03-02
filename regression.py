# -*- coding: utf-8 -*-
from tech import *
from macro import *
import industryA
import industryB

if __name__ == '__main__':
    '''
    1.1 计算技术指标的暴露矩阵和因子收益率
    '''
    tech_fname = 'hushen_tech.xlsx'
    tecnical_index_list = ['close', 'high', 'low', 'trade', 'growth', 'ev']
    # 指标值和收盘价
    [close, data, ret] = clean_data(tech_fname, tecnical_index_list)
    # factor used for regression:['buy_signal', 'EMA', 'RSI',
    # 'KDJ', 'vol', 'trade', 'sell_signal', 'MTM','ev','William']
    [tech_significant_list, tech_loading] = fb_reg_over_time(ret, data)
    print tech_significant_list
    # 选择被判断为显著的因子进行第二步回归
    tech_loading = tech_loading[['buy_signal', 'RSI', 'KDJ', 'vol', 'trade', 'sell_signal', 'MTM','William']]
    tech_factor_return = fb_reg_over_stock(tech_loading, ret)
    print tech_factor_return.shape


    '''
    1.2 通过已知的宏观指标的收益率计算暴露矩阵
    '''


    macro_fname='macro.xlsx'
    macro_index_list=['Ind_growth','CPI','Ex_inport','Deficit','M2','Conf','USD','Mkt_return']
    macro_data=load_macro_index(macro_fname,macro_index_list)
    [macro_loading,macro_significant_list]=macro_regression(ret,macro_data,macro_index_list)
    print macro_significant_list
    # 根据显著股票数，选取其中的CPI,Mkt_return因子
    macro_loading=macro_loading[['CPI','Mkt_return']]
    # 同样的方法选取macro_data中显著因子对应的因子收益率
    macro_factor_return=macro_data[['CPI','Mkt_return']]


    '''
    1.3A (行业因子方法A)从excel导入已经在matlab中算好的行业因子暴露矩阵
    '''


    industry_fname='industry.xlsx'
    industry_list=['Caijue','Huagong','Gangtie','Yousejinshu','Jiancai','Jianzhuzhuangshi','Dianqishebei',
    'Jixieshebei','Guofangjungong','Qiche','Jiadian','Fangzhifuzhuang','Qinggongzhizao','Shangyemaoyi',
    'Nonglinmuyu','Shipinyinliao','Xiuxianfuwu','Yiliaoshengwu','Gonggongshiye','Jiaotongyunshu',
    'Fangdichan','Dianzi','Jisuanji','Chuanmei','Tongxin','Yinhang','Feiyinhangjinrong']
    ind_loading=industryA.load_industry_loading(industry_fname,industry_list)
    [ind_significant_list,ind_factor_return]=industryA.reg_industry(ret,ind_loading,industry_list)
    print ind_significant_list
    # 保留显著天数大于50天的
    reserve_ind_index_list=[]
    for i in range(len(industry_list)):
        if ind_significant_list[industry_list[i]]>50:
            reserve_ind_index_list=reserve_ind_index_list+[industry_list[i]]
    ind_factor_return=ind_factor_return[reserve_ind_index_list]
    ind_loading=ind_loading[reserve_ind_index_list]


    # '''
    # 1.3B (行业因子方法B)读入的已知的行业收益率作为自变量，股票收益率作为因变量进行回归得到因子暴露矩阵
    # '''


    # industry_close_fname='industry_close.xlsx'
    # industry_list=['801710.SI','801711.SI','801712.SI','801713.SI','801720.SI','801721.SI','801722.SI',
    #  '801723.SI','801724.SI','801725.SI','801730.SI','801731.SI','801732.SI','801733.SI',
    #  '801734.SI','801740.SI','801741.SI','801742.SI','801743.SI','801744.SI','801750.SI',
    #  '801751.SI','801752.SI','801760.SI','801761.SI','801770.SI','801780.SI','801790.SI',]


    # '''
    # 1.3C (行业因子方法C)从excel导入已经在matlab中算好的行业因子暴露矩阵
    # '''
    #
    #
    # industry_loading_fname='industry.xlsx'
    # industry_close_fname='industry_close.xlsx'
    # # 在industry.xlsx中，因子（列）的顺序是industry_list描述的
    # # 在industry_close.xlsx中，因子（列）的顺序是industry_code描述的
    # # 这两个顺序并不一样，其对应关系保存在 code_industry_pair.txt 中
    # #首先按照各自的顺序把因子暴露矩阵和行业收益率矩阵读进来
    # industry_list=['Caijue','Huagong','Gangtie','Yousejinshu','Jiancai','Jianzhuzhuangshi','Dianqishebei',
    # 'Jixieshebei','Guofangjungong','Qiche','Jiadian','Fangzhifuzhuang','Qinggongzhizao','Shangyemaoyi',
    # 'Nonglinmuyu','Shipinyinliao','Xiuxianfuwu','Yiliaoshengwu','Gonggongshiye','Jiaotongyunshu',
    # 'Fangdichan','Dianzi','Jisuanji','Chuanmei','Tongxin','Yinhang','Feiyinhangjinrong']
    #
    # industry_code=['801710.SI','801711.SI','801712.SI','801713.SI','801720.SI','801721.SI','801722.SI',
    #  '801723.SI','801724.SI','801725.SI','801730.SI','801731.SI','801732.SI','801733.SI',
    #  '801734.SI','801740.SI','801741.SI','801742.SI','801743.SI','801744.SI','801750.SI',
    #  '801751.SI','801752.SI','801760.SI','801761.SI','801770.SI','801780.SI','801790.SI',]
    # # 读取因子暴露矩阵
    # ind_loading=industryA.load_industry_loading(industry_loading_fname,industry_list)
    # # 读取行业收盘价
    # ind_close=industryB.load_industry_close(industry_close_fname,industry_code)
    # # 按照因子暴露矩阵的列顺序来重新排列收盘价的列，并修改列名
    # # 从收盘价计算行业收益率并截取其中最后192行，即从2000年开始的数据

    '''
    2.1 合并各类因子的暴露矩阵和收益率，计算delta
    '''
    # 合并因子暴露矩阵
    X=pd.concat([tech_loading,macro_loading,ind_loading],axis=1)
    print X.shape
    # 合并收益率
    factor_return=pd.concat([tech_factor_return,macro_factor_return,ind_factor_return],axis=1)
    # 计算协方差
    V = np.cov(ret.transpose().values)
    F = np.cov(factor_return.transpose().values)
    explained_cov = X.values.dot(np.dot(F, X.transpose().values))
    delta = pd.DataFrame(V - explained_cov)
    delta.to_csv('E:\\QuantProject\\result_demo\\delta.csv')