quant project
procedure:
1.运行regression.py 从hushen_tech.xlsx中按sheet读取['close', 'high', 'low', 'trade', 'growth', 'ev']并计算技术指标
在运行过程中，tech.procedure函数产生template.txt文件，这个文件是按照2000年收盘价非0的准则筛选出的列（下标从1开
始以方便matlab的create_dummy.m读取）
2.运行create_dummy.m脚本 读入Raw_Hushen_Hangqing.mat文件，创建industry.xlsx用于存放dummy矩阵（即每个股票所属的行业
）用作下一次regression.py中的行业因子暴露矩阵
3.再次运行regression.py