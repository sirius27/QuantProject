clc;
clear all;
%将数据导入结构体
data=load('E:\金融\量化课题\Raw_Hushen_Hangqing');
%提取结构体的所有成员，即矩阵的名字
names=fieldnames(data);
len=size(names);
data_index=[];
codes_index=[];
fields_index=[];
%找出名字中含有'data'的成员变量在names中的索引
for i=1:len
    containData=regexp(names(i),'.+data.+');
    if (containData{1}==1)
        data_index=[data_index,i];
    end
end
%按照上面的索引从names中提取出所有含有'data'的成员变量
datanames=names(data_index);
%全部改成小写
filenames=lower(data.w_wsd_fields_0);
file_num=size(filenames,1);
data_num=size(datanames,1);
%对于每个成员变量，从第一列开始提取出每个矩阵中的同一列（也就是同一个指标的），拼成一个新的矩阵
%新的矩阵就是某一个指标每天（一行）在每个股票上（一列）的数据
for i =1:file_num
    temp=[];
    for j =1:data_num
        mat_name=datanames(j);
        mat=eval(['data.',mat_name{1}]);
        temp=[temp,mat(:,i)];
    end
    %将这个拼好的新矩阵暂存到files中去
    files{i}=temp;
end

for i = 1:file_num-1
    xlswrite('E:\金融\量化课题\hushen_tech.xlsx',files{i},filenames{i},'B4')
end