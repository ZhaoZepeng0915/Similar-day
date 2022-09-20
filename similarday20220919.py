#!/usr/bin/env python
# coding: utf-8
# # 导入需要的库

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from IPython.core.interactiveshell import InteractiveShell

plt.rcParams['font.family'] = 'SimHei'  # 防止画图时中文乱码
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
InteractiveShell.ast_node_interactivity = 'all'


# In[2]:


class SimilarCompare(object):
    def __init__(self, data1):
        self.data = data1
        self.df_prepare = pd.DataFrame()
        self.pow_data = pd.DataFrame()
        self.temp_data = pd.DataFrame()
        self.temp_corr_compare = pd.DataFrame()
        self.hum_corr_compare = pd.DataFrame()
        self.temp_corr_max_date = str()
        self.hum_corr_max_date = str()
        self.similar_data_power = pd.DataFrame()
        self.temp_corr = 0
        self.hum_corr = 0
    def data_prepare(self):
        df = self.data
        '''
            对数据进行预处理，去掉有缺失值的行;创建新的索引
            创建新的两列，分别为日期和时间列
        '''
        df.dropna(axis=0, how='all', inplace=True)
        df = df.reset_index(drop=True)
        df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
        # 生成日期列、时间列
        df['date'] = df['create_time'].dt.date
        df['h-m-s'] = df['create_time'].dt.time
        df.fillna(method="ffill", inplace = True)
        self.df_prepare = df
        return self.df_prepare

    def power_data_describe(self):
        df = self.data_prepare()
        data_sum = pd.DataFrame()
        for i in range(1, 733):
            if i == 1:
                data_sum = df.groupby(['date'])[df.columns[i]].sum()
            else:
                data_sum = pd.concat([data_sum, pd.DataFrame(df.groupby(['date'])[df.columns[i]].sum())], axis=1)

        def sum_columns(columns):
            label_1 = columns
            label_2 = list(data_sum[label_1].sum(axis=1))  # 按照列相加
            return label_2

        data_sum['风机合相有功功率和(kw)'] = sum_columns(['zsjc0054_1_风机合相有功功率Pt', 'zsjc0131_2_风机合相有功功率Pt',
                                                 'zsjc0207_3_风机合相有功功率Pt', 'zsjc0283_4_风机合相有功功率Pt',
                                                 'zsjc0359_5_风机合相有功功率Pt', 'zsjc0435_6_风机合相有功功率Pt'])
        data_sum['主机电功率和(kw)'] = sum_columns(['zsjc0649_1号主机电功率', 'zsjc0657_2号主机电功率', 'zsjc0665_3号主机电功率',
                                              'zsjc0673_4号主机电功率', 'zsjc0681_5号主机电功率', 'zsjc0689_6号主机电功率'])
        data_sum['冷冻泵电功率和(kw)'] = sum_columns(['zsjc0697_1号冷冻泵电功率', 'zsjc0705_2号冷冻泵电功率',
                                               'zsjc0713_3号冷冻泵电功率', 'zsjc0721_4号冷冻泵电功率'])
        data_sum['风机主机冷冻泵电功率和(kw)'] = sum_columns(['风机合相有功功率和(kw)', '主机电功率和(kw)', '冷冻泵电功率和(kw)'])

        data_sum['风机总能耗(kw)'] = round(data_sum['风机合相有功功率和(kw)'] / 60, 2)
        data_sum['主机总能耗(kw)'] = round(data_sum['主机电功率和(kw)'] / 60, 2)
        data_sum['冷冻泵总能耗(kw)'] = round(data_sum['冷冻泵电功率和(kw)'] / 60, 2)
        data_sum['风机主机冷冻泵总能耗(kw)'] = round(data_sum['风机主机冷冻泵电功率和(kw)'] / 60, 2)

        self.pow_data = data_sum
        return self.pow_data

    def temp_data_extract(self):
        df = self.data_prepare()
        temp_data = (df[['date', 'h-m-s', 'zsjc0461_室外干球温度', 'zsjc0462_室外相对湿度']]).copy()
        temp_data['date'] = temp_data['date'].astype(str)
        temp_data.set_index(['date'], inplace=True)
        self.temp_data = temp_data
        return self.temp_data

    def temp_data_corr(self, date):
        temp = self.temp_data_extract()
        date_list = list(set(temp.index))
        date_list.remove(date)
        choose_df = temp.loc[date]  # 选择日的数据
        date_list_1 = []

        # 先筛选出和选择日的室外温度最相近的一些日期
        for i in date_list:
            average_delt_temp = {}
            temp_day = pd.merge(choose_df, temp.loc[i], on='h-m-s')
            temp_day['delt_temp'] = temp_day['zsjc0461_室外干球温度_x'] - temp_day['zsjc0461_室外干球温度_y']
            average_delt_temp[i] = temp_day['delt_temp'].mean(axis=0)
            if abs(average_delt_temp[i]) < 0.5:
                date_list_1.append(i)

        temp_corr_compare = {}  # 用于存放干球温度相关系数
        hum_corr_compare = {}  # 用于存放相对湿度相关系数
        for j in date_list_1:
            choose_other = pd.merge(choose_df, temp.loc[j], on='h-m-s')
            corr = choose_other.corr()
            temp_corr_compare[j] = [corr.loc['zsjc0461_室外干球温度_x', 'zsjc0461_室外干球温度_y']]
            hum_corr_compare[j] = [corr.loc['zsjc0462_室外相对湿度_x', 'zsjc0462_室外相对湿度_y']]
        temp_corr_max_date = max(temp_corr_compare, key=temp_corr_compare.get)
        hum_corr_max_date = max(hum_corr_compare, key=hum_corr_compare.get)

        self.temp_corr = temp_corr_compare[temp_corr_max_date]
        self.hum_corr = hum_corr_compare[temp_corr_max_date]
        self.temp_corr_compare, self.hum_corr_compare = temp_corr_compare, hum_corr_compare
        self.temp_corr_max_date, self.hum_corr_max_date = temp_corr_max_date, hum_corr_max_date
        return self.temp_corr_max_date

    # 选择日与相似日温度比较图、湿度比较图
    def temp_hum_compare(self, date):
        temp = self.temp_data_extract()
        choose_df = temp.loc[date]
        # 将两个相似日的表合并
        similar_date = self.temp_data_corr(date)
        temp_hum_similar_df = pd.merge(choose_df, temp.loc[similar_date], on='h-m-s')
        temp_hum_similar_df['h-m-s'] = temp_hum_similar_df['h-m-s'].astype(str)
        temp_hum_similar_df.set_index(['h-m-s'], inplace=True)
        plt.figure(dpi=200, figsize=(10, 5))

        # 温度比较图
        plt.subplot(2, 1, 1)
        plt.plot(temp_hum_similar_df['zsjc0461_室外干球温度_x'], 'g', label='T of Choose Day:{}'.format(date))
        plt.plot(temp_hum_similar_df['zsjc0461_室外干球温度_y'], 'b', label='T of Similar Day:{}'.format(similar_date))
        plt.legend(fontsize=8)
        plt.xlabel('Time', fontsize=8)
        plt.xticks(range(0, len(temp_hum_similar_df), 60), rotation=30)
        plt.ylabel('Temperature/℃', fontsize=8)
        plt.title('Temperature Compare', fontsize=11)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.7)
        plt.tick_params(labelsize=8)
        # 湿度比较图
        plt.subplot(2, 1, 2)
        plt.plot(temp_hum_similar_df['zsjc0462_室外相对湿度_x'], 'g', label='H of Choose Day:{}'.format(date))
        plt.plot(temp_hum_similar_df['zsjc0462_室外相对湿度_y'], 'b', label='H of Similar Day:{}'.format(similar_date))
        plt.legend(fontsize=8)
        plt.xlabel('Time', fontsize=8)
        plt.xticks(range(0, len(temp_hum_similar_df), 60), rotation=30)
        plt.ylabel('Humidity/%', fontsize=8)
        plt.title('Humidity Compare', fontsize=11)
        plt.subplots_adjust(top=0.85, bottom=0.15)
        plt.tick_params(labelsize=8)
        return None

    # 选择日与相似日的风机逐时功率、主机逐时功率以及冷冻水泵逐时功率的对比
    def power_compare(self, date):
        choose_date = date
        similar_date = self.temp_data_corr(choose_date)
        dff = self.data_prepare()
        power_data = (dff[['date', 'h-m-s', 'zsjc0054_1_风机合相有功功率Pt', 'zsjc0131_2_风机合相有功功率Pt',
                              'zsjc0207_3_风机合相有功功率Pt', 'zsjc0283_4_风机合相有功功率Pt',
                              'zsjc0359_5_风机合相有功功率Pt', 'zsjc0435_6_风机合相有功功率Pt', 'zsjc0649_1号主机电功率',
                              'zsjc0657_2号主机电功率', 'zsjc0665_3号主机电功率', 'zsjc0673_4号主机电功率', 'zsjc0681_5号主机电功率',
                              'zsjc0689_6号主机电功率', 'zsjc0697_1号冷冻泵电功率', 'zsjc0705_2号冷冻泵电功率',
                              'zsjc0713_3号冷冻泵电功率', 'zsjc0721_4号冷冻泵电功率']]).copy()
        power_data['date'] = power_data['date'].astype(str)
        power_data.set_index(['date'], inplace=True)

        # 将各个风机的逐时功率求和、主机和冷冻水泵同理
        power_data['power_fans'] = list((power_data[['zsjc0054_1_风机合相有功功率Pt', 'zsjc0131_2_风机合相有功功率Pt',
                                                     'zsjc0207_3_风机合相有功功率Pt', 'zsjc0283_4_风机合相有功功率Pt',
                                                     'zsjc0359_5_风机合相有功功率Pt', 'zsjc0435_6_风机合相有功功率Pt']]).sum(axis=1))
        power_data['power_chillers'] = list((power_data[['zsjc0649_1号主机电功率', 'zsjc0657_2号主机电功率', 'zsjc0665_3号主机电功率',
                                                         'zsjc0673_4号主机电功率', 'zsjc0681_5号主机电功率',
                                                         'zsjc0689_6号主机电功率']]).sum(axis=1))
        power_data['power_cooling_pump'] = list((power_data[['zsjc0697_1号冷冻泵电功率', 'zsjc0705_2号冷冻泵电功率',
                                                             'zsjc0713_3号冷冻泵电功率', 'zsjc0721_4号冷冻泵电功率']]).sum(axis=1))

        # 将选择日与相似日的数据提取并合并
        power_merge = pd.merge(power_data.loc[choose_date], power_data.loc[similar_date], on='h-m-s')
        power_merge['h-m-s'] = power_merge['h-m-s'].astype(str)
        power_merge.set_index(['h-m-s'], inplace=True)


        # 单个泵逐时功率比较图
        plt.figure(dpi=200, figsize=(10, 5))
        plt.plot(power_merge['zsjc0697_1号冷冻泵电功率_x'], 'g', linewidth=0.5, label='Choose Day:{},1号冷冻泵'.format(date))
        plt.plot(power_merge['zsjc0705_2号冷冻泵电功率_x'], color ='#008B8B', linewidth=0.5, label='Choose Day:{},2号冷冻泵'.format(date))
        plt.plot(power_merge['zsjc0713_3号冷冻泵电功率_x'], color ='#00EEEE', linewidth=0.5, label='Choose Day:{},3号冷冻泵'.format(date))
        plt.plot(power_merge['zsjc0721_4号冷冻泵电功率_x'], color ='#00C5CD', linewidth=0.5, label='Choose Day:{},4号冷冻泵'.format(date))
        plt.plot(power_merge['zsjc0697_1号冷冻泵电功率_y'], 'r', linewidth=0.5, label='Choose Day:{},1号冷冻泵'.format(date))
        plt.plot(power_merge['zsjc0705_2号冷冻泵电功率_y'], color ='#FFCC00', linewidth=0.5, label='Choose Day:{},2号冷冻泵'.format(date))
        plt.plot(power_merge['zsjc0713_3号冷冻泵电功率_y'], color ='#FF9900', linewidth=0.5, label='Choose Day:{},3号冷冻泵'.format(date))
        plt.plot(power_merge['zsjc0721_4号冷冻泵电功率_y'], color ='#FF6600', linewidth=0.5, label='Choose Day:{},4号冷冻泵'.format(date))

        plt.legend(fontsize=8)
        plt.xlabel('Time', fontsize=10)
        plt.xticks(range(0, len(power_merge), 60), rotation=30)
        plt.ylabel('Power/kW', fontsize=10)
        plt.title('Comparison between power of pumps', fontsize=12)
        plt.tick_params(labelsize=8)
        plt.tight_layout()

        # 风机逐时功率比较图
        plt.figure(dpi=200, figsize=(10, 5))
        plt.plot(power_merge['power_fans_x'], 'g', linewidth=0.5, label='Choose Day:{}'.format(date))
        plt.plot(power_merge['power_fans_y'], 'b', linewidth=0.5, label='Similar Day:{}'.format(similar_date))
        plt.legend(fontsize=8)
        plt.xlabel('Time', fontsize=10)
        plt.xticks(range(0, len(power_merge), 60), rotation=30)
        plt.ylabel('Power/kW', fontsize=10)
        plt.title('Comparison between power of fans', fontsize=12)
        plt.tick_params(labelsize=8)
        plt.tight_layout()

        # 主机逐时功率比较图
        plt.figure(dpi=200, figsize=(10, 5))
        plt.plot(power_merge['power_chillers_x'], 'g', linewidth=0.5, label='Choose Day:{}'.format(date))
        plt.plot(power_merge['power_chillers_y'], 'b', linewidth=0.5, label='Similar Day:{}'.format(similar_date))
        plt.legend(fontsize=8)
        plt.xlabel('Time', fontsize=10)
        plt.xticks(range(0, len(power_merge), 60), rotation=30)
        plt.ylabel('Power_chillers/kW', fontsize=10)
        plt.title('Comparison between power of chillers', fontsize=12)
        plt.tick_params(labelsize=8)
        plt.tight_layout()

        # 冷冻泵逐时功率比较图
        plt.figure(dpi=200, figsize=(10, 5))
        plt.plot(power_merge['power_cooling_pump_x'], 'g', linewidth=0.5, label='Choose Day:{}'.format(date))
        plt.plot(power_merge['power_cooling_pump_y'], 'b', linewidth=0.5, label='Similar Day:{}'.format(similar_date))
        plt.legend(fontsize=8)
        plt.xlabel('Time', fontsize=10)
        plt.xticks(range(0, len(power_merge), 60), rotation=30)
        plt.ylabel('Power_cooling_pump/kW', fontsize=10)
        plt.title('Comparison between power of cooling pumps', fontsize=12)
        plt.tick_params(labelsize=8)
        plt.tight_layout()
        return None

    # 选择日与相似日功率消耗总结对比表
    def similar_data_power_compare(self, date):
        power_data = self.power_data_describe()
        similar_date = self.temp_data_corr(date)
        power_data.index = power_data.index.astype('str')
        similar_data_power = power_data.loc[power_data.index.isin([date, similar_date])]
        self.similar_data_power = similar_data_power
        return self.similar_data_power

    # 增加文字说明
    def set_label(self, rects):
        for rect in rects:
            height = rect.get_height()  # 获取⾼度
            plt.text(x=rect.get_x() + rect.get_width() / 2,  # ⽔平坐标
                     y=height + 0.3,  # 竖直坐标
                     s=height,  # ⽂本
                     ha='center')  # ⽔平居中

    # 选择日与相似日功率柱形图比较
    def power_bar_compare(self, date):
        similar_power_data = self.similar_data_power_compare(date)
        similar_date = self.temp_data_corr(date)
        plt.figure(dpi=200, figsize=(10, 8))
        bar_width = 0.4

        plt.subplot(1, 1, 1)
        xlabel2 = ['风机总能耗(kw)', '主机总能耗(kw)', '冷冻泵总能耗(kw)', '风机主机冷冻泵总能耗(kw)']
        kwh1 = list(similar_power_data.loc[date, xlabel2])
        kwh2 = list(similar_power_data.loc[similar_date, xlabel2])
        x2 = np.arange(len(xlabel2))
        kwh_bar1 = plt.bar(x=x2 - bar_width / 2, height=kwh1, width=bar_width, label='Choose Date {}'.format(date))
        kwh_bar2 = plt.bar(x=x2 + bar_width / 2, height=kwh2, width=bar_width,
                           label='Similar Date {}'.format(similar_date))
        self.set_label(kwh_bar1)
        self.set_label(kwh_bar2)
        # x轴刻度标签位置不进行计算
        plt.xticks(x2, labels=xlabel2)
        plt.legend(fontsize=8)
        plt.title('各类设备的总能耗对比柱状图', fontsize=12)
        plt.tick_params(labelsize=8)
        plt.show()
        return None


# In[3]:

if __name__ == "__main__":
    data = pd.read_csv(r'原始数据\zsjc_data_2022-06-03_2022-09-08.csv', low_memory=False)
    aaa = SimilarCompare(data)
    clean_df = aaa.data_prepare()
    power_df = aaa.power_data_describe()
    temp_df = aaa.temp_data_extract()
    choose_day = str(input("Please input your date selected, eg:2022-08-02:\t"))
    temp_corr_df = aaa.temp_data_corr(choose_day)
    print('Temperature correlation coefficient between choose day and similar day:\n', aaa.temp_corr)
    print('Maximum hum correlation coefficient between choose day and similar day:\n', aaa.hum_corr)
    aaa.temp_hum_compare(choose_day)
    aaa.power_compare(choose_day)
    compare_df = aaa.similar_data_power_compare(choose_day)
    aaa.power_bar_compare(choose_day)

# In[ ]:
