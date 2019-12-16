# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         HLK-20C02(4℃)
# Description:  
# Author:       shichao
# Date:         2019/7/17
#-------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import time
import datetime

# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


# 读取数据
def read_df(file_name):
    file_dir = './raw_data/'
    file_path = os.path.join(file_dir, file_name)
    file_path = open(file_path)
    df_file = pd.read_csv(file_path)
    df_file['DateTime'] = pd.to_datetime(df_file['DateTime'])
    df_file = df_file.sort_values(by='DateTime')
    return df_file


# 对 x 增加特征：date_ymd 年月日、把年月日时分秒转化为秒数utc时间
def add_features(df_file):
    date_list = []
    for date in list(df_file['DateTime']):
        date_str = str(date).split(' ')[0]
        date_list.append(date_str)
    df_file['date_ymd'] = date_list
    time_stamp_list = []
    for time_stamp in list(df_file['DateTime']):
        time_s = time.mktime(time.strptime(str(time_stamp), '%Y-%m-%d %H:%M:%S'))
        # time_s = time.mktime(time.strptime(time_stamp, '%Y/%m/%d %H:%M:%S'))
        time_stamp_list.append(time_s)
    df_file['time_stamp'] = time_stamp_list
    # date_ymdh_list = []
    # for time_stamp in list(df_file['DateTime']):
    #     date_ymdh = str(time_stamp).split(':')[0]
    #     date_ymdh_list.append(date_ymdh)
    # df_file['date_ymdh'] = date_ymdh_list
    return df_file



# 画图：补全缺失值后，画图与原图比较
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
def draw_plot(dataset_12_plot):
    #date_time = 'date_ymd'
    temperature = 'Temperature'
    date_ymd = 'date_ymd'
    plt.figure(1, figsize=(26, 13))
    # 获取坐标轴
    ax = plt.gca()
    #plt.plot(dataset_12_plot[date_time], dataset_12_plot[temperature], 'red', marker='o')
    plt.plot(dataset_12_plot[temperature], 'red', marker='o')
    for label in ax.get_xticklabels():
        # 横轴标签旋转 30°
        label.set_rotation(30)
        label.set_horizontalalignment('right')
        # 显示图例
    plt.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))  # 设置时间标签显示格式
    ax.xaxis.set_major_locator(mdates.HourLocator())  # X轴的间隔为小时
    png_dir = './repair_png/'
    date_ymd = str(dataset_12_plot[date_ymd][1])
    png_path = os.path.join(png_dir, date_ymd + '_' + str(len(dataset_12_plot)) + '.png')
    plt.savefig(png_path)
    plt.show()


# 对缺失的温度进行插补
def repair_tem(df_data_by_date, sample_frequency):
    """
    :param df_data_by_date:
    :param sample_frequency: 采样频率
    :return:
    """
    # 去除重复列，默认所有列无重复记录
    #df_data_by_date.duplicated()
    df_data_by_date = df_data_by_date.reset_index(drop=True)
    term_list_1 = list(df_data_by_date['Temperature'])
    term_list_date = list(df_data_by_date['date_ymd'])
    n = len(term_list_1)
    date_list = []
    temp_temp_list = []
    # 采样频率
    time_n = 3 # 时间间隔， 3 分钟
    for i in range(n):
        if (i >= 0 and i + 1 <= n - 1):
            temp_temp_list.append(term_list_1[i])
            date_list.append(term_list_date[i])
            # 对中间缺失的温度值进行插补
            if (df_data_by_date.loc[i + 1]['time_stamp'] - df_data_by_date.loc[i]['time_stamp'] >= (sample_frequency + time_n) * 60):
                #n_temp = int(np.ceil((df_data_by_date.loc[i + 1]['time_stamp'] - df_data_by_date.loc[i]['time_stamp']) / (sample_frequency * 60.0)))
                # 四舍五入取整
                n_temp = int(((df_data_by_date.loc[i + 1]['time_stamp'] - df_data_by_date.loc[i]['time_stamp']) / (sample_frequency * 60.0)) + 0.5)
                for j in range(n_temp - 1):
                    temp_temp = (df_data_by_date.loc[i + 1]['Temperature'] + df_data_by_date.loc[i]['Temperature']) / 2
                    temp_temp_list.append(temp_temp)
                    date_list.append(term_list_date[-1])
    temp_temp_list.append(term_list_1[-1])
    date_list.append(term_list_date[-1])
    # 如果开始连续缺失数量少于 30%, 用均值补齐
    df_data_by_date = df_data_by_date.reset_index(drop=True)
    #date_ = term_list_date[1]
    # 看是否中间补全
    if(len(temp_temp_list) < int(24*60/sample_frequency)):
        # 开头缺失
        continue_list = []
        time_s = time.mktime(time.strptime(str(term_list_date[1]), '%Y-%m-%d')) # 当天开始时间 0 时 0 分 0 秒
        if(df_data_by_date.loc[0]['time_stamp'] - time_s > (sample_frequency + time_n) * 60):
            # 开头缺失
            n_temp = int(np.ceil((df_data_by_date.loc[0]['time_stamp'] - time_s) / (sample_frequency * 60.0)))
            for j in range(n_temp - 1):
            #for j in range(int(24*60/sample_frequency) - len(term_list_1)):
                continue_list.append(round(np.mean(term_list_1), 2))
                date_list.append(term_list_date[-1])
            continue_list.extend(temp_temp_list)
            temp_temp_list = continue_list
        # 结尾缺失
        # 获取下一天的日期
        date_end = pd.to_datetime(term_list_date[1]) + datetime.timedelta(days=1)
        time_end = time.mktime(time.strptime(str(date_end), '%Y-%m-%d %X'))
        if(time_end - df_data_by_date.loc[len(df_data_by_date)-1]['time_stamp'] >= (sample_frequency + time_n) * 60):
            # 结尾缺失
            for j in range(int(24*60/sample_frequency) - len(term_list_1)):
                continue_list.append(round(np.mean(term_list_1), 2))
                date_list.append(term_list_date[-1])
            temp_temp_list.extend(continue_list)
    df_repair = pd.DataFrame()
    df_repair['date_ymd'] = date_list
    df_repair['Temperature'] = temp_temp_list
    return df_repair



# 对温度做分段常数逼近处理，下采样
def constant_appro_low(df_data_by_date_tem):
    df_data_by_date_tem = df_data_by_date_tem.reset_index(drop=True)
    df_appro = pd.DataFrame()
    date_index = pd.date_range(end = '01/01/2019', periods=len(df_data_by_date_tem), freq='D')
    temperature = 'Temperature'
    date_ymd = 'date_ymd'
    df_appro[temperature] = df_data_by_date_tem[temperature]
    df_appro.index = date_index
    # 下采样，取均值
    df_appro_low = pd.DataFrame()
    # 一个小时聚合一次常值
    df_appro_low[temperature] = df_appro[temperature].resample(rule='6D').mean()
    #date_list = df_data_by_date_tem.loc[:len(df_appro_low)-1][date_ymd]
    #df_appro_low[date_ymd] = list(date_list)
    # 差分，做一阶差分
    df_appro_diff = pd.DataFrame()
    df_appro_diff[temperature] = df_appro_low.loc[:][temperature].diff(1) # 1 阶差分
    df_appro_diff[date_ymd] = list(df_data_by_date_tem.loc[:len(df_appro_diff)-1][date_ymd])
    df_appro_diff = df_appro_diff.dropna()
    df_appro_diff = df_appro_diff.reset_index(drop=True)
    df_appro_diff[temperature] = df_appro_diff[temperature].apply(lambda x: round(x, 2))
    return df_appro_diff




# 对 x 进行特征提取,
from tsfresh import extract_relevant_features
from tsfresh import extract_features
import tsfresh as tsf
def get_features(df_appro):
    #extracted_features = extract_features(df_appro, column_id='date_ymd')
    #ts = pd.Series(x)  # 数据x假设已经获取
    ts = df_appro['Temperature']
    # 一阶差分绝对和
    abs_sum = tsf.feature_extraction.feature_calculators.absolute_sum_of_changes(ts)
    abs_sum = round(abs_sum, 2)
    # 各阶自相关系数的聚合统计特征
    param_statis = [{'f_agg': 'mean', 'maxlag': 2}]
    diff_statis = tsf.feature_extraction.feature_calculators.agg_autocorrelation(ts, param_statis)
    diff_statis = diff_statis[0][1]
    diff_statis = round(diff_statis, 2)
    # ADF 检测统计值
    param_adf = [{'attr': 'pvalue'}]
    adf = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, param_adf)
    adf = adf[0][1]
    adf = round(adf, 2)
    # 峰度
    peak = tsf.feature_extraction.feature_calculators.kurtosis(ts)
    peak = round(peak, 2)
    # 时序数据复杂度
    complexity = tsf.feature_extraction.feature_calculators.cid_ce(ts, True)
    complexity = round(complexity, 2)
    # 线性回归分析
    param_line = [{'attr': 'pvalue'}]
    line = tsf.feature_extraction.feature_calculators.linear_trend(ts, param_line)
    line = list(zip(line))[0][0][1]
    line = round(line, 2)
    # 分组熵
    bin_entropy = tsf.feature_extraction.feature_calculators.binned_entropy(ts, 10)
    bin_entropy = round(bin_entropy, 2)
    # 近似熵
    appro_entropy = tsf.feature_extraction.feature_calculators.approximate_entropy(ts, 6, 0.1)
    appro_entropy = round(appro_entropy, 2)
    # 傅里叶变换频谱统计量
    param_fly = [{'aggtype': 'skew'}]
    fly = tsf.feature_extraction.feature_calculators.fft_aggregated(ts, param_fly)
    fly = list(zip(fly))[0][0][1]
    fly = round(fly, 2)
    # 傅里叶变换系数
    param_fly_change = [{'coeff': 2, 'attr': 'angle'}]
    fly_change = tsf.feature_extraction.feature_calculators.fft_coefficient(ts, param_fly_change)
    fly_change = list(zip(fly_change))[0][0][1]
    fly_change = round(fly_change, 2)
    # 小坡变换
    param_cwt = [{'widths': tuple([2, 2, 2]), 'coeff': 2, 'w': 2}]
    cwt = tsf.feature_extraction.feature_calculators.cwt_coefficients(ts, param_cwt)
    cwt = list(zip(cwt))[0][0][1]
    cwt = round(cwt, 2)
    return abs_sum, adf, peak, complexity, line, bin_entropy, appro_entropy, fly, fly_change, cwt



# 对每天的温度进行特征提取, 分段特征; 统计特征；熵特征; 第 3 阶段
def get_features_everday(df_data, sample_frequence):
    date_ymd_str = 'date_ymd'
    temperature = 'Temperature'
    date_ymd_field = df_data[date_ymd_str]
    date_ymds = []
    for i in date_ymd_field:
        if i not in date_ymds:
            date_ymds.append(i)
    df_features = pd.DataFrame()
    for date_ymd in date_ymds:
        # date_ymd = '2019-04-17'
        abs_sum_list = []
        adf_list = []
        peak_list = []
        complexity_list = []
        line_list = []
        bin_entropy_list = []
        appro_entropy_list = []
        fly_list = []
        fly_change_list = []
        cwt_list = []
        date_ymd_list = []
        df_data_by_date = df_data[df_data[date_ymd_str] == date_ymd]
        # 删除重复记录
        df_data_by_date = df_data_by_date.drop_duplicates()
        # 缺失值大于 30% 的直接舍弃
        #sample_frequence = 10 # 采样频率是 10 min
        data_num = int((24*60)/sample_frequence)
        abandon_percent = 0.3
        abondon_thr = int(data_num * (1 - abandon_percent))
        if (len(df_data_by_date) <= abondon_thr):
            continue
        # # 将异常温度升高的波峰的温度值用当天的温度均值做替换, 此台设备没有可替换的异常温度值
        # df_data_by_date = replace_abnorm_temp(df_data_by_date)
        # 用插值补全温度值
        df_data_by_date_tem = repair_tem(df_data_by_date, sample_frequency)
        # draw_plot(df_data_by_date_tem)
        # 如果是连续性缺失，则舍弃
        if (len(df_data_by_date_tem) <= abondon_thr):
            continue
        # 分段常数逼近，重采样，下采样，df_data_by_date_tem.resample()
        df_appro = constant_appro_low(df_data_by_date_tem)
        temp_array = np.array(list(df_appro[temperature]))
        # 分段常数逼近特征列名: 23 个 , 每 60 分钟取一次均值
        columns_name = ['t' + str(x) for x in range(0, len(temp_array))]
        temp_array = temp_array.reshape(1, len(columns_name))
        df_x = pd.DataFrame(temp_array, columns=columns_name)
        # thresh 提取特征
        abs_sum, adf, peak, complexity, line, bin_entropy, appro_entropy, fly, fly_change, cwt = get_features(df_appro)
        abs_sum_list.append(abs_sum)
        adf_list.append(adf)
        peak_list.append(peak)
        complexity_list.append(complexity)
        line_list.append(line)
        bin_entropy_list.append(bin_entropy)
        appro_entropy_list.append(appro_entropy)
        fly_list.append(fly)
        fly_change_list.append(fly_change)
        cwt_list.append(cwt)
        date_ymd_list.append(date_ymd)
        # 统计特征列名： 8
        df_x['abs_sum'] = abs_sum_list
        df_x['adf'] = adf_list
        df_x['peak'] = peak_list
        df_x['complexity'] = complexity_list
        df_x['line'] = line_list
        df_x['fly'] = fly_list
        df_x['fly_change'] = fly_change_list
        df_x['cwt'] = cwt_list
        # 信息熵特征：数据片段相似性  2 个
        df_x['bin_entropy'] = bin_entropy_list
        df_x['appro_entropy'] = appro_entropy_list
        df_x[date_ymd_str] = date_ymd_list
        df_features = pd.concat([df_features, df_x], axis=0, sort=False)
    return df_features




# 提取目标 y : 之后手动去确认一天的正常除霜时间和跨夜的除霜时间：
# correct 栏位是人工确认白天的除霜时间，accross 栏位是跨夜的除霜时间
# correct 和 accross 是人工添加的两个栏位，待确认整理后，才可进行下一步
def get_targets(file_name_):
    file_label_dir = './label_data/'
    file_label_path = os.path.join(file_label_dir, file_name_ + '.csv')
    file_label_path = open(file_label_path)
    df_file = pd.read_csv(file_label_path)
    if 'Unnamed: 5' in df_file.columns:
        del df_file['Unnamed: 5']
    df_file['hum_label'] = df_file['hum_label'].astype(str)
    df_file_label = df_file[df_file['hum_label']=='1.0']
    label_preprocess_dir = 'targets/'
    file_name_label = file_name_ + '_label' + '.csv'
    label_preprocess_path = os.path.join(label_preprocess_dir, file_name_label)
    #df_file_label.to_csv(label_preprocess_path, index=False)
    return df_file_label


# 给 label y 增加辅助列
def add_y_col(df_file):
    date_list = []
    for date in list(df_file['DateTime']):
        date_str = str(date).split(' ')[0]
        date_list.append(date_str)
    df_file['date_ymd'] = date_list
    return df_file


# 对 y 进行补全、筛选处理：
# 将白天和跨夜的除霜时间进行整理，按照时间顺序进行排序。手动确认日期。
def complete_targets(file_name_label, n_cnt):
    """
    :param file_name_label: 挑出的 y 标签
    :param n_cnt: 除霜次数
    :return:
    """
    label_dir = './targets/'
    label_path = os.path.join(label_dir, file_name_label)
    label_path = open(label_path)
    df_label = pd.read_csv(label_path)
    df_label['correct'] = df_label['correct'].astype(str)
    #df_label['across'] = df_label['across'].astype(str)
    df_laebl_temp_1 = df_label[df_label['correct']=='1.0']
    #df_laebl_temp_2 = df_label[df_label['across']=='1.0']
    # 对 label y 进行整理
    df_laebl_temp_1 = df_laebl_temp_1.reset_index(drop=True)
    #df_laebl_temp_2 = df_laebl_temp_2.reset_index(drop=True)
    df_laebl_temp_1 = add_y_col(df_laebl_temp_1)
    #df_laebl_temp_2 = add_y_col(df_laebl_temp_2)
    # 保存在 ./targets/ 下人工检查日期: 处理跨夜的日期
    save_dir = './targets/'
    save_path_1 = os.path.join(save_dir, file_name_label.split('.')[0] + '_1.csv')
    #df_laebl_temp_1.to_csv(save_path_1, index=False)
    save_path_2 = os.path.join(save_dir, file_name_label.split('.')[0] + '_2.csv')
    #df_laebl_temp_2.to_csv(save_path_2, index=False)
    print ()


# 把 label y 按日期变成一行
def label_to_hor(df_data, n_cnt):
    df_data = df_data.reset_index(drop=True)
    date_ymd_field = df_data['date_ymd']
    date_ymds = []
    for i in date_ymd_field:
        if i not in date_ymds:
            date_ymds.append(i)
    df_features = pd.DataFrame()
    for date_ymd in date_ymds:
        date_list = []
        df_data_by_date = df_data[df_data['date_ymd'] == date_ymd]
        date_list.append(date_ymd)
        temp_array = np.array(list(df_data_by_date['DateTime']))
        temp_array = temp_array.reshape(1, n_cnt*2)
        df_y = pd.DataFrame(temp_array)
        df_y['date_ymd'] = date_list
        df_features = pd.concat([df_features, df_y], axis=0)
    return df_features



# 对 y 进行合并排序，将人工手动的跨夜和白天除霜数据整理为一个目标文件。修改 date_ymd 栏位
def combine_label(file_name_, n_cnt):
    file_name_label_1 = file_name_ +  '_label_1.csv'
    label_dir = './targets/'
    label_path_1 = os.path.join(label_dir, file_name_label_1)
    label_path_1 = open(label_path_1)
    df_label_1 = pd.read_csv(label_path_1)
    #file_name_label_2 = file_name_ + '_label_2.csv'
    #label_path_2 = os.path.join(label_dir, file_name_label_2)
    #label_path_2 = open(label_path_2)
    #df_label_2 = pd.read_csv(label_path_2)
    # 检查日期，人工检查
    import collections
    a = collections.Counter(df_label_1['date_ymd'])
    #b = collections.Counter(df_label_2['date_ymd'])
    # 把 label y 按照每天日期排列成一行
    df_label_y_1 = label_to_hor(df_label_1, n_cnt)
    #df_label_y_2 = label_to_hor(df_label_2, n_cnt)
    #df_y = pd.concat([df_label_y_1, df_label_y_2], axis=0)
    df_y = pd.concat([df_label_y_1], axis=0)
    df_y['date_ymd'] = pd.to_datetime(df_y['date_ymd'])
    df_y = df_y.sort_values(by='date_ymd')
    #df_y.to_csv(label_dir + file_name_label_1.split('_')[0] + '_y.csv', index=False)
    print()



# 整合 x 和 y, 与匹配的日期对应起来, 对应产出在 label_preprocess_2/ 文件夹下
def merge_features_target(file_name):
    features_dir = './features/'
    target_dir = './targets/'
    features_path = os.path.join(features_dir, file_name + '_x_1.csv') # 第 2 阶段
    features_path = open(features_path)
    target_path = os.path.join(target_dir, file_name + '_y.csv')
    target_path = open(target_path)
    df_features = pd.read_csv(features_path)
    df_target = pd.read_csv(target_path)
    # 添加辅助列 date_ymd_1: 为了匹配用昨天的温度预测下一天的时间
    df_target['date_ymd'] = pd.to_datetime(df_target['date_ymd'])
    df_target = df_target.sort_values(by='date_ymd')
    # 获取当前日期前一天日期
    before_days = 1
    #before_days_date = now_date + datetime.timedelta(days=-before_days)
    df_target['date_ymd'] = df_target['date_ymd'].apply(lambda x: x + datetime.timedelta(days=-before_days))

    df_target['date_ymd'] = df_target['date_ymd'].astype(str)
    # 时间字符串格式转换
    df_features['date_ymd'] = pd.to_datetime(df_features['date_ymd'])
    df_features = df_features.sort_values(by='date_ymd')
    df_features['date_ymd'] = df_features['date_ymd'].astype(str)
    # 合并 x 和 y
    df_data = pd.merge(df_features, df_target, on='date_ymd')
    data_merge_dir = './fea_tar_data/'
    data_merge_path = os.path.join(data_merge_dir, file_name + '_feature_target_1.csv') # 第 1 阶段 x 和 y 合并后的成果
    del df_data['date_ymd']
    # df_data.to_csv(data_merge_path, index=False)
    print ()










if __name__ == '__main__':
    # 处理特征 X
    file_name = 'HLK-20C02(4℃).csv'
    file_name_ = os.path.splitext(file_name)[0]
    # # 读取数据
    # df_file = read_df(file_name)
    # # 增加辅助列
    # df_file = add_features(df_file)
    # # 获取 x 特征
    # sample_frequency = 10 # 采样频率 10 min
    # df_features = get_features_everday(df_file, sample_frequency) # 第 1 阶段
    # # df_features.to_csv('./features/' + file_name_ +'_x_1.csv', index=False)  # 保存第 1 阶段 x
    # 提取 y: 人工确认当天除霜时间和跨夜除霜时间。
    df_file_y = get_targets(file_name_)
    # 处理标签 Y ：人工手动整理跨夜除霜数据
    file_name_label = file_name_ + '_label' + '.csv'
    n_cnt = 5 # 除霜次数
    complete_targets(file_name_label, n_cnt)
    # 将手动整理的跨夜和白天除霜时间，根据日期整合一个文件 y
    # combine_label(file_name_, n_cnt)
    # 整合 x 和 y
    merge_features_target(file_name_)

    print ()
