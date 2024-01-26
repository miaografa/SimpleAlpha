import pandas as pd
import numpy as np
import os
import pandas_ta
from pandas_ta.core import adx, cci, macd, rsi, obv, vwap
import cufflinks


class DataLoader(object):
    '''
    只负责读取处理好的特征数据，位于features里
    '''

    def __init__(self):
        self.years = ['2020', '2021', '2022', '2023']
        self.months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        self.df_dict = {}
        for year in self.years:
            self.df_dict[year] = {}
            for month in self.months:
                self.df_dict[year][month] = []  # 存放每个月份的数据
        self.result_recorder = ResultRecorder()
        return

    def read_file(self, temp_path):
        '''
        读取单个数据文档/
        '''
        # 从本地读取价量数据
        binance_price_df = pd.read_csv(temp_path)
        binance_price_df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume',
                                    'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        return binance_price_df

    def init_month_df(self, root_path):
        '''
        读取某个月份的所有特征文件，保存在self.df_dict中。
        加载目标文件夹下的所有zip文件。这些文件读取后会被保存在dataloader中。
        注意，目前默认读取binance的k line格式的数据。
        :param root_path: 根文件夹
        '''
        # 遍历根文件夹下的所有文件夹
        for filename in os.listdir(root_path):
            file_path = os.path.join(root_path, filename)
            # 检查是否是文件夹
            if not 'zip' in filename and not 'csv' in filename:
                if os.path.isdir(file_path):
                    self.init_month_df(file_path)
            else:
                year, month = get_time_from_filename(filename)
                temp_df = self.read_file(file_path)
                if len(temp_df) > 400:
                    if year in self.years and month in self.months:
                        self.df_dict[year][month].append(temp_df)
        return

    def get_valid_year_month(self)->list:
        '''
        输出有数据保存的年月列表
        '''
        output_list = []
        for year in self.years:
            for month in self.months:
                if len(self.get_month_df(year, month)) > 0:
                    output_list.append((year, month))
        return output_list

    def convert_year_month(self, year, month):
        '''
        将年月转换为字符串
        '''
        if type(year) != str:
            year = str(year)
        if type(month) != str:
            month = str(month)
        if len(month) == 1:
            month = '0' + month
        return year, month

    def get_month_df(self, year, month) -> list:
        '''
        获取某个月份的所有特征文件
        '''
        year, month = self.convert_year_month(year, month)
        return self.df_dict[year][month]

    def set_month_df(self, year, month, df_list:list):
        '''
            设置某个月份的所有特征文件
        '''
        year, month = self.convert_year_month(year, month)
        self.df_dict[year][month] = df_list
        return

    def get_all_df(self) -> list:
        '''
        获取所有特征文件
        '''
        all_df = []
        for year in self.years:
            for month in self.months:
                all_df += self.df_dict[year][month]
        return all_df

    def set_record_df(self, kind:str, factor_name:str, result_df:pd.DataFrame):
        '''
        保存结果
        :param kind: 有三类：[Info, SingleFactorML, OverallML]
        :param factor_name: 特征名称
        :param result_df: 结果
        '''
        # 使用 assert 检查值是否在列中
        assert isinstance(result_df, pd.DataFrame), 'record_df must be a DataFrame'
        assert kind in ['Info', 'ML'], \
            f'kind:{kind} 不在范围内[Info, ML]'
        self.result_recorder.set_record(kind=kind, factor_name=factor_name, result_df=result_df)
        return

    def get_record_df(self, kind, factor_name) -> pd.DataFrame:
        '''
        获取所有特征文件
        '''
        # assert kind in ['Info', 'SingleFactorML', 'OverallML'], \
        #     f'kind:{kind} 不在范围内[Info, SingleFactorML, OverallML]'
        return self.result_recorder.get_record(kind, factor_name)


class ResultRecorder(object):
    '''
    用于记录结果
    重要属性：
        _record_keys_df: pd.DataFrame 用于记录所有的key
        record_dict: 用于记录所有的结果
    '''
    def __init__(self):
        self._record_keys_df = pd.DataFrame(index=['Info', 'SingleFactorML', 'OverallML'])
        self.record_dict = {}

    def set_record(self, kind:str, factor_name:str, result_df:pd.DataFrame):
        '''
        保存结果
        :param kind:
        :param factor_name:
        :param result_df:
        :return:
        '''
        record_key = kind+'_'+factor_name
        while record_key in self.record_dict.keys():
            # warning
            record_key += '_'  # 防止重复，如果重复旧的key会被覆盖，但是rusult还在。
        self._record_keys_df.loc[kind, factor_name] = record_key  # 保存key
        self.record_dict[record_key] = result_df
        return

    def get_record(self, kind, factor_name):
        '''
        获取结果
        :param kind:
        :param factor_name:
        :return:
        '''
        key = self._record_keys_df.loc[kind, factor_name]  # 获取key
        assert not pd.isna(key), f'记录kind:{kind}, factor:{factor_name}, 不存在或者未保存。'
        info_df = self.record_dict[key]
        return info_df


def get_time_from_filename(filename):
    year = filename[-11:-7]
    month = filename[-6:-4]
    return year, month


# def log_return(series):
#     return np.log(series).diff()

def compute_future_rtn(data_df_list, periods):
    '''
    计算未来periods个时间单位的rtn
    :param data_df:
    :param periods:
    :return:
    '''
    if type(data_df_list) != list:
        data_df_list = [data_df_list]

    for data_df in data_df_list:
        for i in periods:
            # 首先计算均线，然后错位10个（也就是对应于未来的均价）
            temp_mean_future = data_df[['close']].ewm(span=i, adjust=False).mean().shift(-i)
            # 计算未来收益
            pct_change = (temp_mean_future - data_df[['close']]) / data_df[['close']]
            # 计算真实波幅
            std = pct_change.ewm(96, min_periods=5).std()
            data_df['fwd_rtn_' + str(i)] = pct_change
            data_df['fwd_rtn_' + str(i) + '_real'] = pct_change / std
            data_df['fwd_rtn_' + str(i) + '_bool'] = pct_change > 0
            data_df['fwd_rtn_' + str(i) + '_abs'] = data_df['fwd_rtn_' + str(i) + '_real'].abs()

        data_df['fwd_rtn_1'] = (data_df['close'].shift(-1) - data_df['close']) / data_df['close']
        data_df['fwd_rtn_1_real'] = data_df['fwd_rtn_1'] / data_df['fwd_rtn_1'].ewm(96, min_periods=5).std()
        data_df['fwd_rtn_1_bool'] = data_df['fwd_rtn_1'] > 0
    return data_df_list


def compute_future_rtn_for_all(dl:DataLoader, periods=(5, 10, 20, 50)):
    '''
    计算Data_Loader中所有数据的未来rtn
    '''
    for year in dl.years:
        for month in dl.months:
            data_df_list = dl.get_month_df(year, month)
            data_df_list = compute_future_rtn(data_df_list, periods)
            dl.set_month_df(year, month, data_df_list)
    return


if __name__ =='__main__':
    dl = DataLoader()
    dl.init_month_df('../data/15m_data/')