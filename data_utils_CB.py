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
        self.df_dict = {}
        self.result_recorder = ResultRecorder()
        self.bond_list = ['110052', '110053', '110058', '111006', '111013', '111016',
       '113011', '113016', '113025', '113027', '113044', '113057',
       '113537', '113585', '113588', '113595', '113597', '113626',
       '113672', '113676', '118021', '118035', '118037', '123012',
       '123013', '123015', '123018', '123025', '123031', '123034',
       '123046', '123077', '123083', '123098', '123105', '123116',
       '123118', '123134', '123136', '123148', '123173', '123176',
       '123177', '123181', '123187', '123191', '123194', '123197',
       '123200', '123201', '123205', '123206', '123207', '123209',
       '123218', '123220', '127014', '127021', '127029', '127057',
       '127058', '127065', '127079', '127080', '127089', '127090',
       '127091', '128025', '128040', '128041', '128044', '128070',
       '128074', '128075', '128078', '128079', '128082', '128095',
       '128101', '128111', '128114', '128145']   
        return

    def read_file(self, temp_path):
        '''
        读取单个数据文档/
        '''
        # 从本地读取价量数据
        price_df = pd.read_csv(temp_path)
        return price_df

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
                    if filename == '2022_raw':
                        self.init_month_df(file_path)
            else:
                if filename[:6] in self.bond_list:  # 在需要的范围内才考虑
                    temp_df = self.read_file(file_path)
                    if len(temp_df) > 400:
                        self.df_dict[filename] = temp_df
        return

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
    data_df_list = list(dl.df_dict.values())
    compute_future_rtn(data_df_list, periods)
    return


if __name__ =='__main__':
    dl = DataLoader()
    dl.init_month_df('../data/15m_data/')