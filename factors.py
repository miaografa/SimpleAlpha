'''
计算一些简单特征用于测试
'''
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pandas_ta.core import adx, cci, macd, rsi, obv, vwap, atr, bop, ohlc4
from data_utils import DataLoader

def handle_dataloader_data(factor_func:callable):
    '''
    为DataLoader中的所有数据添加因子的装饰器
    :param factor_func: 因子的计算函数，输入DataFrame，输出计算好factor的DataFrame
    :param dl: DataLoader
    '''
    def add_factors_for_dl(dl, *args, **kwargs):
        for year in dl.years:
            for month in dl.months:
                data_df_list = dl.get_month_df(year, month)
                result_list = []
                for df in data_df_list:
                    result_list.append(factor_func(df, *args, **kwargs))
                dl.set_month_df(year, month, result_list)
        return
    return add_factors_for_dl


def handle_list_data(df_list, factor_func:callable, **kwargs):
    '''
    一般用不到
    按照公式加入特征，对象是df构成的list。
    :param df_list:  list of DataFrame
    :param factor_func: 因子的计算函数，输入DataFrame，输出计算好factor的DataFrame
    :return:
    '''
    if type(df_list) != list:
        df_list = [df_list]
    result_list = []
    for df in df_list:
        result_list.append(factor_func(df, **kwargs))
    return result_list


@handle_dataloader_data
def add_new_factor_demo(raw_df:pd.DataFrame, factor_name)-> pd.DataFrame:
    '''
    加入新因子的模板。
    :param raw_df: 原始数据
    :param factor_name: 因子名称
    :return: 原始数据+因子
    '''
    new_factor = (raw_df['close'] - raw_df['open']) / (raw_df['high'] - raw_df['low'])
    raw_df[factor_name] = new_factor
    return raw_df


@handle_dataloader_data
def add_basic_factors(raw_df)->pd.DataFrame:
    '''
    对于一个df 计算原始技术指标
    :param raw_df: 原始数据
    :return: 原始数据+技术指标
        对于大部分技术指标，实际上是在原有基础上增加了新的列。
    '''
    data_df = raw_df[['open', 'close', 'high', 'low', 'volume']]
    data_df.index = pd.to_datetime(raw_df.open_time, unit='ms')
    temp_factor_df = pd.DataFrame()  # 暂存特征，避免index不一致导致的错误
    temp_factor_df['trend_adx'] = adx(data_df['high'], data_df['low'], data_df['close'])['ADX_14']
    temp_factor_df['trend_cci'] = cci(data_df['high'], data_df['low'], data_df['close'])
    temp_factor_df['macd'] = macd(data_df['close'])['MACD_12_26_9']
    temp_factor_df['momentum_rsi'] = rsi(data_df['close'])
    temp_factor_df['volume_obv'] = obv(data_df['close'], data_df['volume'])
    temp_factor_df['volume_vwap'] = vwap(data_df['high'], data_df['low'], data_df['close'], data_df['volume'])
    temp_factor_df['volatility_atr'] = atr(data_df['high'], data_df['low'], data_df['close'])
    # (open - close)/(high - low)
    temp_factor_df['bop'] = bop(data_df['open'], data_df['high'], data_df['low'], data_df['close'])
    temp_factor_df.reset_index(drop=True, inplace=True)
    factor_df = pd.concat([raw_df, temp_factor_df], axis=1)
    return factor_df