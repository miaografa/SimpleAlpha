import pandas as pd
import numpy as np

def cal_factor_ic(data_df, factor_colname, target_cols=None, use_real_rtn=False) -> pd.Series:
    '''
    计算因子IC
    :return:
    use_real_rtn: 使用去除波动率偏差的真实收益率
    '''
    if target_cols == None:
        periods = (1, 5, 10, 20, 50, 100)
        target_cols = ['fwd_rtn_' + str(i) for i in periods]
        if use_real_rtn:
            target_cols = [x+"_real" for x in target_cols]
    result_series = data_df[target_cols].corrwith(data_df[factor_colname], method='spearman')
    return result_series


def cal_batch_factor_ic(data_df_list, factor_colname, target_cols=None, record_prefix=None, use_real_rtn=False) -> pd.DataFrame:
    """
    计算一系列DataFrame的因子信息系数（Factor IC）并返回一个记录DataFrame。

    Parameters:
    - dataframes (list of pd.DataFrame): 包含因子数据的DataFrame列表。

    Returns:
    - pd.DataFrame: 包含信息系数的记录DataFrame。
    """
    # 实现函数的代码
    result_df = pd.DataFrame()
    for i, df in enumerate(data_df_list):
        result_series = cal_factor_ic(df, factor_colname, target_cols, use_real_rtn=use_real_rtn)
        if record_prefix is not None:
            result_series.name = record_prefix + '_' + str(i)
        else:
            result_series.name = str(i)
        result_df = result_df.append(result_series)
    return result_df


def cal_icir_df(result_df) -> pd.DataFrame:
    '''
    从一个result里计算IC，IR
    result df 可以是一个月的所有IC，也可以是多个月的平均ic构成的df
    '''
    ICIR_df = pd.DataFrame(index=result_df.columns)
    ICIR_df['ic_mean'] = result_df.mean().T
    ICIR_df['ic_std'] = result_df.std().T
    ICIR_df['ir'] = ICIR_df['ic_mean'] / ICIR_df['ic_std']
    return ICIR_df.T


def cal_monthly_mean_ic(dl, factor_name:str='momentum_rsi', target_cols=None, record_prefix=None, use_real_rtn=False) -> pd.DataFrame:
    '''
    计算每个月的平均IC
    '''
    result_df = pd.DataFrame()
    for year in dl.years:
        for month in dl.months:
            temp_month_df_list = dl.get_month_df(year, month)
            if len(temp_month_df_list) == 0:
                continue  # 长度为0，说明这个月没有数据，跳过
            month_factor_ic = cal_batch_factor_ic(temp_month_df_list, factor_name, target_cols, record_prefix, use_real_rtn=use_real_rtn)  # 计算一个月的所有df的IC
            month_mean_ic = cal_icir_df(month_factor_ic).loc['ic_mean']  # 计算月度IC均值 Series
            if record_prefix is not None:
                month_mean_ic.name = record_prefix + '_' + str(year) + '_' + str(month)
            else:
                month_mean_ic.name = str(year) + '_' + str(month)
            result_df = result_df.append(month_mean_ic)
    return result_df



if __name__ == '__main__':
    from factors import add_basic_factors
    from data_utils import compute_future_rtn_for_all, DataLoader

    dl = DataLoader()
    dl.init_month_df('../data/15m_data/')

    # test batch ic, for single month
    # test_df_list = dl.get_month_df(2021, 10)
    # print(len(test_df_list))
    # test_df_list = data_utils.compute_future_rtn(test_df_list)
    # test_df_list = add_basic_features(test_df_list)
    # print(cal_batch_factor_ic(test_df_list, 'trend_adx'))

    # test batch ic, for all months
    compute_future_rtn_for_all(dl)
    add_basic_factors(dl)

    month_mean_ic = cal_monthly_mean_ic(dl, 'trend_adx')

    print(month_mean_ic)
    print('-' * 20)
    monthly_icir = cal_icir_df(month_mean_ic)
    print(monthly_icir)

