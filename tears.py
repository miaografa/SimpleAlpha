import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import warnings


import factors
import data_utils
import performance as perf
import utils
import plotting

class GridFigure(object):
    """
    It makes life easier with grid plots
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(14, rows * 7))
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return subplt

    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.gs = None


@plotting.customize
def create_information_tear_sheet(
    factor_data
):
    """
    Creates a tear sheet for information analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame -
        A DataFrame indexed by date (level 0),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    """




if __name__ == '__main__':
    dl = data_utils.DataLoader()
    dl.init_month_df('../data/15m_data/')
    # 计算未来收益率
    data_utils.compute_future_rtn_for_all(dl)
    # 计算因子
    tests_factor_name = 'test_factor'
    factors.add_new_factor_demo(dl, tests_factor_name)
    # 计算因子IC
    month_mean_ic = perf.cal_monthly_mean_ic(dl, factor_name=tests_factor_name)
    # 保存记录
    dl.set_record_df(kind='Info', factor_name=tests_factor_name, result_df=month_mean_ic)
    # 读取记录，并且展示作图
    month_mean_ic = dl.get_record_df(kind='Info', factor_name=tests_factor_name)
    plotting.plot_information_table(month_mean_ic)
    plotting.plot_ic_qq(month_mean_ic)
    plotting.plot_ic_ts(month_mean_ic)
    plotting.plot_ic_hist(month_mean_ic)
    plt.show()