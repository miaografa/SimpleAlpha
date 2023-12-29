'''
记录机器学习的performance
'''
from itertools import product
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, auc
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from data_utils import DataLoader


class MLFactorEvaluator:
    def __init__(self, method, record_names:list, factor_names:list, cv=10, params_dict={}):
        assert len(record_names) == len(factor_names), 'record_names和factor_names长度不一致'
        self.record_names = record_names
        self.factor_dict = dict(zip(record_names, factor_names))
        self.record_dict = dict()
        self.cv = cv
        self.method = method
        if len(params_dict) == 0:
            self.params_dict = {'max_depth': 5}
        else:
            self.params_dict = params_dict


    def evaluate(self, dataloader:DataLoader, y_col):
        '''
        评估所有的因子
        '''
        self.valid_year_month = dataloader.get_valid_year_month()  # 有效的年月列表
        # 进行时序cv划分
        self.train_test_indices = self.time_series_split(len(self.valid_year_month))   # train_test_indices = [(train_ind, test_ind), ...]
        # 评估每一个因子
        for record_name in self.record_names:
            auc_list = []
            f1_list = []
            factor_name = self.factor_dict[record_name]  # 取出对应的因子名
            for cv_idx in trange(len(self.train_test_indices)):
                train_ind, test_ind = self.train_test_indices[cv_idx]
                # 获取数据
                train_df = self.get_data(dataloader, train_ind)
                test_df = self.get_data(dataloader, test_ind)
                y_train = train_df[y_col]
                y_test = test_df[y_col]
                x_train = train_df[factor_name]
                x_test = test_df[factor_name]
                # 训练并评估
                f1, auc_score = self.train_and_evaluate(x_train, y_train, x_test, y_test)
                f1_list.append(f1)
                auc_list.append(auc_score)
            # 记录结果
            temp_record_df = pd.DataFrame({'f1': f1_list, 'auc': auc_list})
            self.record_dict[record_name] = temp_record_df
        return

    def time_series_split(self, time_length:int) -> list:
        '''
        生成时间序列的索引
        :return:
            [(train_ind, test_ind), ...]
        '''
        tscv = TimeSeriesSplit(n_splits=self.cv, max_train_size=12, test_size=3)
        X = np.ones(time_length)
        train_test_indices = [index for index in tscv.split(X)]
        return train_test_indices

    def get_data(self, dataloader:DataLoader, indices):
        '''
        通过idx 读取对应的月份的数据
        :param dataloader: DataLoader
        :param indices: 训练集索引，或者测试集索引
        :return:
        '''
        valid_year_month = self.valid_year_month
        df_list = []
        for ind in indices:
            year, month = valid_year_month[ind]
            temp_df_list = dataloader.get_month_df(year, month)
            df_list += temp_df_list  # 合并为一个更长的list，每一个元素都是一个df
        data_df = pd.concat(df_list, axis=0)
        data_df = data_df[np.abs(data_df['close_Z']) > 1.07]  # 默认的cut
        data_df.dropna(inplace=True)
        return data_df

    def train_and_evaluate(self, x_train, y_train, x_test, y_test):
        '''
            单次评估模型返回评估结果
        '''
        model = self.method(**self.params_dict)
        # 模型训练
        model.fit(x_train, y_train)
        Y_pred = model.predict(x_test)
        f1 = f1_score(y_test, Y_pred)
        auc_score = auc(y_test, Y_pred)
        return f1, auc_score


    def get_assess_df(self, metric='f1'):
        '''
        获取评估结果的df，没想好怎么写更方便，暂时先能够按照f1，auc的格式输出
        :return:
        '''
        assess_df = pd.DataFrame()
        for record_name in self.record_names:
            temp_record = self.record_dict[record_name][metric]
            assess_df[record_name] = temp_record
        return assess_df


if __name__ =='__main__':
    pass