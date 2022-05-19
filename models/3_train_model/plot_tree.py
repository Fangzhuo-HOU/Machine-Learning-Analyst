# author:Fangzhuo HOU
# contact: 11812411@mail.sustech.edu.cn
# datetime:2021/1/23 13:39
# software: PyCharm

"""
    File specification:

    """
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.figure(dpi=1800)
# 将pandans中的数据导入到DMatrix中
# data = pd.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c'])
# label = pd.DataFrame(np.random.randint(2, size=4))
# dtrain = xgb.DMatrix(data, label=label)

import warnings

warnings.filterwarnings("ignore")



import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# 将pandans中的数据导入到DMatrix中
# data = pd.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c'])
# label = pd.DataFrame(np.random.randint(2, size=4))
# dtrain = xgb.DMatrix(data, label=label)

import warnings

warnings.filterwarnings("ignore")

import xgboost
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from jedi.api.refactoring import inline
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import time
import copy
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import auc, roc_curve
import warnings
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# 添加中文显示
plt.rc('font', family='SimHei', size=13)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True
np.random.seed(42)
warnings.filterwarnings("ignore")


def parameter_tuning(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

    param1 = {'max_depth': [1, 2], 'min_child_weight': [1]}
    gsearch1 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate=0.1,
            n_estimators=500,
            max_depth=4,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:linear',
            # nthread=8,
            seed=6
        ),
        param_grid=param1, scoring='roc_auc', cv=5)
    # cv:交叉验证参数，默认None，使用三折交叉验证。

    gsearch1.fit(X_train, y_train)
    print(gsearch1.scorer_)
    print(gsearch1.best_params_, gsearch1.best_score_)  # 最佳参数（形式是字典型），最高分数（就一个值）
    best_max_depth = gsearch1.best_params_['max_depth']  # 输出的max_depth的values
    best_min_child_weight = gsearch1.best_params_['min_child_weight']  # 同理上面


if __name__ == '__main__':
    data = pd.read_csv(r'C:\PycharmProjects\ML_Analyst_forecast\ML_ANA_FORECAST\data\train_data\raw_train_data.csv',
                       encoding='GBK')
    month_12_data = data[data['month'] == 12]
    # TODO: 净利润

    net_profit = pd.read_csv(
        r'C:\PycharmProjects\ML_Analyst_forecast\ML_ANA_FORECAST\data\raw_data\fundamental\FS_Comins.csv')
    net_profit = net_profit[net_profit['Typrep'] == 'A']
    net_profit['year'] = net_profit.apply(lambda x: int(str(x['Accper'])[:4]), axis=1)
    net_profit['month'] = net_profit.apply(lambda x: int(str(x['Accper'])[5:7]), axis=1)
    net_profit = net_profit[net_profit['month'] == 12]
    net_profit = net_profit[['Stkcd', 'year', 'month', 'B002000000']]
    net_profit.columns = ['stock_code', 'year', 'month', 'net_profit']

    month_12_data = pd.merge(left=month_12_data, right=net_profit, on=['stock_code', 'year', 'month'], how='left')

    y = month_12_data[['net_profit']]
    X = month_12_data[['cpi_mean', 'cpi_std', 'm2_mean', 'm2_std',
                       'electric_mean', 'electric_std', 'gdp_mean', 'gdp_std', 'ind_add_value',
                       'ret', 'std', 'consistent_ana_forecast', 'eps', 'roe', 'incmope', 'opeprf', 'netprf',
                       'ncfbyope', 'ncffrinv', 'ncffrfin', 'incrincce', 'monefd', 'intrecv',
                       'divrecv', 'accrecv', 'othrecv', 'invtr', 'totcurass', 'intanass',
                       'totncurass', 'totass', 'stloan', 'empsalpay', 'divpay', 'taxpay',
                       'intpay', 'otherpay', 'ncurlia1yr', 'totcurlia', 'totncurlia', 'totlia',
                       'shrcap', 'capsur', 'surres', 'retear', 'naps', 'forecast_net_profit']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=8)
    model = xgb.XGBRegressor(
        learning_rate=0.1,
        # 0.01,0.1
        n_estimators=1000,
        # 200,300,1000
        max_depth=1,
        # 3,4
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        # 3,4,5
        objective='reg:linear',
        booster='gbtree'
    )
    model.fit(X_train, y_train)

    y_predication = model.predict(X_test)
    y_all_prediction = pd.DataFrame(model.predict(X))
    y_all_prediction.columns = ['y_prediction']
    month_12_data.reset_index(drop=True, inplace=True)
    final = pd.concat([month_12_data, y_all_prediction], axis=1)

    # print(model.score(X_test, y_test))
    # model.save_model(r'C:\PycharmProjects\ML_Analyst_forecast\ML_ANA_FORECAST\data\0001.model')

    # print(model.feature_importances_)

    model.get_booster().feature_names = list(X.columns)
    # plot_importance(model.get_booster(), max_num_features=20)
    # plt.show()

# # 特征重要性
# # print('数据重要性排序')
# cancer = datasets.load_breast_cancer()
# X = cancer.data
# # 569 rows × 30 columns
# y = cancer.target
# # 569 rows × 1 columns
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5., random_state=8)
#
# xgb_train = xgb.DMatrix(X_train, label=y_train)
# xgb_test = xgb.DMatrix(X_test, label=y_test)
#
# params = {
#     "objective": "binary:logistic",
#     "booster": "gbtree",
#     "eta": 0.1,
#     "max_depth": 5
# }
#
# num_round = 50
#
# watchlist = [(xgb_test, 'eval'), (xgb_train, 'train')]
#
# bst = xgb.train(params, xgb_train, num_round, watchlist)
# # xgb.plot_tree(bst,num_trees=1)
# # plt.show()
#
# # 以fscore为指标
# importance = bst.get_fscore()
# # 以gain和cover为指标
# # importance = bst.get_score(importance_type='gain')
# # importance = bst.get_score(importance_type='cover')
#
# importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
#
# df = pd.DataFrame(importance, columns=['feature', 'fscore'])
#
# # print(df)
#
# df['fscore'] = df['fscore'] / df['fscore'].sum()
# # print(df)
#
# xgb.plot_importance(bst,height=0.5)
# plt.show()





# # 特征重要性
# print('数据重要性排序')
# cancer = datasets.load_breast_cancer()
# X = cancer.data
# # 569 rows × 30 columns
# y = cancer.target
# # 569 rows × 1 columns
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5., random_state=8)
#
# xgb_train = xgb.DMatrix(X_train, label=y_train)
# xgb_test = xgb.DMatrix(X_test, label=y_test)
#
# params = {
#     "objective": "binary:logistic",
#     "booster": "gbtree",
#     "eta": 0.1,
#     "max_depth": 5
# }
#
# num_round = 50
#
# watchlist = [(xgb_test, 'eval'), (xgb_train, 'train')]
#
# bst = xgb.train(params, xgb_train, num_round, watchlist)
#
#
# def plot_tree(xgb_model, filename, rankdir='UT'):
#     """
#     Plot the tree in high resolution
#     :param xgb_model: xgboost trained model
#     :param filename: the pdf file where this is saved
#     :param rankdir: direction of the tree: default Top-Down (UT), accepts:'LR' for left-to-right tree
#     :return:
#     """
#     import xgboost as xgb
#     import os
#     gvz = xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration, rankdir=rankdir)
#     _, file_extension = os.path.splitext(filename)
#     format = file_extension.strip('.').lower()
#     data = gvz.pipe(format=format)
#     full_filename = filename
#     with open(full_filename, 'wb') as f:
#         f.write(data)
#
#
# bst.get_booster().feature_names = [i for i in range(len(X_train))]
# plot_tree(bst, r'test_tree.pdf')
# plt.show()
