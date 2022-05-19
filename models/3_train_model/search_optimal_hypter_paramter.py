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
    print(model.score(X_test, y_test))
    model.save_model(r'C:\PycharmProjects\ML_Analyst_forecast\ML_ANA_FORECAST\data\0001.model')
