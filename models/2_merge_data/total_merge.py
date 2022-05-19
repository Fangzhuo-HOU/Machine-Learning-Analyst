import pandas as pd
import tqdm
import numpy as np
import datetime

basic_table = pd.read_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\temporary_data\raw_dataset_v10.csv')

stock_shenwan_ind_info = pd.read_excel(
    r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\ind\上市公司所属申万行业指数代码对照表.xlsx')
stock_shenwan_ind_info = stock_shenwan_ind_info.dropna()
stock_shenwan_ind_info.columns = ['证券代码', '证券简称', '申万三级行业代码']

basic_table = pd.merge(left=basic_table, right=stock_shenwan_ind_info, on=['证券代码', '证券简称'])
basic_table['year'] = basic_table.apply(lambda x: int(str(x['forecast_date'])[:4]), axis=1)
basic_table['month'] = basic_table.apply(lambda x: int(str(x['forecast_date'])[5:7]), axis=1)

macro_basic_info = pd.read_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\macro\merged_macro_indicator.csv')

ind_basic_info = pd.read_csv(
    r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\ind\clean_shenwan_industry_index.csv')
ind_basic_info['year'] = ind_basic_info.apply(lambda x: int(str(x['date'])[:4]), axis=1)
ind_basic_info['month'] = ind_basic_info.apply(lambda x: int(str(x['date'])[5:7]), axis=1)
del ind_basic_info['date']

analyst_info = pd.read_csv(
    r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\analyst\analyst_consistent_forecast.csv')

fundamental_basic_info = pd.read_csv(
    r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\fundamental\financial_report_data.csv')
fundamental_basic_info = fundamental_basic_info[fundamental_basic_info['AccStd'] == 1]
fundamental_basic_info = fundamental_basic_info[['A_Stkcd', 'Reporttype', 'EndDt', 'Infopubdt', 'EPS', 'ROE', 'Incmope',
                                                 'OpePrf', 'Netprf', 'NCFbyope', 'NCFfrinv', 'NCFfrfin', 'IncrinCCE',
                                                 'MoneFd', 'Intrecv', 'Divrecv',
                                                 'Accrecv', 'Othrecv', 'Invtr', 'Totcurass', 'Intanass', 'TotNcurass',
                                                 'Totass', 'STloan', 'Empsalpay', 'Divpay', 'Taxpay', 'Intpay',
                                                 'Otherpay', 'NCurLia1Yr', 'Totcurlia', 'TotNcurlia', 'Totlia',
                                                 'Shrcap', 'Capsur', 'Surres', 'Retear', 'NAPS'
                                                 ]]

k = ['A_Stkcd', 'Reporttype', 'EndDt', 'Infopubdt', 'EPS', 'ROE', 'Incmope',
     'OpePrf', 'Netprf', 'NCFbyope', 'NCFfrinv', 'NCFfrfin', 'IncrinCCE',
     'MoneFd', 'Intrecv', 'Divrecv',
     'Accrecv', 'Othrecv', 'Invtr', 'Totcurass', 'Intanass', 'TotNcurass',
     'Totass', 'STloan', 'Empsalpay', 'Divpay', 'Taxpay', 'Intpay',
     'Otherpay', 'NCurLia1Yr', 'Totcurlia', 'TotNcurlia', 'Totlia',
     'Shrcap', 'Capsur', 'Surres', 'Retear', 'NAPS'
     ]
fundamental_basic_info.columns = [i.lower() for i in k]

# TODO: merge
basic_table['stock_code'] = basic_table.apply(lambda x: int(str(x['Wind代码'])[:6]), axis=1)

# Macro
basic_table = pd.merge(left=basic_table, right=macro_basic_info, on=['year', 'month'], how='left')

# Industry
del basic_table['所属申万三级行业指数代码']
ind_basic_info.rename(columns={'shenwan_industry_index': '申万三级行业代码'}, inplace=True)

basic_table = pd.merge(left=basic_table, right=ind_basic_info, on=['year', 'month', '申万三级行业代码'], how='left')

# Analyst
analyst_info['year'] = analyst_info.apply(lambda x: int(str(x['forecast_date'])[:4]), axis=1)
analyst_info['month'] = analyst_info.apply(lambda x: int(str(x['forecast_date'])[5:7]), axis=1)

del analyst_info['forecast_date']
basic_table = pd.merge(left=basic_table, right=analyst_info, on=['year', 'month', 'stock_code'], how='left')

# Fundamental
basic_table['frd_year'] = basic_table.apply(lambda x: x['year'] - 1 if x['month'] <= 4 else x['year'], axis=1)
basic_table['frd_quarter'] = basic_table.apply(
    lambda x: "Q4" if x['month'] <= 4 else "Q1" if (x['month'] > 4 and x['month'] < 9)
    else "Q2" if (x['month'] > 8 and x['month'] < 11) else "Q3" if (x['month'] > 10) else np.nan, axis=1)

fundamental_basic_info['stock_code'] = fundamental_basic_info.apply(
    lambda x: int(x['a_stkcd']) if str(x['a_stkcd']).isdigit() else np.nan, axis=1)
fundamental_basic_info = fundamental_basic_info[~fundamental_basic_info['stock_code'].isna()]
del fundamental_basic_info['a_stkcd']
fundamental_basic_info.rename(columns={'reporttype': 'frd_quarter'}, inplace=True)
fundamental_basic_info['frd_year'] = fundamental_basic_info.apply(lambda x: int(str(x['enddt'])[:4]), axis=1)

final = pd.merge(left=basic_table, right=fundamental_basic_info, on=['frd_year', 'frd_quarter', 'stock_code'],
                 how='left')

# basic_table = basic_table[basic_table['year'] < 2020]
# seperated_num = 100
# spread = 2346
# final = pd.DataFrame()
# for i in range(seperated_num):
#     print(i)
#     temp = basic_table[i * spread:(i + 1) * spread]
#     temp2 = fundamental_basic_info[fundamental_basic_info['stock_code'].isin(list(set(temp['stock_code'].to_list())))]
#     temp = pd.merge(left=temp, right=temp2, on=['frd_year', 'frd_quarter','stock_code'], how='left')
#     final = final.append(temp)
print(final.columns)

final = final[['Wind代码', 'forecast_date', '上市日期', '证券代码', '证券简称', '申万三级行业代码', 'year',
               'month', 'stock_code', 'cpi_mean', 'cpi_std', 'm2_mean', 'm2_std',
               'electric_mean', 'electric_std', 'gdp_mean', 'gdp_std', 'ind_add_value',
               'ret', 'std', 'consistent_ana_forecast', 'frd_year', 'frd_quarter',
               'enddt', 'infopubdt', 'eps', 'roe', 'incmope', 'opeprf', 'netprf',
               'ncfbyope', 'ncffrinv', 'ncffrfin', 'incrincce', 'monefd', 'intrecv',
               'divrecv', 'accrecv', 'othrecv', 'invtr', 'totcurass', 'intanass',
               'totncurass', 'totass', 'stloan', 'empsalpay', 'divpay', 'taxpay',
               'intpay', 'otherpay', 'ncurlia1yr', 'totcurlia', 'totncurlia', 'totlia',
               'shrcap', 'capsur', 'surres', 'retear', 'naps']]

# TODO: merge net profit

net_profit = pd.read_excel(r'C:\Users\Finch\Downloads\FS_Comins.xlsx')
net_profit = net_profit[net_profit['Typrep'] == 'A']
net_profit['year'] = net_profit.apply(lambda x: int(str(x['Accper'])[:4]), axis=1)
net_profit['month'] = net_profit.apply(lambda x: int(str(x['Accper'])[5:7]), axis=1)

net_profit = net_profit[net_profit['month'] == 12]
net_profit = net_profit[['Stkcd', '营业利润', 'year']]
net_profit.columns = ['stock_code', 'forecast_net_profit', 'year']

final = pd.merge(left=final, right=net_profit, on=['stock_code', 'year'], how='left')
final = final[~final['forecast_net_profit'].isna()]
final = final[final['eps'] != '年度报告']
final = final[final['eps'] != '跟踪评级报告']
final = final[final['eps'] != '第一季报']
final = final[final['eps'] != '第三季报']
final = final[final['eps'] != '同业存单发行计划']
final = final[final['eps'] != '募集说明书']
final = final[final['eps'] != '半年度报告']
final = final[final['eps'] != '半年报']
final = final[final['eps'] != '其他']
final = final[final['eps'] != '年度报告（东方证券）']
final = final[final['eps'] != '半年报（东方证券）']
final = final[final['eps'] != '半年度报告（兴业银行）']
final = final[final['eps'] != '更正公告']


for i in final.columns[9:]:
    if i == 'frd_quarter':
        continue
    if i == 'frd_year':
        continue
    if i == 'enddt':
        continue
    if i == 'infopubdt':
        continue

    #
    final[i] = final[i].astype(float)

final.to_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\train_data\raw_train_data.csv', index=False, encoding='GBK')
