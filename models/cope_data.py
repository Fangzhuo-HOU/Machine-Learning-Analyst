import pandas as pd
import numpy as np
import tqdm

# TODO: 合并两个表的数据

df1 = pd.read_csv(r'E:\DATAHOUSE\sun\gogoal_rpt_forecast_stk.csv')
df1.columns = [
    'id',
    'report_id',
    'stock_code',
    'stock_name',
    'title',
    'report_type',
    'reliability',
    'organ_id',
    'organ_name',
    'author_name',
    'create_date',
    'report_year',
    'report_quarter',
    'forecast_or',
    'forecast_op',
    'forecast_tp',
    'forecast_np',
    'forecast_eps',
    'forecast_dps',
    'forecast_rd',
    'forecast_pe',
    'forecast_roe',
    'forecast_ev_ebitda',
    'organ_rating_code',
    'organ_rating_content',
    'gg_rating_code',
    'gg_rating_content',
    'target_price_ceiling',
    'target_price_floor',
    'current_price',
    'refered_capital',
    'is_capital_change',
    'currency',
    'settlement_date',
    'language',
    'attention',
    'entrytime',
    'updatetime',
    'tmstamp'
]

df1 = df1[
    ['report_id', 'stock_code', 'stock_name', 'report_type', 'reliability', 'organ_id', 'organ_name', 'author_name',
     'create_date', 'report_year', 'report_quarter', 'forecast_or', 'forecast_op', 'forecast_tp', 'forecast_np',
     'forecast_eps', 'forecast_dps', 'forecast_rd', 'forecast_pe', 'forecast_roe', 'forecast_ev_ebitda', 'language',
     'attention', 'entrytime', 'updatetime']]

df1 = df1[df1['language'] == 0.0]
'''
statistics summary:
                    False     True
forecast_or         827988   49687  *
forecast_op         779895   97780
forecast_tp         756335  121340
forecast_np         847477   30198  *
forecast_eps        865310   12365
forecast_dps        605394  272281
forecast_rd         632260  245415
forecast_pe         758504  119171
forecast_roe        794106   83569
forecast_ev_ebitda  572101  305574

create_time:


'''
df1['create_year'] = df1.apply(lambda x: str(x['create_date'])[:4], axis=1)
print(df1['create_year'].value_counts())

df1.to_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\analyst\table1.csv', index=False)
