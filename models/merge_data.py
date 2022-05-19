import pandas as pd

df1 = pd.read_csv(r'D:\QuantChinaData\sun_back\back2\gogoal_one_bak_DER_REPORT_RESEARCH.csv')
df2 = pd.read_csv(r'D:\QuantChinaData\sun_back\back2\gogoal_one_bak_DER_REPORT_SUBTABLE.csv')

df2.rename(columns={'report_search_id': 'id'}, inplace=True)

df = pd.merge(left=df1, right=df2, on='id')

df['create_year'] = df.apply(lambda x: str(x['create_date'])[:4],axis=1   )
# df['forecast_year'] = df.apply(lambda x: str(x['']))
df['time_year'] = df['time_year'].astype(str)
df = df[df['text8'] == 0]
final = df[df['time_year'] == df['create_year']]
# df1 = df1[
#     ['report_id', 'stock_code', 'stock_name', 'report_type', 'reliability', 'organ_id', 'organ_name', 'author_name',
#      'create_date', 'report_year', 'report_quarter', 'forecast_or', 'forecast_op', 'forecast_tp', 'forecast_np',
#      'forecast_eps', 'forecast_dps', 'forecast_rd', 'forecast_pe', 'forecast_roe', 'forecast_ev_ebitda', 'language',
#      'attention', 'entrytime', 'updatetime']]

final = final[['id','code','code_name','title','content','type_id','organ_id','author','create_date','time_year','quarter','forecast_income',
               'forecast_profit','forecast_income_share','forecast_return_cash_share', 'forecast_return_capital_share',
       'forecast_return_y']]

final.columns =['report_id','stock_code','stock_name','title','content','type_id','organ_id','author_name','create_date','report_year',
                'report_quarter','forecast_or',
               'forecast_np','forecast_income_share','forecast_return_cash_share', 'forecast_return_capital_share',
       'forecast_op']



final.to_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\analyst\table2.csv',index = False)






