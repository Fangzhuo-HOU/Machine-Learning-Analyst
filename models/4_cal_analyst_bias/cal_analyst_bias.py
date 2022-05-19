import pandas as pd
import numpy as np

analyst_forecasts = pd.read_csv(
    r'C:\PycharmProjects\ML_Analyst_forecast\ML_ANA_FORECAST\data\raw_data\analyst\GOGOAL_SINGLE_SHARE_FORECAST_REPORT_201301_202109.csv')
print(analyst_forecasts.columns)
analyst_forecasts['create_month'] = analyst_forecasts.apply(lambda x: int(str(x['create_date'])[5:7]), axis=1)
analyst_forecasts = analyst_forecasts[analyst_forecasts['create_month'] >= 9]
analyst_forecasts = analyst_forecasts[analyst_forecasts['report_quarter'] == 4]

machine_forecasts = pd.read_csv(
    r'C:\PycharmProjects\ML_Analyst_forecast\ML_ANA_FORECAST\data\forecast_result\forecast_result_month12.csv')

machine_forecasts = machine_forecasts[
    ['year', 'month', 'stock_code', 'net_profit', 'y_prediction', 'forecast_net_profit']]

machine_forecasts['analyst_forecast_bias'] = machine_forecasts.apply(
    lambda x: (x['forecast_net_profit'] - x['net_profit']), axis=1)
machine_forecasts['machine_forecast_bias'] = machine_forecasts.apply(
    lambda x: (x['y_prediction'] - x['net_profit']), axis=1)

machine_forecasts['analyst_minus_machine'] = machine_forecasts.apply(
    lambda x: (x['forecast_net_profit'] - x['y_prediction']), axis=1)

machine_forecasts.dropna(inplace=True)
machine_forecasts.drop_duplicates(inplace=True)

machine_forecasts.to_csv(r'C:\PycharmProjects\ML_Analyst_forecast\ML_ANA_FORECAST\analyst_and_machine_bias.csv',
                         index=False)
