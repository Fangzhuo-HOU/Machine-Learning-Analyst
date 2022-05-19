import pandas as pd
import tqdm
import numpy as np

analyst_forecast = pd.read_csv(
    r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\analyst\GOGOAL_SINGLE_SHARE_FORECAST_REPORT_201301_202109.csv')
print(analyst_forecast.columns)
analyst_forecast = analyst_forecast[analyst_forecast['report_quarter'] == 4]

analyst_forecast = analyst_forecast[~analyst_forecast['forecast_np'].isna()]
# forecast_np
analyst_forecast['create_year_month'] = analyst_forecast['create_date'].apply(lambda x: str(x)[:7])

analyst_forecast['create_year'] = analyst_forecast['create_date'].apply(lambda x: int(str(x)[:4]))

analyst_forecast = analyst_forecast[analyst_forecast['create_year'] == analyst_forecast['report_year']]

analyst_forecast = analyst_forecast[
    ['stock_code', 'stock_name', 'author_name', 'create_year_month', 'report_year', 'report_quarter', 'forecast_np']]

grouped = analyst_forecast.groupby(['stock_code', 'create_year_month'])

result_table = pd.DataFrame()

for name, group in tqdm.tqdm(grouped):
    consistent_forecast = group['forecast_np'].median()
    result_table = result_table.append(
        pd.DataFrame({'stock_code': name[0], 'forecast_date': name[1], 'consistent_ana_forecast': consistent_forecast},
                     index=[0]))

result_table.to_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\analyst\analyst_consistent_forecast.csv',
                    index=False)
