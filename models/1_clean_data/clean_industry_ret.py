import pandas as pd

industry_index = pd.read_csv(
    r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\ind\INDUSTRY_DAILY_20100101_20211124.csv')

industry_index.index = industry_index['DATETIME']
del industry_index['DATETIME']
industry_index = industry_index.unstack().reset_index()
industry_index.columns = ['shenwan_industry_index', 'datetime', 'index_value']
industry_index['datetime'] = pd.to_datetime(industry_index['datetime'])
industry_index['year-month'] = industry_index.apply(lambda x: str(x['datetime'])[:7], axis=1)

grouped = industry_index.groupby(['shenwan_industry_index', 'year-month'])

result_table = pd.DataFrame()

for name, group in grouped:
    group = group.sort_values('datetime')
    ret = (group.iloc[-1]['index_value'] - group.iloc[0]['index_value']) / group.iloc[0]['index_value']
    std = group['index_value'].std()

    result_table = result_table.append(
        pd.DataFrame({'shenwan_industry_index': name[0], 'date': name[1], 'ret': ret, 'std': std}, index=[0]))

result_table.to_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\ind\clean_shenwan_industry_index.csv',
                    index=False)
