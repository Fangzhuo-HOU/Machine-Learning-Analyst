import pandas as pd
import tqdm
import numpy as np
import datetime

# TODO: datestamp construction
years = [i for i in range(2010, 2022)]
months = [i for i in range(1, 13)]

date_stamps = []
for year in years:
    for month in months:
        date_stamps.append(str(year) + '-' + str(month))

basic_table = pd.DataFrame(date_stamps)
basic_table.columns = ['date']
basic_table['date'] = pd.to_datetime(basic_table['date'])
basic_table['year'] = basic_table.apply(lambda x: int(str(x['date'])[:4]), axis=1)
basic_table['month'] = basic_table.apply(lambda x: int(str(x['date'])[5:7]), axis=1)

# TODO: merge_macro

month_macro_data = pd.read_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\macro\MONTH_MACRO_2010_2021.csv')
month_macro_data['year'] = month_macro_data.apply(lambda x: int(str(x['date'])[:4]), axis=1)
month_macro_data['month'] = month_macro_data.apply(lambda x: int(str(x['date'])[5:7]), axis=1)

month_macro_data['date'] = pd.to_datetime(month_macro_data['date'])

month_adjusted_result = pd.DataFrame()
for index, row in basic_table.iterrows():
    try:
        begin_date = row['date'] - datetime.timedelta(days=360)
        temp_data = month_macro_data[
            (month_macro_data['date'] >= begin_date) & (month_macro_data['date'] <= row['date'])]
        month_adjusted_result = month_adjusted_result.append(pd.DataFrame({'year': row['year'],
                                                                           'month': row['month'],
                                                                           'observation': len(temp_data),
                                                                           'cpi_mean': temp_data[
                                                                               'CPI: 居住：租赁房房租：当月同比'].mean(),
                                                                           'cpi_std': temp_data[
                                                                               'CPI: 居住：租赁房房租：当月同比'].std(),
                                                                           'm2_mean': temp_data['M2'].mean(),
                                                                           'm2_std': temp_data['M2'].std(),
                                                                           'electric_mean': temp_data[
                                                                               '产业：发电量（同比）'].mean(),
                                                                           'electric_std': temp_data[
                                                                               '产业：发电量（同比）'].std(),
                                                                           }, index=[0]))
    except Exception:
        print('error')
month_adjusted_result = month_adjusted_result[month_adjusted_result['observation'] >= 12]

quarter_macro_data = pd.read_csv(
    r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\macro\QUARTER_MACRO_2010_2021.csv')
quarter_macro_data['year'] = quarter_macro_data.apply(lambda x: int(str(x['date'])[:4]), axis=1)
quarter_macro_data['month'] = quarter_macro_data.apply(lambda x: int(str(x['date'])[5:7]), axis=1)
quarter_macro_data['date'] = pd.to_datetime(quarter_macro_data['date'])

quarter_adjusted_result = pd.DataFrame()
for index, row in basic_table.iterrows():
    try:
        begin_date = row['date'] - datetime.timedelta(days=360)
        temp_data = quarter_macro_data[
            (quarter_macro_data['date'] >= begin_date) & (quarter_macro_data['date'] <= row['date'])]
        quarter_adjusted_result = quarter_adjusted_result.append(pd.DataFrame({'year': row['year'],
                                                                               'month': row['month'],
                                                                               'observation': len(temp_data),
                                                                               'gdp_mean': temp_data['GDP'].mean(),
                                                                               'gdp_std': temp_data['GDP'].std()
                                                                               }, index=[0]))
    except Exception:
        print('error2')
quarter_adjusted_result = quarter_adjusted_result[quarter_adjusted_result['observation'] >= 4]

del month_adjusted_result['observation']
del quarter_adjusted_result['observation']

year_macro_data = pd.read_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\macro\YEAR_MACRO_2010_2021.csv')
year_macro_data['date'] = year_macro_data.apply(lambda x: int(str(x['date'])[:4]) - 1, axis=1)
year_macro_data.columns = ['ind_add_value', 'year']

temp = pd.merge(left=month_adjusted_result, right=quarter_adjusted_result, on=['year', 'month'])
temp = pd.merge(left=temp, right=year_macro_data, on=['year'])

temp.to_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\macro\merged_macro_indicator.csv', index=False)
