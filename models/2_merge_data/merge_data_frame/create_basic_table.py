import pandas as pd
import numpy as np
import datetime
import tqdm

stock_basic_info = pd.read_excel(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\stock_basic_info.xlsx')

stock_basic_info.columns = ['证券代码', '证券简称', '上市日期', 'Wind代码',
                            '所属申万三级行业指数代码']

final_data_frame = pd.DataFrame()

date_stamp = [datetime.datetime(year=2013, month=1, day=1),
              datetime.datetime(year=2013, month=2, day=1),
              datetime.datetime(year=2013, month=3, day=1),
              datetime.datetime(year=2013, month=4, day=1),
              datetime.datetime(year=2013, month=5, day=1),
              datetime.datetime(year=2013, month=6, day=1),
              datetime.datetime(year=2013, month=7, day=1),
              datetime.datetime(year=2013, month=8, day=1),
              datetime.datetime(year=2013, month=9, day=1),
              datetime.datetime(year=2013, month=10, day=1),
              datetime.datetime(year=2013, month=11, day=1),
              datetime.datetime(year=2013, month=12, day=1),
              datetime.datetime(year=2014, month=1, day=1),
              datetime.datetime(year=2014, month=2, day=1),
              datetime.datetime(year=2014, month=3, day=1),
              datetime.datetime(year=2014, month=4, day=1),
              datetime.datetime(year=2014, month=5, day=1),
              datetime.datetime(year=2014, month=6, day=1),
              datetime.datetime(year=2014, month=7, day=1),
              datetime.datetime(year=2014, month=8, day=1),
              datetime.datetime(year=2014, month=9, day=1),
              datetime.datetime(year=2014, month=10, day=1),
              datetime.datetime(year=2014, month=11, day=1),
              datetime.datetime(year=2014, month=12, day=1),
              datetime.datetime(year=2015, month=1, day=1),
              datetime.datetime(year=2015, month=2, day=1),
              datetime.datetime(year=2015, month=3, day=1),
              datetime.datetime(year=2015, month=4, day=1),
              datetime.datetime(year=2015, month=5, day=1),
              datetime.datetime(year=2015, month=6, day=1),
              datetime.datetime(year=2015, month=7, day=1),
              datetime.datetime(year=2015, month=8, day=1),
              datetime.datetime(year=2015, month=9, day=1),
              datetime.datetime(year=2015, month=10, day=1),
              datetime.datetime(year=2015, month=11, day=1),
              datetime.datetime(year=2015, month=12, day=1),
              datetime.datetime(year=2016, month=1, day=1),
              datetime.datetime(year=2016, month=2, day=1),
              datetime.datetime(year=2016, month=3, day=1),
              datetime.datetime(year=2016, month=4, day=1),
              datetime.datetime(year=2016, month=5, day=1),
              datetime.datetime(year=2016, month=6, day=1),
              datetime.datetime(year=2016, month=7, day=1),
              datetime.datetime(year=2016, month=8, day=1),
              datetime.datetime(year=2016, month=9, day=1),
              datetime.datetime(year=2016, month=10, day=1),
              datetime.datetime(year=2016, month=11, day=1),
              datetime.datetime(year=2016, month=12, day=1),
              datetime.datetime(year=2017, month=1, day=1),
              datetime.datetime(year=2017, month=2, day=1),
              datetime.datetime(year=2017, month=3, day=1),
              datetime.datetime(year=2017, month=4, day=1),
              datetime.datetime(year=2017, month=5, day=1),
              datetime.datetime(year=2017, month=6, day=1),
              datetime.datetime(year=2017, month=7, day=1),
              datetime.datetime(year=2017, month=8, day=1),
              datetime.datetime(year=2017, month=9, day=1),
              datetime.datetime(year=2017, month=10, day=1),
              datetime.datetime(year=2017, month=11, day=1),
              datetime.datetime(year=2017, month=12, day=1),
              datetime.datetime(year=2018, month=1, day=1),
              datetime.datetime(year=2018, month=2, day=1),
              datetime.datetime(year=2018, month=3, day=1),
              datetime.datetime(year=2018, month=4, day=1),
              datetime.datetime(year=2018, month=5, day=1),
              datetime.datetime(year=2018, month=6, day=1),
              datetime.datetime(year=2018, month=7, day=1),
              datetime.datetime(year=2018, month=8, day=1),
              datetime.datetime(year=2018, month=9, day=1),
              datetime.datetime(year=2018, month=10, day=1),
              datetime.datetime(year=2018, month=11, day=1),
              datetime.datetime(year=2018, month=12, day=1),
              datetime.datetime(year=2019, month=1, day=1),
              datetime.datetime(year=2019, month=2, day=1),
              datetime.datetime(year=2019, month=3, day=1),
              datetime.datetime(year=2019, month=4, day=1),
              datetime.datetime(year=2019, month=5, day=1),
              datetime.datetime(year=2019, month=6, day=1),
              datetime.datetime(year=2019, month=7, day=1),
              datetime.datetime(year=2019, month=8, day=1),
              datetime.datetime(year=2019, month=9, day=1),
              datetime.datetime(year=2019, month=10, day=1),
              datetime.datetime(year=2019, month=11, day=1),
              datetime.datetime(year=2019, month=12, day=1),
              datetime.datetime(year=2020, month=1, day=1),
              datetime.datetime(year=2020, month=2, day=1),
              datetime.datetime(year=2020, month=3, day=1),
              datetime.datetime(year=2020, month=4, day=1),
              datetime.datetime(year=2020, month=5, day=1),
              datetime.datetime(year=2020, month=6, day=1),
              datetime.datetime(year=2020, month=7, day=1),
              datetime.datetime(year=2020, month=8, day=1),
              datetime.datetime(year=2020, month=9, day=1),
              datetime.datetime(year=2020, month=10, day=1),
              datetime.datetime(year=2020, month=11, day=1),
              datetime.datetime(year=2020, month=12, day=1),
              datetime.datetime(year=2021, month=1, day=1),
              datetime.datetime(year=2021, month=2, day=1),
              datetime.datetime(year=2021, month=3, day=1),
              datetime.datetime(year=2021, month=4, day=1),
              datetime.datetime(year=2021, month=5, day=1),
              datetime.datetime(year=2021, month=6, day=1),
              datetime.datetime(year=2021, month=7, day=1),
              datetime.datetime(year=2021, month=8, day=1)
              ]

print(len(stock_basic_info))

for index, row in tqdm.tqdm(stock_basic_info.iterrows()):
    for date in date_stamp:
        row['forecast_date'] = date
        final_data_frame = final_data_frame.append(row)

final_data_frame.to_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\temporary_data\raw_dataset_v00.csv', index=False)

# import pandas as pd
#   ...: df1 = pd.read_csv(r'C:\Users\Finch\Desktop\TRD_Week2.csv')
#   ...: df2 = pd.read_csv(r'C:\Users\Finch\Desktop\TRD_Week.csv')
#   ...: df3 = pd.read_csv(r'C:\Users\Finch\Desktop\TRD_Week1.csv')
#   ...: df = df1.append(df2).append(df3)
# df.to_csv(r'C:\PycharmProjects\ML_Analyst_forecast\crash risk\data\stock_weekly_return.csv',index = False)


