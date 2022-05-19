import pandas as pd
import os

files = os.listdir(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\fundamental')
df = pd.DataFrame()
for file in files:
    df = df.append(
        pd.read_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\fundamental' + '\\' + file, encoding='GBK'))

df = df[df.columns[:-1]]
k = [str(i).split('_')[1] for i in df.columns]

df.columns = k

df.rename(columns={'A': 'A_Stkcd'}, inplace=True)
df.rename(columns={'B': 'B_Stkcd'}, inplace=True)
df.rename(columns={'H': 'H_stkcd'}, inplace=True)

df.to_csv(r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\fundamental\RESSET_FUNDAMENTAL_20110101_20210930.csv',index = False)