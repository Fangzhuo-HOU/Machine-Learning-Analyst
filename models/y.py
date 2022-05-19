import pandas as pd

initial_value = 1
df = pd.read_excel(r'C:\Users\Finch\Desktop\净值\1年1阶.xlsx').sort_values(by='日期')
dates = df['日期'].to_list()

df['last'] = df['净值(分红再投)'].shift()
df.dropna(inplace = True)
df['净值变化率(%)'] = (df['净值(分红再投)'] - df['last']) / df['last']
df = df[df['净值变化率(%)'] != '--']


df.reset_index(inplace=True, drop=True)
net_value = []
for index, row in df.iterrows():
    if index == 0:
        net_value.append(initial_value)
    else:
        initial_value = initial_value * (1 + float(row['净值变化率(%)']) / 100)
        net_value.append(initial_value)


result = pd.DataFrame({'date': dates, 'net_value': net_value})
print(result)