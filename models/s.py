import pandas as pd
from WindPy import w

w.start()  # 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒

w.isconnected()  # 判断WindPy是否已经登录成功

# QUARTER
GDP = w.edb("M5567876", "2010-01-01", "2021-11-23", "Fill=Previous")
GDP = pd.DataFrame({'GDP': GDP.Data[0], 'date': GDP.Times})

# MONTH
CPI_house = w.edb('M0000603', '2010-01-01', '2021-11-23', 'Fill=Previous')
CPI_house = pd.DataFrame({'CPI: 居住：租赁房房租：当月同比': CPI_house.Data[0], 'date': CPI_house.Times})

house_city_price = w.edb('S2773070', '2010-01-01', '2021-11-23', 'Fill=Previous')
house_city_price = pd.DataFrame({'城市住房租赁价格指数：全国': house_city_price.Data[0], 'date': house_city_price.Times})

local_debt = w.edb('M5639024', '2010-01-01', '2021-11-23', 'Fill=Previous')
local_debt = pd.DataFrame({'地方政府债务': local_debt.Data[0], 'date': local_debt.Times})

M2 = w.edb('M0001384', '2010-01-01', '2021-11-23', 'Fill=Previous')

M2 = pd.DataFrame({'M2': M2.Data[0], 'date': M2.Times})

electricity = w.edb('S0027013', '2010-01-01', '2021-11-23', 'Fill=Previous')
electricity = pd.DataFrame({'产业：发电量（同比）': electricity.Data[0], 'date': electricity.Times})

df = pd.merge(left=CPI_house, right=M2, on='date')
df = pd.merge(left=df, right=electricity, on='date', how='left')

# YEAR
industry_increase = w.edb('M0041964', '2010-01-01', '2021-11-23', 'Fill=Previous')
industry_increase = pd.DataFrame({'工业增加值': industry_increase.Data[0], 'date': industry_increase.Times})
