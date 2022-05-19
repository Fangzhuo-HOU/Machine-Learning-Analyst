import pandas as pd
import tqdm
import numpy as np

finance_report_data = pd.read_csv(
    r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\fundamental\RESSET_FUNDAMENTAL_20110101_20210930.csv')
finance_report_data = finance_report_data[finance_report_data['Adjflg'] == 1]
finance_report_data = finance_report_data[['A_Stkcd', 'B_Stkcd', 'H_stkcd',
                                           'Lstflg', 'AccStd', 'Reporttype', 'EndDt', 'Infopubdt',
                                           'InfoSource', 'BasEPS', 'DilutEPS', 'BasEPSCut', 'DilutEPSCut', 'EPS',
                                           'ROEByRep', 'ROE', 'ROECut', 'ROEW', 'ROEWCut', 'Incmope', 'Netinvincm',
                                           'Finexp', 'NetincmFVC', 'OpePrf', 'Nopeincm', 'Nopeexp', 'Totalprf',
                                           'IncTax', 'Uncfinvlos', 'NPwiomin', 'Minprf', 'Netprf', 'NRecProLos',
                                           'NetprfCut', 'PrfatISA', 'Marinoustat', 'NCFbyope', 'NCFfropePS',
                                           'NCFfrinv', 'NCFfrfin', 'IncrinCCE', 'EffFERonCCE', 'CCEatend',
                                           'MoneFd', 'TraFinass', 'Intrecv', 'Divrecv', 'Accrecv', 'Othrecv',
                                           'Invtr', 'Totcurass', 'Soldfinass', 'Holdinvatterm', 'Invinest',
                                           'LTequinv', 'Intanass', 'TotNcurass', 'Totass', 'STloan', 'Trafinlia',
                                           'Empsalpay', 'Divpay', 'Taxpay', 'Intpay', 'Otherpay', 'NCurLia1Yr',
                                           'Totcurlia', 'TotNcurlia', 'Totlia', 'Shrcap', 'Capsur', 'Surres',
                                           'Retear', 'SHEwioMin', 'MinSHE', 'TotSHE', 'TotliaSHE', 'NetassISA',
                                           'NAPSbyrep', 'NAPS', 'NAPSadj', 'Totshr']]
finance_report_data.to_csv(
    r'C:\PycharmProjects\ML_Analyst_forecast\data\raw_data\fundamental\financial_report_data.csv', index=False)

#   Adjflg	  调整标识	  Adjustment Flag	数值	  1.
# 0－原始报表；
# 1－最新调整数据。
