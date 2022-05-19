# import pandas as pd
#
# df = pd.read_excel(r'C:\Users\Finch\Desktop\中介组织发育和法律得分.xlsx')
# df.index = df['年份']
# del df['年份']
#
# df = df.unstack().reset_index()
#
# df.columns = ['kind', 'date', 'value']
#
# df['kind'] = df.apply(lambda x: str(x['kind'])[12:], axis=1)
# df['province'] = df.apply(lambda x: str(x['kind']).split('_')[-1], axis=1)
# df['kind'] = df.apply(lambda x: str(str(x['kind']).split('_')[:-1]), axis=1)
#
# df.replace({'kind' :{
#     "['消费者权益保护']":"消费者权益保护",
#     "['中介市场发育度']":"中介市场发育度",
#     "['中介市场发育度', '律师、会计师等市场组织服务条件']":"中介市场发育度_律师、会计师等市场组织服务条件",
#     "['知识产权保护']":"知识产权保护",
#     "['对生产者合法权益保护']":"对生产者合法权益保护",
#     "['知识产权保护', '专利申请受理量除以科技人员']":"知识产权保护_专利申请受理量除以科技人员",
#     "['知识产权保护', '专利申请批准数除以科技人员']":"知识产权保护_专利申请批准数除以科技人员",
#     "['中介市场发育度', '行业协会对企业帮助程度']":"中介市场发育度_行业协会对企业帮助程度",
#                      } },inplace = True)
#
#
#
# grouped = df.groupby(['date','province'])
#
# result = pd.DataFrame()
#
# for name,group in grouped:
#     result = result.append(pd.DataFrame({'年份':name[0],
#                                          'province': name[1],
#                                          '中介市场发育度':
#                                          '中介市场发育度_律师、会计师等市场组织服务条件':
#                                          '对生产者合法权益保护':
#                                          '消费者权益保护':
#                                          },index = [0]))
#     break