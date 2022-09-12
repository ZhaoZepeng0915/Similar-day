import numpy as np
import pandas as pd
#循环所有需要处理的列
def data_prepare(df):
   data = df
   list1 = ['power_fans','power_frrr']
   for c in list1:
      data_processing = pd.DataFrame()
      data_processing['data'] = data.loc[:, c].copy()

      # 对于异常值在一定条件下进行填充
      nan_r, _ = np.where(data_processing == 0)
      df1 = pd.DataFrame(enumerate(nan_r))
      df1['sub'] = df1.loc[:, 1] - df1.loc[:, 0]
      df2 = df1.groupby('sub')

      nan_list = []
      for k, g in df2:
         find_list = list(g.loc[:, 1])
         if len(find_list) < 4:
            nan_list = nan_list + find_list
      for i in nan_list:
         data.loc[i, c] = data.loc[i - 1, c]
   return (data)
power_merge=pd.DataFrame()
power_merge['power_fans']=[1,2,3,5,0,6,0,0,4,5,0,0,0,7,5]
power_merge['power_frrr']=[1,0,3,5,0,0,0,0,4,5,0,1,8,7,5]
print(power_merge)
power_merge_new = data_prepare(power_merge)
print(power_merge_new)