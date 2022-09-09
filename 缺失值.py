import numpy as np
import pandas as pd
# 对于异常值在一定条件下进行填充
power_merge=pd.DataFrame()
power_merge['power_fans']=[1,2,3,5,0,6,0,0,4,5,0,0,0,7,5]

nan_r,_ = np.where(power_merge == 0)
df1 =pd.DataFrame(enumerate(nan_r))
df1['sub'] = df1.loc[:,1]-df1.loc[:,0]
df2 = df1.groupby('sub')

nan_list = []
for k,g in df2:
    find_list = list(g.loc[:,1])
    if len(find_list) < 3:
        nan_list = nan_list + find_list
print('nan_list',nan_list)

for i in nan_list:
    power_merge.loc[i,'power_fans']=power_merge.loc[i-1,'power_fans']
print(power_merge)
