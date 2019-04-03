
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE

from sklearn import preprocessing
from sklearn.pipeline import Pipeline


# #### somte sampling

# In[2]:


def Smoter(X, y, is_random=False):
    if is_random == True:
        random_lst = list(np.random.randint(0, 1000, 4))
    elif is_random == False:
        random_lst = [0] * 4

    print("rs:", random_lst)
    sm = SMOTE(random_state=random_lst[2], kind = 0.24)
    X_smote, y_smote = sm.fit_sample(X, y)

    return X_smote, y_smote


# # load all data

# In[3]:


path = os.getcwd()+'/../data/water/csv/20122018freshwater.csv'

data = pd.read_csv(path, na_values = np.nan)

print(data.dtypes)
print(data.shape)


# In[4]:


drop_columns = []
continuous_features = ['pH', 'DO(mg/l)', 'CODMn(mg/l)', 'NH3-N(mg/l)']
cat_features =['水系', '点位名称', '河流名称']


# In[5]:


# 独热编码
data_dummies = pd.get_dummies(data, columns=cat_features)


# In[6]:


# 舍弃无用特征
data_dummies.drop(drop_columns, 1, inplace=True)


# In[7]:


# 删除空行
data_dummies = data_dummies.dropna(axis=0)


# In[8]:


data_dummies[data_dummies.isnull().values==True]


# In[9]:


X = data_dummies.drop(['本周水质'], axis=1) # Series
y = data_dummies['本周水质'] # Series


# In[10]:


print("水质分布情况:")
print(y.value_counts())
print("\n各特征类型分布情况:")
print(data_dummies.dtypes.value_counts())


# In[11]:


data_dummies.head()


# In[12]:


output_path = os.getcwd()+'/../data/water/csv/20122018freshwater_dummies.csv'
data_dummies.to_csv(output_path, encoding='utf-8', index=False)


# In[13]:


# 再存储只保存数字类特征的data
drop_columns = ['水系', '点位名称', '河流名称']
data.drop(drop_columns, 1, inplace=True)
# 删除空行
data = data.dropna(axis=0)


# In[14]:


data[data.isnull().values==True]


# In[15]:


output_path = os.getcwd()+'/../data/water/csv/20122018freshwater_four_feature.csv'
data.to_csv(output_path, encoding='utf-8', index=False)

