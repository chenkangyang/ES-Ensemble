
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

import sys
sys.path.append("..") 
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json


# hyperparameters

# In[2]:


random_seed = 42
score = 'f1_weighted'


# gc config

# # load data_four_features

# In[3]:


path = os.getcwd()+'/../data/20122018freshwater_four_feature.csv'
data_four_features = pd.read_csv(path, na_values = np.nan)

print(data_four_features.dtypes)
print(data_four_features.shape)


# In[4]:


X = data_four_features.drop(['本周水质'], axis=1) # Series
y = data_four_features['本周水质']-1 # Series


# In[5]:


print("水质分布情况:")
print(y.value_counts())
print("\n各特征类型分布情况:")
print(data_four_features.dtypes.value_counts())


# In[6]:


data_four_features.head()


# In[7]:


print("============ train_test_split ============")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                       stratify = y, random_state = random_seed)
print("80%% train: %d/%d, 20%% test: %d/%d" %(X_train.shape[0], X.shape[0], X_test.shape[0], X.shape[0]))


# ### normalize  train data
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[8]:


clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler()),])
X_train = clean_pipeline.fit_transform(X_train)
X_test = clean_pipeline.fit_transform(X_test)


# In[9]:


y_train = y_train.values
y_test = y_test.values


# ## test gc

# In[10]:


with open("../pkl/CE_97661_10/CE_GCForest.pkl", "rb") as f:
    gc = pickle.load(f)
y_pred = gc.predict(X_test)

print("============= 20122018 datasets' results on test =============")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
i=0
acc_all = np.zeros(6)
for c in cm:
    acc_all[i] = c[i]/np.sum(c)
    print("%d accuaracy: %f" %(i+1, acc_all[i]))
    i=i+1
print("acc:", np.sum(y_test == y_pred)/y_pred.shape[0])
print('f1_weighted', f1_score(y_test, y_pred, average='weighted'))


# In[ ]:




