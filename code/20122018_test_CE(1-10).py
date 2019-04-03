
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import sys
sys.path.append("..") 
from gcforest.gcforest import GCForest


# hyperparameters

# In[2]:


test_size = 0.33
random_seed = 42
cv=5
score = 'f1_weighted'
max_iteration = 100


# # load data_four_features

# In[3]:


path = os.getcwd()+'/../data/20122018freshwater_four_feature.csv'
data_four_features = pd.read_csv(path, na_values = np.nan)

print(data_four_features.dtypes)
print(data_four_features.shape)


# 因为加入gcforest，gcforest默认的y是从0开始计数的，所以将原本从1开始计数的y统一减去1

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                       stratify = y, random_state = random_seed)


# ### normalize  train data
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[8]:


clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler()),])
X_train = clean_pipeline.fit_transform(X_train)
X_test = clean_pipeline.fit_transform(X_test)


# ## TEST CE

# In[9]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, 
                                       stratify = y_train, random_state = random_seed)


# In[10]:


models = [
    "LogisticRegression",
    "LinearDiscriminantAnalysis",
    "SVC",
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "GaussianNB",
    "KNeighborsClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GCForest"
]


y_pred_proba_all = []

import pickle
i=0
for model in models:
    model_name = model
    model_path = "../pkl/CE_97661_10/CE_" + model_name + ".pkl"
    with open(model_path, "rb") as f:
        models[i] = pickle.load(f)
    y_pred_proba = models[i].predict_proba(X_valid)
    y_pred = models[i].predict(X_valid)
    print("%s, valid weighted f1 score:%f" %(model_name, f1_score(y_valid, y_pred, average="weighted")))
    y_pred_proba_all.append(y_pred_proba)
    i = i+1


# 测试集

# In[11]:


population_best_weight = np.load("../npy/CE_best_weights(1-10).npy")

classifier_num = 10


# In[12]:


y_test_pred_proba_all = []
for model in models:
    model_name = model.__class__.__name__
    y_test_pred_proba = model.predict_proba(X_test)
    y_test_pred = model.predict(X_test)
    print("%s, test weighted f1 score:%f" %(model_name, f1_score(y_test, y_test_pred, average="weighted")))
    y_test_pred_proba_all.append(y_test_pred_proba)
    
    
y_test_pred_ensemble_proba = np.zeros((len(y_test), 6)) # 集成器概率向量

# 为每一个基学习器乘上权重
for k in range(classifier_num):
    y_test_pred_ensemble_proba += y_test_pred_proba_all[k] * population_best_weight[k]
y_test_pred_ensemble = np.argmax(y_test_pred_ensemble_proba, axis=1)

print("NCE")
print(classification_report(y_test, y_test_pred_ensemble, digits=4))

cm = confusion_matrix(y_test, y_test_pred_ensemble)
i=0
acc_all = np.zeros(6)
for c in cm:
    acc_all[i] = c[i]/np.sum(c)
    print("%d accuaracy: %f" %(i+1, acc_all[i]))
    i=i+1
print("acc:", np.sum(y_test == y_test_pred_ensemble)/y_test_pred_ensemble.shape[0])
print('f1_weighted', f1_score(y_test, y_test_pred_ensemble, average='weighted'))


# In[23]:


np.around(population_best_weight/np.sum(population_best_weight), 3)


# In[ ]:




