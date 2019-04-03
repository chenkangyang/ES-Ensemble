
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pickle

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


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


# In[13]:


X = data_four_features.drop(['本周水质'], axis=1) # Series
y = data_four_features['本周水质'] # Series


# In[5]:


print("水质分布情况:")
print(y.value_counts())
print("\n各特征类型分布情况:")
print(data_four_features.dtypes.value_counts())


# In[6]:


# Convert series to ndarray
X = X.values
y = y.values


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
    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    SVC(probability=True),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    GaussianNB(),
    KNeighborsClassifier(),
    RandomForestClassifier(random_state=random_seed),
    ExtraTreesClassifier(random_state=random_seed)
]

y_pred_proba_all = []

# for model in models:
#     model_name = model.__class__.__name__
#     model.fit(X_train, y_train)
#     y_pred_proba = model.predict_proba(X_valid)
#     y_pred = model.predict(X_valid)
#     print("model_name: %s, valid weighted f1 score:%f" %(model_name, f1_score(y_valid, y_pred, average="weighted")))
#     y_pred_proba_all.append(y_pred_proba)
    
    
i=0
for model in models:
    model_name = model.__class__.__name__
    with open("../pkl/CE_97661/CE_" + model_name + ".pkl", "rb") as f:
        models[i] = pickle.load(f)
    y_pred_proba = models[i].predict_proba(X_valid)
    y_pred = models[i].predict(X_valid)
    print("model_name: %s, valid weighted f1 score:%f" %(model_name, f1_score(y_valid, y_pred, average="weighted")))
    y_pred_proba_all.append(y_pred_proba)
    i = i+1


# 测试集

# In[11]:


population_best_weight = np.load("../npy/CE_best_weights.npy")
mu = np.load("../npy/CE_best_mu.npy")
sigma = np.load("../npy/CE_best_sigma.npy")

classifier_num = 9


# 载入的9个模型标签为1-6，而argmax后的标签为0-5，需要加1

# In[12]:


y_test_pred_proba_all = []
for model in models:
    model_name = model.__class__.__name__
    y_test_pred_proba = model.predict_proba(X_test)
    y_test_pred = model.predict(X_test)
    print("model_name: %s, test weighted f1 score:%f" %(model_name, f1_score(y_test, y_test_pred, average="weighted")))
    y_test_pred_proba_all.append(y_test_pred_proba)
    
    
y_test_pred_ensemble_proba = np.zeros((len(y_test), 6)) # 集成器概率向量

# 为每一个基学习器乘上权重
for k in range(classifier_num):
    y_test_pred_ensemble_proba += y_test_pred_proba_all[k] * population_best_weight[k]
y_test_pred_ensemble = np.argmax(y_test_pred_ensemble_proba, axis=1)+1

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


# In[ ]:





# In[ ]:





# In[ ]:




