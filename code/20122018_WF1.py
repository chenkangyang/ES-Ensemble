
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pickle

import sys
sys.path.append("..") 
from gcforest.gcforest import GCForest


# hyperparameters

# In[2]:


random_seed = 42
kf=10
score = 'f1_weighted'


# # load data_four_features

# In[3]:


path = os.getcwd()+'/../data/20122018freshwater_four_feature.csv'
data_four_features = pd.read_csv(path, na_values = np.nan)

print(data_four_features.shape)


# - CE(1-9)标签y为1-6
# - CE(1-10)标签y为0-5
# 
# 先进行计算CE(1-9)，在载入CE(1-10)的时候要将标签统一减去1

# In[4]:


X = data_four_features.drop(['本周水质'], axis=1) # Series
y = data_four_features['本周水质'] # Series


# In[5]:


X = data_four_features.drop(['本周水质'], axis=1) # Series
y = data_four_features['本周水质'] # Series
print("水质分布情况:")
print(y.value_counts())
print("\n各特征类型分布情况:")
print(data_four_features.dtypes.value_counts())


# In[6]:


# Convert series to ndarray
X = X.values
y = y.values


# In[7]:


print("============ train_test_split ============")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                       stratify = y, random_state = random_seed)
print("67%% train: %d/%d, 33%% test: %d/%d" %(X_train.shape[0], X.shape[0], X_test.shape[0], X.shape[0]))


# ### normalize  train data
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[8]:


clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler()),])
X_train = clean_pipeline.fit_transform(X_train)
X_test = clean_pipeline.fit_transform(X_test)

print("============ train_valid_split ============")
X_train2, X_valid, y_train2, y_valid = train_test_split(X_train, y_train, test_size=0.25, 
                                       stratify = y_train, random_state = random_seed)


# - X_train2: training set
# - X_valid: validation set
# - X_test: test set

# ## Accuracy on 3 parts of data
# 
# 1. 载入CE（1-9）的子模型，仅计算CE（1-9），标签1-6
# 2. 载入CE（1-10）的子模型覆盖之前的模型，计算10个子模型,标签0-5
# 3. 计算CE（1-10）

# - 载入CE（1-9）的子模型，仅计算CE（1-9），标签1-6

# In[9]:


models = [
    "LogisticRegression",
    "LinearDiscriminantAnalysis",
    "SVC",
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "GaussianNB",
    "KNeighborsClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier"
]
i=0
for model in models:
    model_name = model
    with open("../pkl/CE_97661/CE_" + model_name + ".pkl", "rb") as f:
        models[i] = pickle.load(f)
    i = i+1
# models 不再是字符数组，而是模型数组

population_best_weight = np.load("../npy/CE_best_weights.npy")

classifier_num = 9

# 所有学习器都输出概率向量，最后投票
y_train_pred_proba_all = []
y_valid_pred_proba_all = []
y_test_pred_proba_all = []

# 取训练好的模型，计算各模型”验证集“上输出概率向量
for model in models:
    train_pred_proba = model.predict_proba(X_train2)
    valid_pred_proba = model.predict_proba(X_valid)
    test_pred_proba = model.predict_proba(X_test)
    y_train_pred_proba_all.append(train_pred_proba)
    y_valid_pred_proba_all.append(valid_pred_proba)
    y_test_pred_proba_all.append(test_pred_proba)
    
y_train_pred_ensemble_proba = np.zeros((len(y_train2), 6)) # 初始化集成器概率向量
y_valid_pred_ensemble_proba = np.zeros((len(y_valid), 6)) # 初始化集成器概率向量
y_test_pred_ensemble_proba = np.zeros((len(y_test), 6)) # 初始化集成器概率向量

# 为每一个基学习器乘上权重
for k in range(classifier_num):
    y_train_pred_ensemble_proba += y_train_pred_proba_all[k] * population_best_weight[k]
    y_valid_pred_ensemble_proba += y_valid_pred_proba_all[k] * population_best_weight[k]
    y_test_pred_ensemble_proba += y_test_pred_proba_all[k] * population_best_weight[k]
y_train_pred_ensemble = np.argmax(y_train_pred_ensemble_proba, axis=1) + 1
y_valid_pred_ensemble = np.argmax(y_valid_pred_ensemble_proba, axis=1) + 1
y_test_pred_ensemble = np.argmax(y_test_pred_ensemble_proba, axis=1) + 1

# 计算各水质等级的得分
print("=================CE(1-9)=================")

print(classification_report(y_train2, y_train_pred_ensemble, digits=4))
print(classification_report(y_valid, y_valid_pred_ensemble, digits=4))
print(classification_report(y_test, y_test_pred_ensemble, digits=4))


# - 载入CE（1-10）的子模型覆盖之前的模型，计算10个子模型,标签0-5

# In[10]:


y_train2 = y_train2-1
y_valid = y_valid-1
y_test = y_test-1

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

i=0
for name in models:
    model_path = "../pkl/CE_97661_10/CE_" + name + ".pkl"
    with open(model_path, "rb") as f:
        models[i] = pickle.load(f)
    i = i+1


# In[11]:


for model in models:
    model_name = model.__class__.__name__
    train_pred = model.predict(X_train2)
    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)
    print("=================" + model_name + "=================")
    print(classification_report(y_train2, train_pred, digits=4))
    print(classification_report(y_valid, valid_pred, digits=4))
    print(classification_report(y_test, test_pred, digits=4))


# - 计算CE（1-10）

# In[12]:


population_best_weight = np.load("../npy/CE_best_weights(1-10).npy")

classifier_num = 10

# 所有学习器都输出概率向量，最后投票
y_train_pred_proba_all = []
y_valid_pred_proba_all = []
y_test_pred_proba_all = []

# 取训练好的模型，计算各模型”验证集“上输出概率向量
for model in models:
    train_pred_proba = model.predict_proba(X_train2)
    valid_pred_proba = model.predict_proba(X_valid)
    test_pred_proba = model.predict_proba(X_test)
    y_train_pred_proba_all.append(train_pred_proba)
    y_valid_pred_proba_all.append(valid_pred_proba)
    y_test_pred_proba_all.append(test_pred_proba)
    
y_train_pred_ensemble_proba = np.zeros((len(y_train2), 6)) # 初始化集成器概率向量
y_valid_pred_ensemble_proba = np.zeros((len(y_valid), 6)) # 初始化集成器概率向量
y_test_pred_ensemble_proba = np.zeros((len(y_test), 6)) # 初始化集成器概率向量

# 为每一个基学习器乘上权重
for k in range(classifier_num):
    y_train_pred_ensemble_proba += y_train_pred_proba_all[k] * population_best_weight[k]
    y_valid_pred_ensemble_proba += y_valid_pred_proba_all[k] * population_best_weight[k]
    y_test_pred_ensemble_proba += y_test_pred_proba_all[k] * population_best_weight[k]
y_train_pred_ensemble = np.argmax(y_train_pred_ensemble_proba, axis=1)
y_valid_pred_ensemble = np.argmax(y_valid_pred_ensemble_proba, axis=1)
y_test_pred_ensemble = np.argmax(y_test_pred_ensemble_proba, axis=1)

# 计算各水质等级的得分
print("=================CE(1-10)=================")

print(classification_report(y_train2, y_train_pred_ensemble, digits=4))
print(classification_report(y_valid, y_valid_pred_ensemble, digits=4))
print(classification_report(y_test, y_test_pred_ensemble, digits=4))

