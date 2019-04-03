
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

from sklearn.metrics import f1_score
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
print("水质分布情况:")
print(y.value_counts())
print("\n各特征类型分布情况:")
print(data_four_features.dtypes.value_counts())


# In[5]:


# Convert series to ndarray
X = X.values
y = y.values


# In[6]:


print("============ train_test_split ============")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                       stratify = y, random_state = random_seed)
print("67%% train: %d/%d, 33%% test: %d/%d" %(X_train.shape[0], X.shape[0], X_test.shape[0], X.shape[0]))


# ### normalize  train data
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[7]:


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

# ## Accuracy's MEAN(SD) based on 5 flod cross validation
# 1. 载入CE（1-9）的子模型，仅计算CE（1-9）
# 2. 载入CE（1-10）的子模型覆盖之前的模型，计算10个子模型
# 3. 计算CE（1-10）
# 4. 共得到12个模型的MEAN(SD)

# - 载入CE（1-9）的子模型，仅计算CE（1-9），标签为1-6
# 
# argmax后标签为0-5，需要加1

# In[8]:


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

scores = []
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X_train, y_train):
    K_train_x, K_test_x = X_train[train_index], X_train[test_index]
    K_train_y, K_test_y = y_train[train_index], y_train[test_index]

    # 所有学习器都输出概率向量，最后投票
    y_test_pred_proba_all = []
    # 取训练好的模型，计算各模型”验证集“上输出概率向量
    for model in models:
        model_name = model.__class__.__name__
        y_test_pred_proba = model.predict_proba(K_test_x)
        y_test_pred_proba_all.append(y_test_pred_proba)

    y_test_pred_ensemble_proba = np.zeros((len(K_test_y), 6)) # 初始化集成器概率向量
    
    # 为每一个基学习器乘上权重
    for k in range(classifier_num):
        y_test_pred_ensemble_proba += y_test_pred_proba_all[k] * population_best_weight[k]
    y_test_pred_ensemble = np.argmax(y_test_pred_ensemble_proba, axis=1) + 1
    
    scores.append(f1_score(K_test_y, y_test_pred_ensemble, average="weighted"))
print("CE(1-9)  Weighted F1: %0.2f $\pm$ %0.2f %%" % (np.mean(scores)*100, np.std(scores)*100))


# - 载入CE（1-10）的子模型覆盖之前的模型，计算10个子模型，标签为 0-5

# In[9]:


y_train = y_train-1
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
for model in models:
    model_name = model
    model_path = "../pkl/CE_97661_10/CE_" + model_name + ".pkl"
    with open(model_path, "rb") as f:
        models[i] = pickle.load(f)
    
    # Kfold
    scores = []
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X_train, y_train):
        K_train_x, K_test_x = X_train[train_index], X_train[test_index]
        K_train_y, K_test_y = y_train[train_index], y_train[test_index]
        
        y_pred = models[i].predict(K_test_x)
        scores.append(f1_score(K_test_y, y_pred,average="weighted"))
    print("%s    Weighted F1: %0.2f $\pm$ %0.2f %%" % (model_name, np.mean(scores)*100, np.std(scores)*100))
    i = i+1


# - 计算CE（1-10）

# In[10]:


population_best_weight = np.load("../npy/CE_best_weights(1-10).npy")

classifier_num = 10

scores = []
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X_train, y_train):
    K_train_x, K_test_x = X_train[train_index], X_train[test_index]
    K_train_y, K_test_y = y_train[train_index], y_train[test_index]

    # 所有学习器都输出概率向量，最后投票
    y_test_pred_proba_all = []
    # 取训练好的模型，计算各模型”验证集“上输出概率向量
    for model in models:
        model_name = model.__class__.__name__
        y_test_pred_proba = model.predict_proba(K_test_x)
        y_test_pred_proba_all.append(y_test_pred_proba)

    y_test_pred_ensemble_proba = np.zeros((len(K_test_y), 6)) # 初始化集成器概率向量
    
    # 为每一个基学习器乘上权重
    for k in range(classifier_num):
        y_test_pred_ensemble_proba += y_test_pred_proba_all[k] * population_best_weight[k]
    y_test_pred_ensemble = np.argmax(y_test_pred_ensemble_proba, axis=1)
    
    scores.append(f1_score(K_test_y, y_test_pred_ensemble, average="weighted"))
print("CE(1-10)  Weighted F1: %0.2f $\pm$ %0.2f %%" % (np.mean(scores)*100, np.std(scores)*100))


# In[ ]:





# In[ ]:




