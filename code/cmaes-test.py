
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

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

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import sys
sys.path.append("..") 
from gcforest.gcforest import GCForest
import cmaes as cma


# # 数据以及参数

# In[2]:


random_seed = 42


# 其余基本模型的参数都是sklearn默认的
# 
# 设置深度森林参数：最大层数20，连续5层没有效果（WF1）提升则停止，每一层：4个随机森林，3个决策树，1个逻辑回归

# In[3]:


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = random_seed
    ca_config["max_layers"] = 20
    ca_config["early_stopping_rounds"] = 5
    ca_config["n_classes"] = 6
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "random_state" : random_seed})
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "random_state" : random_seed})
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "random_state" : random_seed})
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "random_state" : random_seed})
    ca_config["estimators"].append({"n_folds": 5, "type": "DecisionTreeClassifier"})
    ca_config["estimators"].append({"n_folds": 5, "type": "DecisionTreeClassifier"})
    ca_config["estimators"].append({"n_folds": 5, "type": "DecisionTreeClassifier"})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


# In[4]:


path = os.getcwd()+'/../data/20122018freshwater_four_feature.csv'
data = pd.read_csv(path, na_values = np.nan)


# In[5]:


# training/valid/test: 0.6/0.2/0.2, 各数据集划分的时候要注意
X = data.drop(['本周水质'], axis=1).values # Series
y = data['本周水质'].values-1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                       stratify = y, random_state = random_seed)
# Z-score
clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler()),])
X_train = clean_pipeline.fit_transform(X_train)
X_test = clean_pipeline.fit_transform(X_test)
X_train2, X_valid, y_train2, y_valid = train_test_split(X_train, y_train, test_size=0.25, 
                                       stratify = y_train, random_state = random_seed)


# # 训练

# In[6]:


config = get_toy_config()

models = [
    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    SVC(probability=True),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    GaussianNB(),
    KNeighborsClassifier(),
    RandomForestClassifier(random_state=random_seed),
    ExtraTreesClassifier(random_state=random_seed),
    GCForest(config)
]

y_pred_proba_all = []

for model in models:
    model_name = model.__class__.__name__
    if model_name == 'GCForest':
        model.fit_transform(X_train2, y_train2, X_test, y_test)
    else:
        model.fit(X_train2, y_train2)
    y_pred_proba = model.predict_proba(X_valid)
    # y_pred = model.predict(X_valid)
    # print("%s, validation set: weighted F1 score:%f, Accuracy:%f" %(model_name, f1_score(y_valid, y_pred, average="weighted"), accuracy_score(y_valid, y_pred)))
    y_pred_proba_all.append(y_pred_proba)


# 深度森林训练结果：4层不再有效果提升
# 
# opt_layer_num=4, weighted_f1_train=99.45%, weighted_f1_test=97.66%
# 
# 60%训练集，20%的测试集仅用做展示

# # 验证

# In[7]:


y_pred_proba_all = np.asarray(y_pred_proba_all)
# np.save("../npy/y_pred_proba_all.npy", y_pred_proba_all)


# In[8]:


# 直接载入各个模型在验证集的概率向量，仅在懒得pre train的时候使用
# y_pred_proba_all = np.load("../npy/y_pred_proba_all.npy")


# In[9]:


es = cma.CMAEvolutionStrategy(10 * [0], 0.5)


# In[10]:


# help(cma)
# help(es)


# In[11]:


# score = "f1_weighted"
score = "accuracy"
while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [cma.ff.water_ensemble(x, y_pred_proba_all, y_valid, metric=score) for x in solutions])
    es.logger.add()  # write data to disc to be plotted
    es.disp()


# In[12]:


es.result_pretty()  # pretty print result
cma.plot()


# ## 得到10个基模型的权重

# In[13]:


weights = np.exp(es.result.xbest)/np.sum(np.exp(es.result.xbest))
# weights = np.load("../npy/cmaes_weights.npy")  # 载入保存的权重，直接载入仅在懒得训练CMAES时使用或者想得到固定的结果

# weights of each base models
print("CMAES weights", weights)
# np.save("../npy/cmaes_weights.npy", weights) # 保存结果


# # 测试

# 代入权重，在3个集合上计算ACC和F1
# 
# - 训练集（总样本数量的60%）X_train2
# - 验证集（总样本数量的20%，调参数，用CMAES得到权重）X_valid
# - 测试集（总样本数的20%）X_test
# - 各模型稳定性基于交叉验证，在除测试集外的80%上做5折 X_train

# ### ACC

# In[14]:


# 取训练好的模型
for model in models:
    model_name = model.__class__.__name__
    train_pred = model.predict(X_train2)
    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)
    print("=================" + model_name + "=================")
    train_cm = confusion_matrix(y_train2, train_pred)
    valid_cm = confusion_matrix(y_valid, valid_pred)
    test_cm = confusion_matrix(y_test, test_pred)
    i=0
    train_acc_all = np.zeros(6)
    for c in train_cm:
        train_acc_all[i] = c[i]/np.sum(c)
        print("%d train_acc: %.2f" %(i+1, 100*train_acc_all[i]))
        i=i+1
    print("average: %.2f" % (100*np.mean(train_acc_all)))
    i=0
    valid_acc_all = np.zeros(6)
    for c in valid_cm:
        valid_acc_all[i] = c[i]/np.sum(c)
        print("%d valid_acc: %.2f" %(i+1, 100*valid_acc_all[i]))
        i=i+1
    print("average: %.2f" % (100*np.mean(valid_acc_all)))
    i=0
    test_acc_all = np.zeros(6)
    for c in test_cm:
        test_acc_all[i] = c[i]/np.sum(c)
        print("%d test_acc: %.2f" %(i+1, 100*test_acc_all[i]))
        i=i+1
    print("average: %.2f" % (100*np.mean(test_acc_all)))


# - 计算CE（1-10）

population_best_weight = weights

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
print("=================CMAES=================")
train_cm = confusion_matrix(y_train2, y_train_pred_ensemble)
valid_cm = confusion_matrix(y_valid, y_valid_pred_ensemble)
test_cm = confusion_matrix(y_test, y_test_pred_ensemble)
i=0
train_acc_all = np.zeros(6)
for c in train_cm:
    train_acc_all[i] = c[i]/np.sum(c)
    print("%d train_acc: %.2f" %(i+1, 100*train_acc_all[i]))
    i=i+1
print("average: %.2f" % (100*np.mean(train_acc_all)))
i=0
valid_acc_all = np.zeros(6)
for c in valid_cm:
    valid_acc_all[i] = c[i]/np.sum(c)
    print("%d valid_acc: %.2f" %(i+1, 100*valid_acc_all[i]))
    i=i+1
print("average: %.2f" % (100*np.mean(valid_acc_all)))
i=0
test_acc_all = np.zeros(6)
for c in test_cm:
    test_acc_all[i] = c[i]/np.sum(c)
    print("%d test_acc: %.2f" %(i+1, 100*test_acc_all[i]))
    i=i+1
print("average: %.2f" % (100*np.mean(test_acc_all)))


# ### F1

# In[15]:


# 所有学习器都输出概率向量，最后投票
y_train_pred_proba_all = []
y_valid_pred_proba_all = []
y_test_pred_proba_all = []

# 取训练好的模型，计算各模型”验证集“上输出概率向量
for model in models:
    model_name = model.__class__.__name__
    train_pred_proba = model.predict_proba(X_train2)
    valid_pred_proba = model.predict_proba(X_valid)
    test_pred_proba = model.predict_proba(X_test)
    train_pred = model.predict(X_train2)
    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)
    print("=================" + model_name + "=================")
    print(classification_report(y_train2, train_pred, digits=4))
    print(classification_report(y_valid, valid_pred, digits=4))
    print(classification_report(y_test, test_pred, digits=4))

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
print("=================CMAES=================")

print(classification_report(y_train2, y_train_pred_ensemble, digits=4))
print(classification_report(y_valid, y_valid_pred_ensemble, digits=4))
print(classification_report(y_test, y_test_pred_ensemble, digits=4))

