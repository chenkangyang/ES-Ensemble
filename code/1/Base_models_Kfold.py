
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

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import sys
sys.path.append("..") 
from gcforest.gcforest import GCForest


# # 数据及参数

# In[2]:


random_seed = 42
cv=5
score = 'f1_weighted'


# In[3]:


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = random_seed
    ca_config["max_layers"] = 10
    ca_config["early_stopping_rounds"] = 3
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


X = data.drop(['本周水质'], axis=1).values # Series
y = data['本周水质'].values.reshape(-1,1) - 1

# 1. 中位数填充缺失值，2.Z-score标准化
clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler())])
X = clean_pipeline.fit_transform(X)


# In[6]:


X.shape


# # k折交叉验证

# In[7]:


# function：使用5折交叉验证统计各类别5次平均后的Acc，5次平均后的F1，和模型的总Acc以及总Weighted F1

# Input: 
#     X: 总样本
#     y: 总样本
#     model: function
#     cv: cross_validation的次数
# Output:
#     Acc_mean, 各类别的Acc
#     F1_mean, 各类别F1
#     Support_mean, 各类别预测样本占总样本的比重
#     Acc, 总Acc
#     F1_weighted 总Weighted F1


def kftrain(X, y, model, cv):
    model_name = model.__class__.__name__

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_class = np.unique(y).shape[0]
    Acc_matrix = np.zeros((n_class, cv)) # 矩阵(6,5) 第i行：类别i的在cv组test data的cv个acc
    F1_matrix = np.zeros((n_class, cv)) # 矩阵(6,5) 第i行：类别i的在cv组test data的cv个acc
    cv_F1_weighted = np.zeros(cv) # 5次的F1_weighted
    cv_Acc = np.zeros(cv) # 5次的Acc
    # 各类别的Support，每个类别的support由5次fold后得到的5个support求平均得到，这里support：预测为该类别的样本占总样本的比例
    # 各类别的F1，由5次fold后得到的5个F1求平均得到
    # 所以，Weighted F1 等于“5次fold得到的5个 Weighted F1 求平均得到” ——等价于—— “6个support*6个F1”得到
    Support_matrix = np.zeros((n_class, cv)) # 矩阵(6,5) 5次fold后各类别的support
    
    k = 0
    skf = StratifiedKFold(n_splits=cv) # 定义5折分层划分器
    
    for train_index, test_index in skf.split(X, y):
        K_train_x, K_test_x = X[train_index], X[test_index]
        K_train_y, K_test_y = y[train_index], y[test_index]
        
        if model_name == 'GCForest':
            model.fit_transform(K_train_x, K_train_y)
        else:
            model.fit(K_train_x, K_train_y)
    
        K_test_y_pred = model.predict(K_test_x)
        
        # 由混淆矩阵计算各类别的Acc
        test_cm = confusion_matrix(K_test_y, K_test_y_pred)
        test_acc_all_class = np.zeros(n_class) # 6个类别上的测试acc
        i = 0
        for c in test_cm:
            test_acc_all_class[i] = c[i]/np.sum(c)
            i += 1
        Acc_matrix[:,k] = test_acc_all_class
        
        # 由classification_report提取f1
        cr = classification_report(K_test_y, K_test_y_pred, digits=4)
        test_f1_all_class = np.zeros(n_class) # 6个类别上的测试f1
        support_all_class = np.zeros(n_class) # 6个类别的support
        i = 0
        for l in range(2,8):
            test_f1_all_class[i] = float(cr.splitlines()[l].split()[3])
            support_all_class[i] = float(cr.splitlines()[l].split()[4])/n_samples
            i = i + 1
        F1_matrix[:,k] = test_f1_all_class
        Support_matrix[:,k] = support_all_class
        
        # cv_F1_weighted[k] = float(cr.splitlines()[9].split()[5]) # 第k折时的Weighted F1
        cv_F1_weighted[k] = f1_score(K_test_y, K_test_y_pred, average="weighted")
        cv_Acc[k] = accuracy_score(K_test_y, K_test_y_pred)
        
        k += 1
    
    Acc_mean = np.mean(Acc_matrix, axis=1) # 6个类别的k-fold平均acc
    
    F1_mean = np.mean(F1_matrix, axis=1) # 6个类别的k-fold平均F1
    Support_mean = np.mean(Support_matrix, axis=1) # 6个类别的k-fold平均support
    
    Acc = np.mean(Acc_mean)
    F1_weighted = np.mean(cv_F1_weighted) # 5个Weighted F1 求平均，并可以证明其等价于 np.sum(F1_mean*Support_mean)
    SD_Acc = np.std(cv_Acc)
    SD_F1 = np.std(cv_F1_weighted)
    
    return Acc_mean, F1_mean, Support_mean, Acc, F1_weighted, SD_Acc, SD_F1


# In[ ]:


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

for model in models:
    model_name = model.__class__.__name__
    print(model_name)
    Acc_class, F1_class, Support_class, Acc, F1_weighted, SD_Acc, SD_F1 = kftrain(X, y, model, 5)
    print("===Accuracy===")
    for i in range(len(Acc_class)):
        print("Class %d: %.2f%%" %(i, Acc_class[i]*100))
    print("Overall: %.2f%%" %(Acc*100))
    print("Mean±SD: %.2f±%.2f%%" %(Acc*100,SD_Acc*100))

    print("===   F1  ===")
    for i in range(len(F1_class)):
        print("Class %d: %.2f%%" %(i, F1_class[i]*100))
    print("F{beta}: %.2f%%" %(F1_weighted*100))
    print("Mean±SD: %.2f±%.2f%%" %(F1_weighted*100,SD_F1*100))
    print("\n===============\n")


# In[ ]:




