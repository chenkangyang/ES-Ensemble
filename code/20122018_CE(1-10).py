
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
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.append("..") 
from gcforest.gcforest import GCForest


# hyperparameters

# In[2]:


random_seed = 42
cv=5
score = 'f1_weighted'


# #### somte sampling

# In[3]:


def Smoter(X, y, is_random=False):
    if is_random == True:
        random_lst = list(np.random.randint(0, 1000, 4))
    elif is_random == False:
        random_lst = [0] * 4

    print("rs:", random_lst)
    sm = SMOTE(random_state=random_lst[2], kind = 0.24)
    X_smote, y_smote = sm.fit_sample(X, y)

    return X_smote, y_smote


# In[4]:


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


# # load data_four_features

# In[5]:


path = os.getcwd()+'/../data/20122018freshwater_four_feature.csv'
data_four_features = pd.read_csv(path, na_values = np.nan)

print(data_four_features.dtypes)
print(data_four_features.shape)


# In[6]:


X = data_four_features.drop(['本周水质'], axis=1) # Series
y = data_four_features['本周水质']-1 # Series


# In[7]:


print("水质分布情况:")
print(y.value_counts())
print("\n各特征类型分布情况:")
print(data_four_features.dtypes.value_counts())


# In[8]:


data_four_features.head()


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                       stratify = y, random_state = random_seed)


# ### normalize  train data
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[10]:


clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler()),])
X_train = clean_pipeline.fit_transform(X_train)
X_test = clean_pipeline.fit_transform(X_test)


# # model selection

# In[11]:


X_train2, X_valid, y_train2, y_valid = train_test_split(X_train, y_train, test_size=0.25, 
                                       stratify = y_train, random_state = random_seed)


# load gc config

# In[12]:


y_train2 = y_train2.values
y_train = y_train.values
y_valid = y_valid.values
y_test = y_test.values


# In[90]:


config = get_toy_config()
model = GCForest(config)

model.fit_transform(X_train2, y_train2, X_test, y_test)
gc_valid_proba = model.predict_proba(X_valid)
gc_pred = model.predict(X_valid)


# In[14]:


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

# 训练固定的基学习器
for model in models:
    model_name = model.__class__.__name__
    if model_name == 'GCForest':
        model.fit_transform(X_train2, y_train2, X_test, y_test)
    else:
        model.fit(X_train2, y_train2)
    y_pred_proba = model.predict_proba(X_valid)
    y_pred = model.predict(X_valid)
    print("%s, validation set: weighted F1 score:%f, Accuracy:%f" %(model_name, f1_score(y_valid, y_pred, average="weighted"), accuracy_score(y_valid, y_pred)))
    y_pred_proba_all.append(y_pred_proba)


# # train weights via NCE Ensemble on validation set

# In[220]:


train_step = 100
classifier_num = 10
population_num = 1000
retain_population_num = 100
max_iteration = 50
population_weights = np.zeros((population_num, classifier_num))
population_retain_weights = np.zeros((retain_population_num, classifier_num))
population_score = []
population_retain_score = []

all_best_weights = np.zeros((max_iteration, classifier_num)) # 某次训练时，所有迭代步骤中最好的种群的权重
all_best_f1s = np.zeros(max_iteration) # 某次训练时，每次迭代都取精英种群中最高的f1，构成这个“最高f1数组”
all_mean_f1s = np.zeros(max_iteration) # 某次训练时，每次迭代都取精英种群f1的均值，构成这个“平均f1数组”
all_best_f1s_mean = np.zeros(train_step) # 每次训练最高f1数组的均值, 即 np.mean(all_best_f1s)
all_best_f1s_std = np.zeros(train_step) # 每次训练最高f1数组的标准差, 即 np.std(all_best_f1s)
all_mean_f1s_mean = np.zeros(train_step)
all_mean_f1s_std = np.zeros(train_step)

mu = np.zeros(classifier_num)
sigma = np.ones(classifier_num)


# In[221]:


# 在验证集集上: 训练每个基学习器的投票参数
for i in range(max_iteration):
    print("Iteration: %d" %(i))
    # 该次迭代的所有种群们
    population_score = np.zeros(population_num)
    population_weights = np.zeros((population_num, classifier_num))
    # 该次迭代的优势种群们
    population_retain_score = np.zeros(retain_population_num)
    population_retain_weights = np.zeros((retain_population_num, classifier_num))
    
    # 生成所有种群
    for j in range(classifier_num):
        w = np.random.normal(mu[j], sigma[j]+700/(i+1), population_num)
        # w = np.random.normal(mu[j], sigma[j], population_num)
        population_weights[:,j] = w
        
    # 映射所有种群的权重至[0:1]    
    for j in range(population_num):
        w2 = np.zeros(classifier_num)
        for k in range(classifier_num):
            w2[k] = np.exp(-population_weights[j][k]*population_weights[j][k])
            # w2[k] = np.exp(population_weights[j][k])/np.sum(np.exp(population_weights[j]))
        population_weights[j] = w2
    
    # 计算所有种群得分
    for j in range(population_num):
        y_pred_ensemble_proba = np.zeros((len(y_valid), 6)) # 集成器概率向量
        # 为每一个基学习器乘上权重
        for k in range(classifier_num):
            y_pred_ensemble_proba += y_pred_proba_all[k] * population_weights[j][k]
        y_pred_ensemble = np.argmax(y_pred_ensemble_proba, axis=1)
        f1 = f1_score(y_valid, y_pred_ensemble, average="weighted")
        population_score[j] = f1

    # 所有种群得分按降序排列
    retain_index = np.argsort(-np.array(population_score))[:retain_population_num]
    
    # 记录该次迭代中的优势种群们
    population_retain_weights = population_weights[retain_index]
    population_retain_score = np.array(population_score)[retain_index]
    
    # 记录每次迭代最好的种群和value
    all_best_weights[i] = population_retain_weights[0]
    all_best_f1s[i] = population_retain_score[0]
    all_mean_f1s[i] = np.mean(population_retain_score)
    # 更新mu，sigma为优势种群们的分布
    mu = np.mean(population_retain_weights, axis = 0)
    sigma = np.std(population_retain_weights, axis = 0) #default: ddof = 0, The divisor used in calculations is N - ddof
#     print("mu\n",mu)
#     print("sigma\n", sigma)
#     print("Weighted F1 Score after rank")
#     print(population_retain_score)
#     print("Weights")
#     print(population_retain_weights)


# In[222]:


last_weight = population_retain_weights[0]
last_f1 = all_best_f1s[-1]
best_f1 = all_best_f1s[np.argmax(all_best_f1s)]
best_weight = population_retain_weights[np.argmax(all_best_f1s)]
print("Last f1: %f" % (last_f1))
print("Best f1: %f" % (best_f1))
print("Last Weight: %s" %(last_weight))
print("Best Weight: %s" %(best_weight))
print("Last mu\n", mu)
print("Last sigma\n", sigma)


# In[248]:


plt.figure(figsize=(10,6))
plt.plot(all_best_f1s[:25], 'b', label = 'Best weighted F1 score of elite samples')
plt.plot(all_mean_f1s[:25], 'r', label = 'Mean weighted F1 score of elite samples')
plt.xlabel('Iteration')
plt.ylabel('Weighted F1 score')
plt.legend(frameon=False)
plt.grid(True)
plt.savefig('../img/weighed_F1_iteration(1-10).eps',format='eps')


# 测试集

# In[226]:


y_test_pred_proba_all = []
for model in models:
    model_name = model.__class__.__name__
    y_test_pred_proba = model.predict_proba(X_test)
    y_test_pred = model.predict(X_test)
    print("model_name: %s, test accuracy:%f, weighted f1 score:%f" %(model_name, accuracy_score(y_test, y_test_pred), f1_score(y_test, y_test_pred, average="weighted")))
    y_test_pred_proba_all.append(y_test_pred_proba)
    
    
y_test_pred_ensemble_proba = np.zeros((len(y_test), 6)) # 集成器概率向量

# 为每一个基学习器乘上权重
for k in range(classifier_num):
    y_test_pred_ensemble_proba += y_test_pred_proba_all[k] * best_weight[k]
y_test_pred_ensemble = np.argmax(y_test_pred_ensemble_proba, axis=1)

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


# In[227]:


# np.save("../npy/CE_best_weights(1-10).npy", best_weight)
# np.save("../npy/CE_best_mu(1-10).npy", mu)
# np.save("../npy/CE_best_sigma(1-10).npy", sigma)
# import picklex
# for model in models:
#     model_name = model.__class__.__name__
#     with open("../pkl/CE_" + model_name + ".pkl", "wb") as f:
#         pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


# # plot errbar on validation set

# In[228]:


for ii in range(train_step):
    # 在验证集集上: 训练每个基学习器的投票参数
    print("Train step: %d" %(ii))
    for i in range(max_iteration):
        # print("Iteration: %d" %(i))
        # 该次迭代的所有种群们
        population_score = np.zeros(population_num)
        population_weights = np.zeros((population_num, classifier_num))
        # 该次迭代的优势种群们
        population_retain_score = np.zeros(retain_population_num)
        population_retain_weights = np.zeros((retain_population_num, classifier_num))

        # 生成所有种群
        for j in range(classifier_num):
            w = np.random.normal(mu[j], sigma[j]+700/(i+1), population_num)
            # w = np.random.normal(mu[j], sigma[j], population_num)
            population_weights[:,j] = w

        # 映射所有种群的权重至[0:1]    
        for j in range(population_num):
            w2 = np.zeros(classifier_num)
            for k in range(classifier_num):
                w2[k] = np.exp(-population_weights[j][k]*population_weights[j][k])
                # w2[k] = np.exp(population_weights[j][k])/np.sum(np.exp(population_weights[j]))
            population_weights[j] = w2

        # 计算所有种群得分
        for j in range(population_num):
            y_pred_ensemble_proba = np.zeros((len(y_valid), 6)) # 集成器概率向量
            # 为每一个基学习器乘上权重
            for k in range(classifier_num):
                y_pred_ensemble_proba += y_pred_proba_all[k] * population_weights[j][k]
            y_pred_ensemble = np.argmax(y_pred_ensemble_proba, axis=1)
            f1 = f1_score(y_valid, y_pred_ensemble, average="weighted")
            population_score[j] = f1

        # 所有种群得分按降序排列
        retain_index = np.argsort(-np.array(population_score))[:retain_population_num]

        # 记录该次迭代中的优势种群们
        population_retain_weights = population_weights[retain_index]
        population_retain_score = np.array(population_score)[retain_index]

        # 记录每次迭代最好的种群和value，以及精英种群中的平均value
        all_best_weights[i] = population_retain_weights[0] # i次迭代，精英种群中具有最高value的权重
        all_best_f1s[i] = population_retain_score[0] # i次迭代，精英种群中最高的value
        all_mean_f1s[i] = np.mean(population_retain_score) # i次迭代，精英种群中的平均value
        # 更新mu，sigma为优势种群们的分布
        mu = np.mean(population_retain_weights, axis = 0)
        sigma = np.std(population_retain_weights, axis = 0) #default: ddof = 0, The divisor used in calculations is N - ddof
    best_f1 = np.max(all_best_f1s)
    print("Best f1 of %d iterations: %f" % (max_iteration, best_f1))
    all_best_f1s_mean[ii] = np.mean(all_best_f1s)
    all_best_f1s_std[ii] = np.std(all_best_f1s)
    all_mean_f1s_mean[ii] = np.mean(all_mean_f1s)
    all_mean_f1s_std[ii] = np.std(all_mean_f1s)


# In[229]:


all_best_f1s_mean
all_mean_f1s_mean


# In[251]:


plt.figure(figsize=(10,6))
plt.errorbar(np.arange(train_step)+1, all_best_f1s_mean, yerr=all_best_f1s_std, fmt="--bo", elinewidth=1,
            label = 'MEAN ± SD of best weighted F1 score on validation set')
plt.xlabel('Training Step')
plt.ylabel('Weighted F1 score')
plt.legend(frameon=False)
plt.savefig('../img/MEANSD_of_best.eps',format='eps')


# In[252]:


plt.figure(figsize=(10,6))
plt.errorbar(np.arange(train_step)+1, all_mean_f1s_mean, yerr=all_mean_f1s_std, fmt="--ro", elinewidth=1,
            label = 'MEAN ± SD of mean weighted F1 score on validation set')
plt.xlabel('Training step')
plt.ylabel('Weighted F1 score')

plt.savefig('../img/MEANSD_of_mean.eps',format='eps')
plt.legend(frameon=False)

