
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


# train 80%, test 20%

# In[9]:


print("============ train_test_split ============")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                       stratify = y, random_state = random_seed)
print("80%% train: %d/%d, 20%% test: %d/%d" %(X_train.shape[0], X.shape[0], X_test.shape[0], X.shape[0]))


# ### normalize  train data
# 
# fulfill the Na with median, then standardized the data, output type ndarray

# In[10]:


clean_pipeline = Pipeline([('imputer', preprocessing.Imputer(missing_values='NaN',strategy="median")),
                           ('std_scaler', preprocessing.StandardScaler()),])
X_train = clean_pipeline.fit_transform(X_train)
X_test = clean_pipeline.fit_transform(X_test)


# # model train & test

# In[11]:


y_train = y_train.values
y_test = y_test.values


# In[12]:


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


# In[16]:


test_entries = []
train_entries = []

for model in models:
    model_name = model.__class__.__name__
    if model_name == 'GCForest':
        model.fit_transform(X_train, y_train, X_test, y_test)
    else:
        model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    f1_test = f1_score(y_test, y_test_pred, average='weighted')
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    train_entries.append((model_name, f1_train, acc_train))
    test_entries.append((model_name, f1_test, acc_test))
    # print("%s, test set: weighted F1 score:%f, Accuracy:%f" %(model_name, f1_score(y_valid, y_pred, average="weighted"), accuracy_score(y_valid, y_pred)))
    
train_df = pd.DataFrame(train_entries, columns=['model_name', 'train_f1_weighted', 'train_accuracy'])
test_df = pd.DataFrame(test_entries, columns=['model_name', 'test_f1_weighted', 'test_accuracy'])


# In[17]:


train_df


# In[18]:


test_df


# In[27]:


for model in models:
    model_name = model.__class__.__name__
    y_pred = model.predict(X_test)
    print("=================" + model_name + "=================")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    i=0
    acc_all = np.zeros(6)
    for c in cm:
        acc_all[i] = c[i]/np.sum(c)
        print("%d accuaracy: %f" %(i, acc_all[i]))
        i=i+1
    print("acc:", np.sum(y_test == y_pred)/y_pred.shape[0])
    print('f1_weighted', f1_score(y_test, y_pred, average='weighted'))


# In[28]:


for model in models:
    model_name = model.__class__.__name__
    y_pred = model.predict(X_train)
    print("=================" + model_name + "=================")
    print(classification_report(y_train, y_pred, digits=4))

    cm = confusion_matrix(y_train, y_pred)
    i=0
    acc_all = np.zeros(6)
    for c in cm:
        acc_all[i] = c[i]/np.sum(c)
        print("%d accuaracy: %f" %(i, acc_all[i]))
        i=i+1
    print("acc:", np.sum(y_train == y_pred)/y_pred.shape[0])
    print('f1_weighted', f1_score(y_train, y_pred, average='weighted'))


# MeanSD

# 1. MeanSD weighted F1

# In[30]:


from sklearn.model_selection import StratifiedKFold

for model in models:
    model_name = model.__class__.__name__
    
    # Kfold
    scores = []
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X_train, y_train):
        K_train_x, K_test_x = X_train[train_index], X_train[test_index]
        K_train_y, K_test_y = y_train[train_index], y_train[test_index]
        
        y_pred = model.predict(K_test_x)
        scores.append(f1_score(K_test_y, y_pred,average="weighted"))
    print("%s    Weighted F1: %0.2f ± %0.2f %%" % (model_name, np.mean(scores)*100, np.std(scores)*100))


# 2. MeanSD Acc

# In[31]:


for model in models:
    model_name = model.__class__.__name__
    
    # Kfold
    scores = []
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X_train, y_train):
        K_train_x, K_test_x = X_train[train_index], X_train[test_index]
        K_train_y, K_test_y = y_train[train_index], y_train[test_index]
        
        y_pred = model.predict(K_test_x)
        scores.append(accuracy_score(K_test_y, y_pred))
    print("%s    Acc: %0.2f ± %0.2f %%" % (model_name, np.mean(scores)*100, np.std(scores)*100))


# In[ ]:




