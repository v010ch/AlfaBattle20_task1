#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from glob import glob

import pandas as pd
import numpy as np
from collections import Counter

import pickle
import gc

from tqdm import tqdm
tqdm.pandas()


# In[2]:


from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, make_scorer


# In[3]:


from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import catboost as cb
from catboost import CatBoostClassifier


# In[ ]:





# In[4]:


DATA = os.path.join('.', 'data')
DATA_OWN = os.path.join('.', 'data_own')
CLICKSTREAM = 'alfabattle2_abattle_clickstream'
MODELS = os.path.join('.', 'models')
UTILS = os.path.join('.', 'utils')
SUBM = os.path.join('.', 'submissions')


# In[ ]:





# # load data / submit / features

# In[5]:


load_dtypes = pickle.load(open(os.path.join(UTILS, 'load_dtypes.pkl'), 'rb'))


# In[6]:


using_features = pickle.load(open(os.path.join(DATA_OWN, 'using_features.pkl'), 'rb'))


# In[7]:


#data = pd.read_csv(os.path.join(DATA_OWN, 'data_pred.csv'), parse_dates=['timestamp'])
data = pd.read_csv(os.path.join(DATA_OWN, 'data_pred.csv'),  usecols=using_features, dtype=load_dtypes)
data.head()


# In[8]:


if data.isnull().values.any() == True:
    print('I have a Bad news for you!')
    data.fillna(0, inplace = True)


# In[ ]:





# In[9]:


subm = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_sample_prediction.csv'))
subm.head()


# In[ ]:





# # load models

# In[12]:


#clf_sgd = pickle.load(open(os.path.join(MODELS, 'clf_sgd.pkl'), 'rb'))
clf_mlp = pickle.load(open(os.path.join(MODELS, 'clf_mlp.pkl'), 'rb'))
#clf_lr = pickle.load(open(os.path.join(MODELS, 'clf_lr.pkl'), 'rb'))
#clf_lrsvc = pickle.load(open(os.path.join(MODELS, 'clf_lrsvc.pkl'), 'rb'))
#clf_svc = pickle.load(os.paht.join(MODELS, 'clf_svc.pkl'))
#clf_lgbm = pickle.load(open(os.path.join(MODELS, 'clf_lgbm.pkl'), 'rb'))

#clf_cb  = cb.load_model(os.paht.join(MODELS, 'clf_cb.cbm'), format='cbm')


# In[16]:


clf_xgb = xgb.Booster()
clf_xgb.load_model(os.path.join(MODELS, 'clf_xgb.json'))
target_encoder = pickle.load(open(os.path.join(UTILS, 'oe_target.pkl'), 'rb'))


# In[ ]:





# In[29]:





# # make predictions

# In[ ]:





# In[11]:


#pred_sgd = clf_sgd.predict(data[using_features])


# In[13]:


pred_mlp = clf_mlp.predict(data[using_features])


# In[13]:


#pred_svc = clf_svc.predict(data[using_features])


# In[12]:


#pred_lr = clf_lr.predict(data[using_features])


# In[13]:


#pred_lrsvc = clf_lrsvc.predict(data[using_features])


# In[26]:


#pred_lgbm = clf_lgbm.predict(data[using_features])


# In[16]:


#pred_cb = clf_cb.predict(data[using_features])


# In[18]:


pred_xgb_int = clf_xgb.predict(xgb.DMatrix(data[using_features]))


# In[24]:


pred_xgb = target_encoder.inverse_transform(pred_xgb_int.reshape(-1, 1))


# In[ ]:




%%time
pred_stack = [''] * pred_sgd.shape[0]
for idx in range(pred_sgd.shape[0]):
    cnt = Counter([
                   pred_sgd[idx], 
                   #pred_mlp[idx], 
                   pred_lr[idx], 
                   pred_lrsvc[idx],
                   pred_lgbm[idx],
               ]).most_common()
    pred_stack[idx] = cnt[0][0]
    
    #if cnt[0][0] == cnt[1][0]:
    #    print('terrible!')
# In[ ]:





# # make submit

# In[23]:


#subm.prediction = pred_stack


# In[26]:


#subm.prediction = pred_sgd
#subm.prediction = pred_mlp
#subm.prediction = pred_lr
#subm.prediction = pred_lrsvc
#subm.prediction = pred_lgbm
subm.prediction = pred_xgb


# In[27]:


subm.to_csv(os.path.join(SUBM, 'subm_client_6diff_lt_relat_xgb.csv'), index = False)


# In[ ]:





# In[ ]:




