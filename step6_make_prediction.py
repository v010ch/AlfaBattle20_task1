#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
from glob import glob

import pandas as pd
import numpy as np
from collections import Counter

import pickle
import gc

from tqdm import tqdm
tqdm.pandas()


# In[17]:


from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, make_scorer


# In[18]:


from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import catboost as cb
from catboost import CatBoostClassifier


# In[ ]:





# In[19]:


DATA = './data'
DATA_OWN = './data_own'
CLICKSTREAM = 'alfabattle2_abattle_clickstream'
MODELS = './models'
UTILS = './utils'
SUBM = './submissions'


# In[ ]:





# # load data / submit / features

# In[20]:


load_dtypes = pickle.load(open(os.path.join(UTILS, 'load_dtypes.pkl'), 'rb'))


# In[21]:


using_features = pickle.load(open(os.path.join(DATA_OWN, 'using_features.pkl'), 'rb'))


# In[22]:


#data = pd.read_csv(os.path.join(DATA_OWN, 'data_pred.csv'), parse_dates=['timestamp'])
data = pd.read_csv(os.path.join(DATA_OWN, 'data_pred.csv'),  usecols=using_features, dtype=load_dtypes)
data.head()


# In[23]:


if data.isnull().values.any() == True:
    print('I have a Bad news for you!')
    data.fillna(0, inplace = True)


# In[ ]:





# In[24]:


subm = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_sample_prediction.csv'))
subm.head()


# In[ ]:





# # load models

# In[25]:


clf_sgd = pickle.load(open(os.path.join(MODELS, 'clf_sgd.pkl'), 'rb'))
#clf_mlp = pickle.load(os.paht.join(MODELS, 'clf_mlp.pkl'))
clf_lr = pickle.load(open(os.path.join(MODELS, 'clf_lr.pkl'), 'rb'))
clf_lrsvc = pickle.load(open(os.path.join(MODELS, 'clf_lrsvc.pkl'), 'rb'))
#clf_svc = pickle.load(os.paht.join(MODELS, 'clf_svc.pkl'))
clf_lgbm = pickle.load(open(os.path.join(MODELS, 'clf_lgbm.pkl'), 'rb'))

#clf_cb  = cb.load_model(os.paht.join(MODELS, 'clf_cb.cbm'), format='cbm')


# In[ ]:





# # make predictions

# In[ ]:





# In[11]:


pred_sgd = clf_sgd.predict(data[using_features])


# In[12]:


#pred_mlp = clf_mlp.predict(data[using_features])


# In[13]:


#pred_svc = clf_svc.predict(data[using_features])


# In[12]:


pred_lr = clf_lr.predict(data[using_features])


# In[13]:


pred_lrsvc = clf_lrsvc.predict(data[using_features])


# In[26]:


pred_lgbm = clf_lgbm.predict(data[using_features])


# In[16]:


#pred_cb = clf_cb.predict(data[using_features])


# In[ ]:





# In[22]:


get_ipython().run_cell_magic('time', '', "pred_stack = [''] * pred_sgd.shape[0]\nfor idx in range(pred_sgd.shape[0]):\n    cnt = Counter([\n                   pred_sgd[idx], \n                   #pred_mlp[idx], \n                   pred_lr[idx], \n                   pred_lrsvc[idx],\n                   pred_lgbm[idx],\n               ]).most_common()\n    pred_stack[idx] = cnt[0][0]\n    \n    #if cnt[0][0] == cnt[1][0]:\n    #    print('terrible!')")


# In[ ]:





# # make submit

# In[23]:


#subm.prediction = pred_stack


# In[27]:


#subm.prediction = pred_sgd
#subm.prediction = pred_lr
#subm.prediction = pred_lrsvc
subm.prediction = pred_lgbm


# In[28]:


subm.to_csv(os.path.join(SUBM, 'subm_client_6diff_lt_lgbm.csv'), index = False)


# In[ ]:





# In[ ]:




