#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from glob import glob
import pandas as pd
import numpy as np
import itertools
from collections import Counter


import pickle
import gc

from tqdm import tqdm
tqdm.pandas()


# In[3]:


from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, make_scorer


# In[4]:


from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier


# In[ ]:





# In[ ]:





# In[5]:


DATA = './data'
DATA_OWN = './data_own'
CLICKSTREAM = 'alfabattle2_abattle_clickstream'
MODELS = './models'
UTILS = './utils'


# In[ ]:





# ## load data, features

# In[6]:


get_ipython().run_cell_magic('time', '', "data_train = pd.read_csv(os.path.join(DATA_OWN, 'data_train.csv'), parse_dates=['timestamp'])\ndata_train.head()")


# In[7]:


(data_train.isnull().values.any())


# In[7]:


target = data_train.multi_class_target


# In[10]:


get_ipython().run_cell_magic('time', '', 'X_train, X_val, y_train, y_val = train_test_split(data_train, target, test_size=0.33, random_state=42)\nX_train.shape, X_val.shape, y_train.shape, y_val.shape')


# In[11]:


using_features = pickle.load(open(os.path.join(DATA_OWN, 'using_features.pkl'), 'rb'))


# In[ ]:





# In[43]:


#f1 = make_scorer(f1_score , average='macro')


# In[ ]:





# ## check classifiers with check time of work

# In[ ]:





# In[ ]:


#clf_sgd_log   = SGDClassifier(loss = 'log', class_weight='balanced', n_jobs=-1)

#clf_knn  = KNeighborsClassifier()
#clf_svc  = SVC()
#clf_gaus = GaussianNB()


# In[12]:


get_ipython().run_cell_magic('time', '', "#clf_sgd_hinge = SGDClassifier(loss = 'hinge', class_weight='balanced', n_jobs=-1)\nclf_sgd_hinge = SGDClassifier(loss = 'hinge', n_jobs=-1)\nclf_sgd_hinge.fit(X_train[using_features], y_train)\npred_sgd_hinge = clf_sgd_hinge.predict(X_val[using_features])\nprint(len(set(pred_sgd_hinge)), set(pred_sgd_hinge))\nprint(f1_score(y_val, pred_sgd_hinge, average  = 'micro'))")


# In[ ]:


#%%time
#clf_sgd_perc  = SGDClassifier(loss = 'perceptron', class_weight='balanced', n_jobs=-1)
#clf_sgd_perc  = SGDClassifier(loss = 'perceptron', n_jobs=-1)
#clf_sgd_perc.fit(X_train[using_features], y_train)
#pred_sgd_perc = clf_sgd_perc.predict(X_val[using_features])
#rint(len(set(pred_sgd_perc)), set(pred_sgd_perc))
#rint(f1_score(y_val, pred_sgd_perc, average  = 'micro'))


# In[ ]:


get_ipython().run_cell_magic('time', '', "# 9 min\n\nclf_mlp  = MLPClassifier((200, 50), learning_rate = 'adaptive', activation='logistic', verbose = True)\nclf_mlp.fit(X_train[using_features], y_train)\npred_mlp = clf_mlp.predict(X_val[using_features])\nprint(set(pred_mlp))\nprint(f1_score(y_val, pred_mlp, average  = 'micro'))")


# In[ ]:


#%%time
# TOOO LONG

#clf_svc.fit(X_train[using_features], y_train)
#pred_svc = clf_svc.predict(X_val[using_features])
#print(set(pred_svc))
#print(f1_score(y_val, pred_svc, average  = 'micro'))


# In[ ]:


#%%time
# 24 min
#
#clf_rf   = RandomForestClassifier(n_jobs = -1, verbose = 1) 
#clf_rf.fit(X_train[using_features], y_train)
#pred_rf = clf_rf.predict(X_val[using_features])
#print(set(pred_rf))
#print(f1_score(y_val, pred_rf, average  = 'micro'))


# In[ ]:


#%%time
#clf_gaus.fit(X_train[using_features], y_train)
#pred_gaus = clf_gaus.predict(X_val[using_features])
#print(set(pred_gaus))
#print(f1_score(y_val, pred_gaus, average  = 'micro'))


# In[15]:


#%%time
#
#
#adaboost_inp_clf = DecisionTreeClassifier()
#clf_ab   = AdaBoostClassifier(adaboost_inp_clf)
#clf_ab.fit(X_train[using_features], y_train)
#pred_ab = clf_ab.predict(X_val[using_features])
#rint(len(set(pred_ab)), set(pred_ab))
#print(f1_score(y_val, pred_ab, average  = 'micro'))


# In[13]:


get_ipython().run_cell_magic('time', '', "\nclf_lr = LogisticRegression(n_jobs = -1, verbose = 1) #‘liblinear’,\nclf_lr.fit(X_train[using_features], y_train)\npred_lr = clf_lr.predict(X_val[using_features])\nprint(len(set(pred_lr)), set(pred_lr))\nprint(f1_score(y_val, pred_lr, average  = 'micro'))")


# In[14]:


get_ipython().run_cell_magic('time', '', "\nclf_lrsvc = LinearSVC(verbose = 1) # loss = ‘hinge’\nclf_lrsvc.fit(X_train[using_features], y_train)\npred_lrsvc = clf_lrsvc.predict(X_val[using_features])\nprint(len(set(pred_lrsvc)), set(pred_lrsvc))\nprint(f1_score(y_val, pred_lrsvc, average  = 'micro'))")


# In[1]:


#%%time
#
#clf_lrsvc_hinge = LinearSVC( loss = 'hinge')
#clf_lrsvc_hinge.fit(X_train[using_features], y_train)
#pred_lrsvc_hinge = clf_lrsvc_hinge.predict(X_val[using_features])
#print(len(set(pred_lrsvc_hinge)), set(pred_lrsvc_hinge))
#print(f1_score(y_val, pred_lrsvc_hinge, average  = 'micro'))


# In[ ]:





# In[ ]:


#%%time
#clf_cb = CatBoostClassifier()
#clf_cb.fit(X_train[using_features], y_train)
#pred_cb = clf_cb.predict(X_val[using_features])
#print(set(pred_cb))
#print(f1_score(y_val, pred_cb, average  = 'micro'))


# In[ ]:





# ### saw how many give us stack

# In[19]:


get_ipython().run_cell_magic('time', '', "pred_stack = [''] * pred_sgd_hinge.shape[0]\nfor idx in range(pred_sgd_hinge.shape[0]):\n    cnt = Counter([\n                   pred_sgd_hinge[idx], \n                   #pred_mlp[idx], \n                   pred_lr[idx], \n                   pred_lrsvc[idx]\n               ]).most_common()\n    pred_stack[idx] = cnt[0][0]\n    \n    if cnt[0][0] == cnt[1][0]:\n        print('terrible!')")


# In[21]:


print(f1_score(y_val, pred_stack, average  = 'micro'))


# In[ ]:





# In[ ]:





# In[ ]:





# ## classifiers GridSearch

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## study best models on full data

# In[ ]:





# In[23]:


get_ipython().run_cell_magic('time', '', "clf_sgd_hinge = SGDClassifier(loss = 'hinge', n_jobs=-1)\nclf_sgd_hinge.fit(data_train[using_features], target)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "clf_mlp  = MLPClassifier((200, 50), learning_rate = 'adaptive', activation='logistic', verbose = True)\nclf_mlp.fit(data_train[using_features], target)")


# In[24]:


get_ipython().run_cell_magic('time', '', 'clf_lr = LogisticRegression(n_jobs = -1, verbose = 1)\nclf_lr.fit(data_train[using_features], target)')


# In[25]:


get_ipython().run_cell_magic('time', '', 'clf_lrsvc = LinearSVC(verbose = True) # loss = ‘hinge’\nclf_lrsvc.fit(data_train[using_features], target)')


# In[ ]:





# In[ ]:





# ## save models

# In[26]:


get_ipython().run_cell_magic('time', '', "pickle.dump(clf_sgd_hinge, open(os.path.join(MODELS, 'clf_sgd.pkl'), 'wb'))\n#pickle.dump(clf_mlp, open(os.path.join(MODELS, 'clf_mlp.pkl'), 'wb'))\npickle.dump(clf_lr, open(os.path.join(MODELS, 'clf_lr.pkl'), 'wb'))\npickle.dump(clf_lrsvc, open(os.path.join(MODELS, 'clf_lrsvc.pkl'), 'wb'))\n#pickle.dump(clf_svc, open(os.path.join(MODELS, 'clf_svc.pkl'), 'wb'))\n#pickle.dump(clf_rf,  open(os.path.join(MODELS, 'clf_rf.pkl'),  'wb'))\n            \n#clf_cb.save_model(os.path.join(MODELS, 'clf_cb.cbm'), format='cbm')")


# In[ ]:




