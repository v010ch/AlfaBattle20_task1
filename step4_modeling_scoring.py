#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from catboost import CatBoostClassifier


# In[ ]:





# In[ ]:





# In[4]:


DATA = './data'
DATA_OWN = './data_own'
CLICKSTREAM = 'alfabattle2_abattle_clickstream'
MODELS = './models'
UTILS = './utils'


# In[ ]:





# ## load data, features

# In[6]:


load_dtypes = pickle.load(open(os.path.join(UTILS, 'load_dtypes.pkl'), 'rb'))


# In[7]:


using_features = pickle.load(open(os.path.join(DATA_OWN, 'using_features.pkl'), 'rb'))
len(using_features)


# In[ ]:





# In[13]:


load_col = list(using_features)
load_col.extend(['multi_class_target'])


# In[ ]:





# In[15]:


get_ipython().run_cell_magic('time', '', "#data_train = pd.read_csv(os.path.join(DATA_OWN, 'data_train.csv'), parse_dates=['timestamp'], usecols=using_features, dtype=load_dtypes)\ndata_train = pd.read_csv(os.path.join(DATA_OWN, 'data_train.csv'), usecols=load_col, dtype=load_dtypes)\ndata_train.head()")


# In[16]:


(data_train.isnull().values.any())


# In[17]:


target = data_train.multi_class_target
data_train.drop('multi_class_target', axis = 1, inplace = True)


# In[18]:


get_ipython().run_cell_magic('time', '', 'X_train, X_val, y_train, y_val = train_test_split(data_train, target, test_size=0.33, random_state=42)\nX_train.shape, X_val.shape, y_train.shape, y_val.shape')


# In[19]:


#using_features = pickle.load(open(os.path.join(DATA_OWN, 'using_features.pkl'), 'rb'))
#len(using_features)


# In[ ]:





# In[20]:


#f1 = make_scorer(f1_score , average='macro')


# In[ ]:





# ## check classifiers with check time of work

# In[ ]:





# In[21]:


#clf_sgd_log   = SGDClassifier(loss = 'log', class_weight='balanced', n_jobs=-1)

#clf_knn  = KNeighborsClassifier()
#clf_svc  = SVC()
#clf_gaus = GaussianNB()


# In[22]:


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


get_ipython().run_cell_magic('time', '', "# 9 min\n# \n\nclf_mlp  = MLPClassifier((200, 50), learning_rate = 'adaptive', activation='logistic', verbose = True)\nclf_mlp.fit(X_train[using_features], y_train)\npred_mlp = clf_mlp.predict(X_val[using_features])\nprint(len(set(pred_mlp)), set(pred_mlp))\nprint(f1_score(y_val, pred_mlp, average  = 'micro'))")


# In[23]:


get_ipython().run_cell_magic('time', '', "# 81 - 8min 23s\n\n\nclf_lr = LogisticRegression(n_jobs = -1, verbose = 1) #‘liblinear’,\nclf_lr.fit(X_train[using_features], y_train)\npred_lr = clf_lr.predict(X_val[using_features])\nprint(len(set(pred_lr)), set(pred_lr))\nprint(f1_score(y_val, pred_lr, average  = 'micro'))")


# In[24]:


get_ipython().run_cell_magic('time', '', "# 81 - 17min 29s\n\nclf_lrsvc = LinearSVC(verbose = 1) # loss = ‘hinge’\nclf_lrsvc.fit(X_train[using_features], y_train)\npred_lrsvc = clf_lrsvc.predict(X_val[using_features])\nprint(len(set(pred_lrsvc)), set(pred_lrsvc))\nprint(f1_score(y_val, pred_lrsvc, average  = 'micro'))")


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

# In[41]:


get_ipython().run_cell_magic('time', '', "pred_stack = [''] * pred_sgd_hinge.shape[0]\ncollisions = 0\nfor idx in range(pred_sgd_hinge.shape[0]):\n    cnt = Counter([\n                   pred_sgd_hinge[idx], \n                   #pred_mlp[idx], \n                   pred_lr[idx], \n                   pred_lrsvc[idx]\n               ]).most_common()\n    pred_stack[idx] = cnt[0][0]\n    \n    if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:\n        collisions += 1\n    #    print('terrible!')\n    #    print(cnt[0], cnt[1])")


# In[42]:


print(f1_score(y_val, pred_stack, average  = 'micro'), (collisions/pred_sgd_hinge.shape[0]))


# In[48]:


stack_df = pd.DataFrame({'SGD':pred_sgd_hinge, 'logit':pred_lr, 'linear_svc':pred_lrsvc})
cat_ftrs = ['SGD', 'logit', 'linear_svc']


# In[54]:


#cb_stack = CatBoostClassifier(loss_function='Logloss', cat_features=cat_ftrs)
cb_stack = CatBoostClassifier(cat_features=cat_ftrs)
cb_stack.fit(stack_df, y_val)
pred_stack_cb = cb_stack.predict(X_val[using_features])
print(len(set(pred_stack_cb)), set(pred_stack_cb))
print(f1_score(y_val, pred_stack_cb, average  = 'micro'))


# In[59]:


pred_stack_cb = cb_stack.predict(stack_df)
print(len(set(pred_stack_cb[:, 0])), set(pred_stack_cb[:, 0]))
print(f1_score(y_val, pred_stack_cb[:, 0], average  = 'micro'))


# In[ ]:


#dataset = cb.Pool(X, y, cat_features=['d'])


# In[ ]:





# ## classifiers GridSearch

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## study best models on full data

# In[ ]:





# In[43]:


get_ipython().run_cell_magic('time', '', "clf_sgd_hinge = SGDClassifier(loss = 'hinge', n_jobs=-1)\nclf_sgd_hinge.fit(data_train[using_features], target)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "clf_mlp  = MLPClassifier((200, 50), learning_rate = 'adaptive', activation='logistic', verbose = True)\nclf_mlp.fit(data_train[using_features], target)")


# In[44]:


get_ipython().run_cell_magic('time', '', '# 81 - 13min 29s\n\nclf_lr = LogisticRegression(n_jobs = -1, verbose = 2)\nclf_lr.fit(data_train[using_features], target)')


# In[45]:


get_ipython().run_cell_magic('time', '', '#81 - 27min 9s\n\nclf_lrsvc = LinearSVC(verbose = True) # loss = ‘hinge’\nclf_lrsvc.fit(data_train[using_features], target)')


# In[ ]:





# In[ ]:





# ## save models

# In[60]:


get_ipython().run_cell_magic('time', '', "pickle.dump(clf_sgd_hinge, open(os.path.join(MODELS, 'clf_sgd.pkl'), 'wb'))\n#pickle.dump(clf_mlp, open(os.path.join(MODELS, 'clf_mlp.pkl'), 'wb'))\npickle.dump(clf_lr, open(os.path.join(MODELS, 'clf_lr.pkl'), 'wb'))\npickle.dump(clf_lrsvc, open(os.path.join(MODELS, 'clf_lrsvc.pkl'), 'wb'))\n#pickle.dump(clf_svc, open(os.path.join(MODELS, 'clf_svc.pkl'), 'wb'))\n#pickle.dump(clf_rf,  open(os.path.join(MODELS, 'clf_rf.pkl'),  'wb'))\n            \n#clf_cb.save_model(os.path.join(MODELS, 'clf_cb.cbm'), format='cbm')")


# In[ ]:




