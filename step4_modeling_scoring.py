#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
from glob import glob
import pandas as pd
import numpy as np
import itertools
from collections import Counter


import pickle
import gc

from tqdm.notebook import tqdm
#tqdm.pandas()


# In[13]:


from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, make_scorer


# In[14]:


from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb


# In[ ]:





# In[ ]:





# In[15]:


DATA = os.path.join('.', 'data')
DATA_OWN = os.path.join('.', 'data_own')
CLICKSTREAM = 'alfabattle2_abattle_clickstream'
MODELS = os.path.join('.', 'models')
UTILS = os.path.join('.', 'utils')
SUBM = os.path.join('.', 'submissions')


# In[ ]:





# ## load data, features

# In[16]:


load_dtypes = pickle.load(open(os.path.join(UTILS, 'load_dtypes.pkl'), 'rb'))


# In[17]:


using_features = pickle.load(open(os.path.join(DATA_OWN, 'using_features.pkl'), 'rb'))
len(using_features)


# In[ ]:





# In[18]:


load_col = list(using_features)
load_col.extend(['client_pin', 'timestamp', 'multi_class_target'])


# In[ ]:





# In[19]:


get_ipython().run_cell_magic('time', '', "data_train = pd.read_csv(os.path.join(DATA_OWN, 'data_train.csv'), parse_dates=['timestamp'], usecols=load_col, dtype=load_dtypes)\n#data_train = pd.read_csv(os.path.join(DATA_OWN, 'data_train.csv'), usecols=load_col, dtype=load_dtypes)\ndata_train.head()")


# In[20]:


(data_train.isnull().values.any())


# In[21]:


target = data_train.multi_class_target
data_train.drop('multi_class_target', axis = 1, inplace = True)


# In[22]:


get_ipython().run_cell_magic('time', '', 'X_train, X_val, y_train, y_val = train_test_split(data_train, target, test_size=0.33, random_state=42)\nX_train.shape, X_val.shape, y_train.shape, y_val.shape')


# In[ ]:


#using_features = pickle.load(open(os.path.join(DATA_OWN, 'using_features.pkl'), 'rb'))
#len(using_features)


# In[ ]:





# In[ ]:


#f1 = make_scorer(f1_score , average='macro')


# In[ ]:





# ## check classifiers with check time of work

# In[ ]:





# In[ ]:


#clf_sgd_log   = SGDClassifier(loss = 'log', class_weight='balanced', n_jobs=-1)

#clf_knn  = KNeighborsClassifier()
#clf_svc  = SVC()
#clf_gaus = GaussianNB()


# In[ ]:


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


get_ipython().run_cell_magic('time', '', "# 9 min\n# 91 - 1h 10min 53s\n\nclf_mlp  = MLPClassifier((200, 50), learning_rate = 'adaptive', activation='logistic', verbose = True)\nclf_mlp.fit(X_train[using_features], y_train)\npred_mlp = clf_mlp.predict(X_val[using_features])\nprint(len(set(pred_mlp)), set(pred_mlp))\nprint(f1_score(y_val, pred_mlp, average  = 'micro'))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# 81 - 8min 23s\n# 91 - 8min 40s\n\nclf_lr = LogisticRegression(n_jobs = -1, verbose = 1) #‘liblinear’,\nclf_lr.fit(X_train[using_features], y_train)\npred_lr = clf_lr.predict(X_val[using_features])\nprint(len(set(pred_lr)), set(pred_lr))\nprint(f1_score(y_val, pred_lr, average  = 'micro'))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# 81 - 17min 29s\n# 91 - 4h 4min 35s\nclf_lrsvc = LinearSVC(verbose = 1) # loss = ‘hinge’\nclf_lrsvc.fit(X_train[using_features], y_train)\npred_lrsvc = clf_lrsvc.predict(X_val[using_features])\nprint(len(set(pred_lrsvc)), set(pred_lrsvc))\nprint(f1_score(y_val, pred_lrsvc, average  = 'micro'))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# 81 - 5min 38s\n# 91 - 5min 21s\n\nclf_lgbm = LGBMClassifier( silent = False) # loss = ‘hinge’\nclf_lgbm.fit(X_train[using_features], y_train)\npred_lgbm = clf_lgbm.predict(X_val[using_features])\nprint(len(set(pred_lgbm)), set(pred_lgbm))\nprint(f1_score(y_val, pred_lgbm, average  = 'micro'))")


# ## XGBoost

# In[ ]:





# In[ ]:





# In[ ]:


target_encoder = OrdinalEncoder().fit(y_train.values.reshape(-1, 1))
y_train_int = target_encoder.transform(y_train.values.reshape(-1, 1))
y_val_int   = target_encoder.transform(y_val.values.reshape(-1, 1))


# In[ ]:


pickle.dump(target_encoder, open(os.path.join(UTILS, 'oe_target.pkl'), 'wb'))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'xgb_params = {\n    \'tree_method\':\'gpu_hist\',\n    "objective": "multi:softmax",\n    "eta": 0.3,\n    "num_class": 10,\n    #"max_depth": 10,\n    #"nthread": 4,\n    "eval_metric": "mlogloss",\n    #"print.every.n": 10\n    "verbosity": 2,\n}\n\npart_size = int(X_train.shape[0] / 4)\nprogress_result = {}\ntotal_progress = []\nDeval = xgb.DMatrix(X_val[using_features], label=y_val_int)\n\n\nfor part in range(4):\n    #print(part*part_size, (part+1)*part_size)\n    print(\'part \' + str(part) + \'  eta: \' +  str(xgb_params[\'eta\']))\n    \n    if part != 3:\n        Dtrain = xgb.DMatrix(X_train[using_features][part * part_size : (part + 1) * part_size], \n                             label = y_train_int[part * part_size : (part + 1) * part_size])\n    else:\n        Dtrain = xgb.DMatrix(X_train[using_features][part * part_size : ], \n                             label = y_train_int[part * part_size : ])  \n        \n    eval_sets = [(Dtrain, \'train\'), (Deval, \'eval\')]\n    if part == 0:\n        model = xgb.train(xgb_params, Dtrain, \n                           num_boost_round = 50,\n                           evals = eval_sets,        \n                           #early_stopping_rounds = 10, \n                           evals_result = progress_result, \n                         )\n    else:\n        model = xgb.train(xgb_params, Dtrain, \n                           num_boost_round = 50,\n                           xgb_model = model,\n                           evals = eval_sets,        \n                           #early_stopping_rounds = 10, \n                           evals_result = progress_result,\n                         )\n    \n    total_progress.append(progress_result)\n    \n    ##\n    ##\n    ## THIS PART SHOULD BE CHECKED IN SANDBOX\n    ## DOWNGREADE ETA GIVE ONLY 0.666830 ON \n    ## num_boost_round = 30 AND START ETA 0.5\n    ## train-mlogloss:0.79863 eval-mlogloss:0.80901\n    ## better to start with eta 0.3\n    ##\n    ##\n    #if np.argmin(progress_result[\'eval\'][\'mlogloss\']) < (len(progress_result[\'eval\'][\'mlogloss\']) - 6 - 1):\n    #    #break\n    #    xgb_params.update({\'eta\': xgb_params[\'eta\'] / 2}) #part\n    #    print(\'eta downgraded to \' + str(xgb_params[\'eta\']))\n    \n    # without update/refresh parameters getting 0.669192\n    #xgb_params.update({#\'process_type\': \'update\',    # 0.66732872\n                       #\'updater\'     : \'refresh\',\n                       #\'refresh_leaf\': True\n    #                  })\n    \nprint(\'done!\')')


# In[ ]:


pred_xgb = model.predict(Deval)
print(f1_score(y_val_int, pred_xgb, average  = 'micro'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


#%%time
clf_cb = CatBoostClassifier(task_type="GPU", #with gpu: 5 min vs 5 hour
                            devices='0'
                           )
clf_cb.fit(X_train[using_features], y_train)


# In[31]:


pred_cb = clf_cb.predict(X_val[using_features])
print(set(pred_cb.reshape(-1)))
print(f1_score(y_val, pred_cb, average  = 'micro'))


# In[ ]:





# ### saw how many give us stack

# In[ ]:


get_ipython().run_cell_magic('time', '', "pred_stack = [''] * pred_sgd_hinge.shape[0]\ncollisions = 0\nfor idx in range(pred_sgd_hinge.shape[0]):\n    cnt = Counter([\n                   pred_sgd_hinge[idx], \n                   #pred_mlp[idx], \n                   pred_lr[idx], \n                   pred_lrsvc[idx]\n               ]).most_common()\n    pred_stack[idx] = cnt[0][0]\n    \n    if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:\n        collisions += 1\n    #    print('terrible!')\n    #    print(cnt[0], cnt[1])")


# In[ ]:


print(f1_score(y_val, pred_stack, average  = 'micro'), (collisions/pred_sgd_hinge.shape[0]))


# In[ ]:


stack_df = pd.DataFrame({'SGD':pred_sgd_hinge, 'logit':pred_lr, 'linear_svc':pred_lrsvc})
cat_ftrs = ['SGD', 'logit', 'linear_svc']


# In[ ]:


#cb_stack = CatBoostClassifier(loss_function='Logloss', cat_features=cat_ftrs)
cb_stack = CatBoostClassifier(cat_features=cat_ftrs)
cb_stack.fit(stack_df, y_val)
pred_stack_cb = cb_stack.predict(X_val[using_features])
print(len(set(pred_stack_cb)), set(pred_stack_cb))
print(f1_score(y_val, pred_stack_cb, average  = 'micro'))


# In[ ]:


pred_stack_cb = cb_stack.predict(stack_df)
print(len(set(pred_stack_cb[:, 0])), set(pred_stack_cb[:, 0]))
print(f1_score(y_val, pred_stack_cb[:, 0], average  = 'micro'))


# In[ ]:


#dataset = cb.Pool(X, y, cat_features=['d'])


# In[ ]:





# ## study best models on full data

# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', "# 91 - 45min 1s ????\n\nclf_sgd_hinge = SGDClassifier(loss = 'hinge', n_jobs=-1)\nclf_sgd_hinge.fit(data_train[using_features], target)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# 91 - 1h 28min 38s\nclf_mlp  = MLPClassifier((200, 50), learning_rate = 'adaptive', activation='logistic', verbose = True)\nclf_mlp.fit(data_train[using_features], target)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# 81 - 13min 29s\n# 91 - 1h 13min 58s\nclf_lr = LogisticRegression(n_jobs = -1, verbose = 2)\nclf_lr.fit(data_train[using_features], target)')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#81 - 27min 9s\n\n#clf_lrsvc = LinearSVC(verbose = True) # loss = ‘hinge’\n#clf_lrsvc.fit(data_train[using_features], target)')


# In[ ]:


clf_lgbm = LGBMClassifier( silent = False) # loss = ‘hinge’
clf_lgbm.fit(data_train[using_features], target)


# In[32]:


#%%time
clf_cb = CatBoostClassifier(task_type="GPU", #with gpu: 10 min vs ?? hour
                            devices='0'
                           )
clf_cb.fit(data_train[using_features], target)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'xgb_params = {\n    \'tree_method\':\'gpu_hist\',\n    "objective": "multi:softmax",\n    "eta": 0.3,\n    "num_class": 10,\n    #"max_depth": 10,\n    #"nthread": 4,\n    "eval_metric": "mlogloss",\n    #"print.every.n": 10\n    "verbosity": 2,\n}\n\ntarget_int = target_encoder.transform(target.values.reshape(-1, 1))\npart_size = int(data_train.shape[0] / 4)\n\n#progress_result = {}\n#total_progress = []\n\n\nfor part in range(4):\n    #print(part*part_size, (part+1)*part_size)\n    print(\'part \' + str(part) + \'  eta: \' +  str(xgb_params[\'eta\']))\n    \n    if part != 3:\n        Dtrain = xgb.DMatrix(data_train[using_features][part * part_size : (part + 1) * part_size], \n                             label = target_int[part * part_size : (part + 1) * part_size])\n    else:\n        Dtrain = xgb.DMatrix(data_train[using_features][part * part_size : ], \n                             label = target_int[part * part_size : ])  \n        \n    eval_sets = [(Dtrain, \'train\'), (Deval, \'eval\')]\n    if part == 0:\n        model_xgb = xgb.train(xgb_params, Dtrain, \n                           num_boost_round = 50,\n                           #early_stopping_rounds = 10, \n                         )\n    else:\n        model_xgb = xgb.train(xgb_params, Dtrain, \n                           num_boost_round = 50,\n                           xgb_model = model_xgb,\n                           #early_stopping_rounds = 10, \n                         )\n    \n#    total_progress.append(progress_result)\n    \n    ##\n    ##\n    ## THIS PART SHOULD BE CHECKED IN SANDBOX\n    ## DOWNGREADE ETA GIVE ONLY 0.666830 ON \n    ## num_boost_round = 30 AND START ETA 0.5\n    ## train-mlogloss:0.79863 eval-mlogloss:0.80901\n    ## better to start with eta 0.3\n    ##\n    ##\n    #if np.argmin(progress_result[\'eval\'][\'mlogloss\']) < (len(progress_result[\'eval\'][\'mlogloss\']) - 6 - 1):\n    #    #break\n    #    xgb_params.update({\'eta\': xgb_params[\'eta\'] / 2}) #part\n    #    print(\'eta downgraded to \' + str(xgb_params[\'eta\']))\n    \n    # without update/refresh parameters getting 0.669192\n    #xgb_params.update({#\'process_type\': \'update\',    # 0.66732872\n                       #\'updater\'     : \'refresh\',\n                       #\'refresh_leaf\': True\n    #                  })\n    \nprint(\'done!\')')


# In[ ]:





# In[ ]:





# ## save models

# In[ ]:


get_ipython().run_cell_magic('time', '', "pickle.dump(clf_sgd_hinge, open(os.path.join(MODELS, 'clf_sgd.pkl'), 'wb'))\npickle.dump(clf_mlp, open(os.path.join(MODELS, 'clf_mlp.pkl'), 'wb'))\npickle.dump(clf_lr, open(os.path.join(MODELS, 'clf_lr.pkl'), 'wb'))\npickle.dump(clf_lgbm, open(os.path.join(MODELS, 'clf_lgbm.pkl'), 'wb'))\n\n#lgb.save(clf_lgbm, os.path.join(MODELS, 'clf_lgbm.lgb'), num_iteration = NULL)\n#clf_lgbm.save_model(os.path.join(MODELS, 'clf_lgbm.txt'))\n\n#pickle.dump(clf_lrsvc, open(os.path.join(MODELS, 'clf_lrsvc.pkl'), 'wb'))\n#pickle.dump(clf_svc, open(os.path.join(MODELS, 'clf_svc.pkl'), 'wb'))\n#pickle.dump(clf_rf,  open(os.path.join(MODELS, 'clf_rf.pkl'),  'wb'))\n\n\npickle.dump(clf_lgbm, open(os.path.join(MODELS, 'clf_lgbm.pkl'), 'wb'))")


# In[ ]:


model_xgb.save_model(os.path.join(MODELS, 'clf_xgb.json'))


# In[33]:


clf_cb.save_model(os.path.join(MODELS, 'clf_cb.cbm'), format='cbm')


# In[ ]:





# In[ ]:




