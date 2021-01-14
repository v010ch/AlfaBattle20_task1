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


# In[56]:


from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
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





# ## prepare data, test / training sets

# In[5]:


data = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_train_target.csv'), parse_dates=['timestamp'])
data.head()


# In[6]:


def get_time_of_day(inp_hour):
    if (inp_hour >= 12) and (inp_hour < 18):
        return 'day'
    elif (inp_hour >= 6) and (inp_hour < 12):
        return 'morning'
    elif (inp_hour >= 18) and (inp_hour <= 23):
        return 'evening'
    else:
        return 'night'


# In[7]:


get_ipython().run_cell_magic('time', '', "data['dom']  = data.timestamp.apply(lambda x: x.day)\ndata['dow']  = data.timestamp.apply(lambda x: x.weekday())\ndata['hour'] = data.timestamp.apply(lambda x: x.hour)\ndata['tod']  = data.hour.apply(get_time_of_day)\n\ndata.head()")


# In[8]:


data.keys()[4:]


# In[9]:


target = data['multi_class_target']


# In[10]:


#lb_dom  = LabelBinarizer().fit(data['dom'])
#lb_dow  = LabelBinarizer().fit(data['dow'])
#lb_hour = LabelBinarizer().fit(data['hour'])
#lb_tod  = LabelBinarizer().fit(data['tod'])


# In[11]:


#dom_features  = ['dom_'  + str(el) for el in lb_dom.classes_]
#dow_features  = ['dow_'  + str(el) for el in lb_dow.classes_]
#hour_features = ['hour_' + str(el) for el in lb_hour.classes_]
#tod_features  = ['tod_'  + str(el) for el in lb_tod.classes_]


# In[12]:


#%%time
#dom_prep  = lb_dom.transform(data['dom'])
#dow_prep  = lb_dow.transform(data['dow'])
#hour_prep = lb_hour.transform(data['hour'])
#tod_prep =  lb_tod.transform(data['tod'])

#dom_prep.shape, dow_prep.shape, hour_prep.shape, tod_prep.shape


# saving LabelBinarizer for using in prepare to submit part

# In[13]:


#pickle.dump(lb_dom,  open((os.path.join(UTILS, 'lb_dom.pkl')),  'wb'))
#pickle.dump(lb_dow,  open((os.path.join(UTILS, 'lb_dow.pkl')),  'wb'))
#pickle.dump(lb_hour, open((os.path.join(UTILS, 'lb_hour.pkl')), 'wb'))
#pickle.dump(lb_tod,  open((os.path.join(UTILS, 'lb_tod.pkl')),  'wb'))


# In[ ]:




%%time
data = data.join(pd.DataFrame(dom_prep,  columns = dom_features), how='inner')
data = data.join(pd.DataFrame(dow_prep,  columns = dow_features), how='inner')
data = data.join(pd.DataFrame(hour_prep, columns = hour_features), how='inner')
data = data.join(pd.DataFrame(tod_prep,  columns = tod_features), how='inner')
# In[ ]:





# merge current data with statistics

# In[52]:


client_freq_targ = pd.read_csv(os.path.join(DATA_OWN, 'client_freq.csv'))
client_diff_freq_dow = pd.read_csv(os.path.join(DATA_OWN, 'client_diff_freq_dow.csv'))
client_diff_freq_tod = pd.read_csv(os.path.join(DATA_OWN, 'client_diff_freq_tod.csv'))


# In[15]:


client_freq_targ.keys()


# In[16]:


data = data.merge(client_freq_targ, how= 'left', on='client_pin', validate='many_to_one')


# In[17]:


#client_freq_features = ['client_freq_main_screen', 'client_freq_statement',
#       'client_freq_credit_info', 'client_freq_own_transfer',
#       'client_freq_mobile_recharge', 'client_freq_phone_money_transfer',
#       'client_freq_card2card_transfer', 'client_freq_chat',
#       'client_freq_card_recharge', 'client_freq_invest']
client_freq_features = client_freq_targ.keys()[1:]


# In[ ]:





# In[ ]:


new_df = pd.merge(A_df, B_df,  how='left', left_on='[A_c1,c2]', right_on = '[B_c1,c2]')


# In[ ]:





# In[ ]:





# In[18]:


glob_diff_freq_dom  = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_dom.csv'))
glob_diff_freq_dow  = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_dow.csv'))
glob_diff_freq_hour = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_hour.csv'))
glob_diff_freq_tod  = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_tod.csv'))


# In[19]:


glob_diff_freq_dom.columns  = ['dom_'  + el for el in glob_diff_freq_dom.keys()]
glob_diff_freq_dow.columns  = ['dow_'  + el for el in glob_diff_freq_dow.keys()]
glob_diff_freq_hour.columns = ['hour_' + el for el in glob_diff_freq_hour.keys()]
glob_diff_freq_tod.columns  = ['tod_'  + el for el in glob_diff_freq_tod.keys()]


# In[20]:


glob_diff_freq_dom = glob_diff_freq_dom.rename(  columns={'dom_dom': 'dom'})
glob_diff_freq_dow = glob_diff_freq_dow.rename(  columns={'dow_dow': 'dow'})
glob_diff_freq_hour = glob_diff_freq_hour.rename(columns={'hour_hour': 'hour'})
glob_diff_freq_tod = glob_diff_freq_tod.rename(  columns={'tod_tod': 'tod'})

#glob_freq_dom.keys(), glob_freq_dow.keys(), glob_freq_hour.keys(), glob_freq_tod.keys(), 


# In[21]:


dom_diff_freq_features  = [el for el in glob_diff_freq_dom.keys()[1:]]
dow_diff_freq_features  = [el for el in glob_diff_freq_dow.keys()[1:]]
hour_diff_freq_features = [el for el in glob_diff_freq_hour.keys()[1:]]
tod_diff_freq_features  = [el for el in glob_diff_freq_tod.keys()[1:]]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


data.shape


# In[23]:


data = data.merge(glob_diff_freq_dom, how= 'left', on='dom', validate='many_to_one')
data.shape


# In[24]:


data = data.merge(glob_diff_freq_dow, how= 'left', on='dow', validate='many_to_one')
data.shape


# In[25]:


data = data.merge(glob_diff_freq_hour, how= 'left', on='hour', validate='many_to_one')
data.shape


# In[26]:


data = data.merge(glob_diff_freq_tod, how= 'left', on='tod', validate='many_to_one')
data.shape


# In[ ]:





# In[ ]:





# In[27]:


data.sample(10)


# In[28]:


get_ipython().run_cell_magic('time', '', 'X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.33, random_state=42)\nX_train.shape, X_val.shape, y_train.shape, y_val.shape')


# In[ ]:





# In[ ]:





# In[43]:


#f1 = make_scorer(f1_score , average='macro')


# In[61]:


using_features = []
using_features.extend(client_freq_features)
using_features.extend(dom_diff_freq_features)
using_features.extend(dow_diff_freq_features)
using_features.extend(hour_diff_freq_features)
using_features.extend(tod_diff_freq_features)

print(len(using_features), using_features)


# In[ ]:





# ## check classifiers with check time of work
SGDClassifier
MLPClassifier
KNeighborsClassifier
SVC
RandomForestClassifier, AdaBoostClassifier
GaussianNB
# In[ ]:





# In[ ]:


#clf_sgd_log   = SGDClassifier(loss = 'log', class_weight='balanced', n_jobs=-1)

#clf_knn  = KNeighborsClassifier()
#clf_svc  = SVC()
clf_gaus = GaussianNB()


# In[47]:


get_ipython().run_cell_magic('time', '', "#clf_sgd_hinge = SGDClassifier(loss = 'hinge', class_weight='balanced', n_jobs=-1)\nclf_sgd_hinge = SGDClassifier(loss = 'hinge', n_jobs=-1)\nclf_sgd_hinge.fit(X_train[using_features], y_train)\npred_sgd_hinge = clf_sgd_hinge.predict(X_val[using_features])\nprint(len(set(pred_sgd_hinge)), set(pred_sgd_hinge))\nprint(f1_score(y_val, pred_sgd_hinge, average  = 'micro'))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#clf_sgd_perc  = SGDClassifier(loss = 'perceptron', class_weight='balanced', n_jobs=-1)\nclf_sgd_perc  = SGDClassifier(loss = 'perceptron', n_jobs=-1)\nclf_sgd_perc.fit(X_train[using_features], y_train)\npred_sgd_perc = clf_sgd_perc.predict(X_val[using_features])\nprint(len(set(pred_sgd_perc)), set(pred_sgd_perc))\nprint(f1_score(y_val, pred_sgd_perc, average  = 'micro'))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# 9 min\n\nclf_mlp  = MLPClassifier((200, 50), learning_rate = 'adaptive', activation='logistic', verbose = True)\nclf_mlp.fit(X_train[using_features], y_train)\npred_mlp = clf_mlp.predict(X_val[using_features])\nprint(set(pred_mlp))\nprint(f1_score(y_val, pred_mlp, average  = 'micro'))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# TOOO LONG\n\n#clf_knn.fit(X_train[using_features], y_train)\n#pred_knn = clf_knn.predict(X_val[using_features])\n#print(set(pred_knn))\n#print(f1_score(y_val, pred_knn, average  = 'micro'))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# TOOO LONG\n\n#clf_svc.fit(X_train[using_features], y_train)\n#pred_svc = clf_svc.predict(X_val[using_features])\n#print(set(pred_svc))\n#print(f1_score(y_val, pred_svc, average  = 'micro'))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# 24 min\n\nclf_rf   = RandomForestClassifier(n_jobs = -1, verbose = 1) \nclf_rf.fit(X_train[using_features], y_train)\npred_rf = clf_rf.predict(X_val[using_features])\nprint(set(pred_rf))\nprint(f1_score(y_val, pred_rf, average  = 'micro'))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "clf_gaus.fit(X_train[using_features], y_train)\npred_gaus = clf_gaus.predict(X_val[using_features])\nprint(set(pred_gaus))\nprint(f1_score(y_val, pred_gaus, average  = 'micro'))")


# In[51]:


get_ipython().run_cell_magic('time', '', "\n\nadaboost_inp_clf = DecisionTreeClassifier()\nclf_ab   = AdaBoostClassifier(adaboost_inp_clf)\nclf_ab.fit(X_train[using_features], y_train)\npred_ab = clf_ab.predict(X_val[using_features])\nprint(len(set(pred_ab)), set(pred_ab))\nprint(f1_score(y_val, pred_ab, average  = 'micro'))")


# In[55]:


get_ipython().run_cell_magic('time', '', "\nclf_lr = LogisticRegression(n_jobs = -1, verbose = 1)\nclf_lr.fit(X_train[using_features], y_train)\npred_lr = clf_lr.predict(X_val[using_features])\nprint(len(set(pred_lr)), set(pred_lr))\nprint(f1_score(y_val, pred_lr, average  = 'micro'))")


# In[64]:


print(len(set(pred_lr)), set(pred_lr))


# In[57]:


get_ipython().run_cell_magic('time', '', "\nclf_lrsvc = LinearSVC() # loss = ‘hinge’\nclf_lrsvc.fit(X_train[using_features], y_train)\npred_lrsvc = clf_lrsvc.predict(X_val[using_features])\nprint(len(set(pred_lrsvc)), set(pred_lrsvc))\nprint(f1_score(y_val, pred_lrsvc, average  = 'micro'))")


# In[60]:


get_ipython().run_cell_magic('time', '', "\nclf_lrsvc_hinge = LinearSVC( loss = 'hinge')\nclf_lrsvc_hinge.fit(X_train[using_features], y_train)\npred_lrsvc_hinge = clf_lrsvc_hinge.predict(X_val[using_features])\nprint(len(set(pred_lrsvc_hinge)), set(pred_lrsvc_hinge))\nprint(f1_score(y_val, pred_lrsvc_hinge, average  = 'micro'))")


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', "clf_cb = CatBoostClassifier()\nclf_cb.fit(X_train[using_features], y_train)\npred_cb = clf_cb.predict(X_val[using_features])\n#print(set(pred_cb))\nprint(f1_score(y_val, pred_cb, average  = 'micro'))")


# In[ ]:





# In[ ]:


pred_list = [pred_sgd, pred_mlp, pred_knn, pred_svc, pred_rf, pred_ab, pred_gaus, pred_cb]


# In[ ]:


#for el in itertools.permutations(pred_list):
#    pred_all = []
#    pred_all = []


# In[ ]:


def set_pred(idx):
    
    return Counter(                    pred_sgd[idx],                    pred_mlp[idx],                    pred_knn[idx],                    pred_svc[idx],                    pred_rf[idx],                    pred_ab[idx],                    pred_gaus[idx],                    pred_cb[idx],                   ).most_common()[0][0]


# In[ ]:


pred_all = [set_pred(el) for el in range(len(y_val))]
print(f1_score(y_val, pred_all, average  = 'micro'))


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


clf_sgd = SGDClassifier(loss = 'hinge', n_jobs=-1)
#clf_sgd_log   = SGDClassifier(loss = 'log', n_jobs=-1)
#clf_sgd  = SGDClassifier(loss = 'perceptron', n_jobs=-1)

clf_mlp  = MLPClassifier()
#clf_knn  = KNeighborsClassifier()
clf_svc  = SVC()
clf_rf   = RandomForestClassifier() 
clf_ab   = AdaBoostClassifier()
clf_gaus = GaussianNB()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf_sgd.fit(data[using_features], target)')


# ## save models

# In[68]:


get_ipython().run_cell_magic('time', '', "#pickle.dump(clf_sgd, open(os.path.join(MODELS, 'clf_sgd.pkl'), 'wb'))\n#pickle.dump(clf_mlp, open(os.path.join(MODELS, 'clf_mlp.pkl'), 'wb'))\n#pickle.dump(clf_knn, open(os.path.join(MODELS, 'clf_knn.pkl'), 'wb'))\n#pickle.dump(clf_lr, open(os.path.join(MODELS, 'clf_lr.pkl'), 'wb'))\npickle.dump(clf_lrsvc, open(os.path.join(MODELS, 'clf_lrsvc.pkl'), 'wb'))\n#pickle.dump(clf_svc, open(os.path.join(MODELS, 'clf_svc.pkl'), 'wb'))\n#pickle.dump(clf_rf,  open(os.path.join(MODELS, 'clf_rf.pkl'),  'wb'))\n#pickle.dump(clf_ab,  open(os.path.join(MODELS, 'clf_ab.pkl'),  'wb'))\n#pickle.dump(clf_gaus,open(os.path.join(MODELS, 'clf_gaus.pkl'),'wb'))\n            \n#clf_cb.save_model(os.path.join(MODELS, 'clf_cb.cbm'), format='cbm')")


# In[ ]:




