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


import catboost as cb


# In[3]:


DATA = './data'
DATA_OWN = './data_own'
CLICKSTREAM = 'alfabattle2_abattle_clickstream'
MODELS = './models'
UTILS = './utils'
SUBM = './submissions'


# In[ ]:





# In[ ]:





# In[4]:


data_pred = pd.read_csv(os.path.join(DATA, 'alfabattle2_prediction_session_timestamp.csv'), parse_dates=['timestamp'])
data_pred.head()


# In[21]:


data_pred.shape


# In[5]:


def get_time_of_day(inp_hour):
    if (inp_hour >= 12) and (inp_hour < 18):
        return 'day'
    elif (inp_hour >= 6) and (inp_hour < 12):
        return 'morning'
    elif (inp_hour >= 18) and (inp_hour <= 23):
        return 'evening'
    else:
        return 'night'


# In[6]:


get_ipython().run_cell_magic('time', '', "data_pred['dom']  = data_pred.timestamp.apply(lambda x: x.day)\ndata_pred['dow']  = data_pred.timestamp.apply(lambda x: x.weekday())\ndata_pred['hour'] = data_pred.timestamp.apply(lambda x: x.hour)\ndata_pred['tod']  = data_pred.hour.apply(get_time_of_day)")


# In[7]:


data_pred.head()


# In[8]:


data_pred.isnull().values.any()


# loading LabelBinarizer

# In[9]:


#lb_dom  = pickle.load(open((os.path.join(UTILS, 'lb_dom.pkl')),  'rb'))
#lb_dow  = pickle.load(open((os.path.join(UTILS, 'lb_dow.pkl')),  'rb'))
#b_hour = pickle.load(open((os.path.join(UTILS, 'lb_hour.pkl')), 'rb'))
#b_tod  = pickle.load(open((os.path.join(UTILS, 'lb_tod.pkl')),  'rb'))


# In[10]:


#dom_features  = ['dom_'  + str(el) for el in lb_dom.classes_]
#dow_features  = ['dow_'  + str(el) for el in lb_dow.classes_]
#hour_features = ['hour_' + str(el) for el in lb_hour.classes_]
#tod_features  = ['tod_'  + str(el) for el in lb_tod.classes_]


# In[ ]:





# In[ ]:





# merge data with statistics

# In[20]:


client_freq_targ = pd.read_csv(os.path.join(DATA_OWN, 'client_freq.csv'))
client_diff_freq_dow = pd.read_csv(os.path.join(DATA_OWN, 'client_diff_freq_dow.csv'))
client_diff_freq_tod = pd.read_csv(os.path.join(DATA_OWN, 'client_diff_freq_tod.csv'))
print(client_freq_targ.isnull().values.any(), client_diff_freq_dow.isnull().values.any(), client_diff_freq_tod.isnull().values.any())


# In[12]:


col = ['client_pin', 'dow']
col.extend(['dow_'+el for el in client_diff_freq_dow.keys()[2:]])
client_diff_freq_dow.columns = col


col = ['client_pin', 'tod']
col.extend(['tod_'+el for el in client_diff_freq_tod.keys()[2:]])
client_diff_freq_tod.columns = col
client_diff_freq_dow.keys(), client_diff_freq_tod.keys()


# In[13]:


data_pred = data_pred.merge(client_freq_targ, how= 'left', on='client_pin', validate='many_to_one')
print(data_pred.isnull().values.any())
data_pred = pd.merge(data_pred, client_diff_freq_dow,  how='left', left_on=['client_pin', 'dow'], right_on = ['client_pin','dow'])
print(data_pred.isnull().values.any())
#data_pred = pd.merge(data_pred, client_diff_freq_tod,  how='left', left_on=['client_pin', 'tod'], right_on = ['client_pin','tod'])
#print(data_pred.isnull().values.any())


# In[19]:


is_NaN = data_pred.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = data_pred[row_has_NaN]

print(rows_with_NaN)


# In[ ]:


data_pred.fillna(0, inplace = True)
data_pred.isnull().values.any()


# In[ ]:


client_freq_features = ['client_freq_main_screen', 'client_freq_statement',
       'client_freq_credit_info', 'client_freq_own_transfer',
       'client_freq_mobile_recharge', 'client_freq_phone_money_transfer',
       'client_freq_card2card_transfer', 'client_freq_chat',
       'client_freq_card_recharge', 'client_freq_invest']


# In[ ]:


glob_diff_freq_dom  = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_dom.csv'))
glob_diff_freq_dow  = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_dow.csv'))
glob_diff_freq_hour = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_hour.csv'))
glob_diff_freq_tod  = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_tod.csv'))


# In[ ]:


glob_diff_freq_dom.columns  = ['dom_'  + el for el in glob_diff_freq_dom.keys()]
glob_diff_freq_dow.columns  = ['dow_'  + el for el in glob_diff_freq_dow.keys()]
glob_diff_freq_hour.columns = ['hour_' + el for el in glob_diff_freq_hour.keys()]
glob_diff_freq_tod.columns  = ['tod_'  + el for el in glob_diff_freq_tod.keys()]


# In[ ]:


glob_diff_freq_dom = glob_diff_freq_dom.rename(  columns={'dom_dom': 'dom'})
glob_diff_freq_dow = glob_diff_freq_dow.rename(  columns={'dow_dow': 'dow'})
glob_diff_freq_hour = glob_diff_freq_hour.rename(columns={'hour_hour': 'hour'})
glob_diff_freq_tod = glob_diff_freq_tod.rename(  columns={'tod_tod': 'tod'})

#glob_freq_dom.keys(), glob_freq_dow.keys(), glob_freq_hour.keys(), glob_freq_tod.keys(), 


# In[ ]:


dom_diff_freq_features  = [el for el in glob_diff_freq_dom.keys()[1:]]
dow_diff_freq_features  = [el for el in glob_diff_freq_dow.keys()[1:]]
hour_diff_freq_features = [el for el in glob_diff_freq_hour.keys()[1:]]
tod_diff_freq_features  = [el for el in glob_diff_freq_tod.keys()[1:]]


# In[ ]:


data_pred.isnull().values.any()


# In[ ]:


data_pred.shape


# In[ ]:


data_pred = data_pred.merge(glob_diff_freq_dom, how= 'left', on='dom', validate='many_to_one')
data_pred = data_pred.merge(glob_diff_freq_dow, how= 'left', on='dow', validate='many_to_one')
data_pred = data_pred.merge(glob_diff_freq_hour, how= 'left', on='hour', validate='many_to_one')
data_pred = data_pred.merge(glob_diff_freq_tod, how= 'left', on='tod', validate='many_to_one')
data_pred.shape


# In[ ]:


data_pred.isnull().values.any()


# ## saving 

# In[ ]:


get_ipython().run_cell_magic('time', '', "data_pred.to_csv(os.path.join(DATA_OWN, 'data_pred.csv'))")


# In[ ]:





# In[ ]:


using_features = pickle.load(open(os.path.join(DATA_OWN, 'using_features.pkl'), 'rb'))


# In[ ]:


clf_sgd = pickle.load(open(os.path.join(MODELS, 'clf_sgd.pkl'), 'rb'))


# In[ ]:


pred_sgd = clf_sgd.predict(data_pred[using_features])


# In[ ]:


data_pred.fillna(0, inplace = True)


# In[ ]:





# In[ ]:




