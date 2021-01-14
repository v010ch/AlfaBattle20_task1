#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
from glob import glob

import pandas as pd
import numpy as np
from collections import Counter

import pickle
import gc

from tqdm import tqdm
tqdm.pandas()


# In[22]:


import catboost as cb


# In[64]:


DATA = './data'
DATA_OWN = './data_own'
CLICKSTREAM = 'alfabattle2_abattle_clickstream'
MODELS = './models'
UTILS = './utils'
SUBM = './submissions'


# In[ ]:





# In[ ]:





# In[3]:


data = pd.read_csv(os.path.join(DATA, 'alfabattle2_prediction_session_timestamp.csv'), parse_dates=['timestamp'])
data.head()


# In[5]:


subm = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_sample_prediction.csv'))
subm.head()


# In[15]:


def get_time_of_day(inp_hour):
    if (inp_hour >= 12) and (inp_hour < 18):
        return 'day'
    elif (inp_hour >= 6) and (inp_hour < 12):
        return 'morning'
    elif (inp_hour >= 18) and (inp_hour <= 23):
        return 'evening'
    else:
        return 'night'


# In[16]:


get_ipython().run_cell_magic('time', '', "data['dom']  = data.timestamp.apply(lambda x: x.day)\ndata['dow']  = data.timestamp.apply(lambda x: x.weekday())\ndata['hour'] = data.timestamp.apply(lambda x: x.hour)\ndata['tod']  = data.hour.apply(get_time_of_day)")


# In[19]:


data.head()


# In[ ]:





# loading LabelBinarizer

# In[93]:


lb_dom  = pickle.load(open((os.path.join(UTILS, 'lb_dom.pkl')),  'rb'))
lb_dow  = pickle.load(open((os.path.join(UTILS, 'lb_dow.pkl')),  'rb'))
lb_hour = pickle.load(open((os.path.join(UTILS, 'lb_hour.pkl')), 'rb'))
lb_tod  = pickle.load(open((os.path.join(UTILS, 'lb_tod.pkl')),  'rb'))


# In[94]:


dom_features  = ['dom_'  + str(el) for el in lb_dom.classes_]
dow_features  = ['dow_'  + str(el) for el in lb_dow.classes_]
hour_features = ['hour_' + str(el) for el in lb_hour.classes_]
tod_features  = ['tod_'  + str(el) for el in lb_tod.classes_]


# In[ ]:





# In[ ]:





# merge data with statistics

# In[46]:


client_freq_targ = pd.read_csv(os.path.join(DATA_OWN, 'client_freq_targ.csv'))
data = data.merge(client_freq_targ, how= 'left', on='client_pin', validate='many_to_one')


# In[47]:


client_freq_features = ['client_freq_main_screen', 'client_freq_statement',
       'client_freq_credit_info', 'client_freq_own_transfer',
       'client_freq_mobile_recharge', 'client_freq_phone_money_transfer',
       'client_freq_card2card_transfer', 'client_freq_chat',
       'client_freq_card_recharge', 'client_freq_invest']


# In[25]:


glob_freq_dom  = pd.read_csv(os.path.join(DATA_OWN, 'glob_freq_dom.csv'))
glob_freq_dow  = pd.read_csv(os.path.join(DATA_OWN, 'glob_freq_dow.csv'))
glob_freq_hour = pd.read_csv(os.path.join(DATA_OWN, 'glob_freq_hour.csv'))
glob_freq_tod  = pd.read_csv(os.path.join(DATA_OWN, 'glob_freq_tod.csv'))


# In[26]:


glob_freq_dom.columns  = ['dom_'  + el for el in glob_freq_dom.keys()]
glob_freq_dow.columns  = ['dow_'  + el for el in glob_freq_dow.keys()]
glob_freq_hour.columns = ['hour_' + el for el in glob_freq_hour.keys()]
glob_freq_tod.columns  = ['tod_'  + el for el in glob_freq_tod.keys()]


# In[27]:


glob_freq_dom = glob_freq_dom.rename(  columns={'dom_dom': 'dom'})
glob_freq_dow = glob_freq_dow.rename(  columns={'dow_dow': 'dow'})
glob_freq_hour = glob_freq_hour.rename(columns={'hour_hour': 'hour'})
glob_freq_tod = glob_freq_tod.rename(  columns={'tod_tod': 'tod'})

#glob_freq_dom.keys(), glob_freq_dow.keys(), glob_freq_hour.keys(), glob_freq_tod.keys(), 


# In[28]:


dom_freq_features  = [el for el in glob_freq_dom.keys()[1:]]
dow_freq_features  = [el for el in glob_freq_dow.keys()[1:]]
hour_freq_features = [el for el in glob_freq_hour.keys()[1:]]
tod_freq_features  = [el for el in glob_freq_tod.keys()[1:]]


# In[ ]:





# In[29]:


data.shape


# In[30]:


data = data.merge(glob_freq_dom, how= 'left', on='dom', validate='many_to_one')
data = data.merge(glob_freq_dow, how= 'left', on='dow', validate='many_to_one')
data = data.merge(glob_freq_hour, how= 'left', on='hour', validate='many_to_one')
data = data.merge(glob_freq_tod, how= 'left', on='tod', validate='many_to_one')
data.shape


# In[ ]:





# In[ ]:





# ## load models

# In[55]:


clf_sgd = pickle.load(open(os.path.join(MODELS, 'clf_sgd.pkl'), 'rb'))
#clf_mlp = pickle.loads(os.paht.join(MODELS, 'clf_mlp.pkl'))
#clf_knn = pickle.loads(os.paht.join(MODELS, 'clf_knn.pkl'))
#clf_svc = pickle.loads(os.paht.join(MODELS, 'clf_svc.pkl'))
#clf_rf  = pickle.loads(os.paht.join(MODELS, 'clf_rf.pkl'))
#clf_ab  = pickle.loads(os.paht.join(MODELS, 'clf_ab.pkl'))
#clf_gaus= pickle.loads(os.paht.join(MODELS, 'clf_gauss.pkl'))

#clf_cb  = cb.load_model(os.paht.join(MODELS, 'clf_cb.cbm'), format='cbm')


# In[ ]:





# ## make predictions

# In[56]:


using_features =                  client_freq_features +                  dom_freq_features +                  dow_freq_features +                  hour_freq_features +                  tod_freq_features 


                 # dates as a features \
                 #dom_features + \
                 #dow_features + \
                 #hour_features + \
                 #tod_features 
                 #
print(using_features)


# In[ ]:





# In[57]:


pred_sgd = clf_sgd.predict(data[using_features])


# In[39]:


#pred_mlp = clf_mlp.predict(data[using_features])


# In[40]:


#pred_knn = clf_knn.predict(data[using_features])


# In[41]:


#pred_svc = clf_svc.predict(data[using_features])


# In[42]:


#pred_rf = clf_rf.predict(data[using_features])


# In[43]:


#pred_ab = clf_ab.predict(data[using_features])


# In[44]:


#pred_gaus = clf_gau.predict(data[using_features])


# In[45]:


#pred_cb = clf_cb.predict(data[using_features])


# In[ ]:





# ## make submit

# In[99]:


subm = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_sample_prediction.csv'))


# In[105]:


subm.to_csv(os.path.join(SUBM, 'subm_client_4glob_sgd.csv'), index = False)


# In[ ]:




