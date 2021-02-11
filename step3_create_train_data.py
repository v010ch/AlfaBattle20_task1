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


# In[ ]:





# In[3]:


DATA = os.path.join('.', 'data')
DATA_OWN = os.path.join('.', 'data_own')
CLICKSTREAM = 'alfabattle2_abattle_clickstream'
MODELS = os.path.join('.', 'models')
UTILS = os.path.join('.', 'utils')
SUBM = os.path.join('.', 'submissions')


# In[ ]:





# ## prepare data, test / training sets

# In[4]:


data = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_train_target.csv'), parse_dates=['timestamp'])
data.head()


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


get_ipython().run_cell_magic('time', '', "data['dom']  = data.timestamp.apply(lambda x: x.day)\ndata['dow']  = data.timestamp.apply(lambda x: x.weekday())\ndata['hour'] = data.timestamp.apply(lambda x: x.hour)\ndata['tod']  = data.hour.apply(get_time_of_day)\n\ndata.head()")


# In[7]:


data.keys()[4:]


# In[8]:


#target = data['multi_class_target']


# In[9]:


#lb_dom  = LabelBinarizer().fit(data['dom'])
#lb_dow  = LabelBinarizer().fit(data['dow'])
#lb_hour = LabelBinarizer().fit(data['hour'])
#lb_tod  = LabelBinarizer().fit(data['tod'])


# In[10]:


#dom_features  = ['dom_'  + str(el) for el in lb_dom.classes_]
#dow_features  = ['dow_'  + str(el) for el in lb_dow.classes_]
#hour_features = ['hour_' + str(el) for el in lb_hour.classes_]
#tod_features  = ['tod_'  + str(el) for el in lb_tod.classes_]


# In[11]:


#%%time
#dom_prep  = lb_dom.transform(data['dom'])
#dow_prep  = lb_dow.transform(data['dow'])
#hour_prep = lb_hour.transform(data['hour'])
#tod_prep =  lb_tod.transform(data['tod'])

#dom_prep.shape, dow_prep.shape, hour_prep.shape, tod_prep.shape


# saving LabelBinarizer for using in prepare to submit part

# In[12]:


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





# ## merge current data with client statistics

# In[13]:


get_ipython().run_cell_magic('time', '', "client_freq_targ = pd.read_csv(os.path.join(DATA_OWN, 'client_freq.csv'))\nclient_diff_freq_dow = pd.read_csv(os.path.join(DATA_OWN, 'client_diff_freq_dow.csv'))\nclient_diff_freq_tod = pd.read_csv(os.path.join(DATA_OWN, 'client_diff_freq_tod.csv'))")


# In[14]:


col = ['client_pin', 'dow']
col.extend(['dow_'+el for el in client_diff_freq_dow.keys()[2:]])
client_diff_freq_dow.columns = col


col = ['client_pin', 'tod']
col.extend(['tod_'+el for el in client_diff_freq_tod.keys()[2:]])
client_diff_freq_tod.columns = col
client_diff_freq_dow.keys(), client_diff_freq_tod.keys()


# In[ ]:





# In[15]:


data = data.merge(client_freq_targ, how= 'left', on='client_pin', validate='many_to_one')
data.shape


# In[16]:


#client_freq_features = ['client_freq_main_screen', 'client_freq_statement',
#       'client_freq_credit_info', 'client_freq_own_transfer',
#       'client_freq_mobile_recharge', 'client_freq_phone_money_transfer',
#       'client_freq_card2card_transfer', 'client_freq_chat',
#       'client_freq_card_recharge', 'client_freq_invest']
#client_freq_features = client_freq_targ.keys()[1:]


# In[17]:


#data.head()


# In[ ]:





# In[18]:


data = pd.merge(data, client_diff_freq_dow,  how='left', left_on=['client_pin', 'dow'], right_on = ['client_pin','dow'])
data.shape


# In[19]:


#data.head()


# In[ ]:





# In[20]:


data = pd.merge(data, client_diff_freq_tod,  how='left', left_on=['client_pin', 'tod'], right_on = ['client_pin','tod'])
data.shape


# In[21]:


#new_df = pd.merge(A_df, B_df,  how='left', left_on='[A_c1,c2]', right_on = '[B_c1,c2]')


# In[ ]:





# In[ ]:





# ## merge current data with global statistics

# In[22]:


glob_diff_freq_dom  = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_dom.csv'))
glob_diff_freq_dow  = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_dow.csv'))
glob_diff_freq_hour = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_hour.csv'))
glob_diff_freq_tod  = pd.read_csv(os.path.join(DATA_OWN, 'glob_diff_freq_tod.csv'))


# In[23]:


glob_diff_freq_dom.columns  = ['dom_'  + el for el in glob_diff_freq_dom.keys()]
glob_diff_freq_dow.columns  = ['dow_'  + el for el in glob_diff_freq_dow.keys()]
glob_diff_freq_hour.columns = ['hour_' + el for el in glob_diff_freq_hour.keys()]
glob_diff_freq_tod.columns  = ['tod_'  + el for el in glob_diff_freq_tod.keys()]


# In[24]:


glob_diff_freq_dom = glob_diff_freq_dom.rename(  columns={'dom_dom': 'dom'})
glob_diff_freq_dow = glob_diff_freq_dow.rename(  columns={'dow_dow': 'dow'})
glob_diff_freq_hour = glob_diff_freq_hour.rename(columns={'hour_hour': 'hour'})
glob_diff_freq_tod = glob_diff_freq_tod.rename(  columns={'tod_tod': 'tod'})

#glob_freq_dom.keys(), glob_freq_dow.keys(), glob_freq_hour.keys(), glob_freq_tod.keys(), 


# In[25]:


dom_diff_freq_features  = [el for el in glob_diff_freq_dom.keys()[1:]]
dow_diff_freq_features  = [el for el in glob_diff_freq_dow.keys()[1:]]
hour_diff_freq_features = [el for el in glob_diff_freq_hour.keys()[1:]]
tod_diff_freq_features  = [el for el in glob_diff_freq_tod.keys()[1:]]


# In[ ]:





# In[ ]:





# In[26]:


data.shape


# In[27]:


data = data.merge(glob_diff_freq_dom, how= 'left', on='dom', validate='many_to_one')
data.shape


# In[28]:


data = data.merge(glob_diff_freq_dow, how= 'left', on='dow', validate='many_to_one')
data.shape


# In[29]:


data = data.merge(glob_diff_freq_hour, how= 'left', on='hour', validate='many_to_one')
data.shape


# In[30]:


data = data.merge(glob_diff_freq_tod, how= 'left', on='tod', validate='many_to_one')
data.shape


# In[ ]:





# In[ ]:





# ### Add last known target

# In[31]:


last_target = pd.read_csv(os.path.join(DATA_OWN, 'last_target_begore.csv'), parse_dates=['timestamp'])
last_target.drop('Unnamed: 0', axis = 1, inplace = True)


# In[32]:


get_ipython().run_cell_magic('time', '', "data = data.merge(last_target, how= 'left', on=['client_pin', 'timestamp'], validate='one_to_one')\n#last_target_begore")


# In[33]:


get_ipython().run_cell_magic('time', '', "lb_last_target  = LabelBinarizer().fit(data['last_target_begore'])\nlast_targetdom_features  = ['lt_' + str(el) for el in lb_last_target.classes_]\nlt_prep = lb_last_target.transform(data['last_target_begore'])\ndata = data.join(pd.DataFrame(lt_prep,  columns = last_targetdom_features), how='inner')\ndata.drop('last_target_begore', axis = 1, inplace = True)")


# In[34]:


data.head()


# In[35]:


pickle.dump(lb_last_target, open(os.path.join(UTILS, 'lb_last_target.pkl'), 'wb'))


# In[ ]:





# In[ ]:





# ### ADD relations time spend past target was

# In[42]:


data_relations = pd.read_csv(os.path.join(DATA_OWN, 'relations_time_past_targ.csv'), parse_dates=['timestamp'])
data_relations.drop('Unnamed: 0', inplace = True, axis = 1)


# In[43]:


data = data.merge(data_relations, how= 'left', on=['client_pin', 'timestamp'], validate='one_to_one')


# In[ ]:


#data_relations.head()


# In[ ]:





# In[ ]:





# In[45]:


#data.sample(10)


# In[ ]:





# In[ ]:





# In[ ]:


#f1 = make_scorer(f1_score , average='macro')


# ### create using features list
using_features = []
using_features.extend(client_freq_features)
using_features.extend(dom_diff_freq_features)
using_features.extend(dow_diff_freq_features)
using_features.extend(hour_diff_freq_features)
using_features.extend(tod_diff_freq_features)

print(len(using_features), using_features)
# In[46]:


using_features = data.keys()[8:]
print(len(using_features))
#print(using_features)


# In[ ]:





# ## saving

# In[51]:


get_ipython().run_cell_magic('time', '', "#81 - 8min 57s\n#91 - 9min 14s\n\ndata.to_csv(os.path.join(DATA_OWN, 'data_train.csv'))")


# In[52]:


get_ipython().run_cell_magic('time', '', "pickle.dump(using_features, open(os.path.join(DATA_OWN, 'using_features.pkl'), 'wb'))")


# In[ ]:





# In[ ]:





# #### Set dtypes for reduse mem usage

# In[40]:


get_ipython().run_cell_magic('time', '', "using_features = pickle.load(open(os.path.join(DATA_OWN, 'using_features.pkl'), 'rb'))")


# In[47]:


print("{")
for el in using_features:
    if el.startswith('lt'):
        print("'" + el + "': np.int8,")
    else:
        print("'" + el + "': np.float32, # np.int8,")
print("}")


# In[48]:


load_dtypes = {
'client_freq_main_screen': np.float32, # np.int8,
'client_freq_statement': np.float32, # np.int8,
'client_freq_credit_info': np.float32, # np.int8,
'client_freq_own_transfer': np.float32, # np.int8,
'client_freq_mobile_recharge': np.float32, # np.int8,
'client_freq_phone_money_transfer': np.float32, # np.int8,
'client_freq_card2card_transfer': np.float32, # np.int8,
'client_freq_chat': np.float32, # np.int8,
'client_freq_card_recharge': np.float32, # np.int8,
'client_freq_invest': np.float32, # np.int8,
'dow_diff_client_freq_main_screen': np.float32, # np.int8,
'dow_diff_client_freq_statement': np.float32, # np.int8,
'dow_diff_client_freq_credit_info': np.float32, # np.int8,
'dow_diff_client_freq_own_transfer': np.float32, # np.int8,
'dow_diff_client_freq_mobile_recharge': np.float32, # np.int8,
'dow_diff_client_freq_phone_money_transfer': np.float32, # np.int8,
'dow_diff_client_freq_card2card_transfer': np.float32, # np.int8,
'dow_diff_client_freq_chat': np.float32, # np.int8,
'dow_diff_client_freq_card_recharge': np.float32, # np.int8,
'dow_diff_client_freq_invest': np.float32, # np.int8,
'tod_diff_client_freq_main_screen': np.float32, # np.int8,
'tod_diff_client_freq_statement': np.float32, # np.int8,
'tod_diff_client_freq_credit_info': np.float32, # np.int8,
'tod_diff_client_freq_own_transfer': np.float32, # np.int8,
'tod_diff_client_freq_mobile_recharge': np.float32, # np.int8,
'tod_diff_client_freq_phone_money_transfer': np.float32, # np.int8,
'tod_diff_client_freq_card2card_transfer': np.float32, # np.int8,
'tod_diff_client_freq_chat': np.float32, # np.int8,
'tod_diff_client_freq_card_recharge': np.float32, # np.int8,
'tod_diff_client_freq_invest': np.float32, # np.int8,
'dom_diff_glob_freq_main_screen': np.float32, # np.int8,
'dom_diff_glob_freq_statement': np.float32, # np.int8,
'dom_diff_glob_freq_credit_info': np.float32, # np.int8,
'dom_diff_glob_freq_own_transfer': np.float32, # np.int8,
'dom_diff_glob_freq_mobile_recharge': np.float32, # np.int8,
'dom_diff_glob_freq_phone_money_transfer': np.float32, # np.int8,
'dom_diff_glob_freq_card2card_transfer': np.float32, # np.int8,
'dom_diff_glob_freq_chat': np.float32, # np.int8,
'dom_diff_glob_freq_card_recharge': np.float32, # np.int8,
'dom_diff_glob_freq_invest': np.float32, # np.int8,
'dow_diff_glob_freq_main_screen': np.float32, # np.int8,
'dow_diff_glob_freq_statement': np.float32, # np.int8,
'dow_diff_glob_freq_credit_info': np.float32, # np.int8,
'dow_diff_glob_freq_own_transfer': np.float32, # np.int8,
'dow_diff_glob_freq_mobile_recharge': np.float32, # np.int8,
'dow_diff_glob_freq_phone_money_transfer': np.float32, # np.int8,
'dow_diff_glob_freq_card2card_transfer': np.float32, # np.int8,
'dow_diff_glob_freq_chat': np.float32, # np.int8,
'dow_diff_glob_freq_card_recharge': np.float32, # np.int8,
'dow_diff_glob_freq_invest': np.float32, # np.int8,
'hour_diff_glob_freq_main_screen': np.float32, # np.int8,
'hour_diff_glob_freq_statement': np.float32, # np.int8,
'hour_diff_glob_freq_credit_info': np.float32, # np.int8,
'hour_diff_glob_freq_own_transfer': np.float32, # np.int8,
'hour_diff_glob_freq_mobile_recharge': np.float32, # np.int8,
'hour_diff_glob_freq_phone_money_transfer': np.float32, # np.int8,
'hour_diff_glob_freq_card2card_transfer': np.float32, # np.int8,
'hour_diff_glob_freq_chat': np.float32, # np.int8,
'hour_diff_glob_freq_card_recharge': np.float32, # np.int8,
'hour_diff_glob_freq_invest': np.float32, # np.int8,
'tod_diff_glob_freq_main_screen': np.float32, # np.int8,
'tod_diff_glob_freq_statement': np.float32, # np.int8,
'tod_diff_glob_freq_credit_info': np.float32, # np.int8,
'tod_diff_glob_freq_own_transfer': np.float32, # np.int8,
'tod_diff_glob_freq_mobile_recharge': np.float32, # np.int8,
'tod_diff_glob_freq_phone_money_transfer': np.float32, # np.int8,
'tod_diff_glob_freq_card2card_transfer': np.float32, # np.int8,
'tod_diff_glob_freq_chat': np.float32, # np.int8,
'tod_diff_glob_freq_card_recharge': np.float32, # np.int8,
'tod_diff_glob_freq_invest': np.float32, # np.int8,
'lt_card2card_transfer': np.int8,
'lt_card_recharge': np.int8,
'lt_chat': np.int8,
'lt_credit_info': np.int8,
'lt_first_appear': np.int8,
'lt_invest': np.int8,
'lt_main_screen': np.int8,
'lt_mobile_recharge': np.int8,
'lt_own_transfer': np.int8,
'lt_phone_money_transfer': np.int8,
'lt_statement': np.int8,
'relations_time_past_targ_main_screen': np.float32, # np.int8,
'relations_time_past_targ_statement': np.float32, # np.int8,
'relations_time_past_targ_credit_info': np.float32, # np.int8,
'relations_time_past_targ_own_transfer': np.float32, # np.int8,
'relations_time_past_targ_mobile_recharge': np.float32, # np.int8,
'relations_time_past_targ_phone_money_transfer': np.float32, # np.int8,
'relations_time_past_targ_card2card_transfer': np.float32, # np.int8,
'relations_time_past_targ_chat': np.float32, # np.int8,
'relations_time_past_targ_card_recharge': np.float32, # np.int8,
'relations_time_past_targ_invest': np.float32, # np.int8,
}


# In[50]:


pickle.dump(load_dtypes, open(os.path.join(UTILS, 'load_dtypes.pkl'), 'wb'))


# In[ ]:


#load_dtypes = pickle.load(open(os.path.join(UTILS, 'load_dtypes.pkl'), 'rb'))


# In[ ]:





# In[ ]:




