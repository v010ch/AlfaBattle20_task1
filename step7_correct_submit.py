#!/usr/bin/env python
# coding: utf-8

# In[32]:


import os
from glob import glob

import pandas as pd
import numpy as np
from collections import Counter

import pickle
import gc

from tqdm import tqdm
tqdm.pandas()


# In[33]:


DATA = './data'
DATA_OWN = './data_own'
CLICKSTREAM = 'alfabattle2_abattle_clickstream'
MODELS = './models'
UTILS = './utils'
SUBM = './submissions'


# In[ ]:





# In[ ]:





# ### Correction if client never used predicted function

# In[34]:


load_col_for_check = ['client_pin', 'timestamp']
data_corr = pd.read_csv(os.path.join(DATA_OWN, 'data_pred.csv'),  usecols=load_col_for_check, parse_dates=['timestamp'])


# In[35]:


subm = pd.read_csv(os.path.join(SUBM, 'subm_client_6diff_lt_relat_mlp.csv'))


# In[36]:


#for idx in data.index:
#    if data.loc[0].client_pin != subm.loc[0].client_pin:
#       print('What?')


# In[37]:


client_used_target_df = pd.read_csv(os.path.join(DATA_OWN, 'client_target_time_mean.csv'))
client_used_target_df.drop('Unnamed: 0', axis = 1, inplace = True)
client_used_target_df.tail()


# In[38]:


last_target = pd.read_csv(os.path.join(DATA_OWN, 'last_target_begore.csv'), parse_dates=['timestamp'])
last_target.drop('Unnamed: 0', axis = 1, inplace = True)
last_target.head()


# In[39]:


#data_check['predicted'] = pred_sgd_full
data_corr = data_corr.merge(subm, how='left', on='client_pin', validate='one_to_one')
data_corr = data_corr.merge(last_target, how='left', on=['client_pin', 'timestamp'], validate='one_to_one')
data_corr = data_corr.merge(client_used_target_df, how='left', on='client_pin', validate='many_to_one')


# In[40]:


data_corr.head()


# In[41]:


def correct_low_info_client_predict(inp_row):
    
    global n_corrected
    
    if inp_row['mean_' + inp_row.prediction] == 0:
        n_corrected += 1
        return inp_row.last_target_begore
    else:
        return inp_row.prediction


# In[42]:


def check_client_low_info(inp_data):
    
    inp_data['corr_prediction'] = inp_data.progress_apply(correct_low_info_client_predict, axis = 1)
    
    return inp_data.corr_prediction


# In[43]:


get_ipython().run_cell_magic('time', '', '\nn_corrected = 0\npred_new = check_client_low_info(data_corr)\nprint(n_corrected)')


# In[45]:


subm.prediction = pred_new


# In[46]:


subm.to_csv(os.path.join(SUBM, 'subm_client_6diff_lt_relat_corr_mlp.csv'), index = False)


# In[ ]:




