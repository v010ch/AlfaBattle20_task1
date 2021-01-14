#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
tqdm.pandas()

import pickle
import pyarrow.parquet as pq
import dask


# In[2]:


get_ipython().run_line_magic('pylab', 'inline')


# In[3]:


DATA = './data'
DATA_OWN = './data_own'
CLICKSTREAM = 'alfabattle2_abattle_clickstream'


# In[ ]:





# In[4]:


data = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_train_target.csv'), parse_dates=['timestamp']).sort_values(by = ['timestamp'])
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)
data.index.name = 'ind'
data.head()


# In[57]:


data.shape, data.client_pin.nunique()


# In[ ]:





# In[5]:


data['dom']  = data.timestamp.apply(lambda x: x.day)


# In[16]:


data['dom'].hist(bins = 31)


# In[7]:


data['dow']  = data.timestamp.apply(lambda x: x.weekday())


# In[17]:


data['dow'].hist(bins = 7)


# In[13]:


data['hour'] = data.timestamp.apply(lambda x: x.hour)


# In[18]:


data['hour'].hist(bins = 24)


# In[19]:


def get_time_of_day(inp_hour):
    if (inp_hour >= 12) and (inp_hour < 18):
        return 'day'
    elif (inp_hour >= 6) and (inp_hour < 12):
        return 'morning'
    elif (inp_hour >= 18) and (inp_hour <= 23):
        return 'evening'
    else:
        return 'night'


# In[20]:


data['tod']  = data.hour.apply(get_time_of_day)


# In[21]:


data['tod'].hist(bins = 4)


# In[ ]:





# In[27]:


def week_moment(inp_dow):
    if (inp_dow == 5) or (inp_dow == 6):
        return 'weekend'
    else:
        return 'weekdays'


# In[28]:


data['week_moment']  = data.dow.apply(week_moment)


# In[29]:


data['week_moment'].hist()


# In[ ]:





# In[74]:


get_ipython().run_cell_magic('time', '', "first_note = data.groupby('client_pin').timestamp.agg(['min', 'max'])\nmax(first_note['min']), max(first_note['max'])")


# In[76]:


get_ipython().run_cell_magic('time', '', "\ncnt = 0\nappear_date = pd.DataFrame(columns = ['cur_date', 'cnt'])\nfirst_note = [el.replace(hour=0, minute=0, second = 0) for el in first_note['min']]\nfirst_note_ind = sorted(list(set(first_note)))\n#for el in first_note['min']:\nfor el in first_note_ind:\n    cnt += first_note.count(el)\n    appear_date = appear_date.append({'cur_date': el, 'cnt': cnt}, ignore_index = True)\n\n#appear_date")


# In[77]:


appear_date['cnt'].plot()


# In[86]:


q = 0.9
np.quantile(appear_date['cnt'], q, axis = 0), data.client_pin.nunique() - np.quantile(appear_date['cnt'], q, axis = 0)


# In[ ]:





# In[ ]:





# In[ ]:




