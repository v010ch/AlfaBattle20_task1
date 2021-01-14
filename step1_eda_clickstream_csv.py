#!/usr/bin/env python
# coding: utf-8

# In[100]:


import pandas as pd

import os
from glob import glob
from collections import Counter
import pickle

#from tqdm import tqdm, tqdm_notebook
from tqdm.notebook import trange, tqdm
#tqdm.pandas()


# In[19]:


import pyarrow.parquet as pq


# In[126]:


get_ipython().run_line_magic('pylab', 'inline')


# In[108]:


DATA = './data'
DATA_OWN = './data_own'
CLICKSTREAM = 'alfabattle2_abattle_clickstream'


# #device_is_webview   always   True
# 
# timestamp - дата и время совершения события    
# application_id - идентификатор приложения    
# client	- Идентификатор клиента     
# session_id - Идентификатор сессии    
# event_type - Тип события    
# event_category - Категория события    
# event_name - Имя события    
# event_label - Дополнительный атрибут события    
# device_screen_name - Имя экрана на котором произошло событие    
# timezone - Часовой пояс    
# device_is_webview - Флаг того что страница открыта внутри webview    
# page_urlhost - Домен страницы    
# page_urlpath_full - Путь страницы    
# net_connection_type - Тип подключения    
# net_connection_tech - Технология подключения    

# In[4]:


files = sorted(glob(os.path.join(DATA, CLICKSTREAM, 'part*')))
files


# In[5]:


get_ipython().run_cell_magic('time', '', "data = pd.read_parquet(files[0])#, engine='fastparquet')\ndata.shape")


# In[13]:


data.keys()


# In[27]:


data.application_id.unique()


# In[1]:


#data.info()


# In[14]:


data.head()


# In[15]:


cat_columns = ['application_id', 'event_type', 'device_is_webview', 'page_urlhost', 'net_connection_type', 'net_connection_tech']
# event_category - 355
# event_name - 10396
# event_label - 6370
# device_screen_name - 536
# timezone - 169
# page_urlpath_full - 14208
# 


# In[16]:


data.net_connection_tech.value_counts().keys()


# In[17]:


df.dtypes


# In[65]:


dft = pq.read_table(files[0], use_pandas_metadata=True)
df = dft.to_pandas(categories=cat_columns )

for el in files:
    #data = pd.read_parquet(el, columns = cat_columns)
    #print(data.shape)
    #print(data.dtypes)
    dft = pq.read_table(el, use_pandas_metadata=True)
    df = dft.to_pandas(categories=cat_columns )
    
    print(df.application_id.value_counts())
    
    #for col_el in cat_columns:
        
    
    
    del df
# In[27]:


del data


# In[ ]:





# In[21]:


def get_freq_dict(paths_to_files, column_of_interest):
    
    size_full = 0
    #col_data = Counter()
    col_data = []
    for file in tqdm(paths_to_files):
        tmp_data = pd.read_parquet(file)
        #tmp_data = tmp_data[column_of_interest]    
        
        #for idx in range(tmp_data.shape[0]):
        #col_data += Counter(tmp_data)
        col_data += list(tmp_data[column_of_interest])
        #kjhjhk  up  kjjkjkj.k
        size_full += tmp_data.shape[0]
        del tmp_data
    
    print(size_full)
    #print(len(set(col_data)))
    col_data = Counter(col_data)
    freq_dict = {}
    for idx, el in enumerate(col_data.most_common()):
        freq_dict[el[0]] = (idx, el[1])
        
    #{'site': n, 'site2: m, 'site3': k}
    return freq_dict


# In[109]:


get_ipython().run_cell_magic('time', '', "site_freq = get_freq_dict(files, 'page_urlhost')\nlen(site_freq)")


# In[30]:


'LTE' in site_freq


# In[ ]:


'timestamp', 'application_id', 'client', 'session_id', 'event_type',
'event_category', 'event_name', 'event_label', 'device_screen_name',
'timezone', 'device_is_webview', 'page_urlhost', 'page_urlpath_full',
'net_connection_type', 'net_connection_tech'


# 120.025.286 total records
# 
# 
# 131.676 page_urlpath_full    
# 14 page_urlhost    
# 23 application_id    
# 80.376 client    
# 9.676.500 session_id    
# 5 event_type    
# 376 event_category    
# 83.163 event_name    
# 36.678 event_label    
# 607 device_screen_name    
# 302 timezone    
# 2 device_is_webview    
# 4 net_connection_type    
# 18 net_connection_tech    

# old data
# 
# 120.025.286 total records
# 
# 131.677   page_urlpath_full    
# 15 page_urlhost    
# 23       application_id    
# 80.376    client    
# 9.676.500  session_id    
# 5 event_type    
# 377 event_category    
# 83.160 event_name    
# 36.679 event_label    
# 608 device_screen_name    
# 

# In[5]:


def check_ft_changed_in_sess(paths_to_files, col_of_interest):
    
    
    ret_list  = []
   
    for file in tqdm(paths_to_files):
        tmp_data = pd.read_parquet(file)[['session_id', col_of_interest]]
        #print(tmp_data.keys())
        #tmp_data = tmp_data[['session_id', col_of_interest]]
        
        
        for sess in tqdm(set(tmp_data.session_id), leave = False):
            tmp_df = tmp_data[tmp_data.session_id == sess]
            
            ret_list.append(len(set(tmp_df[col_of_interest])))
            
            
        del tmp_data
        
    return  ret_list


# In[ ]:


list_all_changes = check_ft_changed_in_sess(files, 'application_id')


# In[ ]:





# In[38]:


get_ipython().run_cell_magic('time', '', "data[data.session_id == '5b145a41510b7af3d717ff6a8243ebe8']")


# In[ ]:





# In[7]:


DATA


# In[20]:


sub = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_train_target.csv'))
sub.shape


# In[15]:


sub.head(3)


# In[9]:


sub.multi_class_target.value_counts()


# In[23]:


sub.multi_class_target.hist()


# In[34]:


sample_pred = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_sample_prediction.csv'))
sample_pred.shape


# In[14]:


sample_pred.head(3)


# In[12]:


sample_pred.prediction.value_counts()


# In[24]:


sample_pred.prediction.hist()


# In[27]:


hz = pd.read_csv(os.path.join(DATA, 'alfabattle2_prediction_session_timestamp.csv'))
hz.shape


# In[29]:


hz.head(3)


# In[32]:


same = 0
for el in hz.index:
    if hz.loc[el, 'client_pin'] in site_freq:
        same += 1


# In[33]:


same


# In[35]:


same2 = 0
for el in hz.index:
    if sample_pred.loc[el, 'client_pin'] in site_freq:
        same2 += 1


# In[36]:


same2


# In[91]:


answer_time_chack = data.groupby(['client', 'session_id']).timestamp.agg(min_val = 'min', max_val = 'max')


# In[42]:


hz.head()


# In[93]:


answer_time_chack.index.client


# In[99]:


answer_time_chack.loc[('0014a49ec89e3a43098375b107f8ff2e')].max_val.sort_values()


# In[87]:


hz[hz.client_pin == '0014a49ec89e3a43098375b107f8ff2e']


# In[6]:


data.keys()


# In[17]:


get_ipython().run_cell_magic('time', '', "data.groupby('session_id').timestamp.agg(['min', 'max', 'size'])")


# In[102]:


sess_file = []
for idx, file in tqdm(enumerate(files)):
    tmp_data = pd.read_parquet(file)
    tmp_sess_list = []
    sess_file.append( list(set(tmp_data.session_id)))


# In[105]:


len(sess_file[0])


# In[110]:


with open(os.path.join(DATA_OWN, 'sess_files.pickle'), 'wb') as f:
    pickle.dump(sess_file, f)


# In[111]:


with open(os.path.join(DATA_OWN, 'site_freq.pickle'), 'wb') as f:
    pickle.dump(site_freq, f)


# In[113]:


data.session_id.apply(lambda x: site_freq)


# In[114]:


site_freq


# In[ ]:


invest                  8055
mobile_recharge         8028
statement               7995
chat                    7989
card2card_transfer      7965
phone_money_transfer    7941
card_recharge           7936
main_screen             7893
own_transfer            7759
credit_info             7707


# In[124]:


get_ipython().run_cell_magic('time', '', "\nfirst_site = []\nsecond_site = []\nsess_len = []\nsess_id = ''\n\nfor file in (files):\n    sess_tmp = []\n    tmp_data = pd.read_parquet(file)\n    for idx, el in tqdm(enumerate(tmp_data.session_id)):\n        #if el not in sess_tmp:\n        if el != sess_id:\n            #sess_tmp.append(el)\n            sess_id = el\n            first_site.append(tmp_data.loc[idx, 'page_urlhost'])\n            if (idx+1) < tmp_data.shape[0]:\n                second_site.append(tmp_data.loc[idx+1, 'page_urlhost'])\n            else:\n                second_site.append('None')\n            ")


# In[ ]:





# In[125]:


with open(os.path.join(DATA_OWN, 'first_site.pickle'), 'wb') as f:
    pickle.dump(first_site, f)
    
with open(os.path.join(DATA_OWN, 'second_site.pickle'), 'wb') as f:
    pickle.dump(second_site, f)


# In[ ]:


hist(second_site)


# In[ ]:




