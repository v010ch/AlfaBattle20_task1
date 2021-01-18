#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import gc
from glob import glob
from tqdm import tqdm
tqdm.pandas()

import pickle
#import pyarrow.parquet as pq
#import dask


# In[2]:


get_ipython().run_line_magic('pylab', 'inline')


# In[3]:


DATA = './data'
DATA_OWN = './data_own'
CLICKSTREAM = 'alfabattle2_abattle_clickstream'


# In[ ]:





# In[37]:


get_ipython().run_cell_magic('time', '', "data_train = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_train_target.csv'), parse_dates=['timestamp']).sort_values(by = ['timestamp'])\ndata_train.reset_index(inplace = True)\ndata_train.drop('index', axis = 1, inplace = True)\ndata_train.index.name = 'ind'\ndata_train.head()")


# In[38]:


data_train.shape


# form features for future using: day of month / day of week / hour / times of day

# In[39]:


def get_time_of_day(inp_hour):
    if (inp_hour >= 12) and (inp_hour < 18):
        return 'day'
    elif (inp_hour >= 6) and (inp_hour < 12):
        return 'morning'
    elif (inp_hour >= 18) and (inp_hour <= 23):
        return 'evening'
    else:
        return 'night'


# In[40]:


def week_moment(inp_dow):
    if (inp_dow == 5) or (inp_dow == 6):
        return 'weekend'
    else:
        return 'weekdays'


# In[ ]:





# In[41]:


def holiday():
    return 'holiday'
    return 'pre-holiday'
    return 'after_the_holiday'


# In[42]:


get_ipython().run_cell_magic('time', '', "# 51.3 s\n\ndata_train['dom']  = data_train.timestamp.apply(lambda x: x.day)\ndata_train['dow']  = data_train.timestamp.apply(lambda x: x.weekday())\ndata_train['hour'] = data_train.timestamp.apply(lambda x: x.hour)\ndata_train['tod']  = data_train.hour.apply(get_time_of_day)\ndata_train['week_moment'] = data_train.dow.apply(week_moment)")


# calculate time difference (in hours) between sessions inside each client for future using

# In[43]:


get_ipython().run_cell_magic('time', '', "# 49.2 s\n\ndata_train['time_diff'] = data_train.groupby(['client_pin']).timestamp.diff()#.agg('size')\ndata_train['time_diff'] = data_train.time_diff.apply(lambda x: x.total_seconds() / 3600)\ndata_train['time_diff'].fillna(0, inplace = True)")


# In[ ]:





# In[44]:


data_train.head()


# In[12]:


#data.groupby(['client_pin']).multi_class_target.agg(['nunique', 'size'])['size'].hist()


# In[13]:


['last_before', 'most_common', 'mc_last_24hours', 'mc_last_week', 'days_since_last_'
 'days_since_last_mobile_recharge', 'days_since_last_statement', 'days_since_last_phone_money_transfer', 
 'days_since_last_chat', 'days_since_last_invest', 'days_since_last_main_screen', 'days_since_last_own_transfer', 
 'days_since_last_card_recharge', 'days_since_last_credit_info', 'days_since_last_card2card_transfer'
]


# In[14]:


#data.multi_class_target.unique()


# In[ ]:





# In[ ]:





# In[ ]:




main_screen             2280763
statement                922569
credit_info              498698
own_transfer             290077
mobile_recharge          266485
phone_money_transfer     232911
card2card_transfer       193378
chat                     184775
card_recharge            138616
invest                    57078
# In[ ]:





# In[15]:


def glob_freq_blank(inp_series, feature):
    tmp = inp_series.value_counts()
    return (tmp[feature]) / sum(tmp.values)


# #### global frequency of target for all clients

# In[16]:


def glob_freq_main_screen(inp_series):
    return glob_freq_blank(inp_series, 'main_screen')
    
def glob_freq_statement(inp_series):
    return glob_freq_blank(inp_series, 'statement')
    
def glob_freq_credit_info(inp_series):
    return glob_freq_blank(inp_series, 'credit_info')
    
def glob_freq_own_transfer(inp_series):
    return glob_freq_blank(inp_series, 'own_transfer')

def glob_freq_mobile_recharge(inp_series):
    return glob_freq_blank(inp_series, 'mobile_recharge')
          
def glob_freq_phone_money_transfer(inp_series):
    return glob_freq_blank(inp_series, 'phone_money_transfer')                             
     
def glob_freq_card2card_transfer(inp_series):
    return glob_freq_blank(inp_series, 'card2card_transfer')
       
def glob_freq_chat(inp_series):
    return glob_freq_blank(inp_series, 'chat')
                     
def glob_freq_card_recharge(inp_series):
    return glob_freq_blank(inp_series, 'card_recharge')
    
def glob_freq_invest(inp_series):
    return glob_freq_blank(inp_series, 'invest')


# In[17]:


glob_for_agg = [glob_freq_main_screen,     glob_freq_statement, 
                glob_freq_credit_info,     glob_freq_own_transfer,
                glob_freq_mobile_recharge, glob_freq_phone_money_transfer, 
                glob_freq_card2card_transfer, glob_freq_chat, 
                glob_freq_card_recharge,   glob_freq_invest
               ]


# In[18]:


get_ipython().run_cell_magic('time', '', "glob_freq = data_train.multi_class_target.agg(glob_for_agg)\nglob_freq.to_csv(os.path.join(DATA_OWN, 'glob_freq.csv'))\nglob_freq.head()")


# In[ ]:





# #### global frequency of target for all clients for every day of month

# In[19]:


get_ipython().run_cell_magic('time', '', "# 5.66 s\n# 5.99 s\n\nglob_freq_dom = data_train.groupby('dom').multi_class_target.agg(glob_for_agg)\nglob_freq_dom.to_csv(os.path.join(DATA_OWN, 'glob_freq_dom.csv'))\nglob_freq_dom.head()")


# In[ ]:





# #### global frequency of target for all clients for every day of week

# In[20]:


get_ipython().run_cell_magic('time', '', "# 5.44 s\n# 5.56 s\n\nglob_freq_dow = data_train.groupby('dow').multi_class_target.agg(glob_for_agg)\nglob_freq_dow.to_csv(os.path.join(DATA_OWN, 'glob_freq_dow.csv'))\nglob_freq_dow")


# In[ ]:





# #### global frequency of target for all clients for every hour

# In[21]:


get_ipython().run_cell_magic('time', '', "# 5.73 s\n# 5.52 s\n\nglob_freq_hour = data_train.groupby('hour').multi_class_target.agg(glob_for_agg)\nglob_freq_hour.to_csv(os.path.join(DATA_OWN, 'glob_freq_hour.csv'))\n#glob_freq_hour.head()\nglob_freq_hour.sample(5)")


# In[ ]:





# #### ???global frequency of target for all clients for every times of day????

# In[22]:


get_ipython().run_cell_magic('time', '', "# 5.7 s\n# 5.65 s\n\nglob_freq_tod = data_train.groupby('tod').multi_class_target.agg(glob_for_agg)\nglob_freq_tod.to_csv(os.path.join(DATA_OWN, 'glob_freq_tod.csv'))\n#glob_freq_tod.head()\nglob_freq_tod")


# In[ ]:





# ## get difference from global freq target

# In[23]:


get_ipython().run_cell_magic('time', '', "# 154 ms\n\nglob_diff_freq_dom  = pd.DataFrame(index = glob_freq_dom.index)\nglob_diff_freq_dow  = pd.DataFrame(index = glob_freq_dow.index)\nglob_diff_freq_hour = pd.DataFrame(index = glob_freq_hour.index)\nglob_diff_freq_tod  = pd.DataFrame(index = glob_freq_tod.index)\n\nfor el in glob_freq.keys():\n    glob_diff_freq_dom['diff_' + el]   = glob_freq[el] - glob_freq_dom[el]\n    glob_diff_freq_dow['diff_' + el]   = glob_freq[el] - glob_freq_dow[el]\n    glob_diff_freq_hour['diff_' + el]  = glob_freq[el] - glob_freq_hour[el]\n    glob_diff_freq_tod['diff_' + el]   = glob_freq[el] - glob_freq_tod[el]\n\n    \nglob_diff_freq_dom.to_csv(os.path.join(DATA_OWN, 'glob_diff_freq_dom.csv'))\nglob_diff_freq_dow.to_csv(os.path.join(DATA_OWN, 'glob_diff_freq_dow.csv'))\nglob_diff_freq_hour.to_csv(os.path.join(DATA_OWN, 'glob_diff_freq_hour.csv'))\nglob_diff_freq_tod.to_csv(os.path.join(DATA_OWN, 'glob_diff_freq_tod.csv'))")


# In[24]:


del glob_diff_freq_dom
del glob_diff_freq_dow
del glob_diff_freq_hour
del glob_diff_freq_tod

gc.collect()


# In[ ]:





# In[ ]:





# In[25]:


def client_freq_blank(inp_series, feature):
    tmp = inp_series.value_counts()
    if feature in tmp.keys():
        return (tmp[feature]) / sum(tmp.values)
    else:
        return 0


# #### client frequency of target for all dataset

# In[26]:


def client_freq_main_screen(inp_series):
    return client_freq_blank(inp_series, 'main_screen')
    
def client_freq_statement(inp_series):
    return client_freq_blank(inp_series, 'statement')
    
def client_freq_credit_info(inp_series):
    return client_freq_blank(inp_series, 'credit_info')
    
def client_freq_own_transfer(inp_series):
    return client_freq_blank(inp_series, 'own_transfer')

def client_freq_mobile_recharge(inp_series):
    return client_freq_blank(inp_series, 'mobile_recharge')
          
def client_freq_phone_money_transfer(inp_series):
    return client_freq_blank(inp_series, 'phone_money_transfer')                             
     
def client_freq_card2card_transfer(inp_series):
    return client_freq_blank(inp_series, 'card2card_transfer')
       
def client_freq_chat(inp_series):
    return client_freq_blank(inp_series, 'chat')
                     
def client_freq_card_recharge(inp_series):
    return client_freq_blank(inp_series, 'card_recharge')
    
def client_freq_invest(inp_series):
    return client_freq_blank(inp_series, 'invest')


# In[27]:


cli_for_agg = [client_freq_main_screen, client_freq_statement, 
               client_freq_credit_info, client_freq_own_transfer,
               client_freq_mobile_recharge, client_freq_phone_money_transfer, 
               client_freq_card2card_transfer, client_freq_chat, 
               client_freq_card_recharge, client_freq_invest
              ]


# In[28]:


#%%timeit
#client_freq_main_screen(data.multi_class_target)


# In[45]:


get_ipython().run_cell_magic('time', '', "# 5min 46s\n\nclient_freq = data_train.groupby('client_pin').multi_class_target.agg(cli_for_agg)\nclient_freq.to_csv(os.path.join(DATA_OWN, 'client_freq.csv'))\nclient_freq.head(10)")


# In[ ]:





# #### client frequency of target for day of week

# In[47]:


get_ipython().run_cell_magic('time', '', "# 32min 55s\n\nclient_freq_dow = data_train.groupby(['client_pin', 'dow']).multi_class_target.agg(cli_for_agg)\n# after read_csv multilevel will disapper\nclient_freq_dow.to_csv(os.path.join(DATA_OWN, 'client_freq_dow.csv'))\nclient_freq_dow.head(10)")


# In[ ]:





# #### client frequency of target for times of day

# In[34]:


get_ipython().run_cell_magic('time', '', "# 19min 58s\n\nclient_freq_tod = data_train.groupby(['client_pin', 'tod']).multi_class_target.agg(cli_for_agg)\n# after read_csv multilevel will disapper\nclient_freq_tod.to_csv(os.path.join(DATA_OWN, 'client_freq_tod.csv'))\nclient_freq_tod.head(10)")


# In[ ]:





# ## get difference from client freq target

# In[48]:


get_ipython().run_cell_magic('time', '', "# 11.3 s\n\n#client_diff_freq_dom  = pd.DataFrame(index = client_freq_dom.index)\nclient_diff_freq_dow  = pd.DataFrame(index = client_freq_dow.index)\n#client_diff_freq_hour = pd.DataFrame(index = client_freq_hour.index)\nclient_diff_freq_tod  = pd.DataFrame(index = client_freq_tod.index)\n\nfor el in client_freq.keys():\n#    client_diff_freq_dom['diff_' + el]   = client_freq[el] - client_freq_dom[el]\n    client_diff_freq_dow['diff_' + el]   = client_freq[el] - client_freq_dow[el]\n#    client_diff_freq_hour['diff_' + el]  = client_freq[el] - client_freq_hour[el]\n    client_diff_freq_tod['diff_' + el]   = client_freq[el] - client_freq_tod[el]\n    \n#client_diff_freq_dom.to_csv(os.path.join(DATA_OWN, 'client_diff_freq_dom.csv'))\nclient_diff_freq_dow.to_csv(os.path.join(DATA_OWN, 'client_diff_freq_dow.csv'))\n#client_diff_freq_hour.to_csv(os.path.join(DATA_OWN, 'client_diff_freq_hour.csv'))\nclient_diff_freq_tod.to_csv(os.path.join(DATA_OWN, 'client_diff_freq_tod.csv'))")


# In[49]:


#del client_diff_freq_dom
#del client_diff_freq_hour

del client_freq
del client_freq_dow
del client_diff_freq_dow
del client_freq_tod
del client_diff_freq_tod

gc.collect()


# In[50]:


del data_train
gc.collect()


# In[ ]:




main_screen             2280763
statement                922569
credit_info              498698
own_transfer             290077
mobile_recharge          266485
phone_money_transfer     232911
card2card_transfer       193378
chat                     184775
card_recharge            138616
invest                    57078
# In[ ]:





# In[ ]:





# when we get features below we should get them for test and train data    
# for thaqt we concat this dataframes

# In[56]:


get_ipython().run_cell_magic('time', '', "data_train = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_train_target.csv'), parse_dates=['timestamp'])#.sort_values(by = ['timestamp'])\ndata_test = pd.read_csv(os.path.join(DATA, 'alfabattle2_prediction_session_timestamp.csv'), parse_dates=['timestamp'])#.sort_values(by = ['timestamp'])\ndata = pd.concat([data_train, data_test])\ndata.sort_values(by = ['timestamp'], inplace = True)\n\ndata.reset_index(inplace = True)\ndata.drop('index', axis = 1, inplace = True)\ndata.index.name = 'ind'\n#data.head()\ndata.tail()")


# ## get last target before this session

# In[57]:


def last_target(inp_ser):
    global dict_last_targets
    
    ret = dict_last_targets[inp_ser.client_pin.values[0]]
    dict_last_targets[inp_ser.client_pin.values[0]] = inp_ser.multi_class_target.values[0]
    
    return ret


# In[60]:


get_ipython().run_cell_magic('time', '', "# 5min 47s\n\ndict_last_targets = {el: 'first_appear' for el in data.client_pin.unique()}\ndata_new = data.groupby(['client_pin', 'timestamp']).progress_apply(last_target)#.value_counts()\ndata_new = pd.DataFrame(data_new.reset_index())\ndata_new.columns = ['client_pin', 'timestamp', 'last_target_begore']\ndata_new.shape")


# In[61]:


data_new.to_csv(os.path.join(DATA_OWN, 'last_target_begore.csv'))
pickle.dump(dict_last_targets, open(os.path.join(DATA_OWN, 'client_last_target.pkl'), 'wb'))


# In[62]:


data_new.head()


# In[63]:


data_new.last_target_begore.value_counts()


# In[ ]:





# In[65]:


data_check = pd.merge(data, data_new,  how='left', left_on=['client_pin', 'timestamp'], right_on = ['client_pin','timestamp'])
print(data_check.isnull().values.any())
del data_check
gc.collect()


# In[66]:


#data_check[data_check.client_pin == data_check.client_pin.unique()[682]]


# In[67]:


del data_new
gc.collect()


# In[ ]:





# # USED UNTIL THIS LINE

# In[ ]:





# In[ ]:





# # time past after last known target for this user

# In[ ]:





# In[ ]:


target_dict = {'main_screen': 0, 
               'statement': 1,
               'credit_info': 2,
               'own_transfer': 3,
               'mobile_recharge': 4,
               'phone_money_transfer': 5,
               'card2card_transfer': 6,
               'chat': 7,
               'card_recharge': 8,
               'invest': 9
}

def target_time_diff(inp_data):

    #print(inp_data.shape)
    #print(inp_data.client_pin)
    #ret_series = pd.Series([[0]]*inp_data.shape[0]
    #                       , index = inp_data.index)
    ret_series = pd.Series(['']*inp_data.shape[0]
                           , index = inp_data.index
                           , dtype = 'object'
                          )
    last_known = [0] * len(target_dict)
    #last_known = np.array([0] * len(target_dict))
    
    #print(ret_series.dtype)
    for idx, val in enumerate(inp_data.index):
        time  = inp_data['time_diff'][val]
        field = inp_data['multi_class_target'][val]
        
        last_known = [x+time for x in last_known]
        #ret_series.loc[val] = last_known
        #ret_series.iloc[idx] = np.array(last_known, dtype = object)
        ret_series.iloc[idx] = ' '.join([str(el) for el in last_known])
        #update last known time:
        last_known[target_dict[field]] = 0
            
    #print(ret_series.shape)
    
    return ret_series

#data.client_pin.unique()
gg1 = '70783113e4f4117935d9f746237fce3e'
gg2 = '989f645a00e3e8a179036cf5fd5be29d'
gg3 = '586fe605f61e81581efe72d9d3dc237d'
gg4 = 'd1007b6356164d9150b57f35e42b7812'
gg5 = '9193bff6506617d47c92d3029807ea2a'
gg6 = '1021464b1a37f1f1c1bff1cc9dd24cf1'


#first = data[data.client_pin == gg].groupby('client_pin')
first = data.query('client_pin == @gg1 or client_pin == @gg2 or client_pin == @gg6 or \
            client_pin == @gg3 or client_pin == @gg4 or client_pin == @gg5').groupby('client_pin')
# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', "# 34min 39s\n#tt = first.apply(target_time_diff)\n\ntime_past_data = data.groupby('client_pin').progress_apply(target_time_diff)\ntime_past_data = time_past_data.reset_index()\ntime_past_data.columns = ['client_pin', 'merge_index', 'last_for_all_sessions']")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# 9.73 s\n# 7.72 s\n\n#pickle.dump(time_past_data, open(os.path.join(DATA_OWN, 'time_past_no_processing.pkl'), 'wb'))\ntime_past_data = pickle.load(open(os.path.join(DATA_OWN, 'time_past_no_processing.pkl'), 'rb'))   ")


# In[ ]:


get_ipython().run_cell_magic('time', '', "time_past_data['last_for_all_sessions_split'] = time_past_data.last_for_all_sessions.apply(lambda x: x.split())\n\nfor el in target_dict.keys():\n    #print(el)\n    time_past_data['time_past_last_' + el] = time_past_data['last_for_all_sessions_split'].apply(lambda x: round(float(x[target_dict[el]]), 4))\n    ")


# In[ ]:


#time_past_data.drop('last_for_all_sessions', axis = 1, inplace = True)
#time_past_data.drop('last_for_all_sessions_split', axis = 1, inplace = True)
time_past_data


# In[ ]:


get_ipython().run_cell_magic('time', '', "time_past_data.to_csv(os.path.join(DATA_OWN, 'time_past_last_target.csv'))\n#time_past_data = pd.read_csv(os.path.join(DATA_OWN, 'time_past_last_target.csv'))")


# In[ ]:


average time between targets


# In[ ]:





# In[ ]:


target_dict.keys()


# In[ ]:


def client_used_target(inp_series, targ):
    
    #print(inp_series.shape)
    #print(inp_series.values)
    #print(inp1)
    #print(inp_series.unique())
    #print(len(inp_series.unique()))
    if targ in inp_series.unique():
        return 1
    else:
        return 0


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', "\nclient_used_target_df = pd.DataFrame(index = data.client_pin.unique())\nfor el in target_dict.keys():\n    tmp_df = data.groupby('client_pin').multi_class_target.agg(client_used_target, targ=el)\n    tmp_df.name = 'has_' + el\n    client_used_target_df = client_used_target_df.merge( tmp_df, how='left', left_index=True, \n                                                        right_index=True, validate='one_to_one')\n")


# In[ ]:


client_used_target_df


# In[ ]:


#client_used_target_df.to_csv(os.path.join(DATA_OWN, 'client_used_target.csv'))

client_used_target_df = pd.read_csv(os.path.join(DATA_OWN, 'client_used_target.csv'))


# In[ ]:





# In[ ]:





# # !!!!! HERE !!!!!

# In[ ]:


#%%time
#with open(os.path.join(DATA_OWN, 'last_known2.pkl'), 'wb') as fd_last_known:
#    pickle.dump(tt, fd_last_known, protocol = 2)


# In[ ]:


tt.index.name = ('client_pin', 'ind')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#tt.to_csv(os.path.join(DATA_OWN, 'client_last_target.csv'))\ndd = pd.read_csv(os.path.join(DATA_OWN, 'client_last_target.csv'))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "with open(os.path.join(DATA_OWN, 'last_known.pkl'), 'rb') as fd_last_known:\n    tt2 = pickle.load(fd_last_known, protocol = 2)")


# In[ ]:


tt.index


# In[ ]:


dd.columns = ['client_pin', 'ind', 'last_known']
dd.set_index('ind', inplace=True)


# In[ ]:


dd.loc[4]


# In[ ]:


data.loc[4]


# In[ ]:


dd.head()


# In[ ]:


for el in tqdm(target_dict.keys()):
    dd['last_known_' + el] = dd.last_known.apply(lambda x: x[target_dict[el]])


# In[ ]:


target_dict.keys()


# In[ ]:


dd


# In[ ]:


dd.loc[0, 'last_known']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from dask.distributed import Client
from dask import delayed

client = Client(n_workers=8)


# In[ ]:


def prep_ds(inp_data):
    
    #global data
    #tmp_data = data.query('client_pin == @inp_client_pin')
    
    #return data.query('client_pin == @inp_client_pin').nunique()
    
    indexes = inp_data.index
    
    dict_freq = {'mobile_recharge': [0], 'statement': [0], 'phone_money_transfer': [0], 
             'chat': [0], 'invest': [0], 'main_screen': [0], 'own_transfer': [0], 
             'card_recharge': [0], 'credit_info': [0], 'card2card_transfer': [0]}
    
    tmp_value_counts = inp_data.multi_class_target.value_counts()
    sum_sess = inp_data.shape[0]
    for el in tmp_value_counts.keys():
        dict_freq[el] = [tmp_value_counts[el] / sum_sess]
    
    feature_dict = {}
    for idx, el in enumerate(indexes):
        feature_dict['last_before'] = 
        #'most_common',
        #'mc_last_24hours',
        #'mc_last_week',
        'days_since_last_days_since_last_mobile_recharge',
        'days_since_last_statement',
        'days_since_last_phone_money_transfer',
        'days_since_last_chat',
        'days_since_last_invest',
        'days_since_last_main_screen',
        'days_since_last_own_transfer',
        'days_since_last_card_recharge',
        'days_since_last_credit_info',
        'days_since_last_card2card_transfer'
    
    return inp_data.nunique()


# In[ ]:


get_ipython().run_cell_magic('time', '', "uniqq = []\nfor el in tqdm(data.client_pin.unique()[:100]):\n    #uniqq.append(data.query('client_pin == @el').nunique())\n    #ret = delayed(prep_ds)(el)\n    ret = delayed(prep_ds)(data.query('client_pin == @el'))\n    uniqq.append(ret)\n    \nss = sum(uniqq)\nansw = ss.compute()\n#answ = sum(uniqq)\n#ret.visalize()\nprint(answ)")


# In[ ]:


#ret.visualize()
client.close()


# In[ ]:


data.client_pin.unique()[:5]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




