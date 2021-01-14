#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import pandas as pd
import numpy as np
import gc
from glob import glob
from tqdm import tqdm
tqdm.pandas()

#import pickle
#import pyarrow.parquet as pq
#import dask


# In[2]:


get_ipython().run_line_magic('pylab', 'inline')


# In[3]:


DATA = './data'
DATA_OWN = './data_own'
CLICKSTREAM = 'alfabattle2_abattle_clickstream'


# In[ ]:





# In[4]:


get_ipython().run_cell_magic('time', '', "data = pd.read_csv(os.path.join(DATA, 'alfabattle2_abattle_train_target.csv'), parse_dates=['timestamp']).sort_values(by = ['timestamp'])\ndata.reset_index(inplace = True)\ndata.drop('index', axis = 1, inplace = True)\ndata.index.name = 'ind'\ndata.head()")


# In[5]:


data.shape


# form features for future using: day of month / day of week / hour / times of day

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


# In[29]:


def week_moment(inp_dow):
    if (inp_dow == 5) or (inp_dow == 6):
        return 'weekend'
    else:
        return 'weekdays'


# In[ ]:





# In[37]:


def holiday():
    return 'holiday'
    return 'pre-holiday'
    return 'after_the_holiday'


# In[31]:


get_ipython().run_cell_magic('time', '', "# 51.3 s\n\ndata['dom']  = data.timestamp.apply(lambda x: x.day)\ndata['dow']  = data.timestamp.apply(lambda x: x.weekday())\ndata['hour'] = data.timestamp.apply(lambda x: x.hour)\ndata['tod']  = data.hour.apply(get_time_of_day)\ndata['week_moment'] = data.dow.apply(week_moment)")


# calculate time difference (in hours) between sessions inside each client for future using

# In[8]:


get_ipython().run_cell_magic('time', '', "# 49.2 s\n\ndata['time_diff'] = data.groupby(['client_pin']).timestamp.diff()#.agg('size')\ndata['time_diff'] = data.time_diff.apply(lambda x: x.total_seconds() / 3600)\ndata['time_diff'].fillna(0, inplace = True)")


# In[ ]:





# In[32]:


data.head()


# In[10]:


#data.groupby(['client_pin']).multi_class_target.agg(['nunique', 'size'])['size'].hist()


# In[11]:


['last_before', 'most_common', 'mc_last_24hours', 'mc_last_week', 'days_since_last_'
 'days_since_last_mobile_recharge', 'days_since_last_statement', 'days_since_last_phone_money_transfer', 
 'days_since_last_chat', 'days_since_last_invest', 'days_since_last_main_screen', 'days_since_last_own_transfer', 
 'days_since_last_card_recharge', 'days_since_last_credit_info', 'days_since_last_card2card_transfer'
]


# In[ ]:


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





# In[72]:


def glob_freq_blank(inp_series, feature):
    tmp = inp_series.value_counts()
    return (tmp[feature]) / sum(tmp.values)


# #### global frequency of target for all clients

# In[13]:


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


# In[14]:


glob_for_agg = [glob_freq_main_screen,     glob_freq_statement, 
                glob_freq_credit_info,     glob_freq_own_transfer,
                glob_freq_mobile_recharge, glob_freq_phone_money_transfer, 
                glob_freq_card2card_transfer, glob_freq_chat, 
                glob_freq_card_recharge,   glob_freq_invest
               ]


# In[41]:


get_ipython().run_cell_magic('time', '', "glob_freq = data.multi_class_target.agg(glob_for_agg)\nglob_freq.to_csv(os.path.join(DATA_OWN, 'glob_freq.csv'))\nglob_freq.head()")


# In[ ]:





# #### global frequency of target for all clients for every day of month

# In[55]:


get_ipython().run_cell_magic('time', '', "# 5.66 s\n# 5.99 s\n\nglob_freq_dom = data.groupby('dom').multi_class_target.agg(glob_for_agg)\nglob_freq_dom.to_csv(os.path.join(DATA_OWN, 'glob_freq_dom.csv'))\nglob_freq_dom.head()")


# In[58]:


#['dom_' + str(el) for el in glob_freq_dom.keys()]


# In[ ]:


#del glob_freq_dom
#gc.collect()


# In[ ]:





# #### global frequency of target for all clients for every day of week

# In[16]:


get_ipython().run_cell_magic('time', '', "# 5.44 s\n# 5.56 s\n\nglob_freq_dow = data.groupby('dow').multi_class_target.agg(glob_for_agg)\nglob_freq_dow.to_csv(os.path.join(DATA_OWN, 'glob_freq_dow.csv'))\nglob_freq_dow")


# In[ ]:


#del glob_freq_dow
#gc.collect()


# In[ ]:





# #### global frequency of target for all clients for every hour

# In[17]:


get_ipython().run_cell_magic('time', '', "# 5.73 s\n# 5.52 s\n\nglob_freq_hour = data.groupby('hour').multi_class_target.agg(glob_for_agg)\nglob_freq_hour.to_csv(os.path.join(DATA_OWN, 'glob_freq_hour.csv'))\n#glob_freq_hour.head()\nglob_freq_hour.sample(5)")


# In[ ]:


#del glob_freq_hour
#gc.collect()


# In[ ]:





# #### ???global frequency of target for all clients for every times of day????

# In[18]:


get_ipython().run_cell_magic('time', '', "# 5.7 s\n# 5.65 s\n\nglob_freq_tod = data.groupby('tod').multi_class_target.agg(glob_for_agg)\nglob_freq_tod.to_csv(os.path.join(DATA_OWN, 'glob_freq_tod.csv'))\n#glob_freq_tod.head()\nglob_freq_tod")


# In[ ]:


#del glob_freq_tod
#gc.collect()


# In[47]:


#glob_freq.keys()


# In[60]:





# ## get difference from global freq target

# In[67]:


get_ipython().run_cell_magic('time', '', "# 154 ms\n\nglob_diff_freq_dom  = pd.DataFrame(index = glob_freq_dom.index)\nglob_diff_freq_dow  = pd.DataFrame(index = glob_freq_dow.index)\nglob_diff_freq_hour = pd.DataFrame(index = glob_freq_hour.index)\nglob_diff_freq_tod  = pd.DataFrame(index = glob_freq_tod.index)\n\nfor el in glob_freq.keys():\n    glob_diff_freq_dom['diff_' + el]   = glob_freq[el] - glob_freq_dom[el]\n    glob_diff_freq_dow['diff_' + el]   = glob_freq[el] - glob_freq_dow[el]\n    glob_diff_freq_hour['diff_' + el]  = glob_freq[el] - glob_freq_hour[el]\n    glob_diff_freq_tod['diff_' + el]   = glob_freq[el] - glob_freq_tod[el]\n    #glob_diff_freq_dom['diff_' + el]  = glob_freq_dom[el].apply(diff_freq_main_screen)\n    #glob_diff_freq_dow['diff_' + el]  = glob_freq_dow[el].apply()\n    #glob_diff_freq_hour['diff_' + el] = glob_freq_hour[el].apply()\n    #glob_diff_freq_tod['diff_' + el]  = glob_freq_tod[el].apply()\n    \nglob_diff_freq_dom.to_csv(os.path.join(DATA_OWN, 'glob_diff_freq_dom.csv'))\nglob_diff_freq_dow.to_csv(os.path.join(DATA_OWN, 'glob_diff_freq_dow.csv'))\nglob_diff_freq_hour.to_csv(os.path.join(DATA_OWN, 'glob_diff_freq_hour.csv'))\nglob_diff_freq_tod.to_csv(os.path.join(DATA_OWN, 'glob_diff_freq_tod.csv'))")


# In[69]:


#del glob_diff_freq_dom
#del glob_diff_freq_dow
#del glob_diff_freq_hour
#del glob_diff_freq_tod

#gc.collect()


# In[ ]:





# In[ ]:





# In[20]:


def client_freq_blank(inp_series, feature):
    tmp = inp_series.value_counts()
    if feature in tmp.keys():
        return (tmp[feature]) / sum(tmp.values)
    else:
        return 0


# #### client frequency of target for all dataset

# In[21]:


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


# In[22]:


cli_for_agg = [client_freq_main_screen, client_freq_statement, 
               client_freq_credit_info, client_freq_own_transfer,
               client_freq_mobile_recharge, client_freq_phone_money_transfer, 
               client_freq_card2card_transfer, client_freq_chat, 
               client_freq_card_recharge, client_freq_invest
              ]


# In[23]:


#%%timeit
#client_freq_main_screen(data.multi_class_target)


# In[42]:


get_ipython().run_cell_magic('time', '', "# 5min 46s\n\nclient_freq = data.groupby('client_pin').multi_class_target.agg(cli_for_agg)\nclient_freq.to_csv(os.path.join(DATA_OWN, 'client_freq.csv'))\nclient_freq.head(10)")


# In[ ]:


#del client_freq
#gc.collect()


# In[ ]:





# #### client frequency of target for day of week

# In[25]:


get_ipython().run_cell_magic('time', '', "# 32min 55s\n\nclient_freq_dow = data.groupby(['client_pin', 'dow']).multi_class_target.agg(cli_for_agg)\n# after read_csv multilevel will disapper\nclient_freq_dow.to_csv(os.path.join(DATA_OWN, 'client_freq_dow.csv'))\nclient_freq_dow.head(10)")


# In[ ]:


#del client_freq_dow
#gc.collect()


# In[ ]:





# #### client frequency of target for times of day

# In[28]:


get_ipython().run_cell_magic('time', '', "# \n\nclient_freq_tod = data.groupby(['client_pin', 'tod']).multi_class_target.agg(cli_for_agg)\n# after read_csv multilevel will disapper\nclient_freq_tod.to_csv(os.path.join(DATA_OWN, 'client_freq_tod.csv'))\nclient_freq_tod.head(10)")


# In[ ]:


#del client_freq_tod
#gc.collect()


# In[ ]:





# ## get difference from client freq target

# In[70]:


get_ipython().run_cell_magic('time', '', "# 11.3 s\n\n#client_diff_freq_dom  = pd.DataFrame(index = client_freq_dom.index)\nclient_diff_freq_dow  = pd.DataFrame(index = client_freq_dow.index)\n#client_diff_freq_hour = pd.DataFrame(index = client_freq_hour.index)\nclient_diff_freq_tod  = pd.DataFrame(index = client_freq_tod.index)\n\nfor el in client_freq.keys():\n#    client_diff_freq_dom['diff_' + el]   = client_freq[el] - client_freq_dom[el]\n    client_diff_freq_dow['diff_' + el]   = client_freq[el] - client_freq_dow[el]\n#    client_diff_freq_hour['diff_' + el]  = client_freq[el] - client_freq_hour[el]\n    client_diff_freq_tod['diff_' + el]   = client_freq[el] - client_freq_tod[el]\n    \n#client_diff_freq_dom.to_csv(os.path.join(DATA_OWN, 'client_diff_freq_dom.csv'))\nclient_diff_freq_dow.to_csv(os.path.join(DATA_OWN, 'client_diff_freq_dow.csv'))\n#client_diff_freq_hour.to_csv(os.path.join(DATA_OWN, 'client_diff_freq_hour.csv'))\nclient_diff_freq_tod.to_csv(os.path.join(DATA_OWN, 'client_diff_freq_tod.csv'))")


# In[71]:


#del client_diff_freq_dom
#del client_diff_freq_hour

#del client_diff_freq_dow
#del client_diff_freq_tod

#gc.collect()


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





# # USED UNTIL THIS LINE

# In[ ]:





# In[ ]:


data[data.client_pin == gg]


# In[ ]:





# # time past after last known target for this user

# In[ ]:


data.time_diff


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
    ret_series = pd.Series([[0]]*inp_data.shape[0]
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
        ret_series.iloc[idx] = np.array(last_known, dtype = object)
        #update last known time:
        last_known[target_dict[field]] = 0
            
    #print(ret_series.shape)
    
    return ret_series


# In[ ]:


#first = data[data.client_pin == gg].groupby('client_pin')
first = data.query('client_pin == @gg1 or client_pin == @gg2 or             client_pin == @gg3 or client_pin == @gg4 or client_pin == @gg5').groupby('client_pin')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#.group['multi_class_target']\n#tt = first.apply(target_time_diff)\n\ntt = data.groupby('client_pin').progress_apply(target_time_diff)")


# In[ ]:


tt = tt.reset_index()


# In[ ]:


tt.set_index('ind', inplace=True)


# In[ ]:


tt.columns=['client_pin', 'last_known']


# In[ ]:


with open(os.path.join(DATA_OWN, 'last_known.pkl'), 'wb') as fd_last_known:
    pickle.dump(tt, fd_last_known, protocol = 2)


# In[ ]:


get_ipython().run_cell_magic('time', '', "with open(os.path.join(DATA_OWN, 'last_known2.pkl'), 'wb') as fd_last_known:\n    pickle.dump(tt, fd_last_known, protocol = 2)")


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




