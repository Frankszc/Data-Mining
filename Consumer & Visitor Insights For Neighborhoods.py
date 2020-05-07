
# coding: utf-8

# # Dataset：“cbg_patterns.csv”

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[3]:


data_df = pd.read_csv('./cbg_patterns.csv', index_col=0)


# ## Data summary

# In[4]:


data_df.head()


# In[5]:


data_df.dtypes


# In[6]:


data_df.shape#数据的大小


# ## missing value：

# In[9]:


print("cbg_patterns缺失值数量：\n",data_df.isnull().sum())#数据集各列（属性）缺失值数量统计


# In[7]:


data_df['related_same_day_brand'].value_counts()


# In[11]:


data_df['related_same_month_brand'].value_counts()


# In[12]:


data_df['top_brands'].value_counts()


# In[19]:


data_df[['raw_visit_count','raw_visitor_count','distance_from_home']].describe()##数值属性的5数概括


# ## Histogram：

# In[16]:


data_df['raw_visit_count'].hist()


# In[17]:


data_df['raw_visitor_count'].hist()


# In[18]:


data_df['distance_from_home'].hist()


# In[20]:


def plot_feature_distribution(df, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(1,5,figsize=(18,6))

    for feature in features:
        i += 1
        plt.subplot(1,5,i)
        sns.kdeplot(df[feature], bw=0.5,label=feature)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


# In[23]:


features = ['date_range_start','date_range_end','raw_visit_count','raw_visitor_count','distance_from_home']
plot_feature_distribution(data_df, features)


# ## Box figure

# In[31]:


data_df['raw_visit_count'].plot(kind='box', notch=True, grid=True)
plt.show()


# In[32]:


data_df['raw_visitor_count'].plot(kind='box', notch=True, grid=True)
plt.show()


# In[33]:


data_df['distance_from_home'].plot(kind='box', notch=True, grid=True)
plt.show()


# ## 数据缺失处理

# In[34]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[35]:


get_ipython().run_cell_magic('time', '', 'missing_data(data_df)')


# ## Directly culled missing values：

# In[28]:


del_df = data_df.dropna()


# In[29]:


del_df.shape


# In[30]:


print("cbg_patterns缺失值数量：\n",del_df.isnull().sum())#数据集各列（属性）缺失值数量统计


# In[38]:


del_df.head()


# ## Fill high frequency value ：

# In[32]:


from collections import Counter
from math import isnan

miss_features = ['raw_visit_count','raw_visitor_count','distance_from_home']
fill_df = data_df

for col in miss_features:
    word_counts = Counter(fill_df[col])
    top = word_counts.most_common(1)[0][0]
    if type(top) != str:
        if isnan(top):
            top = word_counts.most_common(2)[1][0]
    print(top, type(top))
    temp = fill_df[col].fillna(top)
    fill_df[col] = temp
fill_df.head()


# In[33]:


print("cbg_patterns缺失值数量：\n",fill_df.isnull().sum())#数据集各列（属性）缺失值数量统计


# In[41]:


missing_data(fill_df)


# ## Similarity padding between objects：

# In[43]:


data_df.corr()


# In[44]:


data_df.corr('spearman')


# ## The correlation padding between the properties：

# In[ ]:


from fancyimpute import KNN
fill_knn = KNN(k=3).fit_transform(dfc)
dfc_fill_knn = pd.DataFrame(fill_knn)

