#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv("Walmart_Store_sales.csv")


# In[4]:


df.head(10)


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.dtypes


# ## Which store has maximum sale?

# In[13]:


retail_data=df


# In[14]:


print(df[df.Weekly_Sales == df.Weekly_Sales.max()])


# ## Which store has maximum standard deviation i.e., the sales vary a lot. Also, find out the coefficient of mean to standard deviation

# In[15]:


maxstd=pd.DataFrame(df.groupby('Store').agg({'Weekly_Sales':['std','mean']}))
#Just resetting the index.
maxstd = maxstd.reset_index()
# coefficient of Variance = mean / standard Deviation
maxstd['COV'] = (maxstd[('Weekly_Sales','std')]/maxstd[('Weekly_Sales','mean')]) *100
maxstd.loc[maxstd[('Weekly_Sales','std')]==maxstd[('Weekly_Sales','std')].max()]


# ## Which store has a good quarterly growth rate?

# In[16]:


# create a new column which shows the year and quarter
df['quarter'] = pd.PeriodIndex(df.Date, freq='Q')
T2012Q2 = df.loc[df['quarter'] == "2012Q2", ["Weekly_Sales", 'Store']]
T2012Q3 = df.loc[df['quarter'] == "2012Q3", ["Weekly_Sales", 'Store']]
T2012Q2_sum_per_store = pd.DataFrame(T2012Q2.groupby('Store')['Weekly_Sales'].sum())
T2012Q2_sum_per_store.reset_index(inplace=True)
T2012Q3_sum_per_store = pd.DataFrame(T2012Q3.groupby('Store')['Weekly_Sales'].sum())
T2012Q3_sum_per_store.reset_index(inplace=True)
T2012Q2_sum_per_store['Weekly_Sales_Q3'] = T2012Q3_sum_per_store['Weekly_Sales']
T2012Q2_sum_per_store['Growth Rate'] = ((T2012Q2_sum_per_store.Weekly_Sales_Q3 - T2012Q2_sum_per_store.Weekly_Sales)/T2012Q2_sum_per_store.Weekly_Sales)*100
T2012Q2_sum_per_store.loc[T2012Q2_sum_per_store['Growth Rate']==T2012Q2_sum_per_store['Growth Rate'].max()]


# ## Find out the holiday that has the higher sales than the mean sales in non-holiday season ll together

# In[20]:


import datetime as dt
retail_data.Date = pd.to_datetime(retail_data.Date)


# In[21]:


Holiday_Week = retail_data.loc[retail_data['Holiday_Flag']==1]


# In[22]:


from matplotlib import pyplot as plt
import time
retail_data['Date'] = pd.to_datetime(retail_data.Date)


# In[23]:


def plot_line(df, holiday_dates, holiday_label):
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(df['Date'], df['Weekly_Sales'], label=holiday_label)
    
    for day in holiday_dates:
        day = dt.datetime.strptime(day, '%d-%m-%Y').date()
        plt.axvline(x=day, linestyle='--', c='r')
        
    plt.title(holiday_label)
    x_dates = df['Date'].dt.strftime('%Y-%m-%d').sort_values().unique()
    plt.show()


# In[24]:


total_sales = retail_data.groupby('Date')['Weekly_Sales'].sum().reset_index()
Super_Bowl=['12-2-2010', '11-2-2011', '10-2-2012']
Labor_Day = ['10-9-2010', '9-9-2011', '7-9-2012']
Thanksgiving = ['26-11-2010', '25-11-2011', '23-11-2012']
Christmas = ['31-12-2010', '30-12-2011', '28-12-2012']


# In[25]:


plot_line(total_sales, Super_Bowl, 'Super Bowl')


# In[26]:


plot_line(total_sales, Labor_Day, 'Labour Day')


# In[27]:


plot_line(total_sales, Thanksgiving, 'Thanksgiving Week')


# In[28]:


plot_line(total_sales, Christmas, 'Christmas')


# In[29]:


from matplotlib import pyplot as plt
pd.DatetimeIndex(retail_data['Date'])
monthly = retail_data.groupby(pd.Grouper(key='Date', freq='1M')).sum()# groupby each 1 month
monthly=monthly.reset_index()
fig, ax = plt.subplots(figsize=(10,6))
X = monthly['Date']
Y = monthly['Weekly_Sales']
plt.plot(X,Y)
plt.title('Month Wise Sales')
plt.xlabel('Months')
plt.ylabel('Weekly_Sales')


# In[30]:


quarterly = retail_data.groupby(pd.Grouper(key='Date', freq='3M')).sum()# groupby each 3e month
quarterly=quarterly.reset_index()
fig, ax = plt.subplots(figsize=(10,6))
X = quarterly['Date']
Y = quarterly['Weekly_Sales']
plt.plot(X,Y)
plt.title('Quarterly Sales')
plt.xlabel('Months')
plt.ylabel('Weekly_Sales')


# In[31]:


semesterly = retail_data.groupby(pd.Grouper(key='Date', freq='3M')).sum()# groupby each 3e month
semesterly=semesterly.reset_index()
fig, ax = plt.subplots(figsize=(10,6))
X = semesterly['Date']
Y = semesterly['Weekly_Sales']
plt.plot(X,Y)
plt.title('Semester Sales')
plt.xlabel('Years')
plt.ylabel('Weekly_Sales')


# In[32]:


# A little feature engineering
import seaborn as sns
corr = retail_data.corr()

plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True)
plt.show()


# In[33]:


retail_data['Day']=retail_data['Date'].dt.day
retail_data = retail_data.drop('Date',axis=1)


# In[34]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, r2_score


# In[35]:


df_v2 = pd.get_dummies(retail_data, columns = ['Holiday_Flag','Store'])
y = df_v2['Weekly_Sales']
X= df_v2.drop(['Weekly_Sales', 'quarter'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[36]:


ln_model = LinearRegression()
ln_model.fit(X_train, y_train)


# In[37]:


y_pred = ln_model.predict(X_test)
print("r2 score:", r2_score(y_test,y_pred))


# ### Submitted by SAMBIT MAHANTA

# In[ ]:




