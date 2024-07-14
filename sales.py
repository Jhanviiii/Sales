#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[43]:


df = pd.read_csv(r"C:\Users\jhanv\Downloads\advertising.csv")


# In[44]:


df.info()


# In[45]:


df.describe()


# In[46]:


df.head()


# In[47]:


df.tail()


# In[48]:


df.shape


# In[49]:


df.isnull().sum()


# In[50]:


total_budgets = df[['TV', 'Radio', 'Newspaper']].sum()
print("Total advertising budgets:")
print(f"TV: {total_budgets['TV']}")
print(f"Radio: {total_budgets['Radio']}")
print(f"Newspaper: {total_budgets['Newspaper']}")


# In[51]:


sns.histplot(df['TV'],bins=10)


# In[52]:


labels = ['TV', 'Radio', 'Newspaper']
sizes = [29408.5, 4652.8, 6110.8]
plt.figure(figsize=(3,5))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Advertising Budget Proportions')
plt.show()


# In[53]:


df['TV'].plot.hist(bins=10, color="turquoise", xlabel="TV")


# In[54]:


df['Radio'].plot.hist(bins=10, color="teal", xlabel="Radio")
plt.figure(figsize=(2,2))


# In[55]:


df['Newspaper'].plot.hist(bins=10,color="beige", xlabel="newspaper")
plt.figure(figsize=(3,3))


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size = 0.3,random_state=0)


# In[57]:


print(X_train)


# In[58]:


print(y_train)


# In[59]:


print(X_test)


# In[60]:


print(y_test)


# In[61]:


from sklearn.linear_model import LogisticRegression
model = LinearRegression()
plt.figure(figsize=(1,1))
model.fit(X_train,y_train)


# In[62]:


res= model.predict(X_test)
print(res)


# In[ ]:




