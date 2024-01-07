#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
######## Take you mpg data and load it over here - 
data = pd.read_csv('E:/Python docs/bangalore house price prediction OHE-data.csv')
print(data.head(2))


# In[7]:


##### KNN Regression
y = data['price']
X = data[['bath','balcony','total_sqft_int', 'bhk', 'price_per_sqft', 'area_typeSuper built-up  Area', 'area_typeBuilt-up  Area', 'area_typePlot  Area', 'availability_Ready To Move']]


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=123)


# In[10]:


y_train.shape


# In[11]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


# In[12]:


model = KNeighborsRegressor(n_neighbors=2) ###### default
print("The model is loaded")


# In[13]:


###### Training the Model
model_fitting = model.fit(X_train, y_train)
print('Model Training is completed')


# In[14]:


####### Getting the Training Score
round(model_fitting.score(X_train, y_train),2)


# In[15]:


###### Prediction  - Testing Data
pred = model_fitting.predict(X_test)
results = r2_score(y_test,pred)
print(results)  #### 0 to 1


# In[16]:


##### How to choose the value of k 
# 1. sqrt(n)
# 2. Error method  - Which is used


# In[17]:


data.shape


# In[18]:


import math
math.sqrt(318)  ###### Will this be correct


# In[19]:


##### Error Method -  k =  (1,21) It will give me the error present in those k
error = []
k = []
for i in range(1,10,2): #### K values  #### stepover
    model = KNeighborsRegressor(n_neighbors=i)
    model_fit = model.fit(X_train,y_train)
    err = 1 - round(model_fit.score(X_train,y_train),2)
    error.append(err)
    k.append(i)


# In[20]:


pd.DataFrame({'K':k, 'error':error})


# In[21]:


##### How to Save the Model ---
import joblib


# In[22]:


joblib.dump(model,'KNN_Reg_model.sav')


# In[26]:


X_test['Actual'] = y_test
X_test['Pred'] = pred


# In[27]:


X_test


# In[ ]:




