#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


# In[4]:


np.random.seed(0)


# In[8]:


iris=load_iris() #creating objects called iris with iris data
df=pd.DataFrame(iris.data,columns=iris.feature_names) #creating a dataframe


# In[9]:


df.head()


# In[10]:


print(df)


# In[11]:


print(iris)


# In[32]:


df['Species']=pd.Categorical.from_codes(iris.target,iris.target_names) #adding a new column for the species name
df.tail(10)


# In[35]:


df=df.drop('Specis',axis=1)


# In[36]:


df=df.drop('is_train',axis=1)


# In[37]:


df.head()


# In[38]:


#creating test and train data
df['is_train']=np.random.uniform(0,1,len(df))<=0.75
df.head()


# In[39]:


train,test=df[df['is_train']==True],df[df['is_train']==False]
print('Number of observation in training_data',len(train))
print('Number of observation in testing_data',len(test))


# In[22]:


from sklearn.model_selection import train_test_split
X=df.iloc[:,0:4]
y=df.iloc[:,5]


# In[23]:


X.head()


# In[25]:


X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[26]:


X_train.shape


# In[28]:


X_test.shape


# In[30]:


#create a list of the features columns name
features=df.columns[:4]
features


# In[41]:


#converting each species name into integers
y=pd.factorize(train['Species'])[0]
y


# In[43]:


#creating a random forest classifier 
clf=RandomForestClassifier(n_jobs=2,random_state=0)
clf.fit(train[features],y)


# class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

# In[60]:


output=clf.predict(test[features])
print(output)


# In[45]:


#viewing the predicted probabilities of the first 10 obervations
clf.predict_proba(test[features])[0:10]


# In[47]:


#viewing the predicted probabilities of the next 10 obervations
clf.predict_proba(test[features])[10:20]


# In[48]:


#mapping names of the plants for each predicted plant class
pred=iris.target_names[clf.predict(test[features])]


# In[49]:


pred[0:10] #view predicted species for 10 observations


# In[51]:


pred[10:20]  #view predicted species for next 10 observations


# In[54]:


test['Species'].head(10)


# In[56]:


#creating confusion matrix
pd.crosstab(test['Species'],pred,rownames=['Actual species'],colnames=['Predicted Species'])


# In[ ]:




