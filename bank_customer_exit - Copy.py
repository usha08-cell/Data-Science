#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd


# In[22]:


data=pd.read_csv(r"C:\Users\POOJA.K\Documents\dsp(assignment)\bank\Churn_Modelling.csv")
data


# In[23]:


data.isna().sum()


# In[24]:


x=data.iloc[:,3:13].values
y=data.iloc[:,13].values


# In[25]:


from sklearn.preprocessing import LabelEncoder
LabelEncoder_x1=LabelEncoder()
x[:,1]=LabelEncoder_x1.fit_transform(x[:,1])
x[:,2]=LabelEncoder_x1.fit_transform(x[:,2])


# In[26]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7, random_state=0)


# In[18]:


#from sklearn.tree import DecisionTreeClassifier
#tree_model=DecisionTreeClassifier(criterion="entropy")
#tree_model.fit(x_train,y_train)
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=50, metric='minkowski', p=2)  
classifier.fit(x_train, y_train) 


# In[19]:


y_pred= classifier.predict(x_test) 
y_pred


# In[20]:


from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[ ]:




