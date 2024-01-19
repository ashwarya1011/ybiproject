#!/usr/bin/env python
# coding: utf-8

# # Women Cloth Review Prediction with Multinomial Naive Bayes
# 
# 

# ##### The multinomial Naive Bayes classifier is suitable for classification with discrete features(e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts.However, in practice, fractional counts such as tf-idf may also work.

# # Import Library

# In[1]:


import pandas as pd


# In[2]:


import numpy as np 


# In[3]:


import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns


# # Import Dataset

# In[4]:


df=pd.read_csv('https://raw.githubusercontent.com/YBIFoundation/ProjectHub-MachineLearning/main/Women%20Clothing%20E-Commerce%20Review.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[8]:


df.shape


# # Missing Values

# #### Remove missing values in Reviews columns with No Review Text

# In[9]:


df.isna().sum()


# In[10]:


df[df['Review']==""]=np.NaN


# In[11]:


df['Review'].fillna("No Review",inplace=True)


# In[13]:


df.isna().sum()


# In[14]:


df['Review']


# # Define Target(y) and Feature(X)

# In[15]:


df.columns


# In[16]:


X=df['Review']


# In[17]:


y=df['Rating']


# In[18]:


df['Rating'].value_counts()


# # Train Test Split

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,stratify=y ,random_state = 2529)


# In[21]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Get Feature Text Conversion to Tokens

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer


# In[27]:


cv= CountVectorizer(lowercase= True, analyzer='word', ngram_range=(2, 3), stop_words= 'english', max_features= 5000)


# In[28]:


X_train = cv.fit_transform(X_train)


# In[30]:


cv.get_feature_names_out()


# In[31]:


X_train.toarray()


# In[78]:


X_test=cv.fit_transform(X_test)


# In[37]:


cv.get_feature_names_out()


# In[38]:


X_test.toarray()


# # Get Model Train

# In[41]:


from sklearn.naive_bayes import MultinomialNB


# In[42]:


model= MultinomialNB()


# In[43]:


model.fit(X_train,y_train)


# # Get Model Prediction

# In[44]:


y_pred=model.predict(X_test)


# In[46]:


y_pred.shape


# In[47]:


y_pred


# # Get Probability of Each Predicted Class

# In[49]:


model.predict_proba(X_test)


# # Get Model Evaluation

# In[51]:


from sklearn.metrics import confusion_matrix, classification_report


# In[52]:


print(confusion_matrix(y_test, y_pred))


# In[53]:


print(classification_report(y_test, y_pred))


# # Recategories Ratings as Poor (0) and Good(1)

# In[56]:


df['Rating'].value_counts()


# #### Re-Rating as 1,2,3 as 0 and 4,5 as 1

# In[58]:


df.replace({'Rating':{1:0, 2:0, 3:0, 4:1, 5:1}},inplace=True)


# In[59]:


y=df['Rating']


# In[60]:


X=df['Review']


# # Train Test Split

# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,stratify=y ,random_state = 2529)


# In[63]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Get Feature Text Conversion to Tokens

# In[64]:


from sklearn.feature_extraction.text import CountVectorizer


# In[65]:


cv= CountVectorizer(lowercase= True, analyzer='word', ngram_range=(2, 3), stop_words= 'english', max_features= 5000)


# In[66]:


X_train = cv.fit_transform(X_train)


# In[67]:


X_test=cv.fit_transform(X_test)


# # Get Model Re-Train

# In[72]:


from sklearn.naive_bayes import MultinomialNB


# In[73]:


model= MultinomialNB()


# In[74]:


model.fit(X_train,y_train)


# # Get Model Prediction

# In[69]:


y_pred=model.predict(X_test)


# In[70]:


y_pred.shape


# In[71]:


y_pred


# # Get Model Evaluation

# In[75]:


from sklearn.metrics import confusion_matrix, classification_report


# In[76]:


print(confusion_matrix(y_test, y_pred))


# In[77]:


print(classification_report(y_test, y_pred))


# In[ ]:




