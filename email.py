#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("spam.csv",encoding = "ISO-8859-1")
df.head()

df.isnull().sum()
# In[16]:


df1 = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1)


# In[17]:


df1.head()

df1.rename(columns = {"v1" : "Spam or Ham", "v2":"Mail Message"},inplace = True)
df1.head()
# In[19]:


df1.shape


# In[20]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df1['Spam or Ham'] = encoder.fit_transform(df1['Spam or Ham'])


# In[21]:


df1.duplicated().sum()


# In[22]:


df1 = df1.drop_duplicates(keep = 'first')


# In[23]:


df1.duplicated().sum()


# In[26]:


X =  df1["Mail Message"]
y =df1["Spam or Ham"]


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_count = cv.fit_transform(X_train.values)
X_train_count.toarray()


# In[29]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)


# In[30]:


from sklearn.metrics import confusion_matrix , recall_score , precision_score
from sklearn.metrics import accuracy_score


# In[31]:


mail_ham = ['Same. Wana plan a trip sometme then']
mail_ham_count = cv.transform(mail_ham)
y_pred = model.predict(mail_ham_count)
y_pred


# In[32]:


model.score(X_train_count,y_train)


# In[33]:


X_test_count = cv.transform(X_test)
model.score(X_test_count,y_test)


# In[ ]:




