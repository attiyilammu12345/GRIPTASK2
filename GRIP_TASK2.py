#!/usr/bin/env python
# coding: utf-8

# # <font color=blue>GRIP-THE SPARKS FOUNDATION</font>
# 

# # <font color=red>Data Science and Bussiness Analytics</font>

# # <font color=green>Task 2:Prediction using unsupervised ML</font>

# # <font color=dark violet>Performed by: ATTIYIL AMMU PRASHANTH</font>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("D:/Iris.csv")
data.head()


# In[3]:


import seaborn as sns
sns.pairplot(data,hue='Species')


# In[4]:


p = data.iloc[:,[0,1,2,3,4]].values


# In[5]:


from sklearn.cluster import KMeans


# In[6]:


w = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300, n_init=10,random_state=0)
    kmeans.fit(p)
    w.append(kmeans.inertia_)


# In[7]:


plt.plot(range(1,11),w)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squares')
plt.show()


# In[8]:


kmeans = KMeans(n_clusters = 3,init = 'k-means++',max_iter = 300, n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(p)
y_kmeans


# In[9]:


plt.scatter(p[y_kmeans==0,0], p[y_kmeans==0,1],s = 100,c = 'red', label = 'Iris-setosa')
plt.scatter(p[y_kmeans==1,0], p[y_kmeans==1,1],s = 100,c = 'blue', label = 'Iris-versicolour')
plt.scatter(p[y_kmeans==2,0], p[y_kmeans==2,1],s = 100,c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroid')
plt.title("Cluster of Species")
plt.legend()
plt.show()


# In[10]:


KModel = kmeans.fit(p)
KModel


# In[11]:


KModel.labels_


# In[12]:


KModel.cluster_centers_


# In[ ]:




