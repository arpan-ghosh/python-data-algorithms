
# coding: utf-8

# # Note: I have reviewed this individual's GitHub and implemented similar code in my principal component analysis: https://ocefpaf.github.io/python4oceanographers/blog/2014/12/01/PCA/
# 

# Consider the data set in file food.csv, describing the eating habits of the population England,
# Wales, Scotland and Ireland. Calculate the Principal components and analyze the results.

# In[92]:

# Using pandas library for CSV reading and table manipulation
import pandas
import numpy as np
import matplotlib.pyplot as plt
from  pandas  import  DataFrame
from  sklearn.decomposition  import  PCA


# In[153]:

# Reading food.csv dataset from workspace folder and storing into variable food
#food = pandas.read_csv('data.csv', index_col='Food')
file = pandas.read_csv('/home/idies/workspace/AS.171.205/data/food.csv',header=None)
file.columns=['Food', 'England', 'Wales', 'Scotland', 'N.Ireland']
foodData = file.transpose()
newFD = file.iloc[:,1:].values #remove first row
values = foodData.iloc[1:,:].values # 4 x 17 matrix


# In[275]:

#to check that the new data is what we want
newFD


# In[136]:

# Quick data exploration of food, will print all rows
foodData.head(17)


# In[154]:

# Summary of numerical fields of food
file.describe()


# In[276]:

#checking if the transposed data is what we want
values


# In[284]:

#normalizing the values and using PCA
from  sklearn.decomposition  import  PCA

pca = PCA(n_components=None)
values_norm = normalize(values)
pca.fit(values_norm)


# In[13]:

from pandas import DataFrame
#No transposition needed, unlike the link I referenced above since I 
#already transposed the data above
output = DataFrame(pca.components_)
output.index = ['Principal Component %s' % pc for pc in output.index + 1]
output.columns = ['Time Series %s' % pc for pc in output.columns + 1]
output


# In[301]:

#newValus = float(output.values)
PCs = np.dot(output.values, newFD)


# In[303]:

marker = dict(linestyle='none', marker='D', markersize=7, color='black', alpha=5)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(PCs[0], np.zeros_like(PCs[0]),
        label="Scores", **marker)
[ax.text(x, y, t) for x, y, t in zip(PCs[0], output.values[1, :], file.ix[:,1:])]
plt.grid(True)
ax.set_title("Principal Component Projection")
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Projection on Vector")

_ = ax.set_ylim(-.5, .5)
marker = dict(linestyle='None', marker='D', markersize= 5, alpha=2)


# In[217]:

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(PCs[0], PCs[1], label="Scores", **marker)
plt.grid(True)

ax.set_title("Principal Component Against The Other")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

text = [ax.text(x, y, t) for x, y, t in
        zip(PCs[0], PCs[1]+0.5, file.ix[:,1:])]


# In[216]:

percent = pca.explained_variance_ratio_ * 100

percent = DataFrame(percent, columns=['Variance Ratio %'], index=['PC %s' % pc for pc in np.arange(len(percent)) + 1])
graph = percent.plot(kind='barh')
plt.grid(True)


# In[274]:

series1 = output['Time Series 1']
series1.index = output.index
graph = series1.plot(kind='bar')
plt.title('Time Series 1 per Food')
plt.grid(True)
graph.set_xticks([])


# In[299]:

marker = dict(linestyle='none', marker='o', markersize=7, color='blue', alpha=0.5)
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(output.iloc[:,0], output.iloc[:,1], label="Output", **marker)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.axis([-1, 1, -1, 1])
plt.grid(True)

text = [ax.text(x, y, t) for
        x, y, t in zip(output.iloc[:,0], output.iloc[:,1], file.index)]









