
# coding: utf-8


# ##  Create the adjacency matrix A for the graph X.

# In[1]:

import networkx as nx
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import numpy as np
import scipy as sp
get_ipython().magic('matplotlib inline')


# In[2]:

G=nx.Graph()#  G is an empty Graph


# In[3]:

Nodes=range(11)
G.add_nodes_from(Nodes)
G.remove_node(0)
Edges=[(1,2), (1,4), (1,8), (1,9), (2,6), (2,8), (3,4), (3,6), (4,5), (5,6), (5,9), (5,10), (6,7), (6,9), (7,10), (8,9), (8,11), (9,10), (10,11)]
G.add_edges_from(Edges)


# In[4]:

nx.draw(G, node_color='c',edge_color='k', with_labels=True)


# In[5]:

numpyMatrix = nx.to_numpy_matrix(G)
A = nx.adjacency_matrix(G)
print("The adjacency matrix A for the above graph X is: ")
print()
print(A.todense())


# ## Using the adjacency matrix, calculate the number of closed triangles on X
Let a = the adjacency matrix

The adjacencies in a*a (a2) matrix multiplied are the numbers of 2-length paths
The adjacencies in a2*a matrix multiplied are the numbers of 3-length paths
# In[6]:

A = nx.to_numpy_matrix(G)
C = np.dot(A,A)
D = np.dot(A,C)
# print(nx.triangles(G))
print("The total number of triangles are: ")
#np.trace finds the sum along its diagonal, divide by 6 as # of nodes
# needed to be checked for a triangles to be a closed figure
# (3*2 eliminate redundencies)
np.trace(D) / 6  


# ## Calculate the Laplacian matrix of X, and calculate its eigenvalues and eigenvectors. 

# In[7]:

#The graph Laplacian is the matrix L = D - A,
# where A is the adjacency matrix and D is the
# diagonal matrix of node degrees.
#networkx has a laplacian matrix intenral function 
#that does L=D-A for us
laplacian = nx.laplacian_matrix(G)
print(laplacian.todense())


# In[8]:

from scipy.sparse import csgraph
lap= csgraph.laplacian(numpyMatrix, normed=False)
lap


# In[9]:

eigvectors = np.linalg.eig(lap)
eig_vals, eig_vecs = np.linalg.eig(lap)
print("The eigenvalues of the laplacian of the graph X above is: ")
#two smallest -1.44136522e-15,   1.16842658e+00,
eig_vals


# In[10]:

print("The eigenvectors of the laplacian of the graph X above is: ")
eig_vecs


# ## Problem 4: Use the eigenvectors corresponding to the two lowest eigenvalues (different from 0) as the x,y coordinates to plot the graph.

# In[11]:

#Just to see which are the smallest values
eig_vals_sorted = np.sort(eig_vals)


# In[12]:

print("The sorted eigenvalues of the laplacian of the graph X above is: ")
eig_vals_sorted


# In[29]:

fig, ax = plt.subplots()
x = np.array(eig_vecs[2])
#x = x[0]
y = np.array(eig_vecs[9])
#y = y[0]

n = [1,2,3,4,5,6,7,8,9,10,11]
ax.scatter(x,y)
ax.plot(x,y)

for i, txt in enumerate(n):
    ax.annotate(txt,(eig_vecs[2][i],eig_vecs[9][i]))
    ax.annotate(i+1,(x[i],y[i]))

AArray = np.array(A)

for i in range(0,11):
    for j in range(1+1, 11):
        if (AArray[i][j] == 1):
            
            P1 = [eig_vecs[2][i], eig_vecs[9][j]]
            P2 = [eig_vecs[2][i], eig_vecs[9][j]]
 
            P3 = [x[i], y[j]]
            P4 = [x[i], y[j]]
        
            plt.plot(P3,P4)
            plt.plot(P1,P2)


# In[14]:

print("The eigenvector of the smallest eigenvalue plotted on the x-axis are:", x)


# In[15]:

print("The eigenvector of the second smallest eigenvalue plotted on the x-axis are:", y)

