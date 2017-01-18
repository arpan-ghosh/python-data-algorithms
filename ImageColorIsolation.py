
# coding: utf-8


# In[ ]:

#import statements for all

import pandas
import csv
import numpy
import scipy
from skimage.transform import (hough_line, hough_line_peaks)
from skimage.filters import canny
from skimage import data
from scipy import misc
from numpy import genfromtxt
from IPython.display import Image
from matplotlib import pyplot as plt
from matplotlib.mlab import PCA
from sklearn.decomposition import PCA


# ## Problem 1

# In[ ]:

#nir
nir = pandas.read_csv('/home/idies/workspace/nir.csv',header=None)
nir_arr = numpy.array(nir)

#red
red = pandas.read_csv('/home/idies/workspace/red.csv',header=None)
red_arr = numpy.array(red)

#grn
grn = pandas.read_csv('/home/idies/workspace/grn.csv',header=None)
grn_arr = numpy.array(grn)

scipy.misc.imsave('nir.jpg', nir_arr)
scipy.misc.imsave('red.jpg', red_arr)
scipy.misc.imsave('grn.jpg', grn_arr)

## To view images: 
#Image("/home/idies/workspace/grn.jpg")


# In[ ]:

rows = len(nir_arr[0])
columns = len(nir_arr[0])


# In[ ]:

totalnir = numpy.reshape(nir_arr, (rows*columns,1)).astype(numpy.float)
totalred = numpy.reshape(red_arr, (rows*columns,1)).astype(numpy.float)
totalgrn = numpy.reshape(grn_arr, (rows*columns,1)).astype(numpy.float)

nirMatrix = numpy.matrix(totalnir)
redMatrix = numpy.matrix(totalred)
grnMatrix = numpy.matrix(totalgrn)


# In[ ]:

matrix = numpy.hstack((nirMatrix,redMatrix, grnMatrix))


# In[ ]:

matrix


# In[ ]:

pca = PCA()
pca.fit(matrix)
transform = pca.transform(matrix)


# In[ ]:

transform


# In[ ]:

#Nir
pca1 = transform[:,0]
zeroNir = pca1 < -.14
oneNir = pca1 > -.13
pca1[zeroNir] = -2000
pca1[oneNir] = 0

#Red (Green Parks)
pca2 = transform[:,1]
oneRed = pca2 < .11
zeroRed = pca2 > .15
pca2[oneRed] = 0
pca2[zeroRed] = 1

#Grn
pca3 = transform[:,2]
zeroGrn = pca3 < .000000006
oneGrn = pca3 > .0000000001
pca3[zeroGrn] = 1
pca3[oneGrn] = 0

#Red (River)
transformRed2 = transform[:,1]
oneRed2 = transformRed2 < .0006
zeroRed2 = transformRed2 > .15
transformRed2[oneRed2] = 1
transformRed2[zeroRed2] = 0


# In[ ]:

reshapePCA1 = numpy.reshape(pca1,(512,512))
reshapePCA2 = numpy.reshape(pca2,(512,512))
reshapePCA3 = numpy.reshape(pca3,(512,512))
reshapeRed2 = numpy.reshape(transformRed2,(512,512))


# In[ ]:

GreenPark = plt.imshow(reshapePCA2,cmap='gray')
plt.savefig('Green_Park.jpg')  
GreenPark
print("Green areas corresponding to parks")


# In[ ]:

river = plt.imshow(reshapePCA1, cmap='gray')
plt.savefig('River.jpg')  
river
print("The river Seine")


# ## Problem 2

# In[ ]:

hough = pandas.read_csv('/home/idies/workspace/hough.csv',header=None)
hough = numpy.array(hough)

x = hough[:,0]
y = hough[:,1]

x = (x * 100).astype(int)
y = (y * 100).astype(int)

plot = numpy.zeros((100, 100))
for i in range(len(x)):
    plot[x[i], y[i]] = 100
h, theta, d = hough_line(plot)

plt.imshow(plot)
rows, cols = plot.shape

for _, angle, distance in zip(*hough_line_peaks(h, theta, d, num_peaks = 3)):
    first = ((distance - 0 * numpy.cos(angle)) / numpy.sin(angle))
    second = ((distance - cols * numpy.cos(angle)) / numpy.sin(angle))
    plt.plot((0, cols), (first, second), '-o')
plt.axis((0, cols, rows, 0))
plt.grid(True)
plt.title('Three lines from Hough Transform')

