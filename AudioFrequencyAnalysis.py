# The files noise01.csv to noise10.csv contain a random noise from a real
# instrument, measuring the intensity of light as a function of the voltage on a light
# source. The voltage goes from 0.1V to 1.0V, encoded in the filename. (0.1V, 0.2V, 0.3V,
# 0.4V, 0.5V, 1.0V). 

# Here I prove that the noise is due to the Poisson distribution of the discrete
# photons 

# In[321]:

import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.optimize import curve_fit
from scipy.misc import factorial
from __future__ import division      
import scipy.stats as stats     # for pdfs 


colnames = ['x']


noise1 = pandas.read_csv('/home/idies/workspace/AS.171.205/data/noise_01.csv', names=colnames, header=None)
noise2 = pandas.read_csv('/home/idies/workspace/AS.171.205/data/noise_02.csv', names=colnames, header=None)
noise3 = pandas.read_csv('/home/idies/workspace/AS.171.205/data/noise_03.csv', names=colnames, header=None)
noise4 = pandas.read_csv('/home/idies/workspace/AS.171.205/data/noise_04.csv', names=colnames, header=None)
noise5 = pandas.read_csv('/home/idies/workspace/AS.171.205/data/noise_05.csv', names=colnames, header=None)
noise10 = pandas.read_csv('/home/idies/workspace/AS.171.205/data/noise_10.csv', names=colnames, header=None)


# In[459]:

data = np.random.poisson(100,100000)
la = plt.hist(data, bins=100, normed=True)
print ('                                                Spread of A Random Ideal Poisson')


# In[450]:

noise1Hist = plt.hist(noise1.x, bins=100, normed=True)
print ('                                                            Noise 1')
print("Noise 1 Mean: ",noise1.x.mean())
print("Noise 1 Variance: ",noise1.x.std()*noise1.x.std())
noise1.x.var()


# In[441]:

noise2Hist = plt.hist(noise2.x, bins=100, normed=True)
print ('                                                            Noise 2')


# In[442]:

noise3Hist = plt.hist(noise3.x, bins=100, normed=True)
print ('                                                            Noise 3')


# In[443]:

noise4Hist = plt.hist(noise4.x, bins=100, normed=True)
print ('                                                            Noise 4')


# In[444]:

noise5Hist = plt.hist(noise5.x, bins=100, normed=True)
print ('                                                            Noise 5')


# In[445]:

noise10Hist = plt.hist(noise10.x, bins=100, normed=True)
print ('                                                            Noise 10')


# In[463]:

#All graphs superimposed on one
noise1Hist = plt.hist(noise1.x, bins=100, normed=True)
noise2Hist = plt.hist(noise2.x, bins=100, normed=True)
noise3Hist = plt.hist(noise3.x, bins=100, normed=True)
noise4Hist = plt.hist(noise4.x, bins=100, normed=True)
noise5Hist = plt.hist(noise5.x, bins=100, normed=True)
noise10ist = plt.hist(noise10.x, bins=100, normed=True)


# ## Problem 4
# Take a Cauchy distribution, 1/(1+x2). Show that the distribution of the sum of two
# Cauchy variables is still a Cauchy. Do this numerically as an iPython notebook, through
# generating a few hundred random Cauchy variables

# In[426]:

s = np.random.standard_cauchy(600)
y = np.random.standard_cauchy(600)
t = np.random.standard_cauchy(600)
#Comparing the two separate random cauchy stored in variables, looks very, very, very similar.
#Looks similar when after adding and also when creating cauchy values of double of s and y, so from 300 to 600


#Loop  to show each different cauchy value added together, then sum it back to its original 
for num in range(600):
     s += np.random.standard_cauchy(600)


print()        
        
print('                                           Proof: Random Cauchy Still Cauchy')
#s = s[(s>-25) & (y<25)]
plt.hist(s, bins=100)
plt.show()

print('                                           Random Cauchy of Just "S"')
y = y[(y>-25) & (y<25)]  # just showing part of the distribution for plotting purposes
plt.hist(y, bins=100)
plt.show()

print('                                           Still Cauchy "S+Y=Z"')
z = z[(z>-25) & (z<25)]  # just showing part of the distribution for plotting purposes
plt.hist(z, bins=100)
plt.show()


# A die is rolled 24 times. Use the Central limit theorem to estimate the probability that
# a. The sum is greater than 84
# b. The sum is equal to 84
# c. Perform a hundred numerical realizations to illustrate the result

# 
# 

# In[403]:

import pylab
import random

size = 24

counter84 = 0
count84more = 0

## One die
singleDie = []
sumDie = []
sumOf24 = []
for num in range(100):
    total = 0
    for i in range(size):
        newValue = random.randint(1,6)
        total += newValue
    sumOf24.append(total)
    if (total > 84): count84more += 1
    if (total == 84): counter84 += 1

print ("The Expected Value of One Die Thrown using Central Limit Theorem is 3.5, variance is 35/12")
print ("The mean/average of 24 throws results in an expected value of 84 and variance of 35/12 * 24 = 70")
print ()

print ("This is the Probability Distribution Function of Summing to 84: ", stats.norm(84, sqrt(70)).pdf(84))
print ("This is the Cumulative Distribution Function of Getting above 84: ", stats.norm(84, sqrt(70)).cdf(84))
print ()

print ("Actual Simulation-Num times Total is 84: ", counter84/100)
print ("Actual Simulation-Num times Total Greater 84: ", count84more/100)
print ()

print ("Actual Results for throwing One Die", size, "times, Total 100 Simulations of 24 Throws:")
print ("Actual Mean sample =", pylab.mean(sumOf24))
print ("Actual Median of the sample =", pylab.median(sumOf24))
print ("Actual Standard deviation =", pylab.std(sumOf24))
print ("Actual Simulation-Total sum: ", total)
print
print

pylab.hist(sumOf24, bins=15)
pylab.xlabel('Sum of 24 Throws')
pylab.ylabel('Frequency of Sum')
pylab.show()


# In[ ]:



