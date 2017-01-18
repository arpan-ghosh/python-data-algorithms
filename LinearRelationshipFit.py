# Fit a linear relationship to the data in the files *fit.csv. Use an iPython notebook for
# the fitting (bfit.csv, cfit.csv, dfit.csv, efit.csv).

# In[52]:

# Using pandas library for CSV reading and table manipulation
import pandas
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import datasets
from sklearn import linear_model

# Reading *fit.csv dataset from workspace folder and storing into variable of same name
colnames = ['x','y']
bfit = pandas.read_csv('/home/idies/workspace/AS.171.205/data/bfit.csv', names=colnames, header=None)
cfit = pandas.read_csv('/home/idies/workspace/AS.171.205/data/cfit.csv', names=colnames, header=None)
dfit = pandas.read_csv('/home/idies/workspace/AS.171.205/data/dfit.csv', names=colnames, header=None)
efit = pandas.read_csv('/home/idies/workspace/AS.171.205/data/efit.csv', names=colnames, header=None)


# In[126]:

#I will print the four *fit graphs in scatter form all together. They are labeled accordingly

plt.figure(figsize=(12,8))
fig, axs = plt.subplots(1, 4, sharey=True)
bfit.plot(kind='scatter', x='x', y='y', ax=axs[0], figsize=(16, 8))
cfit.plot(kind='scatter', x='x', y='y', ax=axs[1], figsize=(16, 8))
dfit.plot(kind='scatter', x='x', y='y', ax=axs[2], figsize=(16, 8))
efit.plot(kind='scatter', x='x', y='y', ax=axs[3], figsize=(16, 8))

axs[0].set_title("bfit.csv Scatter")
axs[1].set_title("cfit.csv Scatter")
axs[2].set_title("dfit.csv Scatter")
axs[3].set_title("efit.csv Scatter")


plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# In[136]:

from numpy import *
from scipy.interpolate import *
from matplotlib.pyplot import *
from sklearn import datasets
from sklearn import linear_model

print ('Bfit.csv')
plot(bfit.x,bfit.y,'o')
plot(bfit.x,polyval(p1,bfit.x), 'r-')
plot(bfit.x,polyval(p2,bfit.x), 'b--')
plot(bfit.x,polyval(p3,bfit.x), 'm:')

#The following will give me the y=mx+b format, and print the slope and intercept
p1 = polyfit(bfit.x, bfit.y, 1)
print('The following will give me the y=mx+b format, and print the slope and intercept')
print(p1)
print ()

#Quadratic
p2 = polyfit(bfit.x, bfit.y, 2)
print('The following will give me the quadratic and print its coefficients')
print (p2)
print ()

#Polynomial
p3 = polyfit(bfit.x, bfit.y, 3)
print('The following will give me the polynomial and print its coefficients')
print (p3)

#Actual yfit
yfit = p1[0] * bfit.x + p1[1]
yresid = bfit.y - yfit
SSresid = sum(pow(yresid,2))
SStotal = len(bfit.y) * var(bfit.y)
rsq = 1 - SSresid/SStotal
print ()
print('I am computing the r squared value using the yfit derived')
print(rsq)
print ()

from scipy.stats import *
slope, intercept, r_value, p_value, std_err = linregress(bfit.x, bfit.y)
print('I am computing the r squared value using the scipy method')
print(pow(r_value,2))
print ()
print('This is the pvalue')
print(p_value)
print ('                                             Bfit.csv')



# In[135]:

from numpy import *
from scipy.interpolate import *
from matplotlib.pyplot import *
from sklearn import datasets
from sklearn import linear_model

print ('Cfit.csv')
plot(cfit.x,cfit.y,'o')
plot(cfit.x,polyval(c1,cfit.x), 'r-')
plot(cfit.x,polyval(c2,cfit.x), 'b--')
plot(cfit.x,polyval(c3,cfit.x), 'm:')

#The following will give me the y=mx+b format, and print the slope and intercept
c1 = polyfit(cfit.x, cfit.y, 1)
print('The following will give me the y=mx+b format, and print the slope and intercept')
print(c1)
print ()

#Quadratic
c2 = polyfit(cfit.x, cfit.y, 2)
print('The following will give me the quadratic and print its coefficients')
print (c2)
print ()

#Polynomial
c3 = polyfit(cfit.x, cfit.y, 3)
print('The following will give me the polynomial and print its coefficients')
print (c3)

#Actual yfit
yfit = c1[0] * cfit.x + c1[1]
yresid = cfit.y - yfit
SSresid = sum(pow(yresid,2))
SStotal = len(cfit.y) * var(cfit.y)
rsq = 1 - SSresid/SStotal
print ()
print('I am computing the r squared value using the yfit derived')
print(rsq)
print ()

from scipy.stats import *
slope, intercept, r_value, p_value, std_err = linregress(cfit.x, cfit.y)
print('I am computing the r squared value using the scipy method')
print(pow(r_value,2))
print ()
print('This is the pvalue')
print(p_value)
print ('                                             Cfit.csv')


# In[304]:

from numpy import *
from scipy.interpolate import *
from matplotlib.pyplot import *
from sklearn import datasets
from sklearn import linear_model

print ('dfit.csv')
plot(dfit.x,dfit.y,'o')
plot(dfit.x,polyval(d1,dfit.x), 'r-')
plot(dfit.x,polyval(d2,dfit.x), 'b--')
plot(dfit.x,polyval(d3,dfit.x), 'm:')

#The following will give me the y=mx+b format, and print the slope and intercept
d1 = polyfit(dfit.x, dfit.y, 1)
print('The following will give me the y=mx+b format, and print the slope and intercept')
print(d1)
print ()

#Quadratic
d2 = polyfit(dfit.x, dfit.y, 2)
print('The following will give me the quadratic and print its coefficients')
print (d2)
print ()

#Polynomial
d3 = polyfit(dfit.x, dfit.y, 3)
print('The following will give me the polynomial and print its coefficients')
print (d3)

#Actual yfit
yfit = d1[0] * dfit.x + d1[1]
yresid = dfit.y - yfit
SSresid = sum(pow(yresid,2))
SStotal = len(dfit.y) * var(dfit.y)
rsq = 1 - SSresid/SStotal
print ()
print('I am computing the r squared value using the yfit derived')
print(rsq)
print ()

from scipy.stats import *
slope, intercept, r_value, p_value, std_err = linregress(dfit.x, dfit.y)
print('I am computing the r squared value using the scipy method')
print(pow(r_value,2))
print ()
print('This is the pvalue')
print(p_value)
print ('                                             Dfit.csv')


# In[141]:

from numpy import *
from scipy.interpolate import *
from matplotlib.pyplot import *
from sklearn import datasets
from sklearn import linear_model

print ('efit.csv')
plot(efit.x,efit.y,'o')
plot(efit.x,polyval(c1,efit.x), 'r-')
plot(efit.x,polyval(c2,efit.x), 'b--')
plot(efit.x,polyval(c3,efit.x), 'm:')

#The following will give me the y=mx+b format, and print the slope and intercept
e1 = polyfit(efit.x, efit.y, 1)
print('The following will give me the y=mx+b format, and print the slope and intercept')
print(e1)
print ()

#Quadratic
e2 = polyfit(efit.x, efit.y, 2)
print('The following will give me the quadratic and print its coefficients')
print (e2)
print ()

#Polynomial
e3 = polyfit(efit.x, efit.y, 3)
print('The following will give me the polynomial and print its coefficients')
print (e3)

#Actual yfit
yfit = e1[0] * efit.x + e1[1]
yresid = efit.y - yfit
SSresid = sum(pow(yresid,2))
SStotal = len(efit.y) * var(efit.y)
rsq = 1 - SSresid/SStotal
print ()
print('I am computing the r squared value using the yfit derived')
print(rsq)
print ()

from scipy.stats import *
slope, intercept, r_value, p_value, std_err = linregress(efit.x, efit.y)
print('I am computing the r squared value using the scipy method')
print(pow(r_value,2))
print ()
print('This is the pvalue')
print(p_value)
print ('                                             efit.csv')