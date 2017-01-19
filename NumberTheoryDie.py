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


