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
