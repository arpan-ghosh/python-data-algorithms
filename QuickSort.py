
# coding: utf-8

# ## Quicksort in Python
# 

# In[16]:

# quicksort via recursion

# suppose list = [10, 0, 4, 6, 7, 8, 4]

test = [10, 0, 4, 6, 7, 8, 5]

import numpy as np

def quicksort(list, low, high):
    # when low and high pivot same (list of one element)
    if ((high - low) > 0):
        pivot = partition(list, low, high)
        quicksort(list, low, pivot-1)
        quicksort(list, pivot+1, high)

def partition(list, low, high):
    # set barrier
    barrier = low
    stop = high
    
    for i in range(low, high):
        if (list[i] < list[stop]):
            list[i], list[barrier] = list[barrier], list[i]
            # increment
            barrier += 1
    
    list[stop], list[barrier] = list[barrier], list[stop]
    
    array = np.asarray(barrier);
    print array

    return barrier;

