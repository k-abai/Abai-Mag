# -*- coding: utf-8 -*-
"""Plot a histogram of magnetopause radius values."""


import pandas as pd #imports pandas as pd
import matplotlib.pyplot as plt #imports matplotlib as plt
import numpy as np

data = pd.read_csv('C:\\Users\\kekea\\OneDrive\\Desktop\\MP_list_CSV.txt', header = None ) #read txt file into array
data.columns = data.iloc[0] #names colomns using first row of txt 
data = data[1:] #removing first row 
data_subset = data.iloc[:, 8:11] #extract useful columns

# Attempt to strictly convert all data to numeric
data_subset = data_subset.apply(pd.to_numeric, errors='coerce')
# Drop rows with NaN values
data_subset.dropna(inplace=True)

r = data_subset['r[re]'] #define r
print(data_subset)#print updated data_subset

#plot histogram
plt.title("r frequency")
plt.xlabel("r[re]")
plt.hist(r)
plt.show() 

