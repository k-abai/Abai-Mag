# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:06:20 2024

@author: kekea
"""

import pandas as pd #imports pandas as pd
import matplotlib.pyplot as plt #imports matplotlib as plt
from astropy.visualization import hist

data = pd.read_csv('C:\\Users\\kekea\\OneDrive\\Desktop\\MP_list_CSV.txt', header = None ) #read txt file into array
data.columns = data.iloc[0] #names colomns using first row of txt 
data = data[1:] #removing first row 
# Extract the 9th, 10th, and 13th columns 
data_subset = data.iloc[:, 8:13]


# Attempt to strictly convert all data to numeric
data_subset = data_subset.apply(pd.to_numeric, errors='coerce')
# Drop rows with NaN values
data_subset.dropna(inplace=True)

pdyn = data_subset['pdyn[nPa]']

print(data_subset)#print updated data_subset

fig, ax = plt.subplots(1,2, figsize=(10, 4))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)
for i, bins in enumerate(['scott',20]):
    hist(pdyn, bins=bins, ax=ax[i], histtype='stepfilled',
         alpha=0.2, density=True)
    ax[i].set_xlabel('pdyn[nPa]')
    ax[i].set_title('pdyn frequency',
                    fontdict=dict(family='monospace'))

