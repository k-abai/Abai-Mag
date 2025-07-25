# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 15:23:12 2025

@author: kekea
"""

import numpy as np
import pandas as pd
from timestamp import timestamp

def makedata(df):
    #cut 
    data = df.loc[df['year']>1994,:] 
    #gets timestamp of data
    data_time = timestamp(data) 
    #load sw
    sw_data = pd.read_hdf('omni_data.h5', key = 'omni_1min', mode = 'r') 
    #What data to load
    data['Bx'] = np.interp(data_time,sw_data['Epoch'],sw_data['BX_GSE'])
    data['By'] = np.interp(data_time,sw_data['Epoch'],sw_data['BY_GSM'])
    data['Bz'] = np.interp(data_time,sw_data['Epoch'],sw_data['BZ_GSM'])
    data['Vx'] = np.interp(data_time,sw_data['Epoch'],sw_data['Vx'])
    data['Vy'] = np.interp(data_time,sw_data['Epoch'],sw_data['Vy'])
    data['Vz'] = np.interp(data_time,sw_data['Epoch'],sw_data['Vz'])
    data['proton_density'] = np.interp(data_time,sw_data['Epoch'],sw_data['proton_density'])
    #check and drop NaN columns for training
    data = data.dropna(axis = 0,how = 'any')

    return data