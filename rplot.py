# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:11:00 2025

@author: kekea
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from makedata import makedata

def XYZ_predict (df, model_file = 'my_model2.keras'):
    data = makedata(df)
    #df of useful variables
    df_subset = data.iloc[:, 8:20] 
   
    #r testing array extracts the r[re] column from the dataset data
    r_test = df_subset['r[re]']
    
    #drop r
    df_input = df_subset.drop('r[re]', axis=1) 
    
    #import scalers
    xscaler = joblib.load('input_xscaler.pkl')
    yscaler = joblib.load('input_yscaler.pkl')
    
    #scale inputs
    norm_input = xscaler.transform(df_input)
    
    # Convert input to Df shape
    df_input = pd.DataFrame(norm_input, columns=['lat[gsm rad]','lon[gsm rad]','bz[nT]','pdyn[nPa]', 'tilt[gsm rad]','Bx','By','Bz','Vx','Vy','Vz'])
    
    #loads model makes r prediction array
    new_model = tf.keras.models.load_model(model_file)   
    r_predict = new_model.predict(df_input)  # Make prediction
    
    #inv scale
    r = pd.DataFrame(yscaler.inverse_transform(r_predict))
    df_input_invscale = xscaler.inverse_transform(df_input)
    df_input = pd.DataFrame(df_input_invscale, columns=['lat[gsm rad]','lon[gsm rad]','bz[nT]','pdyn[nPa]', 'tilt[gsm rad]','Bx','By','Bz','Vx','Vy','Vz'])
    #lat, lon
    lat = pd.DataFrame(df_input['lat[gsm rad]'])
    lon = pd.DataFrame(df_input['lon[gsm rad]'])
   #EVERYTHING IS IN DF CONVERT R LAT LON AND COMBINE TO XYZ
    #Convert to XYZ
    #lon is the horizontal angle from the X axis
    #lat is the vertical angle from the Z axis
    X = r * (np.sin(lat)) * np.cos(lon)
    Y = r * (np.sin(lat)) * np.sin(lon)
    Z = r * (np.cos(lat))
    
    XYZ_predict = [X,Y,Z]
    return XYZ_predict
    
#def r_plot (df):
 #   ['R[re]'] = r_predict(df)
    