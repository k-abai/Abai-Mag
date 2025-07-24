# -*- coding: utf-8 -*-
"""Plot predicted versus true magnetopause radii."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from matplotlib.colors import LogNorm  # for log color scale
from makedata import makedata

def valtest(df, model_file='my_model2.keras'):
    """Generate a hexbin plot comparing predictions with target radii.

    Parameters
    ----------
    df : DataFrame
        Input dataset containing measurement values.
    model_file : str, optional
        File path to the trained Keras model to load.

    Returns
    -------
    None
        The plot is displayed directly and nothing is returned.
    """
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
    norm_input = xscaler.transform(df_input)
   
    # Convert input to Df shape
    df_input = pd.DataFrame(norm_input, columns=['lat[gsm rad]','lon[gsm rad]','bz[nT]','pdyn[nPa]', 'tilt[gsm rad]','Bx','By','Bz','Vx','Vy','Vz'])
    
    #loads model makes r prediction array
    new_model = tf.keras.models.load_model(model_file)   
    r_predict = new_model.predict(df_input)  # Make prediction
    
    r_predict = yscaler.inverse_transform(r_predict)

    #makes hexmap
    x = r_test
    y = r_predict
    xlim = x.min(), x.max()
    ylim = y.min(), y.max()

                
             
    hb = plt.hexbin(x, y[:,0], gridsize = 50, cmap='inferno', bins = 'log')   
    plt.xlabel("target - r(re)")
    plt.ylabel("predict - r(re)")
    plt.title("r(re)")
    cb = plt.colorbar(hb,label='counts')

    
    #Add the x = y line
    min_val, max_val = plt.xlim()  
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    plt.show()

    return None
