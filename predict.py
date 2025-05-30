# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 11:46:42 2025

@author: kekea
"""

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib

def abaimp(lat,lon,bz,pdyn,tilt,Bx,By,Bz,Vx,Vy,Vz, model_file = 'my_model2.keras'):
    input_data = [lat,lon,bz,pdyn,tilt,Bx,By,Bz,Vx,Vy,Vz]
    scaler = joblib.load('input_scaler.pkl')
    norm_input = scaler.transform([input_data])

# Convert input to Df shape
    df = pd.DataFrame(norm_input, columns=['lat[gsm rad]','lon[gsm rad]','bz[nT]','pdyn[nPa]', 'tilt[gsm rad]','Bx','By','Bz','Vx','Vy','Vz'])
    df = df.astype(float)
      
# Load model
    new_model = tf.keras.models.load_model(model_file)
        
# Make prediction
    prediction = new_model.predict(df)
    
# Return the predicted value
    return prediction,

#made up test data [1,1,-.8,1.9,-0.2,-3,-0.2,-0.8,-400,3,-8.5]