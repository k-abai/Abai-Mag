# -*- coding: utf-8 -*-
"""
Helper to obtain a single magnetopause distance prediction using the
pre-trained neural network model.

"""
#made up test data [1,1,-.8,1.9,-0.2,-3,-0.2,-0.8,-400,3,-8.5]


import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib

def abaimp(lat, lon, bz, pdyn, tilt, Bx, By, Bz, Vx, Vy, Vz, model_file='my_model2.keras'):
    """Predict the magnetopause radius for a single input sample.

    Parameters
    ----------
    lat, lon, bz, pdyn, tilt, Bx, By, Bz, Vx, Vy, Vz : float
        Environment parameters describing the solar wind conditions.
    model_file : str, optional
        File path to the trained Keras model to use.

    Returns
    -------
    ndarray
        Array with the predicted radius.
    """
    input_data = [lat, lon, bz, pdyn, tilt, Bx, By, Bz, Vx, Vy, Vz]
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
    return prediction