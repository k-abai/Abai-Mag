# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:30:49 2025

@author: kekea
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from makedata import makedata


def synth_mp (pdyn, bx, by, bz, vx, vy, vz, tilt, lat_lim = np.pi/2, lon_lim = 3*np.pi/4, model_file = 'my_model2.keras'):
    #make lat and lon
    lat = np.linspace(-lat_lim, lat_lim, 10000)
    lon = np.linspace(-lon_lim, lon_lim, 10000)
    #make df
    
    #fill data_input
    data_input = pd.DataFrame([])
    data_input['pdyn[nPa]'] = pdyn * np.ones(lat.shape)
    data_input['bz[nT]'] = bz * np.ones(lat.shape)
    data_input['Bz'] = bz * np.ones(lat.shape)
    data_input['Bx'] = bx * np.ones(lat.shape)
    data_input['By'] = by * np.ones(lat.shape)
    data_input['Vx'] = vx * np.ones(lat.shape)
    data_input['Vy'] = vy * np.ones(lat.shape)
    data_input['Vz'] = vz * np.ones(lat.shape)
    data_input['tilt[gsm rad]'] = tilt * np.ones(lat.shape)
    data_input['lat[gsm rad]'] = lat
    #data_input['lon[gsm rad]'] = lon
    data_input['lon[gsm rad]'] = np.zeros(lat.shape)
    
    #reorder to match model
    order = ['lat[gsm rad]', 'lon[gsm rad]', 'bz[nT]', 'pdyn[nPa]', 'tilt[gsm rad]', 'Bx', 'By', 'Bz', 'Vx', 'Vy', 'Vz']
    data_input = data_input[order]
    
    #import scalers
    xscaler = joblib.load('input_xscaler.pkl')
    yscaler = joblib.load('input_yscaler.pkl')
     
    #scale inputs
    norm_input = xscaler.transform(data_input)
      
    #loads model makes r prediction array
    new_model = tf.keras.models.load_model(model_file)   
    r_predict = new_model.predict(norm_input)  # Make prediction
     
    #inv scale
    r = yscaler.inverse_transform(r_predict)
    
    #add r into data_input
    data_input['r'] = r
    
    #create lat, lon as np array
    lat = data_input['lat[gsm rad]'].to_numpy()
    lon = data_input['lon[gsm rad]'].to_numpy()
    r = np.reshape(r,len(r))
    #Convert to XYZ
    X = r* np.cos(lat)* np.cos(lon)
    Y = r* np.cos(lat)* np.sin(lon)
    Z = r* np.sin(lat)
    
    #CREATE PLOTS
    fig = plt.figure(figsize=(10, 8))

    #3D scatter of (X, Y, Z)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(X, Y, Z, c='b', marker = 'o', s = 3, alpha=0.5)
    ax1.set_title("3D Scatter (X, Y, Z)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_aspect('equal')


    #2D scatter of X vs Y
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(X, Y, c='r', marker = 'o', s = 3, alpha=0.5)
    ax2.set_title("XY Projection")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)
    ax2.set_aspect('equal')

    
    #2D scatter of X vs Z
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(X, Z, c='g', marker = 'o', s = 3, alpha=0.5)
    ax3.set_title("XZ Projection")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.grid(True)
    ax3.set_aspect('equal')


    #2D scatter of Z vs Y (equivalently Y vs Z or “ZX”)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(Z, Y, c='m', marker = 'o', s = 3, alpha=0.5)
    ax4.set_title("ZY Projection")
    ax4.set_xlabel("Z")
    ax4.set_ylabel("Y")
    ax4.grid(True)
    ax4.set_aspect('equal')

    
    plt.tight_layout()
    plt.show()
    
    return None
    














