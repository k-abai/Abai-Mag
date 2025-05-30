# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 22:22:50 2025

@author: kekea
"""


import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib 
from makedata import makedata

print(tf.__version__)
#data load from directory

mag_data_path = 'C:/Users/kekea/OneDrive/MP_list_CSV.txt' #path to data
#mag_data = tf.keras.utils.get_file( 'MP_list_CSV.txt','C:/Users/kekea/OneDrive/Desktop/MP_list_CSV.txt')
data = pd.read_csv(mag_data_path) #read in data
print(data.head(10),data.dtypes) #display data preview and dtypes
magdata = makedata(data)
data_subset = magdata.iloc[:, 8:20]

y = data_subset['r[re]'] #Extracts the r[re] column from the dataset data and assigns it to the variable y 
X = data_subset.drop('r[re]', axis=1) #create new dataframe w/out r[re]
#normalize data
xscaler = MinMaxScaler()
yscaler = MinMaxScaler()

X_norm = xscaler.fit_transform(X)
y_norm = yscaler.fit_transform(pd.DataFrame(y))
train, test, train_label, test_label  = train_test_split(X_norm,y_norm,test_size=0.2) #splt x and y in training and test
#train: Training set features (80% of X).
#test: Testing set features (20% of X).
#train_label: Training set labels (80% of y).
#test_label: Testing set labels (20% of y).




model = keras.models.Sequential([
    #keras.layers.Input(shape = (None, 2)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation = 'linear')])

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate = 0.05), loss='mean_squared_error')
history = model.fit(train, train_label,
                    batch_size=32,
                    epochs=160,
                    validation_split=0.1)

xscaler.inverse_transform(X_norm)
yscaler.inverse_transform(y_norm)

y_train = model.predict(train)
y_train = yscaler.inverse_transform(y_train)
train_label = yscaler.inverse_transform(train_label)
print(np.mean((y_train-train_label)**2))

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (RE)')

model.save('my_model2.keras')
joblib.dump(xscaler,'input_xscaler.pkl')
joblib.dump(yscaler,'input_yscaler.pkl')
