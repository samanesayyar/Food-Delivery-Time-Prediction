# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:32:09 2024

@author: Samane
"""
import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import plotly.io as pio
pio.templates.default = "plotly_white"
pio.renderers.default='browser'

data=pd.read_csv("Dataset/deliverytime.txt")
#print(data.head())
print(data.info())
#print(data.isnull().sum())

## Calculating Distance Between Two Latitudes and Longitudes##

#Set the earth's radius
R=6371

#convert degrees to radians
def deg_to_rad(degrees):
    return degrees*(np.pi/180)

#function to calcute the distance between two point
def distcalculate(lat1, lon1, lat2, lon2):
    d_lat=deg_to_rad(lat2-lat1)
    d_lon=deg_to_rad(lon2-lon1)
    a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
# Calculate the distance between each pair of points
data['distance'] = np.nan

for i in range(len(data)):
    data.loc[i, 'distance'] = distcalculate(data.loc[i, 'Restaurant_latitude'], 
                                        data.loc[i, 'Restaurant_longitude'], 
                                        data.loc[i, 'Delivery_location_latitude'], 
                                        data.loc[i, 'Delivery_location_longitude'])
#print(data.head())  

# figure = px.scatter(data_frame = data, 
#                     x="distance",
#                     y="Time_taken(min)", 
#                     size="Time_taken(min)", 
#                     trendline="ols", 
#                     title = "Relationship Between Distance and Time Taken")
# figure.show()
#######
# figure = px.scatter(data_frame = data, 
#                     x="Delivery_person_Ratings",
#                     y="Time_taken(min)", 
#                     size="Time_taken(min)", 
#                     color="distance",
#                     trendline="ols", 
#                     title = "Relationship Between Distance and Time Taken")
# figure.show()


#splitting data
x=np.array(data[["Delivery_person_Age", 
                   "Delivery_person_Ratings", 
                   "distance"]])
y=np.array(data[["Time_taken(min)"]])

x_train, x_tesr, y_train, y_test = train_test_split(x,y, test_size=0.10, random_state=42)

# creating the LSTM neural network model

model=Sequential()
model.add(LSTM(128, return_sequences = True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

# training the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=9)

print("Food Delivery Time Prediction")
a = int(input("Age of Delivery Partner: "))
b = float(input("Ratings of Previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[a, b, c]])
print("Predicted Delivery Time in Minutes = ", model.predict(features))
  