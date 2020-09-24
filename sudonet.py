# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:30:44 2020

@author: Dell
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D 
from tensorflow.keras.layers import Activation

class sudonet(): #https://www.researchgate.net/figure/A-seven-layered-convolutional-neural-network-for-handwritten-digits-recognition_fig3_330872480
    @staticmethod
    def build(width,height,depth,classes):
        model = Sequential()
        input_shape = (width,height,depth)
        
        model.add(Conv2D(32,(5,5),padding = "same", input_shape= input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size= (2,2)))
        
        model.add(Conv2D(64,(3,3),padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size= (2,2)))
        
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        
        return model   
        
        
        
        
        