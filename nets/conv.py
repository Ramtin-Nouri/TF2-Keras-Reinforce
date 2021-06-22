from tensorflow.keras.layers import Dense, Conv2D, Dropout, Multiply, Input, Reshape, Flatten, Lambda, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model,Sequential
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.backend import clip,log

import numpy as np
from tensorflow.python.keras.layers.advanced_activations import Softmax
from nets import nnBase

class NeuralNetwork(nnBase.NNBase):
    
    def __init__(self):
        self.filename = "CNN"
            
    def makeModel(self,inputShape,nActions):

        model = Sequential()
        model.add(Conv2D(8, (3, 3), activation='relu',padding='same',input_shape=inputShape))
        model.add((MaxPooling2D(2,2)))
        model.add(Conv2D(16, (3, 3), activation='relu',padding='same'))
        model.add((MaxPooling2D(2,2)))
        model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
        model.add((MaxPooling2D(2,2)))
        model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        model.add((MaxPooling2D(2,2)))
        model.add(Flatten())
        model.add(Dense(128,activation="sigmoid") )
        if nActions>1:
            model.add(Dense(nActions) )
            model.add(Softmax())
        else:
            model.add(Dense(nActions,activation="sigmoid") )
        

        model.compile(optimizer=RMSprop(learning_rate=0.0001), loss="binary_crossentropy")
        return model