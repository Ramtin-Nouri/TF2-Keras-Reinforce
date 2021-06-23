from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

import numpy as np
from tensorflow.python.keras.layers.advanced_activations import Softmax
from nets import nnBase

class NeuralNetwork(nnBase.NNBase):
    
    def __init__(self):
        #Only sets the name of this class
        self.networkName = "CNN"
            
    def makeModel(self,inputShape,nActions):
        """
            overrides base function
            Create and return a Keras Model
        """
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
