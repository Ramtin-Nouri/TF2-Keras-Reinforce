# import necessary modules from keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from nets import nnBase
"""
Simple FNN with a 200 neuron Dense Layer
"""
class NeuralNetwork(nnBase.NNBase):
    
    def __init__(self):
        self.networkName = "SimpleFNN"
        
    def makeModel(self,inputShape,nActions):
        model = Sequential()
        model.add(Dense(units=200,input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
