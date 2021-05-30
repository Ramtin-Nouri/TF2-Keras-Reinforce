from tensorflow.keras.layers import Dense, Conv2D, Dropout, Multiply, Input, Reshape, Flatten, Lambda, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model,Sequential
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.backend import clip,log

import numpy as np
from tensorflow.python.keras.layers.advanced_activations import Softmax
from nets import nnBase

class NeuralNetwork(nnBase.NNBase):
    
    def __init__(self):
        self.filename = "cnnAutoEncoder"
            
    def makeModelTF1(self,inputShape,nActions):

        model = Sequential()
        model.add(Conv2D(8, (3, 3), activation='relu',padding='same',input_shape=inputShape))
        model.add((MaxPooling2D(2,2)))
        model.add(Conv2D(16, (3, 3), activation='relu',padding='same'))
        model.add((MaxPooling2D(2,2)))
        model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
        model.add((MaxPooling2D(2,2)))
        model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        model.add(Dropout(0.1))
        model.add((MaxPooling2D(2,2)))
        model.add(Flatten())
        model.add(Dense(nActions))
        model.add(Softmax())
        

        model.compile(optimizer='adam', loss=m_loss)
        return model

    def makeModel(self,inputShape,nActions):
        inputs = Input(shape=inputShape)
        conv1 = Conv2D(8,(3,3),activation="relu",padding="valid")(inputs)
        max1 = MaxPooling2D(2,2)(conv1)    
        conv2 = Conv2D(16,(3,3),activation="relu",padding="valid")(max1)
        max2 = MaxPooling2D(2,2)(conv2)
        conv3 = Conv2D(32,(3,3),activation="relu",padding="valid")(max2)
        max3 = MaxPooling2D(2,2)(conv3)   
        conv4 = Conv2D(16,(3,3),activation="relu",padding="valid")(max3)
        max4 = MaxPooling2D(2,2)(conv4)
        flat = Flatten()(max4)
        dense1 = Dense(nActions)(flat)
        soft = Softmax()(dense1)

        model = Model(inputs=inputs,outputs=soft)
        model.compile(optimizer=RMSprop(lr=0.0001), loss=m_loss)
        return model 

#from https://nbviewer.jupyter.org/github/thinkingparticle/deep_rl_pong_keras/blob/master/reinforcement_learning_pong_keras_policy_gradients.ipynb
def m_loss(episode_reward):
    def loss(y_true,y_pred):
        # feed in y_true as actual action taken 
        # if actual action was up, we feed 1 as y_true and otherwise 0
        # y_pred is the network output(probablity of taking up action)
        # note that we dont feed y_pred to network. keras computes it
        
        # first we clip y_pred between some values because log(0) and log(1) are undefined
        tmp_pred = Lambda(lambda x: clip(x,0.05,0.95))(y_pred)
        # we calculate log of probablity. y_pred is the probablity of taking up action
        # note that y_true is 1 when we actually chose up, and 0 when we chose down
        # this is probably similar to cross enthropy formula in keras, but here we write it manually to multiply it by the reward value
        tmp_loss = Lambda(lambda x:-y_true*log(x)-(1-y_true)*(log(1-x)))(tmp_pred)
        # multiply log of policy by reward
        policy_loss= Multiply()([tmp_loss,episode_reward])
        return policy_loss
    return loss
