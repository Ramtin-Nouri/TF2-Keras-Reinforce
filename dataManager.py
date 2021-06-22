import os,random,cv2,skimage,numpy as np
import gym

# preprocessing used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def preprocess_frame_karpathy(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return np.expand_dims(I.astype(np.float),axis=2)


class SingleGym():
    """
        Wrapper around actual OpenAi Gym environent with some processing of the observations
    """
    def __init__(self,env_id,prepro):
        """
        Arguments
        ---------
        env_id: str
            Name of the OpenAi Gym Environment
        prepro: bool
            Whether observations should be simplified with preprocess_frame_karpathy or not
        """
        self.env = gym.make(env_id).env
        self.use_preprocess = prepro
        self.lastObservation = None
    
    def reset(self):
        """
            If use_preprocess is True the outputted observation will run through preprocess_frame_karpathy
            else it will only be normalized 
            and then returned
            
            Returns
            -------
            observation: np.array()
                (preprocessed) initial state observation 
        """
        observation = self.env.reset()
        if self.use_preprocess:
            observation = preprocess_frame_karpathy(observation)
        else:
            observation = np.array(observation)/255
        self.lastObservation = observation
        return observation
    
    def step(self,action):
        """
            The returned observation of env.step() will be handled the same as reset()
            At last it will also be subtracted by the last observation before being returned.
            The other returned values from env.step remain unchanged
            
            Arguments
            --------
            action: int
                See OpenAi Gym Documentation
            
            Returns
            -------
            diff: np.array()
                difference between current (processed) observation and last observation
            For description of returns See OpenAi Gym Documentation
            
            
        """
        observation, reward, done, info = self.env.step(action)
        if self.use_preprocess:
            observation = preprocess_frame_karpathy(observation)
        else:
            observation = np.array(observation)/255
        diff = observation-self.lastObservation
        self.lastObservation = observation
        return diff, reward, done, info

        
#Inspired by https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
def calculateRewards(rewards,gamma=0.99):
    """
        Calculate the new rewards with a discount factor gamma
        
        Arguments
        ---------
        rewards: list of floats
            reward value ouputted by environent for each step in the episode
        gamma: float
            Discount factor (See paper)
    """
    newRewards = []
    for t in range(len(rewards)): #For each reward
        power = 0
        newReward = 0
        for oldReward in rewards[t:]: #For each future reward
            newReward = v_t + gamma ** power * oldReward
            power += 1
        newRewards.append(newReward)
    #Normalize:
    newRewards = np.array(newRewards)
    newRewards -= np.mean(newRewards)
    newRewards /= np.std(newRewards)+10**-9 #so we don't divide by 0
    return newRewards
