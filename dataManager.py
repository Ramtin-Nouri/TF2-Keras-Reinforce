import os,random,cv2,skimage,numpy as np
import gym

class MultiGym:
    def __init__(self, env_id, num_env):
        self.envs = []
        for _ in range(num_env):
            self.envs.append(gym.make(env_id).env)

    def reset(self):
        obs = []
        for env in self.envs:
	        obs.append(env.reset())

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []

        for env, ac in zip(self.envs, actions):
            ob, rew, done, info = env.step(ac)
            obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

            if done:
                env.reset()
	
        return obs, rewards, dones, infos
    


# preprocessing used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
#This feels like cheating tho
def preprocess_frame_karpathy(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()


class SingleGym():
    def __init__(self,env_id,prepro):
        self.env = gym.make(env_id).env
        self.use_preprocess = prepro
        self.lastObservation = None
    
    def reset(self):
        observation = self.env.reset()
        if self.use_preprocess:
            observation = preprocess_frame_karpathy(observation)
        else:
            observation = np.array(observation)/255
        self.lastObservation = observation
        return observation
    
    def step(self,action):
        observation, reward, done, info = self.env.step(action)
        if self.use_preprocess:
            observation = preprocess_frame_karpathy(observation)
        else:
            observation = np.array(observation)/255
        diff = observation-self.lastObservation
        self.lastObservation = observation
        return diff, reward, done, info

        
def calculateRewards(rewards,gamma=0.99):
    newRewards = []
    v_t = 0
    for t in range(len(rewards),0,1):
        v_t = gamma * v_t + rewards[t]
        newRewards.append(v_t)
    newRewards.reverse()
    """
    for t in range(len(rewards)):
        power = 0
        v_t = 0
        for r in rewards[t:]:
            v_t = v_t + gamma ** power * r
            power += 1
        newRewards.append(v_t)
    """ 
    #Normalize:
    newRewards = np.array(newRewards)
    newRewards -= np.mean(newRewards)
    newRewards /= np.std(newRewards)
    return newRewards