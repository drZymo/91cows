import gym
from gym import error, spaces, utils
from swoc1 import SwocEnv
import numpy as np
from draw import DrawObservation

class SwocGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, botId, gameservicePath, portOffset, saveEpisode=False, actionRepeat=4):
        super(SwocGym, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1, shape=(1103,))
        self.env = SwocEnv(botId, gameservicePath, hostname='localhost', portOffset=portOffset)
        self.viewer = None
        self.botId = botId
        self.saveEpisode = saveEpisode
        self.index = 0
        self.actionRepeat = actionRepeat


    def close(self):
        self.env.close()


    def _saveObs(self, obs):
        np.save(f'/home/ralph/swoc2019/episode/{self.botId}-{self.index}-obs.npy', obs)
        self.index += 1


    def _saveAct(self, action, reward, done):
        np.save(f'/home/ralph/swoc2019/episode/{self.botId}-{self.index}-act.npy', [action, reward, done])


    def reset(self):
        fieldObs, botObs = self.env.reset(10, 10)
        self.lastFieldObs = fieldObs
        self.lastBotObs = botObs
        obs = np.hstack([fieldObs.flatten(), botObs])
        if self.saveEpisode: 
            self._saveObs(obs)
        return obs


    def step(self, action):
        totalReward = 0
        for _ in range(self.actionRepeat):
            (fieldObs, botObs), reward, done = self.env.step(action)
            self.lastFieldObs = fieldObs
            self.lastBotObs = botObs
            totalReward += reward
            if reward < 0 or done: break
        obs = np.hstack([self.lastFieldObs.flatten(), self.lastBotObs])
        
        if self.saveEpisode:
            self._saveAct(action, totalReward, done)
            self._saveObs(obs)
        
        return obs, totalReward, done, dict()


    def render(self, mode='human'):
        img = np.array(DrawObservation((self.lastFieldObs, self.lastBotObs), 240, 240))
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen