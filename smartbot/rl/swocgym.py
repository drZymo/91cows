import gym
from gym import error, spaces, utils
from swoc1 import SwocEnv
import numpy as np
from draw import DrawObservation

class SwocGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, botId, gameservicePath, portOffset, saveEpisode=False, actionRepeat=1, fieldWidth=10, fieldHeight=10, oneTarget=False):
        super(SwocGym, self).__init__()
        self.botId = botId
        self.saveEpisode = saveEpisode
        self.actionRepeat = actionRepeat
        self.fieldWidth = fieldWidth
        self.fieldHeight = fieldHeight

        self.action_space = spaces.Discrete(3)
        if oneTarget:
            observationShape = (self.fieldWidth*self.fieldHeight*4 + 3 + 2,)
        else:
            observationShape = (self.fieldWidth*self.fieldHeight*11 + 3 + 2,)
        self.observation_space = spaces.Box(0, 1, shape=observationShape)
        self.env = SwocEnv(botId, gameservicePath, hostname='localhost', portOffset=portOffset, oneTarget=oneTarget)
        self.viewer = None
        self.index = 0


    def close(self):
        self.env.close()


    def _saveObs(self, obs):
        np.save(f'/home/ralph/swoc2019/episode/{self.botId}-{self.index}-obs.npy', obs)
        self.index += 1


    def _saveAct(self, action, reward, done):
        np.save(f'/home/ralph/swoc2019/episode/{self.botId}-{self.index}-act.npy', [action, reward, done])


    def reset(self):
        fieldObs, botObs, targetObs = self.env.reset(self.fieldWidth, self.fieldHeight)
        self.lastFieldObs = fieldObs
        self.lastBotObs = botObs
        self.lastTargetObs = targetObs
        obs = np.hstack([fieldObs.flatten(), botObs, targetObs])
        if self.saveEpisode: 
            self._saveObs(obs)
        return obs


    def step(self, action):
        totalReward = 0
        for _ in range(self.actionRepeat):
            (fieldObs, botObs, targetObs), reward, done = self.env.step(action)
            self.lastFieldObs = fieldObs
            self.lastBotObs = botObs
            self.lastTargetObs = targetObs
            totalReward += reward
            if done: break
        obs = np.hstack([self.lastFieldObs.flatten(), self.lastBotObs, self.lastTargetObs])
        
        if self.saveEpisode:
            self._saveAct(action, totalReward, done)
            self._saveObs(obs)
        
        return obs, totalReward, done, dict()


    def render(self, mode='human', width=240, height=240):
        img = np.array(DrawObservation((self.lastFieldObs, self.lastBotObs), width, height))
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