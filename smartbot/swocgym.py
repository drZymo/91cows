import gym
from gym import error, spaces, utils
from swoc1 import SwocEnv
import numpy as np
from draw import DrawObservation

class SwocGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, botId, portOffset):
        super(SwocGym, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1, shape=(1103,))
        self.env = SwocEnv(botId, hostname='localhost', portOffset=portOffset)
        self.viewer = None

    def close(self):
        self.env.close()

    def reset(self):
        fieldObs, botObs = self.env.reset(10, 10)
        self.lastFieldObs = fieldObs
        self.lastBotObs = botObs
        obs = np.hstack([fieldObs.flatten(), botObs])
        return obs

    def step(self, action):
        (fieldObs, botObs), reward, done = self.env.step(action)
        self.lastFieldObs = fieldObs
        self.lastBotObs = botObs
        obs = np.hstack([fieldObs.flatten(), botObs])
        return obs, reward, done, dict()

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