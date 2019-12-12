import gym
from gym import error, spaces, utils
from swoc1 import SwocEnv
import numpy as np
from draw import DrawObservation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def convertToGrid(obs, fieldWidth, fieldHeight):
    fieldObs, botObs, targetObs = obs

    # 0 = empty, 1 = wall, 2 = bot, 3 = target
    field = np.ones((fieldHeight*2+1, fieldWidth*2+1))

    # empty cells
    for y in range(fieldObs.shape[0]):
        cy = y*2+1
        for x in range(fieldObs.shape[1]):
            cx = x*2+1
            t,r,b,l = fieldObs[y, x]
            field[cy, cx] = 0
            field[cy+1, cx] = 1 if t else 0
            field[cy, cx+1] = 1 if r else 0
            field[cy-1, cx] = 1 if b else 0
            field[cy, cx-1] = 1 if l else 0

    # bot
    bx = int(botObs[0]*fieldWidth)*2 + 1
    by = int(botObs[1]*fieldHeight)*2 + 1
    bx = np.clip(bx, 0, field.shape[1]-1)
    by = np.clip(by, 0, field.shape[0]-1)
    field[by, bx] = 2

    # target
    tx = int(targetObs[0]*fieldWidth)*2 + 1
    ty = int(targetObs[1]*fieldHeight)*2 + 1
    field[ty, tx] = 3

    field = to_categorical(field, num_classes=4)
    return field


class SwocGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, botId, gameservicePath, portOffset, saveEpisode=False, fieldWidth=10, fieldHeight=10):
        super(SwocGym, self).__init__()
        self.botId = botId
        self.saveEpisode = saveEpisode
        self.fieldWidth = fieldWidth
        self.fieldHeight = fieldHeight

        self.action_space = spaces.Discrete(3)
        observationShape = ((self.fieldWidth*2 + 1)*(self.fieldHeight*2 + 1),)
        self.observation_space = spaces.Box(0, 1, shape=observationShape)
        self.env = SwocEnv(botId, gameservicePath, hostname='localhost', portOffset=portOffset, oneTarget=True)
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
        obs = self.env.reset(self.fieldWidth, self.fieldHeight)
        obs = convertToGrid(obs, self.fieldWidth, self.fieldHeight)

        self.lastObs = obs
        if self.saveEpisode: 
            self._saveObs(obs)
        return obs


    def step(self, action):
        totalReward = 0
        for _ in range(4):
            obs, reward, done = self.env.step(action)
            obs = convertToGrid(obs, self.fieldWidth, self.fieldHeight)
            self.lastObs = obs
            totalReward += reward
            if done: break
        obs = self.lastObs
        
        if self.saveEpisode:
            self._saveAct(action, totalReward, done)
            self._saveObs(obs)
        
        return obs, totalReward, done, dict()


    def render(self, mode='human'):
        img = np.argmax(self.lastObs, axis=-1) / 3
        img = np.uint8(plt.get_cmap('viridis')(img)*255)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen