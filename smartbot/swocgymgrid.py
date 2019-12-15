import gym
from gym import error, spaces, utils
from swoc1 import SwocEnv
import numpy as np
from draw import DrawObservation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image

def convertToGrid(obs, fieldWidth, fieldHeight):
    fieldObs, botObs, targetObs = obs

    # 0 = empty, 1 = wall, 2 = bot, 3 = target
    field = np.zeros((fieldHeight*2+1, fieldWidth*2+1))

    # empty cells
    for y in range(fieldObs.shape[0]):
        cy = y*2+1
        for x in range(fieldObs.shape[1]):
            cx = x*2+1
            t,r,b,l = fieldObs[y, x]
            field[cy, cx] = 0
            
            if t:
                field[cy+1, cx-1] = 1
                field[cy+1, cx] = 1
                field[cy+1, cx+1] = 1
            if b:
                field[cy-1, cx-1] = 1
                field[cy-1, cx] = 1
                field[cy-1, cx+1] = 1
            if l:
                field[cy-1, cx-1] = 1
                field[cy, cx-1] = 1
                field[cy+1, cx-1] = 1
            if r:
                field[cy-1, cx+1] = 1
                field[cy, cx+1] = 1
                field[cy+1, cx+1] = 1

    # target
    tx = int(targetObs[0]*fieldWidth*2 + 0.5)  # round to nearest int
    ty = int(targetObs[1]*fieldHeight*2 + 0.5)
    field[ty, tx] = 3

    # bot
    bx = int(botObs[0]*fieldWidth*2 + 0.5)  # round to nearest int
    by = int(botObs[1]*fieldHeight*2 + 0.5)
    bx = np.clip(bx, 0, field.shape[1]-1)
    by = np.clip(by, 0, field.shape[0]-1)
    field[by, bx] = 2

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
        observationShape = ((self.fieldWidth*2 + 1)*(self.fieldHeight*2 + 1)*4,)
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

        while True:
            self.lastRawObs = self.env.reset(self.fieldWidth, self.fieldHeight)
            _, botObs, targetObs = self.lastRawObs
            dist = np.linalg.norm(targetObs - botObs[:2])
            if dist > 0.01:
                break
        
        self.lastObs = convertToGrid(self.lastRawObs, self.fieldWidth, self.fieldHeight)

        pos = np.argwhere(self.lastObs[:,:,2] == 1)[0]
        pos = pos[0] * self.lastObs.shape[1] + pos[1]
        self.visited = [pos]

        if self.saveEpisode: 
            self._saveObs(obs)
        return self.lastObs.flatten()


    def step(self, action):
        totalReward, done = 0, False
        
        # Determine target orientation
        if action == 0: # north
            targetOrientation = 0.75
        elif action == 1: # east
            targetOrientation = 0
        elif action == 2: # south
            targetOrientation = 0.25
        elif action == 3: # west
            targetOrientation = 0.5

        # Rotate to target orientation
        while not done:
            _,_,orientation = self.lastRawObs[1]
            deltaOrientation = targetOrientation - orientation
            if deltaOrientation < 0.5: deltaOrientation += 1.0
            if deltaOrientation > 0.5: deltaOrientation -= 1.0
            if abs(deltaOrientation) < 0.01:
                break

            self.lastRawObs, reward, done = self.env.step(2 if deltaOrientation > 0 else 1)
            totalReward += reward

        # Move two steps forward
        if not done:            
            self.lastRawObs, reward, done = self.env.step(0)
            totalReward += reward
        if not done:            
            self.lastRawObs, reward, done = self.env.step(0)
            totalReward += reward

        # Convert to right observation
        self.lastObs = convertToGrid(self.lastRawObs, self.fieldWidth, self.fieldHeight)

        # Punish if visited before
        pos = np.argwhere(self.lastObs[:,:,2] == 1)[0]
        print(pos)
        pos = pos[0] * self.lastObs.shape[1] + pos[1]
        print(pos)
        if pos in self.visited:
            totalReward = -0.1
        self.visited.append(pos)

        if self.saveEpisode:
            self._saveAct(action, totalReward, done)
            self._saveObs(obs)
        
        return self.lastObs.flatten(), totalReward, done, dict()


    def render(self, mode='human', width=240, height=240):
        img = np.argmax(self.lastObs, axis=-1) / 3
        img = np.uint8(plt.get_cmap('viridis')(img)*255)
        img = np.array(Image.fromarray(img, mode='RGBA').resize((width, height)))

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen