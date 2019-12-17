import gym
from gym import error, spaces, utils
from swoc1 import SwocEnv
import numpy as np
from draw import DrawObservation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image


EMPTY_ID = 0
WALL_ID = 1
TARGET_ID = 2
BOT_ID = 3


def findDistances(maze, target):
    mazeHeight, mazeWidth = maze.shape
    distances = np.full((mazeHeight, mazeWidth), np.inf)

    y,x = target
    cellsToCheck = [(y,x)]
    distances[y,x] = 0

    def checkAndAdd(y, x, distance):
        if distances[y, x] > distance + 1:
            distances[y, x] = distance + 1
            cellsToCheck.append((y, x))

    while cellsToCheck:
        cellY, cellX = cellsToCheck[0]
        cellsToCheck = cellsToCheck[1:]
        
        distance = distances[cellY, cellX]

        if maze[cellY-1,cellX] == EMPTY_ID:
            checkAndAdd(cellY-1, cellX, distance)
        if maze[cellY+1,cellX] == EMPTY_ID:
            checkAndAdd(cellY+1, cellX, distance)
        if maze[cellY,cellX-1] == EMPTY_ID:
            checkAndAdd(cellY, cellX-1, distance)
        if maze[cellY,cellX+1] == EMPTY_ID:
            checkAndAdd(cellY, cellX+1, distance)
    return distances


def GenerateMaze(mazeWidth, mazeHeight, nrWallsToRemove=10):
    # Generate fully walled maze
    maze = np.zeros((mazeHeight*2+1, mazeWidth*2+1))

    for y in range(mazeHeight):
        cy = y*2 + 1
        for x in range(mazeWidth):
            cx = x*2 + 1

            maze[cy-1,cx-1] = WALL_ID
            maze[cy-1,cx] = WALL_ID
            maze[cy-1,cx+1] = WALL_ID
            maze[cy,cx-1] = WALL_ID
            maze[cy,cx] = EMPTY_ID
            maze[cy,cx+1] = WALL_ID
            maze[cy+1,cx-1] = WALL_ID
            maze[cy+1,cx] = WALL_ID
            maze[cy+1,cx+1] = WALL_ID
    visited = set()
    
    def GetUnvisitedNeighbors(position):
        neighbors = []
        y, x = position

        top = (y-1, x)
        if y > 0 and top not in visited:
            neighbors.append(top)

        bottom = (y+1, x)
        if y < mazeHeight-1 and bottom not in visited:
            neighbors.append(bottom)

        left = (y, x-1)
        if x > 0 and left not in visited:
            neighbors.append(left)

        right = (y, x+1)
        if x < mazeWidth-1 and right not in visited:
            neighbors.append(right)

        return neighbors
    
    def BreakOpen(position, neighbor):
        wally = (position[0] + neighbor[0]) / 2
        wallx = (position[1] + neighbor[1]) / 2
        wy = int(wally*2 + 1)
        wx = int(wallx*2 + 1)
        maze[wy,wx] = EMPTY_ID
    
    def BreakOpenANeighbor(position):
        visited.add(position)

        unvisitedNeighbors = GetUnvisitedNeighbors(position)
        while unvisitedNeighbors:
            # Break open the wall to a random neighbor
            neighbor = unvisitedNeighbors[np.random.choice(len(unvisitedNeighbors))]
            BreakOpen(position, neighbor)

            # Start breaking up that neighbor's walls
            BreakOpenANeighbor(neighbor)

            unvisitedNeighbors = GetUnvisitedNeighbors(position)

    # Start generating a maze
    position = (0,0)
    BreakOpenANeighbor(position)
    
    # Remove a number of walls from the inner part of the maze
    if nrWallsToRemove > 0:
        wallsToRemove = [1,1] + np.argwhere(maze[1:-1, 1:-1] == WALL_ID)
        wallsToRemove = np.random.permutation(wallsToRemove)[:nrWallsToRemove]

        for wall in wallsToRemove:
            maze[wall[0], wall[1]] = EMPTY_ID
    
    # Select a random empty cell for the target
    emptyCells = np.argwhere(maze == EMPTY_ID)
    emptyCells = np.random.permutation(emptyCells)
    targetPos = emptyCells[0]
    maze[targetPos[0], targetPos[1]] = TARGET_ID

    # Find random empty cell with range of target
    distances = findDistances(maze, targetPos)
    closestCells = np.argwhere((distances >= 1) & (distances <=2))
    closestCells = np.random.permutation(closestCells)
    botPos = closestCells[0]
    maze[botPos[0], botPos[1]] = BOT_ID

    return maze
    


class MazeGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mazeWidth=10, mazeHeight=10, saveEpisode=False, nrWallsToRemove=20):
        super(MazeGym, self).__init__()
        self.mazeWidth = mazeWidth
        self.mazeHeight = mazeHeight
        self.saveEpisode = saveEpisode
        self.nrWallsToRemove = nrWallsToRemove

        self.action_space = spaces.Discrete(4)
        observationShape = ((self.mazeWidth*2 + 1),(self.mazeHeight*2 + 1),4)
        self.observation_space = spaces.Box(0, 1, shape=observationShape)
        self.viewer = None
        self.episodeIndex = 0


    def close(self):
        pass


    def _saveObs(self, obs):
        np.save(f'/home/ralph/swoc2019/episode/maze-{self.episodeIndex}-obs.npy', obs, allow_pickle=True)
        self.episodeIndex += 1


    def _saveAct(self, action, reward, done):
        np.save(f'/home/ralph/swoc2019/episode/maze-{self.episodeIndex}-act.npy', [action, reward, done], allow_pickle=True)


    def reset(self):
        while True:
            try:
                self.maze = GenerateMaze(self.mazeWidth, self.mazeHeight, nrWallsToRemove=self.nrWallsToRemove)
                break
            except:
                print('bad maze')

        self.botPos = np.argwhere(self.maze == BOT_ID)[0]

        botIndex = self.botPos[0] * self.maze.shape[1] + self.botPos[1]
        self.visited = [botIndex]
        self.totalReward = 0
        self.minTotalReward = (-0.5*self.maze.shape[0]*self.maze.shape[1])

        if self.saveEpisode: 
            self._saveObs(self.maze)
        return to_categorical(self.maze, num_classes=4)


    def step(self, action):
        if action == 0: # up
            nextPos = (self.botPos[0]-1, self.botPos[1])
        elif action == 1: # right
            nextPos = (self.botPos[0], self.botPos[1]+1)
        elif action == 2: # down
            nextPos = (self.botPos[0]+1, self.botPos[1])
        elif action == 3: # left
            nextPos = (self.botPos[0], self.botPos[1]-1)
        
        nextValue = self.maze[nextPos[0], nextPos[1]]
        nextIndex = nextPos[0] * self.maze.shape[1] + nextPos[1]

        reward, done = 0, False
        if nextValue == WALL_ID:
            # Punish and prevent moving through walls
            reward = -0.75
        else:
            # Move to next cell
            self.maze[self.botPos[0], self.botPos[1]] = EMPTY_ID
            self.maze[nextPos[0], nextPos[1]] = BOT_ID
            self.botPos = nextPos

            if nextValue == TARGET_ID: 
                # Reward when at target
                reward = 1
                done = True
            elif nextIndex in self.visited:
                # Punish if visited before
                reward = -0.25

            self.visited.append(nextIndex)

        self.totalReward += reward
        if self.totalReward < self.minTotalReward:
            done = True

        if self.saveEpisode:
            self._saveAct(action, reward, done)
            self._saveObs(self.maze)

        return to_categorical(self.maze, num_classes=4), reward, done, dict()


    def render(self, mode='human', width=400, height=400):
        img = self.maze / 3
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
