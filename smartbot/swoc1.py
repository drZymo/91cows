import numpy as np
import socket
import json
from tensorflow.keras.utils import to_categorical  
from time import sleep
import subprocess

EndOnNegativeReward = True
DiscretePosition = 1/4
DiscreteAngle = (np.pi/2)/4
ClosestDistanceRewardEnabled = False

ActionItemTypes = {
    'Coin': 0,
    'TreasureChest': 1,
    'EmptyChest': 2,
    'MimicChest': 3,
    'SpikeTrap':  4,
    'Bottle': 5,
    'TestTube': 6,
}

Walls = {
    #   t,r,b,l
    0: np.array([0,0,0,0]),
    1: np.array([0,0,0,1]),
    2: np.array([0,1,0,1]),
    3: np.array([1,0,0,1]),
    4: np.array([1,1,0,1]),
}

def getCellWalls(cellType, cellOrient):
    walls = Walls[cellType]
    while cellOrient > 0:
        walls = walls[[1,2,3,0]]
        cellOrient -= 90
    return walls


def getFieldObservation(gameState):
    data = gameState['data']

    walls = np.array([
        [getCellWalls(int(entry['type']), int(entry['orientation'])) for entry in row]
        for row in gameState['data']])

    fieldHeight, fieldWidth = walls.shape[0:2]

    actionItems = gameState['actionItems']
    
    actionItemTypes = np.zeros((fieldHeight, fieldWidth, 7))
    for actionItem in actionItems:
        x = actionItem['x']
        y = actionItem['y']
        actionItemType = ActionItemTypes[actionItem['type']]
        actionItemType = to_categorical(actionItemType, num_classes=7)
        
        x = int(np.floor(x * fieldWidth))
        y = int(np.floor(y * fieldHeight))
        actionItemTypes[y, x] = actionItemType

    field = []
    for y in range(fieldHeight):
        for x in range(fieldWidth):
            ct = walls[y, x]
            at = actionItemTypes[y, x]
            f = np.hstack([ct, at])

            field.append(f)
    field = np.reshape(field, (fieldHeight, fieldWidth, -1))

    return field


def getAngleFromForward(forward):
    theta = np.arctan2(forward[1], forward[0])
    while theta < 0: theta += 2*np.pi
    while theta > 2*np.pi: theta -= 2*np.pi
    return theta / (2*np.pi)# 0...1


def getBotObservation(gameState):
    bots = dict()
    scores = dict()
    for bot in gameState['bots']:
        botId = int(bot['arucoId'])
        position = np.array(bot['position'])
        forward = np.array(bot['forward'])
        right = np.array(bot['right'])
        orientation = getAngleFromForward(forward)
        score = int(bot['score'])

        bot = np.array([position[0], position[1], orientation])
        bots[botId] = bot
        scores[botId] = score
    return bots, scores


def findFieldDistances(field, botPosition):
    fieldHeight, fieldWidth, _ = field.shape
    distances = np.full((fieldHeight, fieldWidth), np.inf)

    x, y = np.floor(botPosition * [fieldWidth, fieldWidth]).astype(int)
    if x < 0 or x > fieldWidth - 1 or y < 0 or y > fieldHeight - 1:
        return distances

    cellsToCheck = [(x,y)]
    distances[y,x] = 0

    def checkAndAdd(x, y, distance):
        if distances[y, x] > distance + 1:
            distances[y, x] = distance + 1
            cellsToCheck.append((x, y))

    while cellsToCheck:
        cellX, cellY = cellsToCheck[0]
        cellsToCheck = cellsToCheck[1:]
        
        walls = field[cellY,cellX,:4] > 0  #top, right, bottom, left
        item = np.any(field[cellY,cellX,4:])
        distance = distances[cellY, cellX]

        if not walls[0]:
            checkAndAdd(cellX, cellY+1, distance)
        if not walls[1]:
            checkAndAdd(cellX+1, cellY, distance)
        if not walls[2]:
            checkAndAdd(cellX, cellY-1, distance)
        if not walls[3]:
            checkAndAdd(cellX-1, cellY, distance)
    return distances


def findClosestItemLocation(field, botPosition):
    distances = findFieldDistances(field, botPosition)
    itemLocations = np.argwhere(np.any(field[:,:,4:], axis=-1))
    itemDistances = distances[itemLocations[:,0], itemLocations[:,1]]
    closestItemIndex = np.argmin(itemDistances)
    return itemLocations[closestItemIndex], itemDistances[closestItemIndex]


class Observer(object):
    def __init__(self, hostname='localhost', portOffset=0):
        self.hostname = hostname
        self.portOffset = portOffset
        self.buffer = ''

    def _getGameState(self):
        # is there a line in the buffer
        # if not, read blocks until there is one
        newLinePos = self.buffer.find('\n')
        while newLinePos < 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.hostname, 9735+self.portOffset))
                data = sock.recv(16384).decode('utf-8')
                self.buffer += data
            newLinePos = self.buffer.find('\n')

        # make sure all old lines are consumed and only the most recent is used
        while newLinePos >= 0:
            line = self.buffer[:newLinePos]
            self.buffer = self.buffer[newLinePos+1:]
            newLinePos = self.buffer.find('\n')

        # conver to game state structure
        gameState = json.loads(line)
        return gameState

    def getObservation(self):
        gameState = self._getGameState()
        field = getFieldObservation(gameState)
        bots, scores = getBotObservation(gameState)
        gameTick = gameState['gameTick']
        return field, bots, scores, gameTick



class GameController(object):
    def __init__(self, gameservicePath, hostname='localhost', portOffset=0):
        self.gameservicePath = gameservicePath
        self.hostname = hostname
        self.portOffset = portOffset
        self.process = None

    def __del__(self):
        if self.process is not None:
            self.process.kill()
            self.process = None

    def close(self):
        if self.process is not None:
            self.process.kill()
            self.process = None

    def _sendRemoteControllerCommand(self, command):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.hostname, 9935+self.portOffset))
            serialized = json.dumps(command)
            sock.sendall(serialized.encode('utf-8'))
        sleep(0.05)

    def createGame(self, fieldWidth, fieldHeight):
        if self.process is not None:
            self.process.kill()
            self.process = None

        # start a new process
        self.process = subprocess.Popen([self.gameservicePath, f'-p {self.portOffset}'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        sleep(0.1) # give it some time to start

        nrCells = fieldWidth * fieldHeight
        #print('Creating game')
        command = {
            'commandType': 'createGame',
            'gameOptions': {
                'mazeSize': [fieldWidth, fieldHeight],
                'numberOfActionItems': {
                    'Bottle': 0, # was: 4
                    'Coin': 1, # was: (nrCells + 49) // 50,
                    'EmptyChest': 0, # was: 1
                    'MimicChest': 0, # was: 1
                    'SpikeTrap': 0, # was: 4
                    'TestTube': 0, # was: 4
                    'TreasureChest': 0 #was: (nrCells + 99) // 100
                },
                'numberOfWallsToRemove': nrCells // 5,
                'removeDeadEnds': True
            }
        }
        self._sendRemoteControllerCommand(command)

    def startGame(self):
        #print('Starting game')
        command = {'commandType': 'startGame'}
        self._sendRemoteControllerCommand(command)

    def stopGame(self):
        #print('Stopping game')
        command = {'commandType': 'stopGame'}
        self._sendRemoteControllerCommand(command)

        # stop the process
        self.process.kill()
        self.process = None


class BotController(object):
    def __init__(self, botId, hostname='localhost', portOffset=0, useDiscretePositions=False, useDiscreteAngles=True):
        self.botId = botId
        self.hostname = hostname
        self.portOffset = portOffset
        self.useDiscretePositions = useDiscretePositions
        self.useDiscreteAngles = useDiscreteAngles

    def reset(self, fieldWidth, fieldHeight):
        self.fieldWidth = fieldWidth
        self.fieldHeight = fieldHeight
        self.botWidth = 0.25 / fieldWidth
        self.botHeight = 0.25 / fieldHeight

        self.x = np.random.uniform(0, fieldWidth)
        self.y = np.random.uniform(0, fieldHeight)
        self.o = np.random.uniform(0, 2*np.pi)

        # start in center of cell and at 90 degree angles
        self.x = np.floor(self.x) + 0.5
        self.y = np.floor(self.y) + 0.5
        self.o = np.round(self.o / (np.pi/2)) * (np.pi/2)

        self._sendUpdate()

    def _sendRobotsState(self, robots):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.hostname, 9635+self.portOffset))
            serialized = json.dumps(robots)
            sock.sendall(serialized.encode('utf-8'))

    def _sendUpdate(self):
        # Clip angle
        if self.o < 0: self.o += 2*np.pi
        if self.o > 2*np.pi: self.o -= 2*np.pi
        
        # Prevent leaving the arena
        self.x = np.clip(self.x, 0, self.fieldWidth)
        self.y = np.clip(self.y, 0, self.fieldHeight)
        
        # Discrete position and angle
        if self.useDiscretePositions:
            self.x = np.round(self.x / DiscretePosition) * DiscretePosition
            self.y = np.round(self.y / DiscretePosition) * DiscretePosition
        if self.useDiscreteAngles:
            self.o = np.round(self.o / DiscreteAngle) * DiscreteAngle

        # Convert to robot positions
        x = self.x / self.fieldWidth
        y = self.y / self.fieldHeight

        # x = x cos r - y sin r
        # y = x sin r + y cos r
        c, s = np.cos(self.o), np.sin(self.o)
        R = np.array(((c, -s), (s, c)))
        robot = {
            'arucoId': self.botId,
            'position': [x, y],
            'xorient': np.matmul(R, [self.botWidth, 0.0]).tolist(),
            'yorient': np.matmul(R, [0.0, self.botHeight]).tolist()
        }
        self._sendRobotsState([{'robot': robot}])

    def moveForward(self):
        c, s = np.cos(self.o), np.sin(self.o)
        R = np.array(((c, -s), (s, c)))
        forward = np.matmul(R, [0.0, DiscretePosition])
        self.x += forward[0]
        self.y += forward[1]
        self._sendUpdate()

    def turnLeft(self):
        self.o += DiscreteAngle
        self._sendUpdate()

    def turnRight(self):
        self.o -= DiscreteAngle
        self._sendUpdate()


class SwocEnv(object):
    def __init__(self, botId, gameservicePath, hostname='localhost', portOffset=0, oneTarget=False):
        self.botId = botId
        self.oneTarget = oneTarget
        self.game = GameController(gameservicePath, hostname, portOffset)
        self.bot = BotController(botId, hostname, portOffset)
        self.observer = Observer(hostname, portOffset)


    def close(self):
        self.game.close()


    def reset(self, fieldWidth, fieldHeight):
        # Create and start a new game
        self.game.createGame(fieldWidth, fieldHeight)
        self.bot.reset(fieldWidth, fieldHeight)
        self.game.startGame()
        
        # Get the first observation
        field, bots, scores, gameTick = self.observer.getObservation()
        bot = bots[self.botId]
        score = scores[self.botId]
        botPosition = bot[0:2]

        self.previousScore = score
        self.done = False

        self.closestItemLocation, self.previousClosestItemDistance = findClosestItemLocation(field, botPosition)

        if self.oneTarget:
            field = field[:,:,:4]
        target = [(self.closestItemLocation[1] + 0.5) / field.shape[0], (self.closestItemLocation[0] + 0.5) / field.shape[1]]
        obs = (field, bot, target)
        return obs


    def step(self, action):
        if self.done: raise "Game done; reset first"
        
        # Perform this action
        if action == 0:
            self.bot.moveForward()
        elif action == 1:
            self.bot.turnLeft()
        elif action == 2:
            self.bot.turnRight()

        # Get and convert new observation        
        field, bots, scores, gameTick = self.observer.getObservation()
        bot = bots[self.botId]
        score = scores[self.botId]
        botPosition = bot[0:2]

        # Compute reward based on score
        reward = score - self.previousScore
        self.previousScore = score

        # Reward shaping
        if reward > 0:
            # picked up coin
            reward = 1.0
            self.done = True
        elif reward < 0:
            # hit a wall
            reward = -0.75
            self.done = True

        # Stop if episode takes too long (1 tick every 50 ms) => 20 ticks/second => 6000 ticks = 300 seconds = 5 minutes
        if gameTick > 6000:
            self.done = True

        #if not self.done:
        #    if reward > 0:
        #        # Find new closest item on positive reward
        #        self.closestItemLocation, self.previousClosestItemDistance = findClosestItemLocation(field, botPosition)
        #    else:
        #        # Find distance to closest item
        #        distances = findFieldDistances(field, botPosition)
        #        closestItemDistance = distances[self.closestItemLocation[0], self.closestItemLocation[1]]
        #
        #        if closestItemDistance < self.previousClosestItemDistance:
        #            # Reward if bot gets closer to the target
        #            reward += 0.1
        #        #elif closestItemDistance > self.previousClosestItemDistance:
        #        #    # Punish (a little) if bot gets further from the target
        #        #    reward -= 0.01
        #        elif closestItemDistance == self.previousClosestItemDistance:
        #            # Punish if bot stays on the same place
        #            reward -= 0.1
        #        self.previousClosestItemDistance = closestItemDistance

        if self.done:
            self.game.stopGame()

        if self.oneTarget:
            field = field[:,:,:4]
        target = [(self.closestItemLocation[1] + 0.5) / field.shape[0], (self.closestItemLocation[0] + 0.5) / field.shape[1]]
        obs = (field, bot, target)
        return obs, reward, self.done
