import numpy as np
import socket
import json
from tensorflow.keras.utils import to_categorical  
from time import sleep

MaxNrOfBots = 8
MaxBotActionRepeat = 100
TooManyActionRepeatsReward = -1
SamePlaceRadius = 0.01
MaxNotMovedCount = 70 # 3 * 70 = 210 degrees
NotMovedReward = -1
EndOnNegativeReward = False


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
    while theta < -np.pi: theta += 2*np.pi
    while theta > np.pi: theta -= 2*np.pi
    return theta / (2*np.pi)# 0...1


def getBotObservation(gameState):
    bots = []
    scores = []
    for b,bot in enumerate(gameState['bots']):
        position = np.array(bot['position'])
        forward = np.array(bot['forward'])
        right = np.array(bot['right'])
        orientation = getAngleFromForward(forward)
        score = int(bot['score'])

        bots.append(np.array([1, position[0], position[1], orientation]))
        scores.append(score)
    return np.array(bots), np.array(scores)


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
    def __init__(self, hostname='localhost', portOffset=0):
        self.hostname = hostname
        self.portOffset = portOffset

    def _sendRemoteControllerCommand(self, command):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.hostname, 9935+self.portOffset))
            serialized = json.dumps(command)
            sock.sendall(serialized.encode('utf-8'))
        sleep(0.1)

    def createGame(self, fieldWidth, fieldHeight):
        #print('Creating game')
        command = {
            'commandType': 'createGame',
            'gameOptions': {
                'mazeSize': [fieldWidth, fieldHeight],
                'numberOfActionItems': {
                    'Bottle': 4,
                    'Coin': 10,
                    'EmptyChest': 1,
                    'MimicChest': 1,
                    'SpikeTrap': 4,
                    'TestTube': 4,
                    'TreasureChest': 2
                },
                'numberOfWallsToRemove': 20,
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


class BotController(object):
    def __init__(self, hostname='localhost', portOffset=0, nrOfBots=1):
        self.hostname = hostname
        self.portOffset = portOffset
        self.nrOfBots = nrOfBots

    def reset(self, fieldWidth, fieldHeight):
        self.x = np.random.randint(0, fieldWidth, (self.nrOfBots,))
        self.y = np.random.randint(0, fieldHeight, (self.nrOfBots,))

        self.x = (self.x + 0.5) / fieldWidth 
        self.y = (self.y + 0.5) / fieldHeight 

        self.o = np.random.uniform(0, 2*np.pi, (self.nrOfBots,))
        self.sendUpdate()

    def _sendRobotsState(self, robots):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.hostname, 9635+self.portOffset))
            serialized = json.dumps(robots)
            sock.sendall(serialized.encode('utf-8'))

    def sendUpdate(self):
        robots = []
        for b in range(self.nrOfBots):
            robot = dict()
            robot['robot'] = dict()
            robot['robot']["arucoId"] = b+1
            robot['robot']["position"] = [self.x[b], self.y[b]]

            # x = x cos r - y sin r
            # y = x sin r + y cos r
            c, s = np.cos(self.o[b]), np.sin(self.o[b])
            R = np.array(((c, -s), (s, c)))
            robot['robot']["xorient"] = np.matmul(R, [0.025, 0.0]).tolist()
            robot['robot']["yorient"] = np.matmul(R, [0.0, 0.025]).tolist()
            robots.append(robot)
        self._sendRobotsState(robots)

    def moveForward(self, id):
        c, s = np.cos(self.o[id]), np.sin(self.o[id])
        R = np.array(((c, -s), (s, c)))
        forward = np.matmul(R, [0.0, 0.025])
        self.x[id] += forward[0] * 0.5
        self.y[id] += forward[1] * 0.5
        # prevent leaving the arena
        self.x = np.clip(self.x, 0, 1)
        self.y = np.clip(self.y, 0, 1)

    def turnLeft(self, id):
        self.o[id] += 3*np.pi/180

    def turnRight(self, id):
        self.o[id] -= 3*np.pi/180



class SwocEnv(object):
    def __init__(self, hostname='localhost', portOffset=0, nrOfBots=1):
        self.game = GameController(hostname, portOffset)
        self.bot = BotController(hostname, portOffset, nrOfBots)
        self.observer = Observer(hostname, portOffset)
        self.nrOfBots = nrOfBots

        self.previousScores = np.zeros((self.nrOfBots, ))
        self.done = False
        self.lastBotAction = np.array([-1] * self.nrOfBots)
        self.lastBotActionCount = np.zeros((self.nrOfBots,))
        self.lastBotPosition = np.zeros((self.nrOfBots, 2))
        self.lastBotPositionCount = np.zeros((self.nrOfBots,))


    def reset(self, fieldWidth, fieldHeight):
        self.game.createGame(fieldWidth, fieldHeight)
        self.bot.reset(fieldWidth, fieldHeight)
        self.game.startGame()
        field, bots, scores, gameTick = self.observer.getObservation()
        obs = (field, bots)

        self.previousScores = scores[:self.nrOfBots]
        self.done = False
        self.lastBotAction = np.array([-1] * self.nrOfBots)
        self.lastBotActionCount = np.zeros((self.nrOfBots,))
        self.lastBotPosition = np.zeros((self.nrOfBots, 2))
        self.lastBotPositionCount = np.zeros((self.nrOfBots,))

        return obs
    
    def step(self, action):
        if self.done: raise "Game done; reset first"
        
        for b,act in enumerate(action):
            # Perform this action
            if act == 0:
                self.bot.moveForward(b)
            elif act == 1:
                self.bot.turnLeft(b)
            elif act == 2:
                self.bot.turnRight(b)

            # Remember how many times this action was performed
            if act == self.lastBotAction[b]:
                self.lastBotActionCount[b] += 1
            else:
                self.lastBotActionCount[b] = 0
            self.lastBotAction[b] = act
        self.bot.sendUpdate()
        
        field, bots, scores, gameTick = self.observer.getObservation()
        obs = (field, bots)
        rewards = scores[:self.nrOfBots] - self.previousScores
        self.previousScores = scores[:self.nrOfBots]

        # End game if negative reward (e.g. hit wall)
        if EndOnNegativeReward and np.any(rewards < 0):
            self.done = True
        else:
            # Punish if same action was performed too many times
            #tooManyRepeats = (self.lastBotActionCount > MaxBotActionRepeat)
            #rewards += tooManyRepeats * TooManyActionRepeatsReward

            ## Increase weight of positive rewards
            #rewards = rewards + (((rewards > 0) * rewards) * 4)

            # Keep track if bot hasn't moved for a while
            for b,botPosition in enumerate(bots[:,1:3]):
                distance = np.sqrt(np.sum((botPosition - self.lastBotPosition[b])**2))
                if (distance < SamePlaceRadius):
                    self.lastBotPositionCount[b] += 1
                else:
                    self.lastBotPosition[b] = botPosition
                    self.lastBotPositionCount[b] = 0

            # Punish if bot didn't move for a while
            notMoved = (self.lastBotPositionCount > MaxNotMovedCount)
            rewards += notMoved * NotMovedReward

        if self.done:
            self.game.stopGame()

        return obs, rewards, self.done
