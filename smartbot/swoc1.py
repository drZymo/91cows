import numpy as np
import socket
import json
from tensorflow.keras.utils import to_categorical  
from time import sleep

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
    def __init__(self, botId, hostname='localhost', portOffset=0):
        self.botId = botId
        self.hostname = hostname
        self.portOffset = portOffset

    def reset(self, fieldWidth, fieldHeight):
        self.xy = np.array([np.random.randint(0, fieldWidth), np.random.randint(0, fieldHeight)])
        self.xy = (self.xy + 0.5) / [fieldWidth, fieldHeight]

        self.o = np.random.uniform(0, 2*np.pi)
        self._sendUpdate()

    def _sendRobotsState(self, robots):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.hostname, 9635+self.portOffset))
            serialized = json.dumps(robots)
            sock.sendall(serialized.encode('utf-8'))

    def _sendUpdate(self):
        # x = x cos r - y sin r
        # y = x sin r + y cos r
        c, s = np.cos(self.o), np.sin(self.o)
        R = np.array(((c, -s), (s, c)))
        robot = {
            'arucoId': self.botId,
            'position': self.xy.tolist(),
            'xorient': np.matmul(R, [0.025, 0.0]).tolist(),
            'yorient': np.matmul(R, [0.0, 0.025]).tolist()
        }
        self._sendRobotsState([{'robot': robot}])

    def moveForward(self):
        c, s = np.cos(self.o), np.sin(self.o)
        R = np.array(((c, -s), (s, c)))
        forward = np.matmul(R, [0.0, 0.025])
        self.xy += forward * 0.5
        # prevent leaving the arena
        self.xy = np.clip(self.xy, 0, 1)
        self._sendUpdate()

    def turnLeft(self):
        self.o += 3*np.pi/180
        self._sendUpdate()

    def turnRight(self):
        self.o -= 3*np.pi/180
        self._sendUpdate()


class SwocEnv(object):
    def __init__(self, botId, hostname='localhost', portOffset=0):
        self.botId = botId
        self.game = GameController(hostname, portOffset)
        self.bot = BotController(botId, hostname, portOffset)
        self.observer = Observer(hostname, portOffset)

        self.previousScore = 0
        self.done = False
        self.lastBotPosition = np.zeros((2,))
        self.lastBotPositionCount = 0


    def reset(self, fieldWidth, fieldHeight):
        self.game.createGame(fieldWidth, fieldHeight)
        self.bot.reset(fieldWidth, fieldHeight)
        self.game.startGame()
        field, bots, scores, gameTick = self.observer.getObservation()
        bot = bots[self.botId]
        score = scores[self.botId]
        obs = (field, bot)

        self.previousScore = score
        self.done = False
        self.lastBotPosition = np.zeros((2,))
        self.lastBotPositionCount = 0
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
        
        field, bots, scores, gameTick = self.observer.getObservation()
        bot = bots[self.botId]
        score = scores[self.botId]
        obs = (field, bot)
        reward = score - self.previousScore
        self.previousScore = score

        # End game if negative reward (e.g. hit wall)
        if EndOnNegativeReward and np.any(rewards < 0):
            self.done = True
        else:
            # Check if bot hasn't moved for a while
            botPosition = bot[0:2]
            distance = np.linalg.norm(botPosition - self.lastBotPosition)
            if (distance < SamePlaceRadius):
                self.lastBotPositionCount += 1
            else:
                self.lastBotPosition = botPosition
                self.lastBotPositionCount = 0

            # Punish if bot didn't move for a while
            notMoved = (self.lastBotPositionCount > MaxNotMovedCount)
            reward += notMoved * NotMovedReward

        if self.done:
            self.game.stopGame()

        return obs, reward, self.done
