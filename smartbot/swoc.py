import numpy as np
import socket
import json
from tensorflow.keras.utils import to_categorical  

NrOfBots = 8

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


def getBotObservation(gameState):
    bots = np.zeros((NrOfBots, 7))
    for b,bot in enumerate(gameState['bots']):
        position = np.array(bot['position'])
        forward = np.array(bot['forward'])
        right = np.array(bot['right'])

        bots[b,0] = 1
        bots[b,1:3] = position
        bots[b,3:5] = forward
        bots[b,5:7] = right
    return bots


class Env(object):
    def __init__(self, hostname='localhost'):
        self.buffer = ''
        self.sock =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((hostname, 9735))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sock.close()

    def _getGameState(self):
        newLinePos = self.buffer.find('\n')
        while newLinePos < 0:
            data = self.sock.recv(16384).decode('utf-8')
            self.buffer += data
            newLinePos = self.buffer.find('\n')

        line = self.buffer[:newLinePos]
        buffer = self.buffer[newLinePos+1:]
        gameState = json.loads(line)

        return gameState


    def getObservation(self):
        gameState = self._getGameState()
        field = getFieldObservation(gameState)
        bots = getBotObservation(gameState)
        gameTick = gameState['gameTick']
        return field, bots, gameTick
