import socket
import json
from time import sleep
from multiprocessing import Process, Pipe, Queue
import numpy as np
import random
import math

from tensorflow.keras.utils import to_categorical  


### TODO
# - Use radians internally

# bot size is 2 * 0.025 = 0.05 = half a cell with 10 cells 
# item size = 2* widthPerCell / 6 = 1/3th of a cell width
# items are always in the center of a cell. only one item per cell -> one-hot encode


NrOfBots = 4
NrOfOtherBots = 8

ServerAddress = 'localhost'

def SendRemoteControllerCommand(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ServerAddress, 9935))
        serialized = json.dumps(command)
        s.sendall(serialized.encode('utf-8'))
    sleep(1)

def CreateGame():
    print('Creating game')
    command = {
        'commandType': 'createGame',
        'gameOptions': {
            'mazeSize': [10,10],
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
    SendRemoteControllerCommand(command)

def StartGame():
    print('Starting game')
    command = {'commandType': 'startGame'}
    SendRemoteControllerCommand(command)

def StopGame():
    print('Stopping game')
    command = {'commandType': 'stopGame'}
    SendRemoteControllerCommand(command)

def SendRobots(robots):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ServerAddress, 9635))
        serialized = json.dumps(robots)
        s.sendall(serialized.encode('utf-8'))

def SendState(x, y, o):
    robots = []
    for b in range(NrOfBots):
        robot = dict()
        robot['robot'] = dict()
        robot['robot']["arucoId"] = b+1
        robot['robot']["position"] = [x[b], y[b]]

        # x = x cos r - y sin r
        # y = x sin r + y cos r
        c, s = np.cos(o[b]), np.sin(o[b])
        R = np.array(((c, -s), (s, c)))
        robot['robot']["xorient"] = np.matmul(R, [0.025, 0.0]).tolist()
        robot['robot']["yorient"] = np.matmul(R, [0.0, 0.025]).tolist()
        robots.append(robot)
    SendRobots(robots)

ActionItemTypes = {
    'Coin': 1,
    'TreasureChest': 2,
    'EmptyChest': 3,
    'MimicChest': 4,
    'SpikeTrap':  5,
    'Bottle': 6,
    'TestTube': 7,
}

def GetFieldObservation(gameState):
    data = gameState['data']

    cellTypes = []
    for column in data:
        cellTypesColumn = []
        for entry in column:
            cellType = int(entry['type'])   # 5 classes
            cellOrient = int(entry['orientation']) // 90    # 4 classes

            cellType = cellType * 4 + cellOrient

            cellTypesColumn.append(cellType)
        cellTypesColumn = np.array(cellTypesColumn)

        cellTypes.append(cellTypesColumn)
    cellTypes = np.array(cellTypes)

    fieldHeight, fieldWidth = cellTypes.shape

    actionItems = gameState['actionItems']
    
    actionItemTypes = np.zeros_like(cellTypes)
    for actionItem in actionItems:
        x = actionItem['x']
        y = actionItem['y']
        actionItemType = ActionItemTypes[actionItem['type']]
        
        x = int(math.floor(x * fieldWidth))
        y = int(math.floor(y * fieldHeight))
        actionItemTypes[y, x] = actionItemType

    field = []
    for y in range(fieldHeight):
        for x in range(fieldWidth):
            ct = to_categorical(cellTypes[y, x], num_classes=20)
            at = to_categorical(actionItemTypes[y, x], num_classes=8)
            f = np.hstack([ct, at])

            field.append(f)
    field = np.reshape(field, (fieldHeight, fieldWidth, -1))

    return field

def GetAngleFromForward(forward):
    theta = math.atan2(forward[1], forward[0])
    if theta < -math.pi: theta += math.tau
    if theta > math.pi: theta -= math.tau
    return theta

def GetBotObservation(gameState, botId):
    me = np.zeros((3,))
    myScore = 0
    others = np.zeros((NrOfOtherBots, 4))
    nrOthers = 0

    bots = gameState['bots']
    for bot in bots:
        arucoId = bot['arucoId'] - 1
        x, y = bot['position']
        forward = bot['forward']
        orientation = GetAngleFromForward(forward) / math.pi # -1..1
        if arucoId == botId:
            me = np.array([x, y, orientation])
            myScore = bot['score']
        else:
            others[nrOthers] = [1, x, y, orientation]
            nrOthers += 1

    others = np.array(others).flatten()
    return np.hstack([me, others]), myScore

buffer = ''

def GetObservation():
    global buffer
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ServerAddress, 9735))

        newLinePos = buffer.find('\n')
        while newLinePos < 0:
            data = s.recv(16384).decode('utf-8')
            buffer += data
            newLinePos = buffer.find('\n')

        line = buffer[:newLinePos]
        buffer = buffer[newLinePos+1:]
        gameState = json.loads(line)

    botObservations = []
    scores = []
    for botId in range(NrOfBots):
        botObs, score = GetBotObservation(gameState, botId)
        botObservations.append(botObs)
        scores.append(score)
    botObservations = np.array(botObservations)
    scores = np.array(scores)

    return GetFieldObservation(gameState), botObservations, scores

def BotRunner(pipe):
    x, y, o = np.random.uniform(0, 1, (NrOfBots,)), np.random.uniform(0, 1, (NrOfBots,)), np.random.uniform(0, math.tau, (NrOfBots,))

    CreateGame()
    SendState(x, y, o)
    StartGame()

    fieldObs, botObs, scores = GetObservation()

    done = False

    while not done:
        previousScores = scores

        pipe.send((fieldObs, botObs))

        # Wait for the action, otherwise automatically stop
        if not pipe.poll(60): return
        commands = pipe.recv()
        # Stop process if -1 is received
        if commands == -1:
            print("stopping process")
            return

        for b,command in enumerate(commands):
            if command == 0:
                # Move forward
                c, s = np.cos(o[b]), np.sin(o[b])
                R = np.array(((c, -s), (s, c)))
                forward = np.matmul(R, [0.0, 0.025])
                x[b] += forward[0] * 0.5
                y[b] += forward[1] * 0.5
            elif command == 1:
                # turn CCW
                o[b] += 3*math.pi/180
            elif command == 2:
                # turn CW
                o[b] -= 3*math.pi/180
            
        outOfArena = (x < 0) | (x > 1) | (y < 0) | (y > 1)
        if np.any(outOfArena):
            print('Warning: One of the bots outside arena')
            rewards = outOfArena * -10
            done = True
        else:
            SendState(x, y, o)
            #sleep(0.1) 

            fieldObs, botObs, scores = GetObservation()
            rewards = scores - previousScores
        pipe.send((rewards, done))

    StopGame()





from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Concatenate


def BuildModel():
    field = Input((10, 10, 28), name='field')
    bots = Input((35,), name='bots')

    f = field
    f = Conv2D(64, 1, strides=1, padding='same', activation='relu', name='conv1a')(f)
    f = Conv2D(64, 3, strides=1, padding='same', activation='relu', name='conv1b')(f)
    f = Conv2D(64, 3, strides=2, padding='same', activation='relu', name='conv1c')(f)

    f = Conv2D(128, 1, strides=1, padding='same', activation='relu', name='conv2a')(f)
    f = Conv2D(128, 3, strides=1, padding='same', activation='relu', name='conv2b')(f)
    f = Conv2D(128, 3, strides=2, padding='same', activation='relu', name='conv2c')(f)

    f = Conv2D(256, 1, strides=1, padding='same', activation='relu', name='conv3a')(f)
    f = Conv2D(256, 3, strides=1, padding='same', activation='relu', name='conv3b')(f)
    f = Conv2D(256, 3, strides=2, padding='same', activation='relu', name='conv3c')(f)

    f = GlobalAveragePooling2D(name='avg')(f)

    b = bots
    b = Dense(128, activation='relu', name='pre1')(b)
    
    h = Concatenate(name='concat')([f, b])
    h = Dense(1024, activation='relu', name='dense1')(h)
    h = Dense(256, activation='relu', name='dense2')(h)
    h = Dense(3, activation='softmax', name='dense3')(h)
    
    output = h
    model = Model([field, bots], output)
    return model

model = BuildModel()
model.summary()



observationsField, observationsBots, actions, rewards, dones = [], [], [], [], []


# Run one episode

pipe_master, pipe_slave = Pipe()
process = Process(target=BotRunner, args=(pipe_slave,))
process.daemon = True
process.start()

done = False
while not done:
    fo, bo = pipe_master.recv()
    fo = np.expand_dims(fo, axis=0)    
    fo = np.repeat(fo, 4, axis=0)
    observationsField.append(fo)
    observationsBots.append(bo)

    commands = model.predict([fo, bo])
    commands = [np.random.choice(len(c), p=c) for c in commands]
    print(f'action: {commands}')
    # back to one-hot
    action = to_categorical(commands, num_classes=3)
    actions.append(action)

    pipe_master.send(commands) 

    reward, done = pipe_master.recv()
    print(f'reward: {reward}')
    rewards.append(reward)
    dones.append(done)

if not done:
    _ = pipe_master.recv()
    print('sending stop')
    pipe_master.send(-1)

process.join()
print('stopped')

observationsField = np.array(observationsField)
observationsBots = np.array(observationsBots)
actions = np.array(actions)
rewards = np.array(rewards)
dones = np.array(dones)

print(observationsField.shape, observationsBots.shape, actions.shape, rewards.shape, dones.shape)
