import socket
import json
from time import sleep
from multiprocessing import Process, Pipe, Queue
import numpy as np
import random
import math
from datetime import datetime


### TODO
# - Use radians internally

# bot size is 2 * 0.025 = 0.05 = half a cell with 10 cells 
# item size = 2* widthPerCell / 6 = 1/3th of a cell width
# items are always in the center of a cell. only one item per cell -> one-hot encode


ServerAddress = 'localhost'
NrOfBots = 8
NrOfOtherBots = 8
LearningRate = 1e-3
AdvantageGamma = 0.99
NrOfEpochs = 10000
WeightsFile = './model_play/weights'
BatchSize = 1024
MaxEpisodeLength = 1100
MaxCommandRepeat = 100
OutOfArenaReward = -1
TooManyRepeatsReward = -1
EpsilonDecay = 0.99
UseEpsilonGreedy = False

epsilon = 0.6




def SendRemoteControllerCommand(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ServerAddress, 9935))
        serialized = json.dumps(command)
        s.sendall(serialized.encode('utf-8'))
    sleep(0.1)

def CreateGame():
    #print('Creating game')
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
    #print('Starting game')
    command = {'commandType': 'startGame'}
    SendRemoteControllerCommand(command)

def StopGame():
    #print('Stopping game')
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

    lastBotCommand = np.array([-1] * NrOfBots)
    lastBotCommandCount = np.zeros((NrOfBots,))

    while not done:
        previousScores = scores

        pipe.send((fieldObs, botObs))

        # Wait for the action, otherwise automatically stop
        if not pipe.poll(60): return
        commands = pipe.recv()
        # Stop process if -1 is received
        if commands is None:
            #print("stopping process")
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

            # Remember how many times this action was performed
            if command == lastBotCommand[b]:
                lastBotCommandCount[b] += 1
            else:
                lastBotCommandCount[b] = 0
            lastBotCommand[b] = command

        outOfArena = (x < 0) | (x > 1) | (y < 0) | (y > 1)
        if np.any(outOfArena):
            #print('Warning: One of the bots outside arena')
            rewards = outOfArena * OutOfArenaReward
            done = True
        else:
            SendState(x, y, o)
            #sleep(0.1) 

            fieldObs, botObs, scores = GetObservation()
            rewards = scores - previousScores
            
            ## Increase weight of positive rewards
            #rewards = rewards + (((rewards > 0) * rewards) * 4)

            # Punish if same action was performed too many times
            tooManyRepeats = (lastBotCommandCount > MaxCommandRepeat)
            if np.any(tooManyRepeats):
                rewards += tooManyRepeats * TooManyRepeatsReward
        
        pipe.send((rewards, done))

    StopGame()



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


def CategoricalVanillaPolicyGradientLoss(advantage):
    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        crossentropy = K.sum(y_true*K.log(y_pred), axis=1, keepdims=True)
        return -K.mean(crossentropy * advantage)
    return loss_fn

def BinaryVanillaPolicyGradientLoss(advantage):
    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        crossentropy = y_true*K.log(y_pred) + (1-y_true)*K.log(1-y_pred)
        return -K.mean(crossentropy * advantage)
    return loss_fn


def BuildModel():
    field = Input((10, 10, 8), name='field')
    bots = Input((35,), name='bots')
num_cl
    f = field
    f = Conv2D(32, 1, strides=1, padding='same', activation='elu', name='conv1a')(f)
    f = Conv2D(32, 3, strides=1, padding='same', activation='elu', name='conv1b')(f)
    f = Conv2D(32, 3, strides=2, padding='same', activation='elu', name='conv1c')(f)

    f = Conv2D(64, 1, strides=1, padding='same', activation='elu', name='conv2a')(f)
    f = Conv2D(64, 3, strides=1, padding='same', activation='elu', name='conv2b')(f)
    f = Conv2D(64, 3, strides=2, padding='same', activation='elu', name='conv2c')(f)

    f = Conv2D(128, 1, strides=1, padding='same', activation='elu', name='conv3a')(f)
    f = Conv2D(128, 3, strides=1, padding='same', activation='elu', name='conv3b')(f)
    f = Conv2D(128, 3, strides=2, padding='same', activation='elu', name='conv3c')(f)

    f = GlobalAveragePooling2D(name='avg')(f)

    b = bots
    b = Dense(128, activation='elu', name='pre1')(b)
    
    h = Concatenate(name='concat')([f, b])
    h = Dense(512, activation='elu', name='dense1')(h)
    h = Dense(128, activation='elu', name='dense2')(h)
    h = Dense(3, activation='softmax', name='dense3')(h)
    
    output = h
    model_play = Model([field, bots], output)


    advantage = Input((1,), name='advantage')
    model_train = Model([field, bots, advantage], output)
    model_train.compile(loss=CategoricalVanillaPolicyGradientLoss(advantage), optimizer=Adam(lr=LearningRate))

    return model_play, model_train


def Discount(values, notDones, discountFactor):
    discountedValues = np.zeros_like(values)
    currentValue = 0
    for i in reversed(range(len(values))):
        currentValue = values[i] + notDones[i] * discountFactor * currentValue
        discountedValues[i] = currentValue

    return discountedValues


def SelectAction(probs):
    global epsilon
    if UseEpsilonGreedy:
        if np.random.uniform() < epsilon:
            return [np.random.choice(len(p)) for p in probs]
        else:
            return np.argmax(probs, axis=1)
    else:
        return [np.random.choice(len(p), p=p) for p in probs]


model_play, model_train = BuildModel()
model_play.summary()
try:
    model_play.load_weights(WeightsFile)
    print(f'Weights loaded from {WeightsFile}')
except:
    # No weights loaded, so remove history
    with open('rewards.log', 'w+') as file:
        file.write('')
    pass

def TrainOnBatch(observationsField, observationsBots, actions, rewards, dones):
    # Save current weights, if this training fails, then we still have a backup
    model_play.save_weights(WeightsFile)

    totalRewards = np.sum(rewards, axis=0)
    with open('rewards.log', 'a+') as file:
        for r in totalRewards:
            file.write(f'{int(r)}\n')
    print(f'total reward {np.sum(totalRewards)}')

    notDones = np.logical_not(dones)

    # Discount rewards over time
    rewards = Discount(rewards, notDones, AdvantageGamma)

    # Normalize rewards
    rewards -= np.mean(rewards)
    rewardSigma = np.std(rewards)
    if rewardSigma > 1e-7:
        rewards /= rewardSigma

    # Reshape into one giant batch
    observationsField = observationsField.reshape((-1,)+observationsField.shape[2:])
    observationsBots = observationsBots.reshape((-1,)+observationsBots.shape[2:])
    actions = actions.reshape((-1,)+actions.shape[2:])
    rewards = rewards.reshape((-1,)+rewards.shape[2:])
    dones = dones.reshape((-1,)+dones.shape[2:])
    #print(observationsField.shape, observationsBots.shape, actions.shape, rewards.shape, dones.shape)

    # Train
    loss = model_train.train_on_batch([observationsField, observationsBots, rewards], actions)

epoch = 0
print(f'Epoch #{epoch+1} (epsilon {epsilon:.4f})', end='')
observationsField, observationsBots, actions, rewards, dones = [], [], [], [], []
batchSize = 0
epochStart = datetime.now()

while epoch < NrOfEpochs:
    # Run one episode
    pipe_master, pipe_slave = Pipe()
    process = Process(target=BotRunner, args=(pipe_slave,))
    process.daemon = True
    process.start()

    episodeLength = 0
    done = False
    while not done and epoch < NrOfEpochs:
        print('.', end='')

        # Get and store current observation
        fo, bo = pipe_master.recv()
        fo = np.expand_dims(fo, axis=0)    
        fo = np.repeat(fo, NrOfBots, axis=0)
        observationsField.append(fo)
        observationsBots.append(bo)

        # Select, perform and store action
        probs = model_play.predict([fo, bo])
        action = SelectAction(probs)
        #print(f'action: {action}')
        pipe_master.send(action) 
        # back to one-hot
        action = to_categorical(action, num_classes=3)
        actions.append(action)

        # Get and store reward
        reward, done = pipe_master.recv()
        #print(f'reward: {reward}')
        rewards.append(reward)
        dones.append([done] * NrOfBots)

        # Next iteration
        batchSize += 1
        episodeLength += 1

        # Train when batch is large enough
        if batchSize >= BatchSize:
            epochDuration = (datetime.now() - epochStart).total_seconds()
            print(f'finished ({epochDuration:.1f})')

            # Train
            observationsField = np.array(observationsField).astype(np.float)
            observationsBots = np.array(observationsBots).astype(np.float)
            actions = np.array(actions).astype(np.float)
            rewards = np.array(rewards).astype(np.float)
            dones = np.array(dones).astype(np.float)

            TrainOnBatch(observationsField, observationsBots, actions, rewards, dones)

            # New epoch
            epoch += 1
            epsilon = EpsilonDecay * epsilon
            print(f'Epoch #{epoch+1} (epsilon {epsilon:.4f})', end='')
            observationsField, observationsBots, actions, rewards, dones = [], [], [], [], []
            batchSize = 0
            epochStart = datetime.now()

        # Stop when episode is too long
        if episodeLength > MaxEpisodeLength:
            break

    if not done:
        print('X', end='')
        _ = pipe_master.recv()
        pipe_master.send(None)
    else:
        print('!', end='')

    process.join()
