import socket
import json
from time import sleep
from multiprocessing import Process, Pipe, Queue
import numpy as np
import random
import math
from datetime import datetime
from swoc import SwocEnv
from tensorflow.keras.utils import to_categorical  


### TODO
# - Use radians internally

# bot size is 2 * 0.025 = 0.05 = half a cell with 10 cells 
# item size = 2* widthPerCell / 6 = 1/3th of a cell width
# items are always in the center of a cell. only one item per cell -> one-hot encode


NrOfBots = 8
NrOfOtherBots = 8
LearningRate = 1e-3
AdvantageGamma = 0.99
NrOfEpochs = 10000
WeightsFile = './model_play/weights'
BatchSize = 1024
MaxEpisodeLength = 1100
EpsilonDecay = 0.99
UseEpsilonGreedy = False

epsilon = 0.6


        

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
    field = Input((10, 10, 11), name='field')
    bots = Input((32,), name='bots')

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


def CombineObservations(fieldObs, botObs):
    fieldObservations = np.repeat([fieldObs], 8, axis=0)

    botObservations = []
    allBots = np.arange(len(botObs))
    for b,bot in enumerate(botObs):
        indices = np.insert(np.delete(allBots, b), 0, b)
        botObservations.append(botObs[indices].flatten())
    botObservations = np.array(botObservations)

    return fieldObservations, botObservations


env = SwocEnv('localhost', nrOfBots=NrOfBots)


epoch = 0
print(f'Epoch #{epoch+1} (epsilon {epsilon:.4f})', end='')
observationsField, observationsBots, actions, rewards, dones = [], [], [], [], []
batchSize = 0
epochStart = datetime.now()

while epoch < NrOfEpochs:
    # Run one episode
    (fieldObs, botObs), done = env.reset(), False
    fieldObs, botObs = CombineObservations(fieldObs, botObs)

    episodeLength = 0
    while not done and epoch < NrOfEpochs:
        print('.', end='')

        # Store current observation
        observationsField.append(fieldObs)
        observationsBots.append(botObs)

        #print(fieldObs.shape, botObs.shape)

        # Select, perform action
        probs = model_play.predict([fieldObs, botObs])
        action = SelectAction(probs)
        #print(f'action: {action}')
        (fieldObs, botObs), reward, done = env.step(action)
        fieldObs, botObs = CombineObservations(fieldObs, botObs)
        # back to one-hot and store
        action = to_categorical(action, num_classes=3)
        actions.append(action)

        # Store reward
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

            # # Train
            # observationsField = np.array(observationsField).astype(np.float)
            # observationsBots = np.array(observationsBots).astype(np.float)
            # actions = np.array(actions).astype(np.float)
            # rewards = np.array(rewards).astype(np.float)
            # dones = np.array(dones).astype(np.float)

            # TrainOnBatch(observationsField, observationsBots, actions, rewards, dones)

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
    else:
        print('!', end='')
