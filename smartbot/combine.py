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
from pathlib import Path


NrOfBots = 1
MaxNrOfBots = 8
LearningRate = 1e-5
WeightsFile = './model_play/weights'


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


def BuildModel():
    field = Input((10, 10, 11), name='field')
    bots = Input((4*MaxNrOfBots,), name='bots')

    f = Flatten(name='flatten')(field)
    h = Concatenate(name='concat')([f, bots])
    h = Dense(2048, activation='elu', kernel_initializer='he_uniform', name='dense1')(h)
    h = Dense(2048, activation='elu', kernel_initializer='he_uniform', name='dense2')(h)
    h = Dense(1024, activation='elu', kernel_initializer='he_uniform', name='dense3')(h)
    h = Dense(1024, activation='elu', kernel_initializer='he_uniform', name='dense4')(h)
    h = Dense(1024, activation='elu', kernel_initializer='he_uniform', name='dense5')(h)
    h = Dense(1024, activation='elu', kernel_initializer='he_uniform', name='dense6')(h)
    h = Dense(512, activation='elu', kernel_initializer='he_uniform', name='dense7')(h)
    h = Dense(3, activation='softmax', name='dense8')(h)
    
    action = h
    model_play = Model([field, bots], action)

    model_play.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LearningRate))

    return model_play



model_play = BuildModel()
model_play.summary()
try:
    model_play.load_weights(WeightsFile)
    print(f'Weights loaded from {WeightsFile}')
except:
    pass


def CombineObservations(fieldObs, botObs):
    fieldObservations = np.repeat([fieldObs], NrOfBots, axis=0)

    botObservations = []
    allBots = np.arange(len(botObs))
    for b,bot in enumerate(botObs):
        indices = np.insert(np.delete(allBots, b), 0, b)
        newObs = botObs[indices].flatten()
        newObs = np.pad(newObs, (0, (4*MaxNrOfBots)-len(newObs)), 'constant', constant_values=0)
        botObservations.append(newObs)
    botObservations = np.array(botObservations)

    return fieldObservations, botObservations


observationsFieldFilename = Path('sessions/observationsField.npy')
observationsBotsFilename = Path('sessions/observationsBots.npy')
actionsFilename = Path('sessions/actions.npy')

if observationsFieldFilename.exists() and observationsBotsFilename.exists() and actionsFilename.exists():
    observationsField = np.load(observationsFieldFilename)
    observationsBots = np.load(observationsBotsFilename)
    actions = np.load(actionsFilename)
else:
    observationsField, observationsBots, actions = [], [], []
    for sessionDir in Path('sessions').glob('session*'):
       for filename in sessionDir.glob('*.npy'):
           (fieldObs, botObs), action = np.load(filename, allow_pickle=True)
           fieldObs, botObs = CombineObservations(fieldObs, botObs)
           action = to_categorical(action, num_classes=3)
    
           observationsField.append(fieldObs)
           observationsBots.append(botObs)
           actions.append(action)
    
    observationsField = np.vstack(observationsField).astype(np.float)
    observationsBots = np.vstack(observationsBots).astype(np.float)
    actions = np.array(actions).astype(np.float)
    
    np.save(observationsFieldFilename, observationsField)
    np.save(observationsBotsFilename, observationsBots)
    np.save(actionsFilename, actions)

I = np.random.permutation(observationsField.shape[0])
observationsField = observationsField[I]
observationsBots = observationsBots[I]
actions = actions[I]

# Train
print(observationsField.shape, observationsBots.shape, actions.shape)
model_play.fit([observationsField, observationsBots], actions, validation_split=0.2, shuffle=True, epochs=50, verbose=2, batch_size=4096)

model_play.save_weights(WeightsFile)
