import numpy as np
from swoc import SwocEnv
from draw import DrawObservation
import pyglet
from pyglet.window import key
from pathlib import Path

env = SwocEnv(nrOfBots=1)

outputDir = Path('sessions')

# Continue from last file, if present
index = 0
prevRecords = sorted([p.name for p in outputDir.glob('record*.npy')])
if prevRecords:
    lastRecord = str(prevRecords[-1])
    index = int(lastRecord[6:-4]) + 1
    print(f'Starting from {index}')


width, height = 800, 800
window = pyglet.window.Window(width=width, height=height)

obs, totalReward, done  = env.reset(), 0, False
rawImage = DrawObservation(obs, width, height)

image = pyglet.image.ImageData(width, height, 'L', rawImage, pitch=-width)

def Save(obs, action):
    global index
    outputFile = outputDir / f'record{index:05d}.npy'
    np.save(outputFile, [obs, action])
    index += 1

@window.event
def on_draw():
    image.blit(0, 0, 0)

def update(dt):
    rawImage = DrawObservation(obs, width, height)
    image.set_data(fmt='L', data=rawImage, pitch=-width)

@window.event
def on_key_press(symbol, modifiers):
    global obs, reward, done, totalReward, index
    if done:
        obs, totalReward, done  = env.reset(), 0, False
    if symbol == key.UP:
        Save(obs, 0)
        obs, reward, done = env.step([0])
        totalReward += reward[0]
    elif symbol == key.RIGHT:
        for _ in range(30 if modifiers & key.MOD_CTRL else 1):
            Save(obs, 1)
            obs, reward, done = env.step([1])
            totalReward += reward[0]
    elif symbol == key.LEFT:
        for _ in range(30 if modifiers & key.MOD_CTRL else 1):
            Save(obs, 2)
            obs, reward, done = env.step([2])
            totalReward += reward[0]

pyglet.clock.schedule_interval(update, 0.05)

pyglet.app.run()
