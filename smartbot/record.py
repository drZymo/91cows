from swoc import SwocEnv
from draw import DrawObservation
import pyglet
from pyglet.window import key

env = SwocEnv(nrOfBots=1)

index = 0

width, height = 800, 800
window = pyglet.window.Window(width=width, height=height)

obs, totalReward, done  = env.reset(), 0, False
rawImage = DrawObservation(obs, width, height)

image = pyglet.image.ImageData(width, height, 'L', rawImage, pitch=-width)

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
        np.save(f'record{index:5d}.npy', [obs, 0])
        obs, reward, done = env.step([0])
        totalReward += reward[0]
        index += 1
    elif symbol == key.LEFT:
        for _ in range(10 if modifiers & key.MOD_CTRL else 1):
            np.save(f'record{index:5d}.npy', [obs, 1])
            obs, reward, done = env.step([1])
            totalReward += reward[0]
            index += 1
    elif symbol == key.RIGHT:
        for _ in range(10 if modifiers & key.MOD_CTRL else 1):
            np.save(f'record{index:5d}.npy', [obs, 2])
            obs, reward, done = env.step([2])
            totalReward += reward[0]
            index += 1

pyglet.clock.schedule_interval(update, 0.05)

pyglet.app.run()
