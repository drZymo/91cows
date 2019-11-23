from swoc import SwocEnv
from draw import DrawObservation
import pyglet
from pyglet.window import key

env = SwocEnv(nrOfBots=1)

width, height = 800, 800
window = pyglet.window.Window(width=width, height=height)

obs, totalReward, done  = env.reset(10, 10), 0, False
rawImage = DrawObservation(obs, width, height)

image = pyglet.image.ImageData(width, height, 'L', rawImage.tobytes(), pitch=-width)

@window.event
def on_draw():
    image.blit(0, 0, 0)

def update(dt):
    rawImage = DrawObservation(obs, width, height)
    image.set_data(fmt='L', data=rawImage.tobytes(), pitch=-width)

@window.event
def on_key_press(symbol, modifiers):
    global obs, reward, done, totalReward, index
    if done:
        obs, totalReward, done  = env.reset(), 0, False
    if symbol == key.UP:
        obs, reward, done = env.step([0])
        totalReward += reward[0]
    elif symbol == key.RIGHT:
        obs, reward, done = env.step([1])
        totalReward += reward[0]
    elif symbol == key.LEFT:
        obs, reward, done = env.step([2])
        totalReward += reward[0]

pyglet.clock.schedule_interval(update, 0.05)

pyglet.app.run()
